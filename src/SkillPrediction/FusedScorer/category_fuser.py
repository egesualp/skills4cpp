"""
bayesian_fuser.py - Bayesian Re-ranking for Job-to-Skill Mapping

Combines IR similarity scores with predicted category probabilities using:
    Bayesian fusion: score_final = similarity_ir × (P_category | job)^w
    Linear fusion:   score_final = (1 - alpha) × similarity_ir + alpha × P_category

Features:
    - Temperature scaling: Apply temperature T to category logits
    - Max-prob threshold: Skip category weighting if max(P) < threshold (flat distribution)
    - Two fusion modes: 'bayesian' (multiplicative) or 'linear' (additive)
    - Joint optimization over multiple hyperparameters

Usage:
    # Bayesian fusion with temperature scaling
    python -m skill_mapping.v2.bayesian_fuser \
        --similarity_scores_json outputs/similarity_scores.json \
        --category_scores_json outputs/category_scores.json \
        --skill_hierarchy_csv data/processed/master_datasets_2/master_skill_complete_hierarchy.csv \
        --jobs_csv data/processed/augmentation/augmented_decorte_occupations_with_desc.csv \
        --output_dir outputs/bayesian_fusion \
        --top_k 100 \
        --fusion_mode bayesian \
        --weights 0.0,0.5,1.0,1.5,2.0 \
        --temperatures 0.1,0.2,0.5,1.0 \
        --thresholds 0.0,0.1,0.2,0.3
    
    # Linear fusion
    python -m skill_mapping.v2.bayesian_fuser \
        --fusion_mode linear \
        --alphas 0.0,0.1,0.3,0.5,0.7 \
        --temperatures 0.1,0.2,0.5,1.0 \
        ...
"""

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Literal, Set, Tuple, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger
from scipy.special import softmax

from .metrics_utils import compute_precision_at_k, compute_recall_at_k, compute_map, compute_mrr


@dataclass
class FusionConfig:
    """Configuration for fusion hyperparameters."""
    fusion_mode: Literal["bayesian", "linear"] = "bayesian"
    weight: float = 1.0  # w for Bayesian fusion
    alpha: float = 0.5   # alpha for linear fusion
    temperature: float = 1.0  # Temperature for scaling
    threshold: float = 0.0   # Max-prob threshold (skip if max(P) < threshold)
    aggregate_mode: Literal["max", "sum"] = "max" # How to aggregate category probabilities for a skill


def load_similarity_scores(json_path: str | Path) -> Dict[str, List[Dict]]:
    """
    Load IR similarity scores from JSON.
    
    Returns:
        Dict mapping job_id -> list of {skill_uri, score, rank}
    """
    logger.info(f"Loading similarity scores from {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    logger.info(f"Loaded similarity scores for {len(data)} jobs")
    return data


def load_category_scores_raw(json_path: str | Path) -> Dict[str, List[Dict]]:
    """
    Load raw category scores from JSON (preserving order for temperature scaling).
    
    Returns:
        Dict mapping job_id -> list of {category, score}
    """
    logger.info(f"Loading category scores from {json_path}")
    with open(json_path, "r") as f:
        raw_data = json.load(f)
    logger.info(f"Loaded category scores for {len(raw_data)} jobs")
    return raw_data


def apply_temperature_scaling(
    category_scores_raw: Dict[str, List[Dict]],
    temperature: float = 1.0,
) -> Dict[str, Dict[str, float]]:
    """
    Apply temperature scaling to category logits/scores.
    
    Temperature scaling adjusts the sharpness of the probability distribution:
    - T < 1: Sharper distribution (more confident)
    - T = 1: Original distribution
    - T > 1: Flatter distribution (less confident)
    
    Formula: P_i = exp(logit_i / T) / sum(exp(logit_j / T))
    
    Note: If input scores are already probabilities, we convert to log-space first.
    
    Args:
        category_scores_raw: {job_id: [{category, score}, ...]}
        temperature: Temperature parameter T
    
    Returns:
        Dict mapping job_id -> {category_name: scaled_probability}
    """
    scaled_data = {}
    
    for job_id, cat_list in category_scores_raw.items():
        categories = [item["category"] for item in cat_list]
        scores = np.array([item["score"] for item in cat_list])
        
        # Check if scores look like probabilities (sum ≈ 1) or logits
        score_sum = scores.sum()
        if 0.99 < score_sum < 1.01 and np.all(scores >= 0):
            # Scores are probabilities - convert to log-space (logits)
            # Add small epsilon to avoid log(0)
            logits = np.log(np.maximum(scores, 1e-10))
        else:
            # Scores are already logits
            logits = scores
        
        # Apply temperature scaling and softmax
        if temperature > 0:
            scaled_logits = logits / temperature
        else:
            # T=0 means argmax (one-hot)
            scaled_logits = np.zeros_like(logits)
            scaled_logits[np.argmax(logits)] = 1e10
        
        scaled_probs = softmax(scaled_logits)
        
        scaled_data[job_id] = {cat: float(prob) for cat, prob in zip(categories, scaled_probs)}
    
    return scaled_data


def build_skill_to_categories(csv_path: str | Path) -> Dict[str, Set[str]]:
    """
    Build skill URI to Level 1 category mapping from hierarchy CSV.
    
    Note: A skill can belong to multiple categories (multi-parent structure).
    
    Returns:
        Dict mapping skill_uri -> set of level1_label categories
    """
    logger.info(f"Loading skill hierarchy from {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Build mapping: skill_uri -> set of level1_labels
    skill_to_cats = defaultdict(set)
    
    for _, row in df.iterrows():
        skill_uri = row["skillUri"]
        level1_label = row.get("level1_label")
        if pd.notna(level1_label) and level1_label:
            # Normalize category name to lowercase for matching
            skill_to_cats[skill_uri].add(level1_label.lower().strip())
    
    logger.info(f"Built category mapping for {len(skill_to_cats)} skills")
    
    # Log category distribution
    all_cats = set()
    for cats in skill_to_cats.values():
        all_cats.update(cats)
    logger.info(f"Found {len(all_cats)} unique Level 1 categories")
    
    return dict(skill_to_cats)


def load_skill_relations(csv_path: str | Path) -> Dict[str, Set[str]]:
    """
    Load occupation-skill relations for evaluation.
    Returns: dict {occupation_uri: {skill_uri, ...}}
    """
    logger.info(f"Loading skill relations from {csv_path}")
    df = pd.read_csv(csv_path)
    
    if "occupationUri" not in df.columns or "skillUri" not in df.columns:
        logger.warning(f"Relation CSV missing columns. Available: {df.columns.tolist()}")
        return {}
    
    relations = {}
    for occ, group in df.groupby("occupationUri"):
        relations[occ] = set(group["skillUri"].tolist())
    
    logger.info(f"Loaded relations for {len(relations)} occupations")
    return relations


def load_jobs(
    csv_path: str | Path,
    id_column: str = "job_id",
    esco_id_column: str = "esco_id",
) -> Tuple[List[str], List[str], Dict[str, str]]:
    """
    Load job IDs and their ESCO occupation URIs for ground truth evaluation.
    Also loads 'split' column if available.
    
    Returns:
        job_ids: List of job identifiers (as strings)
        esco_ids: List of ESCO occupation URIs
        splits: Dict mapping job_id -> split name (e.g., 'train', 'test')
    """
    logger.info(f"Loading jobs from {csv_path}")
    df = pd.read_csv(csv_path)
    # Ensure ID is string
    df[id_column] = df[id_column].astype(str)
    
    df = df.drop_duplicates(subset=['raw_title', 'raw_description', 'esco_id', id_column], keep='first')
    
    job_ids = df[id_column].tolist()
    esco_ids = df[esco_id_column].astype(str).tolist() if esco_id_column in df.columns else [""] * len(df)
    
    splits = {}
    if "split" in df.columns:
        # Normalize split names to lowercase
        splits = dict(zip(df[id_column], df["split"].astype(str).str.lower()))
        logger.info(f"Found splits: {df['split'].value_counts().to_dict()}")
    else:
        logger.warning("No 'split' column found. Using all data.")
    
    logger.info(f"Loaded {len(job_ids)} jobs")
    return job_ids, esco_ids, splits


def load_task_a_scores(json_path: str | Path) -> Dict[str, List[str]]:
    """
    Load Task A occupation predictions.
    Returns: Dict[job_id, list_of_occupation_uris]
    """
    logger.info(f"Loading Task A scores from {json_path}")
    path = Path(json_path)
    
    # Helper to parse lines
    def parse_jsonl(f) -> Dict[str, List[str]]:
        data_map = {}
        for i, line in enumerate(f):
            if not line.strip(): continue
            try:
                item = json.loads(line)
                # Use line index as job_id, assuming 1-to-1 mapping with input CSV 0-indexed IDs
                # Checks for v5 format (predicted_esco_ids) or potential other formats
                if isinstance(item, dict):
                    preds = item.get('predicted_esco_ids', [])
                    if not preds and 'predictions' in item:
                        preds = item['predictions']
                    data_map[str(i)] = preds
                else:
                    # Fallback if line is not a dict? Unlikely for JSONL
                    pass
            except json.JSONDecodeError:
                logger.warning(f"Failed to decode JSON line {i+1}")
        return data_map

    # Try loading based on extension or content
    try:
        if path.suffix == '.jsonl':
            with open(path, 'r') as f:
                data = parse_jsonl(f)
            logger.info(f"Loaded Task A scores for {len(data)} jobs (JSONL mode)")
            return data
            
        # Try standard JSON load
        with open(path, 'r') as f:
            try:
                data = json.load(f)
                logger.info(f"Loaded Task A scores for {len(data)} jobs (JSON mode)")
                return data
            except json.JSONDecodeError:
                # Fallback to JSONL
                logger.info("JSON load failed, retrying as JSONL...")
                f.seek(0)
                data = parse_jsonl(f)
                logger.info(f"Loaded Task A scores for {len(data)} jobs (JSONL fallback)")
                return data
                
    except Exception as e:
        logger.error(f"Failed to load Task A scores: {e}")
        raise e


def compute_fused_scores(
    similarity_scores: Dict[str, List[Dict]],
    category_scores: Dict[str, Dict[str, float]],
    skill_to_categories: Dict[str, Set[str]],
    config: FusionConfig,
    top_k: int,
    aggregate_mode: str = "max",
    task_a_scores: Optional[Dict[str, List[str]]] = None,
    skill_relations: Optional[Dict[str, Set[str]]] = None,
    task_a_k: int = 5,
) -> Tuple[Dict[str, List[Dict]], Dict[str, bool]]:
    """
    Apply fusion of IR scores with category probabilities.
    
    Supports two fusion modes:
    - Bayesian: score_final = sim_ir × (P_category)^w
    - Linear:   score_final = (1 - alpha) × sim_ir + alpha × P_category
    
    Features:
    - Max-prob threshold: If max(P) < threshold, skip category weighting
    - Temperature scaling should be applied beforehand via apply_temperature_scaling()
    
    For skills with multiple categories, we take the MAX probability across
    all categories the skill belongs to.
    
    Args:
        similarity_scores: {job_id: [{skill_uri, score, rank}, ...]}
        category_scores: {job_id: {category: probability}} (already temperature-scaled)
        skill_to_categories: {skill_uri: {category, ...}}
        config: FusionConfig with fusion_mode, weight, alpha, threshold
        top_k: Number of top skills to return after re-ranking
    
    Returns:
        Tuple of:
        - Re-ranked results: {job_id: [{skill_uri, score, rank}, ...]}
        - Threshold mask: {job_id: bool} indicating if category weighting was applied
    """
    fused_results = {}
    threshold_mask = {}  # Track which jobs had category weighting applied
    
    for job_id, skills in similarity_scores.items():
        job_cat_probs = category_scores.get(job_id, {})
        
        # Normalize category names in job_cat_probs to lowercase
        job_cat_probs_lower = {k.lower().strip(): v for k, v in job_cat_probs.items()}
        
        # Check max-prob threshold
        max_job_prob = max(job_cat_probs_lower.values()) if job_cat_probs_lower else 0.0
        use_category_weighting = max_job_prob >= config.threshold
        threshold_mask[job_id] = use_category_weighting
        
        # Task A filtering logic
        candidate_filter = None
        if task_a_scores is not None and skill_relations is not None:
            occs = task_a_scores.get(job_id, [])[:task_a_k]
            candidates = set()
            for occ in occs:
                if occ in skill_relations:
                    candidates.update(skill_relations[occ])
            
            if candidates:
                candidate_filter = candidates
            # Else fallback -> candidate_filter is None (allow all)

        skill_fused = []
        for skill_info in skills:
            skill_uri = skill_info["skill_uri"]
            
            # Apply filtering if active
            if candidate_filter is not None and skill_uri not in candidate_filter:
                continue
                
            sim_score = skill_info["score"]
            
            # Get categories for this skill
            skill_cats = skill_to_categories.get(skill_uri, set())
            
            # Determine if we should apply category weighting
            should_weight = use_category_weighting and skill_cats
            
            if should_weight:
                # Get category probabilities for this skill
                cat_probs = [
                    job_cat_probs_lower.get(cat, 0.0)
                    for cat in skill_cats
                ]
                
                if not cat_probs:
                    # No categories found for this skill, or all had 0.0 probability
                    category_boost_value = 0.0
                elif aggregate_mode == "max":
                    category_boost_value = max(cat_probs)
                elif aggregate_mode == "sum":
                    category_boost_value = sum(cat_probs)
                else:
                    raise ValueError(f"Unknown aggregate mode: {aggregate_mode}")

                if config.fusion_mode == "bayesian":
                    # Bayesian fusion: sim × P^w
                    if config.weight > 0:
                        # Avoid zero probabilities causing issues
                        category_boost_value = max(category_boost_value, 1e-10)
                        fused_score = sim_score * (category_boost_value ** config.weight)
                    else:
                        fused_score = sim_score
                        
                elif config.fusion_mode == "linear":
                    # Linear fusion: (1 - alpha) × sim + alpha × P
                    fused_score = (1 - config.alpha) * sim_score + config.alpha * category_boost_value
                else:
                    raise ValueError(f"Unknown fusion mode: {config.fusion_mode}")
            else:
                # No category weighting: use raw similarity
                fused_score = sim_score
            
            skill_fused.append({
                "skill_uri": skill_uri,
                "score": fused_score,
                "original_score": sim_score,
            })
        
        # Sort by fused score descending and take top_k
        skill_fused.sort(key=lambda x: x["score"], reverse=True)
        skill_fused = skill_fused[:top_k]
        
        # Assign new ranks
        for rank, item in enumerate(skill_fused, start=1):
            item["rank"] = rank
        
        fused_results[job_id] = skill_fused
    
    return fused_results, threshold_mask


# =============================================================================
# VECTORIZED IMPLEMENTATION FOR FAST GRID SEARCH
# =============================================================================

@dataclass
class VectorizedData:
    """Pre-computed matrices for vectorized fusion operations."""
    job_ids: List[str]                    # [n_jobs] job identifiers
    skill_uris: List[str]                 # [n_skills] skill URIs in order
    similarity_matrix: np.ndarray         # [n_jobs, n_skills] similarity scores
    category_logits: np.ndarray           # [n_jobs, n_categories] raw logits
    skill_category_matrix: np.ndarray     # [n_skills, n_categories] binary mapping
    category_names: List[str]             # [n_categories] category names
    skill_has_category: np.ndarray        # [n_skills] bool - whether skill has any category
    candidate_mask: Optional[np.ndarray] = None  # [n_jobs, n_skills] boolean mask (True=candidate, False=filtered)


def prepare_vectorized_data(
    similarity_scores: Dict[str, List[Dict]],
    category_scores_raw: Dict[str, List[Dict]],
    skill_to_categories: Dict[str, Set[str]],
    task_a_scores: Optional[Dict[str, List[str]]] = None,
    skill_relations: Optional[Dict[str, Set[str]]] = None,
    task_a_k: int = 5,
) -> VectorizedData:
    """
    Convert dict-based data structures to numpy matrices for vectorized operations.
    
    This is done ONCE before grid search, enabling fast matrix operations.
    """
    logger.info("Preparing vectorized data structures...")
    
    # Get ordered lists of jobs and skills
    job_ids = list(similarity_scores.keys())
    n_jobs = len(job_ids)
    
    # Get all unique skills and create a consistent ordering
    all_unique_skills = set()
    for skills_list in similarity_scores.values():
        for skill_info in skills_list:
            all_unique_skills.add(skill_info["skill_uri"])
    skill_uris = sorted(list(all_unique_skills))  # Ensure consistent order
    n_skills = len(skill_uris)
    skill_to_idx = {uri: i for i, uri in enumerate(skill_uris)}
    
    # Get category names from first job's category scores
    first_job_cats = category_scores_raw[job_ids[0]]
    category_names = [item["category"].lower().strip() for item in first_job_cats]
    n_categories = len(category_names)
    cat_to_idx = {cat: i for i, cat in enumerate(category_names)}
    
    logger.info(f"  Jobs: {n_jobs}, Skills: {n_skills}, Categories: {n_categories}")
    
    # Build similarity matrix [n_jobs, n_skills]
    similarity_matrix = np.zeros((n_jobs, n_skills), dtype=np.float32)
    for i, job_id in enumerate(job_ids):
        for skill_info in similarity_scores[job_id]:
            uri = skill_info["skill_uri"]
            if uri in skill_to_idx:
                skill_idx = skill_to_idx[uri]
                similarity_matrix[i, skill_idx] = skill_info["score"]
    
    # Build category logits matrix [n_jobs, n_categories]
    # We store raw scores and apply temperature scaling later
    category_logits = np.zeros((n_jobs, n_categories), dtype=np.float32)
    for i, job_id in enumerate(job_ids):
        cat_list = category_scores_raw[job_id]
        for item in cat_list:
            cat_name = item["category"].lower().strip()
            if cat_name in cat_to_idx:
                category_logits[i, cat_to_idx[cat_name]] = item["score"]
    
    # Check if scores are probabilities or logits
    score_sums = category_logits.sum(axis=1)
    if np.allclose(score_sums, 1.0, atol=0.02) and np.all(category_logits >= 0):
        # Convert probabilities to logits
        logger.info("  Detected probabilities, converting to logits...")
        category_logits = np.log(np.maximum(category_logits, 1e-10))
    
    # Build skill-to-category matrix [n_skills, n_categories]
    # For each skill, mark which categories it belongs to
    skill_category_matrix = np.zeros((n_skills, n_categories), dtype=np.float32)
    skill_has_category = np.zeros(n_skills, dtype=bool)
    
    for skill_uri, categories in skill_to_categories.items():
        if skill_uri in skill_to_idx:
            skill_idx = skill_to_idx[skill_uri]
            for cat in categories:
                cat_lower = cat.lower().strip()
                if cat_lower in cat_to_idx:
                    skill_category_matrix[skill_idx, cat_to_idx[cat_lower]] = 1.0
                    skill_has_category[skill_idx] = True
    
    n_with_cat = skill_has_category.sum()
    logger.info(f"  Skills with category mapping: {n_with_cat}/{n_skills}")
    
    # Build candidate mask for Task A filtering
    candidate_mask = None
    if task_a_scores is not None and skill_relations is not None:
        logger.info(f"  Building candidate mask using Task A (top-k={task_a_k})...")
        candidate_mask = np.zeros((n_jobs, n_skills), dtype=bool)
        n_fallback = 0
        
        for i, job_id in enumerate(job_ids):
            occs = task_a_scores.get(job_id, [])[:task_a_k]
            candidates = set()
            for occ in occs:
                if occ in skill_relations:
                    candidates.update(skill_relations[occ])
            
            # Map candidates to indices
            candidate_indices = []
            for skill_uri in candidates:
                if skill_uri in skill_to_idx:
                    candidate_indices.append(skill_to_idx[skill_uri])
            
            if candidate_indices:
                candidate_mask[i, candidate_indices] = True
            else:
                # Fallback: keep all skills (if no candidates found or job not in task_a_scores)
                candidate_mask[i, :] = True
                n_fallback += 1
                
        logger.info(f"  Candidate mask built. Fallback used for {n_fallback}/{n_jobs} jobs.")

    return VectorizedData(
        job_ids=job_ids,
        skill_uris=skill_uris,
        similarity_matrix=similarity_matrix,
        category_logits=category_logits,
        skill_category_matrix=skill_category_matrix,
        category_names=category_names,
        skill_has_category=skill_has_category,
        candidate_mask=candidate_mask,
    )


def _process_config_chunk(
    config_chunk: List[Tuple],
    vectorized_data: VectorizedData,
    fusion_mode: str,
    top_k: int,
    esco_ids: List[str],
    skill_relations: Dict[str, Set[str]],
    skill_uri_to_idx: Dict[str, int],
    job_batch_size: int = 500,
    aggregate_mode: str = "max",
) -> List[Tuple]:
    """
    Process a chunk of configurations in a single worker process.
    This reduces data serialization overhead by sending data once per worker.
    Uses batched vectorized computation to limit memory usage.
    """
    results = []
    skill_uris = vectorized_data.skill_uris
    
    # Pre-compute gold sets once for this worker
    gold_sets = []
    for esco_id in esco_ids:
        if esco_id and esco_id in skill_relations:
            gold_skill_uris = skill_relations[esco_id]
            gold_indices = {
                skill_uri_to_idx[uri]
                for uri in gold_skill_uris
                if uri in skill_uri_to_idx
            }
            gold_sets.append(gold_indices)
        else:
            gold_sets.append(set())
            
    n_jobs = len(vectorized_data.job_ids)
    
    # Cache for identity mapping results (weight=0 or alpha=0) to avoid re-computation
    cached_identity_result = None
    
    for config_tuple in config_chunk:
        # Check for identity mapping (weight <= 0 or alpha == 0)
        is_identity = False
        if fusion_mode == "bayesian":
            if config_tuple[0] <= 0.0:  # weight
                is_identity = True
        elif fusion_mode == "linear":
            if config_tuple[0] == 0.0:  # alpha
                is_identity = True

        if is_identity and cached_identity_result is not None:
            config_key = make_config_key(fusion_mode, config_tuple)
            metrics, formatted_results = cached_identity_result
            # Use cached metrics and results (fast)
            # Note: category_usage_pct in metrics will correspond to the cached config's T/threshold,
            # which might differ from current config. But since weight=0/alpha=0 makes the fusion
            # output independent of category probs, the exact usage stats are less critical than speed.
            results.append((config_tuple, config_key, metrics, formatted_results))
            continue

        if fusion_mode == "bayesian":
            w, t, thres = config_tuple
            config = FusionConfig(fusion_mode="bayesian", weight=w, temperature=t, threshold=thres, aggregate_mode=aggregate_mode)
        else:
            alpha, t, thres = config_tuple
            config = FusionConfig(fusion_mode="linear", alpha=alpha, temperature=t, threshold=thres, aggregate_mode=aggregate_mode)
        
        config_key = make_config_key(fusion_mode, config_tuple)
        
        # Vectorized fusion computation (memory-efficient batched)
        fused_scores, top_k_indices, top_k_scores, use_weighting = compute_fused_scores_vectorized(
            vectorized_data, config, top_k, job_batch_size=job_batch_size, aggregate_mode=aggregate_mode
        )
        
        # Format results (lightweight)
        formatted_results = {}
        for i, job_id in enumerate(vectorized_data.job_ids):
            skills_list = []
            for rank, (skill_idx, score) in enumerate(zip(top_k_indices[i], top_k_scores[i]), start=1):
                skills_list.append({
                    "skill_uri": skill_uris[skill_idx],
                    "score": round(float(score), 6),
                    "rank": rank,
                })
            formatted_results[job_id] = skills_list
            
        # Metrics
        metrics = {}
        
        # Coverage
        found_in_gold = sum(1 for gs in gold_sets if gs)
        metrics["gold_coverage"] = found_in_gold / len(esco_ids) if esco_ids else 0.0
        
        # Full corpus metrics
        metrics["map_full"] = compute_map(top_k_indices, gold_sets, k=None)
        metrics["mrr_full"] = compute_mrr(top_k_indices, gold_sets, k=None)
        
        # Top-K metrics
        for k in [1, 5, 10, 20, 50, 100]:
            if k > top_k:
                continue
            metrics[f"precision@{k}"] = compute_precision_at_k(top_k_indices, gold_sets, k)
            metrics[f"recall@{k}"] = compute_recall_at_k(top_k_indices, gold_sets, k)
            metrics[f"map@{k}"] = compute_map(top_k_indices, gold_sets, k)
            metrics[f"mrr@{k}"] = compute_mrr(top_k_indices, gold_sets, k)
            
        # Category usage stats
        n_with_cat = int(use_weighting.sum())
        metrics["category_usage_pct"] = 100.0 * n_with_cat / n_jobs
        metrics["n_with_category_weighting"] = n_with_cat
        metrics["n_without_category_weighting"] = n_jobs - n_with_cat
        
        if is_identity and cached_identity_result is None:
            cached_identity_result = (metrics, formatted_results)
        
        results.append((config_tuple, config_key, metrics, formatted_results))
        
    return results


def compute_fused_scores_vectorized(
    data: VectorizedData,
    config: FusionConfig,
    top_k: int,
    job_batch_size: int = 500,
    aggregate_mode: str = "max",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Memory-efficient vectorized fusion using job batching.
    
    Avoids creating the huge [n_jobs, n_skills, n_cats] tensor by processing
    jobs in batches. This reduces peak memory from ~12GB to ~750MB per call.
    
    Formula:
        Bayesian: final = sim × (max_cat_prob)^w
        Linear:   final = (1-α) × sim + α × max_cat_prob
    
    Args:
        data: Pre-computed VectorizedData
        config: FusionConfig with parameters
        top_k: Number of top skills per job
        job_batch_size: Number of jobs to process at once (controls memory usage)
    
    Returns:
        fused_scores: None (not returned to save memory)
        top_k_indices: [n_jobs, top_k] indices of top-k skills per job
        top_k_scores: [n_jobs, top_k] scores of top-k skills
        use_weighting: [n_jobs] bool array - whether category weighting was applied
    """
    n_jobs, n_skills = data.similarity_matrix.shape
    
    # Apply temperature scaling: softmax(logits / T)
    if config.temperature > 0:
        scaled_logits = data.category_logits / config.temperature
    else:
        # T=0: argmax (one-hot)
        scaled_logits = np.zeros_like(data.category_logits)
        max_idx = np.argmax(data.category_logits, axis=1)
        scaled_logits[np.arange(n_jobs), max_idx] = 1e10
    
    # Softmax along category axis (numerically stable)
    scaled_logits = scaled_logits - scaled_logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(scaled_logits)
    category_probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)  # [n_jobs, n_cats]
    
    # Apply threshold: if max job prob < threshold, don't use category weighting
    job_max_probs = category_probs.max(axis=1)  # [n_jobs]
    use_weighting = job_max_probs >= config.threshold  # [n_jobs] bool
    
    # Pre-allocate output arrays
    top_k_indices = np.zeros((n_jobs, top_k), dtype=np.int64)
    top_k_scores = np.zeros((n_jobs, top_k), dtype=np.float32)
    
    # Cache skill_category_matrix expanded (only [1, n_skills, n_cats] - small)
    skill_cat_expanded = data.skill_category_matrix[np.newaxis, :, :]  # [1, n_skills, n_cats]
    
    # Process jobs in batches to avoid huge [n_jobs, n_skills, n_cats] tensor
    for batch_start in range(0, n_jobs, job_batch_size):
        batch_end = min(batch_start + job_batch_size, n_jobs)
        batch_size = batch_end - batch_start
        
        # Get batch data
        batch_probs = category_probs[batch_start:batch_end]  # [batch, n_cats]
        batch_sim = data.similarity_matrix[batch_start:batch_end]  # [batch, n_skills]
        batch_use_weight = use_weighting[batch_start:batch_end]  # [batch]
        
        # Compute max category prob per skill for this batch
        # [batch, 1, n_cats] broadcasting with [1, n_skills, n_cats]
        batch_probs_exp = batch_probs[:, np.newaxis, :]  # [batch, 1, n_cats]
        
        # Mask: where skill has no category membership, set to -inf for max operation
        # or 0 for sum operation
        if aggregate_mode == "max":
            masked_probs = np.where(
                skill_cat_expanded > 0,
                batch_probs_exp,
                -np.inf # For max operation, skills without category should not influence max
            )  # [batch, n_skills, n_cats]
            # Max over categories
            aggregated_cat_probs = np.max(masked_probs, axis=2)  # [batch, n_skills]
            # Handle skills with no categories: set their aggregated_prob to 0 (after max(-inf) is still -inf)
            aggregated_cat_probs = np.where(
                data.skill_has_category[np.newaxis, :],
                aggregated_cat_probs,
                0.0
            )
        elif aggregate_mode == "sum":
            # For sum operation, multiply probabilities by skill_category_matrix
            # then sum over categories. Skills without category membership will be 0.
            aggregated_cat_probs = (batch_probs_exp * skill_cat_expanded).sum(axis=2) # [batch, n_skills]
        else:
            raise ValueError(f"Unknown aggregate mode: {aggregate_mode}")
        
        # Compute fused scores for this batch
        if config.fusion_mode == "bayesian":
            if config.weight > 0:
                safe_probs = np.maximum(aggregated_cat_probs, 1e-10)
                weighted_probs = safe_probs ** config.weight
                fused = batch_sim * weighted_probs
            else:
                fused = batch_sim.copy()
        else:  # linear
            fused = (1 - config.alpha) * batch_sim + config.alpha * aggregated_cat_probs
        # After line 650, before line 652, add:
        # Skills without category mapping should keep original similarity (matching non-vectorized behavior)
        fused = np.where(
            data.skill_has_category[np.newaxis, :],
            fused,
            batch_sim
        )
        # Apply threshold mask: where threshold not met, use original similarity
        fused = np.where(batch_use_weight[:, np.newaxis], fused, batch_sim)
        
        # Apply candidate filtering mask if present
        if data.candidate_mask is not None:
            batch_mask = data.candidate_mask[batch_start:batch_end]
            # Set filtered skills to -inf so they don't appear in top-k
            fused = np.where(batch_mask, fused, -np.inf)
        
        # Get top-k for this batch
        if top_k < n_skills:
            batch_top_k_idx = np.argpartition(-fused, top_k, axis=1)[:, :top_k]
            row_idx = np.arange(batch_size)[:, np.newaxis]
            batch_top_k_scores = fused[row_idx, batch_top_k_idx]
            # Sort within top-k
            sort_order = np.argsort(-batch_top_k_scores, axis=1)
            batch_top_k_idx = np.take_along_axis(batch_top_k_idx, sort_order, axis=1)
            batch_top_k_scores = np.take_along_axis(batch_top_k_scores, sort_order, axis=1)
        else:
            sort_order = np.argsort(-fused, axis=1)
            batch_top_k_idx = sort_order[:, :top_k]
            row_idx = np.arange(batch_size)[:, np.newaxis]
            batch_top_k_scores = fused[row_idx, batch_top_k_idx]
        
        # Store results
        top_k_indices[batch_start:batch_end] = batch_top_k_idx
        top_k_scores[batch_start:batch_end] = batch_top_k_scores
    
    # Return None for fused_scores to save memory (not needed after top-k extraction)
    return None, top_k_indices, top_k_scores, use_weighting


def print_metrics_table(
    all_metrics: Dict[Tuple, Dict[str, float]], 
    top_k: int,
    fusion_mode: str,
):
    """
    Print a formatted comparison table of metrics across different hyperparameter combinations.
    
    Args:
        all_metrics: Dict mapping (param_tuple) -> metrics dict
        top_k: Max k for metrics
        fusion_mode: 'bayesian' or 'linear'
    """
    k_values = [k for k in [5, 10, 20, 50] if k <= top_k]
    
    logger.info("=" * 120)
    logger.info(f"FUSION RESULTS - Joint Grid Search ({fusion_mode.upper()} mode)")
    logger.info("=" * 120)
    
    # Sort by MAP_full descending for easier reading
    sorted_configs = sorted(
        all_metrics.items(), 
        key=lambda x: x[1].get("map_full", 0.0), 
        reverse=True
    )
    
    # Determine header based on fusion mode
    if fusion_mode == "bayesian":
        header = f"{'w':<6} {'T':<6} {'thres':<6}"
    else:
        header = f"{'alpha':<6} {'T':<6} {'thres':<6}"
    
    for k in k_values:
        header += f" {'R@'+str(k):<8}"
    header += f" {'MAP_full':<10} {'MRR_full':<10} {'%Cat':<8}"
    logger.info(header)
    logger.info("-" * 120)
    
    # Show top 20 configurations
    for config_tuple, m in sorted_configs[:20]:
        if fusion_mode == "bayesian":
            w, t, thres = config_tuple
            row = f"{w:<6.2f} {t:<6.2f} {thres:<6.2f}"
        else:
            alpha, t, thres = config_tuple
            row = f"{alpha:<6.2f} {t:<6.2f} {thres:<6.2f}"
        
        for k in k_values:
            rec = m.get(f"recall@{k}", 0.0)
            row += f" {rec:<8.4f}"
        
        cat_usage = m.get("category_usage_pct", 0.0)
        row += f" {m.get('map_full', 0.0):<10.4f} {m.get('mrr_full', 0.0):<10.4f} {cat_usage:<8.1f}"
        logger.info(row)
    
    if len(sorted_configs) > 20:
        logger.info(f"... ({len(sorted_configs) - 20} more configurations not shown)")
    
    logger.info("=" * 120)
    
    # Report best configuration
    best_config, best_metrics = sorted_configs[0]
    if fusion_mode == "bayesian":
        w, t, thres = best_config
        logger.info(f"BEST CONFIG: w={w:.2f}, T={t:.2f}, threshold={thres:.2f}")
    else:
        alpha, t, thres = best_config
        logger.info(f"BEST CONFIG: alpha={alpha:.2f}, T={t:.2f}, threshold={thres:.2f}")
    
    logger.info(f"  MAP_full: {best_metrics['map_full']:.4f}")
    logger.info(f"  MRR_full: {best_metrics['mrr_full']:.4f}")
    logger.info(f"  Recall@10: {best_metrics.get('recall@10', 0.0):.4f}")
    logger.info(f"  Recall@50: {best_metrics.get('recall@50', 0.0):.4f}")
    logger.info(f"  Category usage: {best_metrics.get('category_usage_pct', 0.0):.1f}%")


def make_config_key(fusion_mode: str, config_tuple: Tuple) -> str:
    """Create a string key from config tuple for JSON serialization."""
    if fusion_mode == "bayesian":
        w, t, thres = config_tuple
        return f"w{w:.2f}_T{t:.2f}_thres{thres:.2f}"
    else:
        alpha, t, thres = config_tuple
        return f"alpha{alpha:.2f}_T{t:.2f}_thres{thres:.2f}"


def run_grid_search_vectorized(
    vectorized_data: VectorizedData,
    job_ids: List[str],
    esco_ids: List[str],
    skill_relations: Dict[str, Set[str]],
    fusion_mode: str,
    weights: List[float],
    alphas: List[float],
    temperatures: List[float],
    thresholds: List[float],
    top_k: int,
    num_workers: int = 1,
    job_batch_size: int = 500,
    aggregate_mode: str = "max",
) -> Tuple[Dict[Tuple, Dict], Tuple, Dict[str, List[Dict]], Dict[str, Dict[str, List[Dict]]]]:
    """
    Run joint grid search using VECTORIZED operations for speed.
    
    This is ~100x faster than the dict-based version because:
    1. All data is pre-converted to numpy matrices
    2. Temperature scaling uses vectorized softmax
    3. Fusion uses matrix operations
    4. Top-k uses argpartition instead of full sort
    
    Returns:
        all_metrics: Dict mapping config tuple -> metrics
        best_config: Best configuration tuple  
        best_results: Best fused results (dict format)
        all_fused_results: Dict mapping config_key -> {job_id: [{skill_uri, score, rank}, ...]}
    """
    import time
    
    all_metrics = {}
    all_fused_results = {}
    best_config = None
    best_map = -1.0
    best_results = None
    
    n_jobs = len(vectorized_data.job_ids)
    skill_uris = vectorized_data.skill_uris
    
    # Build skill URI to index for evaluation
    skill_uri_to_idx = {uri: i for i, uri in enumerate(skill_uris)}
    
    # Determine parameter grid
    if fusion_mode == "bayesian":
        param_grid = list(product(weights, temperatures, thresholds))
        logger.info(f"Grid search: {len(weights)} weights × {len(temperatures)} temps × {len(thresholds)} thresholds = {len(param_grid)} configs")
    else:
        param_grid = list(product(alphas, temperatures, thresholds))
        logger.info(f"Grid search: {len(alphas)} alphas × {len(temperatures)} temps × {len(thresholds)} thresholds = {len(param_grid)} configs")
    
    start_time = time.time()
    
    if num_workers != 1:
        logger.info(f"Running PARALLEL vectorized grid search with num_workers={num_workers} (chunked)")
        
        # Split param_grid into chunks
        # Use numpy array_split to handle cases where len(param_grid) is not divisible by num_workers
        import math
        chunk_size = math.ceil(len(param_grid) / num_workers)
        chunks = [param_grid[i:i + chunk_size] for i in range(0, len(param_grid), chunk_size)]
        
        logger.info(f"Split {len(param_grid)} configs into {len(chunks)} chunks of size ~{chunk_size}")
        
        results_nested = Parallel(n_jobs=num_workers, verbose=1)(
            delayed(_process_config_chunk)(
                chunk,
                vectorized_data,
                fusion_mode,
                top_k,
                esco_ids,
                skill_relations,
                skill_uri_to_idx,
                job_batch_size,
                aggregate_mode,
            )
            for chunk in chunks
        )
        
        # Flatten results
        results = [item for sublist in results_nested for item in sublist]
        
        for config_tuple, config_key, metrics, formatted_results in results:
            all_metrics[config_tuple] = metrics
            all_fused_results[config_key] = formatted_results
            
            if metrics["map_full"] > best_map:
                best_map = metrics["map_full"]
                best_config = config_tuple
                best_results = formatted_results
                
        total_time = time.time() - start_time
        logger.info(f"Parallel grid search completed in {total_time:.1f}s ({total_time/len(param_grid):.2f}s per config)")
        
        return all_metrics, best_config, best_results, all_fused_results
    
    # Sequential execution
    for idx, params in enumerate(param_grid):
        if fusion_mode == "bayesian":
            w, t, thres = params
            config = FusionConfig(fusion_mode="bayesian", weight=w, temperature=t, threshold=thres)
            config_tuple = (w, t, thres)
        else:
            alpha, t, thres = params
            config = FusionConfig(fusion_mode="linear", alpha=alpha, temperature=t, threshold=thres)
            config_tuple = (alpha, t, thres)
        
        config_key = make_config_key(fusion_mode, config_tuple)
        
        # Vectorized fusion computation (memory-efficient batched)
        fused_scores, top_k_indices, top_k_scores, use_weighting = compute_fused_scores_vectorized(
            vectorized_data, config, top_k, job_batch_size=job_batch_size, aggregate_mode=aggregate_mode
        )
        
        # Convert to dict format for output and evaluation
        formatted_results = {}
        for i, job_id in enumerate(vectorized_data.job_ids):
            skills_list = []
            for rank, (skill_idx, score) in enumerate(zip(top_k_indices[i], top_k_scores[i]), start=1):
                skills_list.append({
                    "skill_uri": skill_uris[skill_idx],
                    "score": round(float(score), 6),
                    "rank": rank,
                })
            formatted_results[job_id] = skills_list
        
        all_fused_results[config_key] = formatted_results
        
        # Evaluate using the indices directly (faster than dict lookup)
        # Build gold sets
        gold_sets = []
        for esco_id in esco_ids:
            if esco_id and esco_id in skill_relations:
                gold_skill_uris = skill_relations[esco_id]
                gold_indices = {
                    skill_uri_to_idx[uri]
                    for uri in gold_skill_uris
                    if uri in skill_uri_to_idx
                }
                gold_sets.append(gold_indices)
            else:
                gold_sets.append(set())
        
        # Compute metrics using indices
        metrics = {}
        
        # Coverage
        found_in_gold = sum(1 for gs in gold_sets if gs)
        metrics["gold_coverage"] = found_in_gold / len(esco_ids) if esco_ids else 0.0
        
        # Full corpus metrics
        metrics["map_full"] = compute_map(top_k_indices, gold_sets, k=None)
        metrics["mrr_full"] = compute_mrr(top_k_indices, gold_sets, k=None)
        
        # Top-K metrics
        for k in [1, 5, 10, 20, 50, 100]:
            if k > top_k:
                continue
            metrics[f"precision@{k}"] = compute_precision_at_k(top_k_indices, gold_sets, k)
            metrics[f"recall@{k}"] = compute_recall_at_k(top_k_indices, gold_sets, k)
            metrics[f"map@{k}"] = compute_map(top_k_indices, gold_sets, k)
            metrics[f"mrr@{k}"] = compute_mrr(top_k_indices, gold_sets, k)
        
        # Category usage stats
        n_with_cat = int(use_weighting.sum())
        metrics["category_usage_pct"] = 100.0 * n_with_cat / n_jobs
        metrics["n_with_category_weighting"] = n_with_cat
        metrics["n_without_category_weighting"] = n_jobs - n_with_cat
        
        all_metrics[config_tuple] = metrics
        
        # Track best
        if metrics["map_full"] > best_map:
            best_map = metrics["map_full"]
            best_config = config_tuple
            best_results = formatted_results
        
        # Progress logging
        if (idx + 1) % 10 == 0 or idx == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / (idx + 1)
            remaining = avg_time * (len(param_grid) - idx - 1)
            logger.info(
                f"  [{idx+1}/{len(param_grid)}] {config_key}: "
                f"MAP={metrics['map_full']:.4f}, R@10={metrics.get('recall@10', 0.0):.4f} "
                f"({avg_time:.1f}s/config, ~{remaining:.0f}s remaining)"
            )
    
    total_time = time.time() - start_time
    logger.info(f"Grid search completed in {total_time:.1f}s ({total_time/len(param_grid):.2f}s per config)")
    
    return all_metrics, best_config, best_results, all_fused_results


def get_data_subset(
    job_ids_subset: List[str],
    similarity_scores: Dict[str, List[Dict]],
    category_scores_raw: Dict[str, List[Dict]],
    job_to_esco: Dict[str, str],
    task_a_scores: Optional[Dict[str, List[str]]] = None,
) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]], List[str], Optional[Dict[str, List[str]]]]:
    """
    Extract subset of data for specific job IDs.
    Returns:
        (subset_similarity, subset_category, subset_esco_ids, subset_task_a)
    """
    # Order determined by job_ids_subset
    subset_sim = {}
    subset_cat = {}
    subset_esco = []
    
    for job_id in job_ids_subset:
        if job_id in similarity_scores and job_id in category_scores_raw:
            subset_sim[job_id] = similarity_scores[job_id]
            subset_cat[job_id] = category_scores_raw[job_id]
            subset_esco.append(job_to_esco.get(job_id, ""))
            
    subset_task_a = None
    if task_a_scores is not None:
        subset_task_a = {}
        for job_id in job_ids_subset:
            if job_id in task_a_scores:
                # We do not strictly enforce intersection here, just pick what is available
                subset_task_a[job_id] = task_a_scores[job_id]
            
    return subset_sim, subset_cat, subset_esco, subset_task_a


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian/Linear re-ranking: fuse IR scores with category probabilities"
    )
    parser.add_argument(
        "--similarity_scores_json",
        type=str,
        required=True,
        help="Path to similarity_scores.json from similarity_scorer.py",
    )
    parser.add_argument(
        "--category_scores_json",
        type=str,
        required=True,
        help="Path to category_scores.json from category_inference.py",
    )
    parser.add_argument(
        "--task_a_scores_json",
        type=str,
        default=None,
        help="Optional path to Task A occupation predictions JSON for candidate filtering.",
    )
    parser.add_argument(
        "--task_a_k",
        type=int,
        default=5,
        help="Number of Task A top occupations to use for candidate filtering (default: 5).",
    )
    parser.add_argument(
        "--skill_hierarchy_csv",
        type=str,
        default="./data/processed/master_datasets_2/master_skill_complete_hierarchy.csv",
        help="Path to skill hierarchy CSV with level1_label column",
    )
    parser.add_argument(
        "--jobs_csv",
        type=str,
        required=True,
        help="Path to jobs CSV with job_id and esco_id columns",
    )
    parser.add_argument(
        "--skill_relations_csv",
        type=str,
        default="./data/esco_datasets/occupationSkillRelations_en.csv",
        help="Path to occupationSkillRelations.csv for evaluation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save output files",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Number of top skills to keep after re-ranking",
    )
    # Fusion mode
    parser.add_argument(
        "--fusion_mode",
        type=str,
        choices=["bayesian", "linear"],
        default="bayesian",
        help="Fusion mode: 'bayesian' (sim × P^w) or 'linear' ((1-α)×sim + α×P)",
    )
    # Bayesian fusion parameters
    parser.add_argument(
        "--weights",
        type=str,
        default="0.0,0.5,1.0,1.5,2.0",
        help="Comma-separated list of weight values (w) for Bayesian fusion",
    )
    # Linear fusion parameters
    parser.add_argument(
        "--alphas",
        type=str,
        default="0.0,0.1,0.2,0.3,0.5,0.7",
        help="Comma-separated list of alpha values for linear fusion",
    )
    # Temperature scaling
    parser.add_argument(
        "--temperatures",
        type=str,
        default="1.0",
        help="Comma-separated list of temperature values for scaling category logits",
    )
    # Max-prob threshold
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.0",
        help="Comma-separated list of max-prob threshold values. If max(P) < threshold, skip category weighting.",
    )
    parser.add_argument(
        "--aggregate",
        type=str,
        choices=["max", "sum"],
        default="max",
        help="Aggregation mode for skill categories: 'max' (take max probability) or 'sum' (sum probabilities).",
    )
    parser.add_argument(
        "--id_column",
        type=str,
        default="job_id",
        help="Column name for job ID in jobs_csv",
    )
    parser.add_argument(
        "--esco_id_column",
        type=str,
        default="esco_id",
        help="Column name for ESCO occupation URI in jobs_csv",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers for grid search. Use -1 for all CPU cores.",
    )
    parser.add_argument(
        "--job_batch_size",
        type=int,
        default=500,
        help="Number of jobs to process per batch (controls memory usage). Lower = less memory, higher = faster.",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        choices=["all", "best", None],
        default=None,
        help="Save the final scores for 'all' configs or only the best one.",
    )
    parser.add_argument(
        "--grid_search_split",
        type=str,
        choices=["train", "validation", "train+validation"],
        default="validation",
        help="Split to use for grid search optimization. Default: 'validation'.",
    )
    args = parser.parse_args()

    # Parse hyperparameter lists
    weights = [float(w.strip()) for w in args.weights.split(",")]
    alphas = [float(a.strip()) for a in args.alphas.split(",")]
    temperatures = [float(t.strip()) for t in args.temperatures.split(",")]
    thresholds = [float(t.strip()) for t in args.thresholds.split(",")]
    aggregate_mode = args.aggregate

    logger.info(f"Fusion mode: {args.fusion_mode}")
    if args.fusion_mode == "bayesian":
        logger.info(f"Weights (w): {weights}")
    else:
        logger.info(f"Alphas: {alphas}")
    logger.info(f"Temperatures: {temperatures}")
    logger.info(f"Thresholds: {thresholds}")
    logger.info(f"Job batch size: {args.job_batch_size} (memory-efficient batching)")
    logger.info(f"Grid search split: {args.grid_search_split}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    similarity_scores = load_similarity_scores(args.similarity_scores_json)
    category_scores_raw = load_category_scores_raw(args.category_scores_json)
    skill_to_categories = build_skill_to_categories(args.skill_hierarchy_csv)
    job_ids, esco_ids, splits = load_jobs(args.jobs_csv, args.id_column, args.esco_id_column)
    skill_relations = load_skill_relations(args.skill_relations_csv)
    
    # Create job_to_esco map
    job_to_esco = dict(zip(job_ids, esco_ids))
    
    # Determine job ID sets
    # Intersection of all data sources
    common_ids = set(job_ids) & set(similarity_scores.keys()) & set(category_scores_raw.keys())
    
    train_ids = {jid for jid in common_ids if splits.get(jid, "").lower() in ["train", "training"]}
    val_ids = {jid for jid in common_ids if splits.get(jid, "").lower() in ["val", "validation", "valid", "dev"]}
    test_ids = {jid for jid in common_ids if splits.get(jid, "").lower() in ["test", "testing"]}
    
    logger.info(f"Data splits found: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    
    # Determine Search Set
    if not splits:
        logger.warning("No splits found. Using ALL common data for both grid search and evaluation.")
        search_ids = list(common_ids)
        eval_ids = list(common_ids)
    else:
        if args.grid_search_split == "train":
            search_ids = list(train_ids)
        elif args.grid_search_split == "validation":
            search_ids = list(val_ids)
        else: # train+validation
            search_ids = list(train_ids | val_ids)
            
        if not search_ids:
             logger.warning(f"No jobs found for grid search split '{args.grid_search_split}'. Falling back to ALL data.")
             search_ids = list(common_ids)
             
        # Determine Eval Set (Test)
        eval_ids = list(test_ids)
        if not eval_ids:
            logger.warning("No test jobs found. Using validation set for evaluation.")
            eval_ids = list(val_ids)
            if not eval_ids:
                logger.warning("No validation jobs found either. Using search set for evaluation.")
                eval_ids = search_ids

    logger.info(f"Grid Search on {len(search_ids)} jobs")
    logger.info(f"Evaluation on {len(eval_ids)} jobs")
    
    # Load Task A scores if provided
    task_a_scores = None
    if args.task_a_scores_json:
        task_a_scores = load_task_a_scores(args.task_a_scores_json)
    
    # Run joint grid search on Search Set
    logger.info("\n" + "=" * 60)
    logger.info("Starting Joint Grid Search")
    logger.info("=" * 60)
    
    search_sim, search_cat, search_esco, search_task_a = get_data_subset(
        search_ids, similarity_scores, category_scores_raw, job_to_esco, task_a_scores
    )
    
    # Prepare vectorized data for fast grid search
    vectorized_data_search = prepare_vectorized_data(
        search_sim,
        search_cat,
        skill_to_categories,
        task_a_scores=search_task_a,
        skill_relations=skill_relations,
        task_a_k=args.task_a_k,
    )

    all_metrics, best_config, _, _ = run_grid_search_vectorized(
        vectorized_data=vectorized_data_search,
        job_ids=search_ids,
        esco_ids=search_esco,
        skill_relations=skill_relations,
        fusion_mode=args.fusion_mode,
        weights=weights,
        alphas=alphas,
        temperatures=temperatures,
        thresholds=thresholds,
        top_k=args.top_k,
        num_workers=args.num_workers,
        job_batch_size=args.job_batch_size,
        aggregate_mode=aggregate_mode,
    )
    
    # Print comparison table
    print_metrics_table(all_metrics, args.top_k, args.fusion_mode)
    
    # Get best config key
    best_config_key = make_config_key(args.fusion_mode, best_config)
    best_metrics = all_metrics[best_config]
    
    # Evaluate on Test Set
    logger.info("\n" + "=" * 60)
    logger.info("Running Evaluation on TEST Set")
    logger.info("=" * 60)
    
    test_sim, test_cat, test_esco, test_task_a = get_data_subset(
        eval_ids, similarity_scores, category_scores_raw, job_to_esco, task_a_scores
    )
    vectorized_data_test = prepare_vectorized_data(
        test_sim, test_cat, skill_to_categories,
        task_a_scores=test_task_a,
        skill_relations=skill_relations,
        task_a_k=args.task_a_k,
    )
    
    if args.fusion_mode == "bayesian":
        w, t, thres = best_config
        weights_test = [w]
        temps_test = [t]
        thres_test = [thres]
        alphas_test = [0.0]
    else:
        alpha, t, thres = best_config
        alphas_test = [alpha]
        temps_test = [t]
        thres_test = [thres]
        weights_test = [0.0]
        
    metrics_test, _, _, _ = run_grid_search_vectorized(
        vectorized_data=vectorized_data_test,
        job_ids=eval_ids,
        esco_ids=test_esco,
        skill_relations=skill_relations,
        fusion_mode=args.fusion_mode,
        weights=weights_test,
        alphas=alphas_test,
        temperatures=temps_test,
        thresholds=thres_test,
        top_k=args.top_k,
        num_workers=1,
        job_batch_size=args.job_batch_size,
        aggregate_mode=aggregate_mode,
    )
    test_metrics_final = list(metrics_test.values())[0]
    logger.success(f"Test MAP: {test_metrics_final['map_full']:.4f}")

    # -------------------------------------------------------------------------
    # Base retrieval (Task B only) metrics on TEST split (no category fusion)
    # This is the direct baseline comparable to v5's "task_b_only" mAP:
    # - rank by raw similarity score
    # - compute AP over the full available ranking per job
    # -------------------------------------------------------------------------
    # Build gold sets aligned with vectorized_data_test ordering
    skill_uri_to_idx_test = {uri: i for i, uri in enumerate(vectorized_data_test.skill_uris)}
    gold_sets_test: List[Set[int]] = []
    for esco_id in test_esco:
        if esco_id and esco_id in skill_relations:
            gold_skill_uris = skill_relations[esco_id]
            gold_sets_test.append(
                {
                    skill_uri_to_idx_test[uri]
                    for uri in gold_skill_uris
                    if uri in skill_uri_to_idx_test
                }
            )
        else:
            gold_sets_test.append(set())

    # Build rectangular indices array from per-job similarity lists
    base_k = max((len(test_sim[jid]) for jid in vectorized_data_test.job_ids), default=0)
    base_indices = np.full((len(vectorized_data_test.job_ids), base_k), -1, dtype=np.int64)
    for i, job_id in enumerate(vectorized_data_test.job_ids):
        skills_list = test_sim.get(job_id, [])
        # Ensure descending by score (don't trust upstream ordering)
        skills_list_sorted = sorted(skills_list, key=lambda x: x.get("score", 0.0), reverse=True)
        row = []
        for item in skills_list_sorted:
            uri = item.get("skill_uri")
            if uri in skill_uri_to_idx_test:
                row.append(skill_uri_to_idx_test[uri])
        # Fill row (truncate if somehow longer than base_k)
        if row:
            base_indices[i, : min(len(row), base_k)] = np.array(row[:base_k], dtype=np.int64)

    base_map_test = compute_map(base_indices, gold_sets_test, k=None)

    # Also compute a richer metric suite for base retrieval (to report alongside fusion).
    # We compute metrics directly over URI rankings (variable length), matching the v5 style.
    gold_uris_test: List[Set[str]] = []
    for esco_id in test_esco:
        if esco_id and esco_id in skill_relations:
            gold_uris_test.append(set(skill_relations[esco_id]))
        else:
            gold_uris_test.append(set())

    base_rankings_test: List[List[str]] = []
    for job_id in vectorized_data_test.job_ids:
        skills_list = test_sim.get(job_id, [])
        skills_list_sorted = sorted(skills_list, key=lambda x: x.get("score", 0.0), reverse=True)
        base_rankings_test.append([it.get("skill_uri") for it in skills_list_sorted if it.get("skill_uri")])

    def _compute_retrieval_metrics(
        rankings: List[List[str]],
        golds: List[Set[str]],
        ks: List[int] = [1, 5, 10, 20, 50, 100],
    ) -> Dict[str, float]:
        """Compute MAP/MRR/Precision/Recall over ranked URI lists (includes empty-gold queries as 0)."""
        n = len(rankings)
        if n == 0:
            return {}

        # Coverage
        n_with_gold = sum(1 for g in golds if g)

        # Accumulators
        sum_map_full = 0.0
        sum_mrr_full = 0.0
        sums = {f"precision@{k}": 0.0 for k in ks}
        sums.update({f"recall@{k}": 0.0 for k in ks})
        sums.update({f"map@{k}": 0.0 for k in ks})
        sums.update({f"mrr@{k}": 0.0 for k in ks})

        for pred, gold in zip(rankings, golds):
            if not gold:
                # v5-style: empty gold contributes 0 to averages
                continue  # keep sums unchanged; we'll divide by n below

            # Full AP / MRR
            hits = 0
            sum_precs = 0.0
            rr = 0.0
            for rank, uri in enumerate(pred, start=1):
                if uri in gold:
                    hits += 1
                    sum_precs += hits / rank
                    if rr == 0.0:
                        rr = 1.0 / rank
            ap_full = sum_precs / len(gold) if gold else 0.0

            sum_map_full += ap_full
            sum_mrr_full += rr

            # Cutoff metrics
            for k in ks:
                topk = pred[:k]
                topk_set = set(topk)
                hit_k = len(topk_set & gold)

                sums[f"precision@{k}"] += hit_k / k
                sums[f"recall@{k}"] += hit_k / len(gold)

                # AP@k
                hits_k = 0
                sum_precs_k = 0.0
                rr_k = 0.0
                for rank, uri in enumerate(topk, start=1):
                    if uri in gold:
                        hits_k += 1
                        sum_precs_k += hits_k / rank
                        if rr_k == 0.0:
                            rr_k = 1.0 / rank
                sums[f"map@{k}"] += (sum_precs_k / len(gold) if gold else 0.0)
                sums[f"mrr@{k}"] += rr_k

        # Note: divide by total number of queries (including empty-gold as 0), to match v5 averaging.
        metrics = {
            "n_jobs": float(n),
            "gold_coverage": float(n_with_gold / n) if n else 0.0,
            "map_full": float(sum_map_full / n),
            "mrr_full": float(sum_mrr_full / n),
        }
        for k in ks:
            metrics[f"precision@{k}"] = float(sums[f"precision@{k}"] / n)
            metrics[f"recall@{k}"] = float(sums[f"recall@{k}"] / n)
            metrics[f"map@{k}"] = float(sums[f"map@{k}"] / n)
            metrics[f"mrr@{k}"] = float(sums[f"mrr@{k}"] / n)
        return metrics

    base_metrics_test = _compute_retrieval_metrics(base_rankings_test, gold_uris_test)

    # -------------------------------------------------------------------------
    # Optionally generate & save scores for ALL data (train+val+test intersection)
    # Controlled by --save_strategy. If omitted (None), we do NOT compute/save scores.
    # -------------------------------------------------------------------------
    save_scores = args.save_strategy in {"best", "all"}
    all_ids_list: List[str] = []
    all_metrics_all_final: Dict = {}
    final_predictions: Dict[str, List[Dict]] = {}
    if save_scores:
        logger.info("\n" + "=" * 60)
        logger.info("Generating Final Scores for ALL Data")
        logger.info("=" * 60)
        
        all_ids_list = sorted(list(common_ids))
        all_sim, all_cat, all_esco, all_task_a = get_data_subset(
            all_ids_list, similarity_scores, category_scores_raw, job_to_esco, task_a_scores
        )
        vectorized_data_all = prepare_vectorized_data(
            all_sim, all_cat, skill_to_categories,
            task_a_scores=all_task_a,
            skill_relations=skill_relations,
            task_a_k=args.task_a_k,
        )
        
        # We use run_grid_search_vectorized again just to reuse the pipeline.
        all_metrics_all, _, _, all_fused_results_final = run_grid_search_vectorized(
             vectorized_data=vectorized_data_all,
             job_ids=all_ids_list,
             esco_ids=all_esco,
             skill_relations=skill_relations,
             fusion_mode=args.fusion_mode,
             weights=weights_test,
             alphas=alphas_test,
             temperatures=temps_test,
             thresholds=thres_test,
             top_k=args.top_k,
             num_workers=1,
             job_batch_size=args.job_batch_size,
             aggregate_mode=aggregate_mode,
        )
        all_metrics_all_final = list(all_metrics_all.values())[0] if all_metrics_all else {}
        
        # The config key for best params
        best_key_in_all = make_config_key(args.fusion_mode, best_config)
        final_predictions = all_fused_results_final[best_key_in_all]

    # Save outputs
    # 1. Save all metrics with grid search results + Test results
    metrics_output = {
        "args": vars(args),
        "fusion_mode": args.fusion_mode,
        "aggregate": args.aggregate,
        "data_splits": {
            "n_common_jobs": len(common_ids),
            "n_train": len(train_ids),
            "n_validation": len(val_ids),
            "n_test": len(test_ids),
            "grid_search_split": args.grid_search_split,
            "n_grid_search_jobs": len(search_ids),
            "n_eval_jobs": len(eval_ids),
            "n_all_jobs_scored": len(all_ids_list) if save_scores else 0,
        },
        "best_config": {
            "key": best_config_key,
            "weight" if args.fusion_mode == "bayesian" else "alpha": best_config[0],
            "temperature": best_config[1],
            "threshold": best_config[2],
        },
        "search_metrics": {
            "map_full": best_metrics["map_full"],
            "mrr_full": best_metrics["mrr_full"],
            "recall@10": best_metrics.get("recall@10", 0.0),
            "recall@50": best_metrics.get("recall@50", 0.0),
            "category_usage_pct": best_metrics.get("category_usage_pct", 0.0),
        },
        "test_metrics": {
            "map_full": test_metrics_final["map_full"],
            "mrr_full": test_metrics_final["mrr_full"],
            "recall@10": test_metrics_final.get("recall@10", 0.0),
            "recall@50": test_metrics_final.get("recall@50", 0.0),
             "category_usage_pct": test_metrics_final.get("category_usage_pct", 0.0),
        },
        "grid_search": {
            "weights" if args.fusion_mode == "bayesian" else "alphas": weights if args.fusion_mode == "bayesian" else alphas,
            "temperatures": temperatures,
            "thresholds": thresholds,
            "n_configs_tested": len(all_metrics),
        },
        "all_results": [
            {
                "config_key": make_config_key(args.fusion_mode, cfg),
                "config": {
                    "weight" if args.fusion_mode == "bayesian" else "alpha": cfg[0],
                    "temperature": cfg[1],
                    "threshold": cfg[2],
                },
                "metrics": m,
            }
            for cfg, m in sorted(all_metrics.items(), key=lambda x: x[1]["map_full"], reverse=True)
        ],
    }
    metrics_path = output_dir / "grid_search_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_output, f, indent=2)
    logger.success(f"Saved grid search metrics to {metrics_path}")
    
    # 2. Save fused scores (Best Config on ALL data) only if requested
    if save_scores:
        best_only_output = {
            "metadata": {
                "fusion_mode": args.fusion_mode,
                "aggregate": args.aggregate,
                "top_k": args.top_k,
                "n_jobs": len(all_ids_list),
                "best_config_key": best_config_key,
                "config": {
                    "weight" if args.fusion_mode == "bayesian" else "alpha": best_config[0],
                    "temperature": best_config[1],
                    "threshold": best_config[2],
                    "test_map": test_metrics_final["map_full"],
                    "all_map": all_metrics_all_final.get("map_full", None) if all_metrics_all_final else None,
                },
            },
            "scores": final_predictions,
        }
        best_path = output_dir / "best_fused_scores.json"
        logger.info("Saving best configuration fused scores for ALL data...")
        with open(best_path, "w") as f:
            json.dump(best_only_output, f, indent=2)
        logger.success(f"Saved best fused scores to {best_path}")

    # 3. Save only the metrics you need to report
    results_output = {
        "best_config": {
            "key": best_config_key,
            "weight" if args.fusion_mode == "bayesian" else "alpha": best_config[0],
            "temperature": best_config[1],
            "threshold": best_config[2],
            "fusion_mode": args.fusion_mode,
            "aggregate": args.aggregate,
        },
        "metrics": {
            "search_split_best_fusion": best_metrics,          # tuned split (e.g., validation)
            "test_split_base_retrieval": base_metrics_test,    # Task B only
            "test_split_best_fusion": test_metrics_final,      # after best fusion
        },
    }
    results_path = output_dir / "results.json"
    logger.info("Saving results.json (useful metrics: base vs best fusion)...")
    with open(results_path, "w") as f:
        json.dump(results_output, f, indent=2)
    logger.success(f"Saved results to {results_path}")
    
    # Final summary
    logger.success(f"\n{'='*60}")
    logger.success(f"Fusion complete! Mode: {args.fusion_mode.upper()}")
    if args.fusion_mode == "bayesian":
        logger.success(f"Best config: w={best_config[0]:.2f}, T={best_config[1]:.2f}, threshold={best_config[2]:.2f}")
    else:
        logger.success(f"Best config: alpha={best_config[0]:.2f}, T={best_config[1]:.2f}, threshold={best_config[2]:.2f}")
    logger.success(f"Best Search MAP: {best_metrics['map_full']:.4f}")
    logger.success(f"Test MAP: {test_metrics_final['map_full']:.4f}")
    logger.success(f"{'='*60}")


if __name__ == "__main__":
    main()

