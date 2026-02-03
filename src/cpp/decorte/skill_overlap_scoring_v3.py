"""
Skill Overlap Scoring for Career Path Prediction (V2 - Unified).

Computes skill-based scores for next occupation prediction:
    S_SKILLS((occ₁,...,occ_N), occ) = |S_career ∩ S(occ)| / |S(occ)|

where:
- S_career = union of all skills from past occupations
- S(occ) = skills required by target occupation (from ESCO occupation-skill relations)

Supports TWO modes:
1. **Job ID mode** (for decorte with raw titles):
   - Requires --skill_scores_file (e.g., best_fused_scores.json)
   - Career history skills looked up via job_id → predicted skills

2. **ESCO mode** (for decorte_esco with ESCO titles):
   - No skill_scores_file needed
   - Career history skills looked up via ESCO title → URI → skills

Key Features:
- Automatically selects mode based on whether --skill_scores_file is provided
- Uses ESCO occupation-skill relations for target skills (both modes)
- Outputs scores in same format as train_cpp_enhanced_v2.py for fusion
"""

import argparse
import os
import sys
import gc
import pickle
from typing import Dict, List, Set, Tuple, Optional
from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm
import json

try:
    from cpp.data_classes import Data
    from cpp.utils import SEP_TOKEN
except ImportError as e:
    print(f"Error: Required modules not found. {e}")
    sys.exit(1)


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_dir: str):
    """Configure logging to both file and stdout."""
    os.makedirs(log_dir, exist_ok=True)
    
    logger.remove()
    logger.add(
        os.path.join(log_dir, "skill_overlap_scoring_v2.log"),
        format="{time} | {level} | {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        enqueue=True
    )
    logger.add(
        sys.stdout,
        format="<green>{time}</green> | <level>{message}</level>",
        level="INFO"
    )


# ============================================================================
# DATA LOADING
# ============================================================================

def load_career_skill_map(skill_scores_file: str, 
                           top_k_skills: Optional[int] = None,
                           min_confidence: Optional[float] = None) -> Tuple[Dict[str, Set[str]], Dict]:
    """
    Load skill mapping for career history jobs from best_fused_scores.json.
    
    Args:
        skill_scores_file: Path to JSON file with skill scores keyed by job_id
        top_k_skills: If provided, only consider the top K skills (by score) for each job_id.
        min_confidence: If provided, only consider skills with a score >= min_confidence.
        
    Returns:
        Dictionary mapping job_id -> set of skill URIs
    """
    logger.info(f"Loading career skill map from: {skill_scores_file}")
    if min_confidence is not None:
        logger.info(f"  > Filtering by minimum confidence: {min_confidence}")
    if top_k_skills is not None:
        logger.info(f"  > Filtering by top-K skills: {top_k_skills}")
    
    scores_dict = {}
    with open(skill_scores_file, 'r') as f:
        try:
            # optimization: if .jsonl, don't try to load as single json object
            if skill_scores_file.endswith('.jsonl'):
                raise json.JSONDecodeError("Force JSONL path", "", 0)
            
            data = json.load(f)
            # Extract scores dictionary
            if 'scores' in data:
                scores_dict = data['scores']
            else:
                scores_dict = data
        except json.JSONDecodeError:
            # Try reading as JSONL (line-delimited JSON)
            logger.info("  Reading as JSONL format...")
            f.seek(0)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    # Expect items to have 'job_id' and a list of skills in 'predictions' or 'scores'
                    job_id = item.get('job_id')
                    raw_skills = item.get('predictions') or item.get('scores')
                    
                    if job_id is not None and raw_skills is not None:
                        scores_dict[str(job_id)] = raw_skills
                except json.JSONDecodeError:
                    continue
    
    # Convert to job_id -> set of skill URIs
    career_skill_map = {}
    
    # Statistics for logging/reporting
    total_skills_before = 0
    total_skills_after_conf = 0
    total_skills_final = 0
    
    for job_id, raw_skill_list in scores_dict.items():
        # Parse skills into a standardized list of (uri, score) tuples
        parsed_skills = []
        
        for item in raw_skill_list:
            uri = None
            score = -float('inf')
            
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                # Format: [uri, score]
                uri = item[0]
                try:
                    score = float(item[1])
                except (ValueError, TypeError):
                    score = 0.0
            elif isinstance(item, dict):
                # Format: {'skill_uri': ..., 'score': ...}
                uri = item.get('skill_uri') or item.get('skillUri')
                val = item.get('score')
                if val is not None:
                    try:
                        score = float(val)
                    except (ValueError, TypeError):
                        pass # keep default lowest
            
            if uri:
                parsed_skills.append((uri, score))
        
        total_skills_before += len(parsed_skills)
        
        # 1. Filter by confidence
        if min_confidence is not None:
            parsed_skills = [s for s in parsed_skills if s[1] >= min_confidence]
        
        total_skills_after_conf += len(parsed_skills)
        
        # 2. Filter top-k if requested
        if top_k_skills is not None and top_k_skills > 0:
            # Sort by score descending
            parsed_skills.sort(key=lambda x: x[1], reverse=True)
            skill_list_filtered = parsed_skills[:top_k_skills]
        else:
            skill_list_filtered = parsed_skills
            
        total_skills_final += len(skill_list_filtered)
        
        # Add URIs to set
        skill_uris = {uri for uri, _ in skill_list_filtered}
        career_skill_map[str(job_id)] = skill_uris
    
    logger.info(f"  ✓ Loaded skill mappings for {len(career_skill_map)} job_ids")
    
    # Statistics
    avg_before = 0
    avg_after_conf = 0
    avg_final = 0
    
    if career_skill_map:
        avg_before = total_skills_before / len(career_skill_map)
        avg_after_conf = total_skills_after_conf / len(career_skill_map)
        avg_final = total_skills_final / len(career_skill_map)
        
        logger.info(f"  > Avg skills per job: initial={avg_before:.1f}" + 
                    (f", after confidence filter={avg_after_conf:.1f}" if min_confidence is not None else "") +
                    (f", final (top-K)={avg_final:.1f}" if top_k_skills is not None else ""))
        
        if min_confidence is not None:
            removed = total_skills_before - total_skills_after_conf
            pct = 100 * removed / total_skills_before if total_skills_before > 0 else 0
            logger.info(f"  > Confidence filter: removed {removed} skills ({pct:.1f}%)")
    else:
        logger.warning("  ⚠️  Career skill map is empty!")
    
    stats = {
        'total_skills_initial': total_skills_before,
        'total_skills_after_confidence': total_skills_after_conf,
        'total_skills_final': total_skills_final,
        'avg_skills_per_job_initial': avg_before,
        'avg_skills_per_job_final': avg_final,
        'removed_by_confidence': total_skills_before - total_skills_after_conf
    }
    
    return career_skill_map, stats


def load_esco_target_skills(occupations_file: str, relations_file: str) -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
    """
    Load ESCO mappings for target occupations:
    1. Title -> URI (from occupations_file)
    2. URI -> Set[SkillURI] (from relations_file)
    
    Args:
        occupations_file: Path to ESCO occupations CSV
        relations_file: Path to ESCO occupation-skill relations CSV
        
    Returns:
        Tuple of (title_to_uri, uri_to_skills)
    """
    logger.info(f"Loading ESCO occupations from {occupations_file}")
    if not os.path.exists(occupations_file):
        raise FileNotFoundError(f"ESCO occupations file not found at {occupations_file}")
    
    occ_df = pd.read_csv(occupations_file)
    
    # Map preferredLabel -> URI (lowercase for matching)
    # Note: We only use preferredLabel since dataset targets are standardized ESCO labels
    title_to_uri = {}
    for _, row in occ_df.iterrows():
        uri = row['conceptUri']
        if pd.notna(row['preferredLabel']):
            pref_label = str(row['preferredLabel']).strip().lower()
            title_to_uri[pref_label] = uri
                    
    logger.info(f"  ✓ Loaded {len(title_to_uri)} ESCO occupation mappings")
    
    # Load skill relations
    logger.info(f"Loading ESCO skill relations from {relations_file}")
    if not os.path.exists(relations_file):
        raise FileNotFoundError(f"ESCO relations file not found at {relations_file}")

    rel_df = pd.read_csv(relations_file)
    
    # Group by occupationUri -> set of skillUri (includes both essential and optional)
    uri_to_skills = rel_df.groupby('occupationUri')['skillUri'].apply(set).to_dict()
    
    logger.info(f"  ✓ Loaded skill mappings for {len(uri_to_skills)} ESCO occupations")
    
    # Statistics
    n_skills_per_occ = [len(s) for s in uri_to_skills.values()]
    logger.info(f"  > Skills per occupation: min={min(n_skills_per_occ)}, max={max(n_skills_per_occ)}, mean={np.mean(n_skills_per_occ):.1f}")
    
    return title_to_uri, uri_to_skills


def get_target_skills(target_title: str, title_to_uri: Dict[str, str], 
                      uri_to_skills: Dict[str, Set[str]]) -> Set[str]:
    """Get skills for a target occupation title (ESCO-based)."""
    title_norm = target_title.strip().lower()
    
    # Try exact match
    uri = title_to_uri.get(title_norm)
    
    if not uri:
        # Handle prefixes like "esco role:" if present
        for prefix in ["esco role:", "role:"]:
            if prefix in title_norm:
                clean_title = title_norm.replace(prefix, "").strip()
                uri = title_to_uri.get(clean_title)
                if uri:
                    break
    
    if uri and uri in uri_to_skills:
        return uri_to_skills[uri]
        
    return set()


def normalize_target_label(label: str) -> str:
    """Normalize target labels for de-duplication and indexing."""
    return label.strip().lower()


def get_career_skills_from_job_ids(job_ids: List[str], 
                                    career_skill_map: Dict[str, Set[str]]) -> Set[str]:
    """
    Get union of all skills from career history using job_ids.
    
    Args:
        job_ids: List of job_id strings for jobs in career history
        career_skill_map: Map from job_id -> set of skill URIs
        
    Returns:
        Union of all skills across the career
    """
    career_skills = set()
    for job_id in job_ids:
        if job_id in career_skill_map:
            career_skills.update(career_skill_map[job_id])
    return career_skills


def get_career_skills_from_history_esco(history_doc: str,
                                         title_to_uri: Dict[str, str],
                                         uri_to_skills: Dict[str, Set[str]]) -> Set[str]:
    """
    Get union of all skills from career history using ESCO title lookup.
    
    This is used for ESCO → ESCO datasets where history contains ESCO titles
    and we can look up skills directly from the ESCO relations.
    
    Args:
        history_doc: History document string (titles separated by SEP_TOKEN)
        title_to_uri: Map ESCO title -> URI
        uri_to_skills: Map ESCO URI -> set of skill URIs
        
    Returns:
        Union of all skills across the career
    """
    career_skills = set()
    
    # Split by SEP_TOKEN to get individual job titles
    titles = history_doc.split(SEP_TOKEN)
    
    for title in titles:
        title = title.strip()
        if not title:
            continue
        
        # Look up skills for this title using ESCO mappings
        skills = get_target_skills(title, title_to_uri, uri_to_skills)
        career_skills.update(skills)
    
    return career_skills


# ============================================================================
# SKILL OVERLAP SCORING
# ============================================================================

def compute_skill_overlap_score(career_skills: Set[str], target_skills: Set[str]) -> float:
    """
    Compute skill overlap score.
    
    Score = |career_skills ∩ target_skills| / |target_skills|
    
    Returns 0.0 if target has no skills (avoid div by zero).
    """
    if len(target_skills) == 0:
        return 0.0
    
    intersection = career_skills & target_skills
    return len(intersection) / len(target_skills)


def compute_skill_overlap_scores_for_split(
    data_pairs: List[Tuple[str, str]],
    job_ids_list: Optional[List[List[str]]],
    all_targets: List[str],
    career_skill_map: Optional[Dict[str, Set[str]]],
    title_to_uri: Dict[str, str],
    uri_to_skills: Dict[str, Set[str]],
    split_name: str = "data",
    use_esco_mode: bool = False
) -> Dict:
    """
    Compute skill overlap scores for all samples in a split.
    
    Args:
        data_pairs: List of (history_doc, target_doc) tuples
        job_ids_list: List of job_id lists (one per sample) - used in job_id mode
        all_targets: List of all possible target labels (sorted)
        career_skill_map: Map from job_id -> set of skill URIs (None in ESCO mode)
        title_to_uri: Map ESCO title -> URI
        uri_to_skills: Map ESCO URI -> set of skill URIs
        split_name: Name for logging
        use_esco_mode: If True, use ESCO title lookup for career skills (no job_id needed)
        
    Returns:
        Dictionary with scores and metadata (compatible with train_cpp_enhanced_v2.py format)
    """
    n_samples = len(data_pairs)
    n_targets = len(all_targets)
    
    mode_str = "ESCO title lookup" if use_esco_mode else "job_id lookup"
    logger.info(f"  > Processing {split_name}: {n_samples} samples, {n_targets} targets (mode: {mode_str})")
    
    # Precompute skills for all targets
    logger.info("    Precomputing skills for all targets...")
    target_skills_map = {}
    n_targets_no_skills = 0
    
    for target in tqdm(all_targets, desc="    Target skills"):
        skills = get_target_skills(target, title_to_uri, uri_to_skills)
        target_skills_map[target] = skills
        if len(skills) == 0:
            n_targets_no_skills += 1
    
    if n_targets_no_skills > 0:
        logger.warning(f"    ⚠️  {n_targets_no_skills}/{n_targets} targets have no ESCO skills")
    
    # Build target label to index mapping (normalize for matching)
    # Use original target string as key but also create normalized lookup
    target_to_idx = {t: i for i, t in enumerate(all_targets)}
    # Also create normalized version for robust matching
    target_norm_to_idx = {normalize_target_label(t): i for i, t in enumerate(all_targets)}
    
    # Compute scores
    logger.info(f"    Computing overlap scores...")
    scores = np.zeros((n_samples, n_targets), dtype=np.float32)
    true_target_indices = []
    unmapped_targets = []
    
    # Statistics
    n_zero_career_skills = 0
    total_career_skills = 0
    
    # Iterate through samples
    for i in tqdm(range(n_samples), desc="    Scoring"):
        history_doc, target_doc = data_pairs[i]
        
        # Get career skills based on mode
        if use_esco_mode:
            # ESCO mode: look up skills directly from history titles
            career_skills = get_career_skills_from_history_esco(
                history_doc, title_to_uri, uri_to_skills
            )
        else:
            # Job ID mode: look up skills via job_id
            job_ids = job_ids_list[i] if job_ids_list else []
            career_skills = get_career_skills_from_job_ids(job_ids, career_skill_map)
        
        total_career_skills += len(career_skills)
        
        if len(career_skills) == 0:
            n_zero_career_skills += 1
        
        # Score all targets
        for j, target in enumerate(all_targets):
            target_skills = target_skills_map[target]
            scores[i, j] = compute_skill_overlap_score(career_skills, target_skills)
        
        # Record true target index - try exact match first, then normalized
        true_idx = target_to_idx.get(target_doc, -1)
        if true_idx == -1:
            # Try normalized lookup
            true_idx = target_norm_to_idx.get(normalize_target_label(target_doc), -1)
        true_target_indices.append(true_idx)
        if true_idx == -1:
            unmapped_targets.append(target_doc)
    
    # Log statistics
    avg_career_skills = total_career_skills / max(n_samples, 1)
    logger.info(f"    ✓ Average career skills per sample: {avg_career_skills:.1f}")
    
    if n_zero_career_skills > 0:
        pct = 100 * n_zero_career_skills / n_samples
        logger.warning(f"    ⚠️  {n_zero_career_skills}/{n_samples} ({pct:.1f}%) samples have no career skills")
    
    n_missing_targets = sum(1 for idx in true_target_indices if idx < 0)
    if n_missing_targets > 0:
        logger.warning(f"    ⚠️  {n_missing_targets}/{n_samples} samples have unmapped true targets")
        logger.warning(f"    ⚠️  Unmapped targets (n={len(unmapped_targets)}): {unmapped_targets}")
    
    return {
        'scores': scores,
        'target_labels': all_targets,
        'true_target_indices': true_target_indices,
        'histories': [h for h, _ in data_pairs],
        'true_targets': [t for _, t in data_pairs],
        'job_ids': job_ids_list,
        'split': split_name,
    }


# ============================================================================
# EVALUATION
# ============================================================================

def calculate_ranking_metrics(scores: np.ndarray, true_target_indices: List[int],
                              k_values: List[int] = [1, 5, 10, 20],
                              batch_size: int = 1000) -> Dict[str, float]:
    """
    Calculate ranking metrics from score matrix using batched processing.
    
    Matches the metric calculation in train_cpp_enhanced_v2.py.
    """
    n_samples = len(true_target_indices)
    
    reciprocal_ranks = []
    hits_at_k = {k: 0 for k in k_values}
    valid_samples = 0
    
    # Process in batches for memory efficiency
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        
        batch_scores = scores[start_idx:end_idx]
        batch_true_indices = true_target_indices[start_idx:end_idx]
        
        # Sort indices in descending order of score
        batch_sorted_indices = np.argsort(batch_scores, axis=1)[:, ::-1]
        
        for i, true_idx in enumerate(batch_true_indices):
            if true_idx < 0:
                continue

            valid_samples += 1
            
            # MRR
            rank_list = list(batch_sorted_indices[i])
            try:
                rank = rank_list.index(true_idx) + 1
                reciprocal_ranks.append(1.0 / rank)
            except ValueError:
                reciprocal_ranks.append(0.0)
            
            # Recall@K
            for k in k_values:
                if true_idx in batch_sorted_indices[i, :k]:
                    hits_at_k[k] += 1
    
    mrr = (np.mean(reciprocal_ranks) if valid_samples > 0 else 0.0)
    
    recall_at_k = {}
    for k in k_values:
        recall_at_k[f'R@{k}'] = hits_at_k[k] / valid_samples if valid_samples > 0 else 0.0
    
    metrics = {'MRR': mrr}
    metrics.update(recall_at_k)
    
    return metrics


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Skill Overlap Scoring for CPP (V2 - Unified)")
    
    # Data paths
    parser.add_argument("--data_type", type=str, default="decorte",
                        help="Dataset type: decorte, decorte_esco, karrierewege, etc.")
    parser.add_argument("--skill_scores_file", type=str, default=None,
                        help="Path to JSON file with skill scores keyed by job_id (e.g., best_fused_scores.json). "
                             "Required for 'decorte' (raw titles). Not needed for 'decorte_esco' (ESCO→ESCO).")
    parser.add_argument("--occupations_file", type=str, 
                        default="data/esco_datasets/occupations_en.csv",
                        help="Path to ESCO occupations CSV")
    parser.add_argument("--relations_file", type=str, 
                        default="data/esco_datasets/occupationSkillRelations_en.csv",
                        help="Path to ESCO occupation-skill relations CSV")
    
    # Data configuration
    parser.add_argument("--no_subspans", action='store_true',
                        help="Disable using all subspans of length at least 2")
    parser.add_argument("--eval_clean_test", action='store_true',
                        help="Evaluate on clean test set (no subspans)")
    parser.add_argument("--last_job_only", action='store_true',
                        help="Use only the last job in history for skill computation")
    parser.add_argument("--top_k_skills", type=int, default=None,
                        help="If provided, use only the top K skills from skill_scores_file. "
                             "Only applicable in job_id lookup mode.")
    parser.add_argument("--min_skill_confidence", type=float, default=None,
                        help="If provided, only consider skills with a score >= this value. "
                             "Only applicable in job_id lookup mode.")
    
    # Output
    parser.add_argument("--output_dir", type=str, 
                        default="results/cpp/skill_overlap_scores",
                        help="Directory to save skill overlap scores")
    parser.add_argument("--log_dir", type=str,
                        default="logs",
                        help="Directory for log files")
    
    # Which splits to process
    parser.add_argument("--splits", type=str, nargs='+', 
                        default=['train', 'val', 'test'],
                        help="Which splits to process (default: train val test)")

    parser.add_argument("--save_scores", action='store_true', help="Saves scores to output_dir.")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Also log to output_dir
    run_log_path = os.path.join(args.output_dir, "skill_overlap_scoring.log")
    logger.add(
        run_log_path,
        format="{time} | {level} | {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="31 days",
        enqueue=True
    )
    logger.info(f"📜 Logging this run to: {run_log_path}")

    logger.info("=" * 80)
    logger.info("Skill Overlap Scoring for Career Path Prediction (V2 - Unified)")
    logger.info("=" * 80)
    logger.info(f"Configuration: {vars(args)}\n")
    
    # Determine mode based on skill_scores_file
    use_esco_mode = args.skill_scores_file is None
    
    if use_esco_mode:
        logger.info("📌 Mode: ESCO title lookup (no skill_scores_file provided)")
        logger.info("   Career skills will be looked up via ESCO title → URI → skills")
        career_skill_map = None
    else:
        logger.info("📌 Mode: Job ID lookup (skill_scores_file provided)")
        logger.info("   Career skills will be looked up via job_id → predicted skills")
    
    # --- Step 1: Load Career Skill Map (job_id -> skills) - only if needed ---
    skill_stats = {}
    if not use_esco_mode:
        logger.info("\n[1/5] Loading career skill map (job_id → skills)...")
        career_skill_map, skill_stats = load_career_skill_map(
            args.skill_scores_file, 
            top_k_skills=args.top_k_skills,
            min_confidence=args.min_skill_confidence
        )
    else:
        logger.info("\n[1/5] Skipping career skill map (using ESCO mode)...")
        career_skill_map = None
    
    # --- Step 2: Load ESCO Target Skills ---
    logger.info("[2/5] Loading ESCO skill mappings...")
    title_to_uri, uri_to_skills = load_esco_target_skills(
        args.occupations_file, args.relations_file
    )
    
    # --- Step 3: Load Data ---
    logger.info("[3/5] Loading dataset...")
    data = Data(
        DATA_TYPE=args.data_type, 
        ONLY_TITLES=True,  # Always use titles for skill mapping
        consider_subspans=not args.no_subspans,
        LOAD_CLEAN_TEST=args.eval_clean_test
    )
    
    # Get data with job_ids (job_ids are used in job_id mode, ignored in ESCO mode)
    if args.eval_clean_test:
        (train_pairs, train_job_ids), (val_pairs, val_job_ids), \
        (test_pairs, test_job_ids), (test_clean_pairs, test_clean_job_ids) = \
            data.get_data_with_job_ids(stage='transformation_finetuning', include_clean_test=True)
    else:
        (train_pairs, train_job_ids), (val_pairs, val_job_ids), \
        (test_pairs, test_job_ids) = \
            data.get_data_with_job_ids(stage='transformation_finetuning', include_clean_test=False)
        test_clean_pairs, test_clean_job_ids = [], []
    
    logger.info(f"  ✓ Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}" + 
                (f", Test (clean): {len(test_clean_pairs)}" if args.eval_clean_test else ""))
    
    # Handle last_job_only filter
    if args.last_job_only:
        logger.info("  > Filtering to last job only...")
        
        def filter_last_job(pairs, job_ids_list):
            new_pairs = []
            new_job_ids = []
            for (h, t), job_ids in zip(pairs, job_ids_list):
                parts = h.split(SEP_TOKEN)
                last_job = parts[-1].strip() if parts else h
                new_pairs.append((last_job, t))
                new_job_ids.append([job_ids[-1]] if job_ids else [])
            return new_pairs, new_job_ids
        
        train_pairs, train_job_ids = filter_last_job(train_pairs, train_job_ids)
        val_pairs, val_job_ids = filter_last_job(val_pairs, val_job_ids)
        test_pairs, test_job_ids = filter_last_job(test_pairs, test_job_ids)
        if args.eval_clean_test:
            test_clean_pairs, test_clean_job_ids = filter_last_job(test_clean_pairs, test_clean_job_ids)
    
    # Collect all targets (unique occupations, normalized for consistency)
    raw_targets = [t for _, t in train_pairs + val_pairs + test_pairs]
    targets_by_norm = {}
    for target in raw_targets:
        norm = normalize_target_label(target)
        if norm and norm not in targets_by_norm:
            targets_by_norm[norm] = target.strip()
    all_targets = [targets_by_norm[norm] for norm in sorted(targets_by_norm.keys())]
    logger.info(f"  ✓ Total unique targets: {len(all_targets)} (from {len(raw_targets)} labels)")
    
    # Check coverage (only relevant in job_id mode)
    if not use_esco_mode:
        logger.info("  > Checking job_id coverage...")
        all_job_ids = set()
        for job_ids in train_job_ids + val_job_ids + test_job_ids:
            all_job_ids.update(job_ids)
        
        covered = sum(1 for jid in all_job_ids if jid in career_skill_map)
        logger.info(f"    Job IDs with skills: {covered}/{len(all_job_ids)} ({100*covered/len(all_job_ids):.1f}%)")
    else:
        # In ESCO mode, check how many history titles map to ESCO
        logger.info("  > Checking ESCO title coverage for histories...")
        all_history_titles = set()
        for h, _ in train_pairs + val_pairs + test_pairs:
            for title in h.split(SEP_TOKEN):
                title = title.strip().lower()
                if title:
                    all_history_titles.add(title)
        
        covered = sum(1 for t in all_history_titles if t in title_to_uri)
        logger.info(f"    History titles with ESCO mapping: {covered}/{len(all_history_titles)} ({100*covered/len(all_history_titles):.1f}%)")
    
    # --- Step 4: Process splits ---
    logger.info("[4/5] Computing skill overlap scores...")
    
    splits_data = {
        'train': (train_pairs, train_job_ids),
        'val': (val_pairs, val_job_ids),
        'test': (test_pairs, test_job_ids),
    }
    if args.eval_clean_test:
        splits_data['test_clean'] = (test_clean_pairs, test_clean_job_ids)
    
    all_metrics = {}
    
    for split_name in args.splits:
        if split_name not in splits_data:
            logger.warning(f"  ⚠️  Unknown split '{split_name}', skipping")
            continue
        
        if split_name == 'test_clean' and not args.eval_clean_test:
            logger.warning(f"  ⚠️  test_clean not loaded, skipping")
            continue
        
        pairs, job_ids = splits_data[split_name]
        
        if len(pairs) == 0:
            logger.warning(f"  ⚠️  {split_name} has no samples, skipping")
            continue
        
        # Compute scores
        scores_dict = compute_skill_overlap_scores_for_split(
            pairs, job_ids, all_targets, 
            career_skill_map, title_to_uri, uri_to_skills,
            split_name=split_name,
            use_esco_mode=use_esco_mode
        )
        
        # Calculate metrics
        metrics = calculate_ranking_metrics(
            scores_dict['scores'], scores_dict['true_target_indices']
        )
        all_metrics[split_name] = metrics
        logger.info(f"    {split_name} metrics: MRR={metrics['MRR']:.4f}, R@1={metrics['R@1']:.4f}, R@5={metrics['R@5']:.4f}, R@10={metrics['R@10']:.4f}")
        
        # Save scores
        if args.save_scores:
            output_path = os.path.join(args.output_dir, f"{split_name}_scores_skill_overlap.pkl")
            with open(output_path, 'wb') as f:
                pickle.dump(scores_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"    ✓ Saved to {output_path} (shape: {scores_dict['scores'].shape})")
        else:
            logger.info(f"    ✓ Saving is not requested.")
        
        # Clear memory
        del scores_dict
        gc.collect()
    
    # Handle test_clean if requested
    if args.eval_clean_test and 'test_clean' in splits_data and 'test_clean' not in args.splits:
        args.splits.append('test_clean')
        pairs, job_ids = splits_data['test_clean']
        
        if len(pairs) > 0:
            scores_dict = compute_skill_overlap_scores_for_split(
                pairs, job_ids, all_targets, 
                career_skill_map, title_to_uri, uri_to_skills,
                split_name='test_clean',
                use_esco_mode=use_esco_mode
            )
            
            metrics = calculate_ranking_metrics(
                scores_dict['scores'], scores_dict['true_target_indices']
            )
            all_metrics['test_clean'] = metrics
            logger.info(f"    test_clean metrics: MRR={metrics['MRR']:.4f}, R@1={metrics['R@1']:.4f}, R@5={metrics['R@5']:.4f}, R@10={metrics['R@10']:.4f}")
            
            if args.save_scores:
                output_path = os.path.join(args.output_dir, "test_clean_scores_skill_overlap.pkl")
                with open(output_path, 'wb') as f:
                    pickle.dump(scores_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(f"    ✓ Saved to {output_path}")
            
            del scores_dict
            gc.collect()
    
    # --- Step 5: Summary ---
    logger.info("\n" + "=" * 80)
    logger.info("SKILL OVERLAP SCORING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Mode: {'ESCO title lookup' if use_esco_mode else 'Job ID lookup'}")
    
    for split_name, metrics in all_metrics.items():
        logger.info(f"\n{split_name.upper()} METRICS:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
    
    # Save metrics to JSON with configuration for reproducibility
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    results_to_save = {
        'timestamp': pd.to_datetime('now').strftime('%Y-%m-%d %H:%M:%S'),
        'args': vars(args),
        'skill_filtering_stats': skill_stats,
        'metrics': all_metrics
    }
    with open(metrics_path, 'w') as f:
        json.dump(results_to_save, f, indent=4)
    logger.info(f"\nSaved detailed metrics and config to: {metrics_path}")

    logger.info("\n" + "=" * 80)
    logger.info(f"Scores saved to: {args.output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

