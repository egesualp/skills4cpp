"""
Skill Overlap Scoring for Career Path Prediction.

Given a career history with ESCO labels, aggregate skills over the career 
(union of all job's skills). Then score a candidate next occupation by skill overlap:
    S_skills = |career_skills ∩ target_skills| / |target_skills|

This produces scores S_skills that can be fused with MLP text scores S_text.
"""

import argparse
import os
import sys
import re
import pickle
import gc
from typing import Dict, List, Set, Tuple, Optional
from loguru import logger
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root and src to path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, "src"))

try:
    from src.cpp.data_classes import Data
    from src.cpp.utils import SEP_TOKEN
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
        os.path.join(log_dir, "skill_overlap_scoring.log"),
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
# ESCO MAPPING
# ============================================================================

def load_esco_mappings(occupations_file: str, relations_file: str) -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
    """
    Load ESCO mappings:
    1. Title -> URI (from occupations_file)
    2. URI -> Set[SkillURI] (from relations_file)
    """
    logger.info(f"Loading ESCO occupations from {occupations_file}")
    if not os.path.exists(occupations_file):
        raise FileNotFoundError(f"ESCO occupations file not found at {occupations_file}")
    
    occ_df = pd.read_csv(occupations_file)
    
    # Map preferredLabel -> URI
    title_to_uri = {}
    for _, row in occ_df.iterrows():
        uri = row['conceptUri']
        # Main label
        if pd.notna(row['preferredLabel']):
            pref_label = str(row['preferredLabel']).strip().lower()
            title_to_uri[pref_label] = uri
        
        # Alt labels
        if pd.notna(row['altLabels']):
            # altLabels are usually newline separated in ESCO CSVs
            alt_labels = str(row['altLabels']).split('\n')
            for alt in alt_labels:
                alt = alt.strip().lower()
                if alt:
                    title_to_uri[alt] = uri
                    
    logger.info(f"Loaded {len(title_to_uri)} title mappings (including altLabels)")
    
    logger.info(f"Loading ESCO skill relations from {relations_file}")
    if not os.path.exists(relations_file):
        raise FileNotFoundError(f"ESCO relations file not found at {relations_file}")

    rel_df = pd.read_csv(relations_file)
    
    # We want "S(occ) (including both essential and optional skills)"
    # Filter columns if needed? No, we take all skills linked to occupation.
    # Group by occupationUri
    
    uri_to_skills = {}
    # Optimization: iterate once
    # Or use pandas groupby
    grouped = rel_df.groupby('occupationUri')['skillUri'].apply(set)
    uri_to_skills = grouped.to_dict()
    
    logger.info(f"Loaded skill mappings for {len(uri_to_skills)} occupations")
    
    return title_to_uri, uri_to_skills


def get_skills_for_title(title: str, title_to_uri: Dict[str, str], uri_to_skills: Dict[str, Set[str]]) -> Set[str]:
    """Get skills for a single job title."""
    title_norm = title.strip().lower()
    
    # Try exact match
    uri = title_to_uri.get(title_norm)
    
    if not uri:
        # Fallback: maybe handle prefixes like "esco role:" here if needed
        # but typically the data loader handles cleaning or we just split by SEP
        # Let's try removing "esco role:" or "role:" just in case
        if "esco role:" in title_norm:
            clean_title = title_norm.replace("esco role:", "").strip()
            uri = title_to_uri.get(clean_title)
        elif "role:" in title_norm:
            clean_title = title_norm.replace("role:", "").strip()
            uri = title_to_uri.get(clean_title)
            
    if uri and uri in uri_to_skills:
        return uri_to_skills[uri]
        
    return set()


def get_skills_from_history(history_doc: str, title_to_uri: Dict[str, str], uri_to_skills: Dict[str, Set[str]]) -> Set[str]:
    """
    Get the union of all skills across a career history string.
    History doc is assumed to be titles separated by SEP_TOKEN.
    """
    career_skills = set()
    
    # Split by SEP_TOKEN
    # Some datasets might have newlines or other separators, but SEP_TOKEN is standard here
    titles = history_doc.split(SEP_TOKEN)
    
    for title in titles:
        skills = get_skills_for_title(title, title_to_uri, uri_to_skills)
        career_skills.update(skills)
            
    return career_skills


# ============================================================================
# SKILL OVERLAP SCORING
# ============================================================================

def compute_skill_overlap_score(career_skills: Set[str], target_skills: Set[str]) -> float:
    """
    Compute skill overlap score.
    
    Score = |career_skills ∩ target_skills| / |target_skills|
    
    Interpretation: How many of the target job's required skills does the candidate possess?
    
    Returns 0.0 if target has no skills (avoid div by zero).
    """
    if len(target_skills) == 0:
        return 0.0
    
    intersection = career_skills & target_skills
    return len(intersection) / len(target_skills)


def compute_skill_overlap_scores_for_split(
    data_pairs: List[Tuple[str, str]],
    all_targets: List[str],
    title_to_uri: Dict[str, str],
    uri_to_skills: Dict[str, Set[str]]
) -> Dict:
    """
    Compute skill overlap scores for all samples in a split.
    
    Args:
        data_pairs: List of (history_doc, target_doc) tuples
        all_targets: List of all possible target labels
        title_to_uri: Map title -> ESCO URI
        uri_to_skills: Map ESCO URI -> Set[SkillURI]
        
    Returns:
        Dictionary with scores and metadata.
    """
    n_samples = len(data_pairs)
    n_targets = len(all_targets)
    
    # Precompute skills for all targets
    logger.info("  > Precomputing skills for all targets...")
    target_skills_map = {}

    all_targets = [t.strip() for t in all_targets]

    for target in tqdm(all_targets, desc="  Target skills"):
        target_skills_map[target] = get_skills_for_title(target, title_to_uri, uri_to_skills)
    
    # Build target label to index mapping
    target_to_idx = {t: i for i, t in enumerate(all_targets)}
    
    # Compute scores
    logger.info(f"  > Computing overlap scores for {n_samples} samples...")
    scores = np.zeros((n_samples, n_targets), dtype=np.float32)
    true_target_indices = []
    
    # Statistics
    n_zero_career_skills = 0
    n_zero_target_skills = 0 # Count targets with no mapped skills
    
    for i, (history_doc, target_doc) in enumerate(tqdm(data_pairs, desc="  Scoring", total=n_samples)):
        # Get career skills from history string
        career_skills = get_skills_from_history(history_doc, title_to_uri, uri_to_skills)
        
        if len(career_skills) == 0:
            n_zero_career_skills += 1
        
        # Score all targets
        for j, target in enumerate(all_targets):
            target_skills = target_skills_map[target]
            scores[i, j] = compute_skill_overlap_score(career_skills, target_skills)
        
        # Record true target index
        true_idx = target_to_idx.get(target_doc.strip(), -1)
        true_target_indices.append(true_idx)
    
    if n_zero_career_skills > 0:
        logger.warning(f"  ⚠️  {n_zero_career_skills}/{n_samples} samples have no career skills (all scores = 0)")
    
    return {
        'scores': scores,
        'target_labels': all_targets,
        'true_target_indices': true_target_indices,
        'histories': [h.strip() for h, _ in data_pairs],
        'true_targets': [t.strip() for _, t in data_pairs],
    }


# ============================================================================
# EVALUATION
# ============================================================================

def calculate_ranking_metrics(scores: np.ndarray, true_target_indices: List[int],
                              k_values: List[int] = [1, 5, 10, 20],
                              batch_size: int = 1000) -> Dict[str, float]:
    """
    Calculate ranking metrics from score matrix using batched processing to save memory.
    """
    n_samples = len(true_target_indices)
    
    reciprocal_ranks = []
    hits_at_k = {k: 0 for k in k_values}
    
    # Process in batches
    for start_idx in tqdm(range(0, n_samples, batch_size), desc="  Calculating metrics"):
        end_idx = min(start_idx + batch_size, n_samples)
        
        # Get batch data
        batch_scores = scores[start_idx:end_idx]
        batch_true_indices = true_target_indices[start_idx:end_idx]
        
        # Sort indices in descending order of score for this batch only
        batch_sorted_indices = np.argsort(batch_scores, axis=1)[:, ::-1]
        
        for i, true_idx in enumerate(batch_true_indices):
            if true_idx < 0:
                reciprocal_ranks.append(0.0)
                continue
                
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
    
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    recall_at_k = {}
    for k in k_values:
        recall_at_k[f'R@{k}'] = hits_at_k[k] / n_samples if n_samples > 0 else 0.0
    
    metrics = {'MRR': mrr}
    metrics.update(recall_at_k)
    
    return metrics


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Skill Overlap Scoring for CPP")
    
    # Data paths
    parser.add_argument("--data_type", type=str, default="decorte")
    # Replaced skill_scores_file with ESCO files
    parser.add_argument("--occupations_file", type=str, default="data/esco_datasets/occupations_en.csv",
                        help="Path to ESCO occupations CSV")
    parser.add_argument("--relations_file", type=str, default="data/esco_datasets/occupationSkillRelations_en.csv",
                        help="Path to ESCO occupation-skill relations CSV")
    
    # Feature configuration
    parser.add_argument("--use_text_description", action='store_true',
                       help="Include job descriptions in data loading (not recommended for title-based skill mapping)")
    parser.add_argument("--last_job_only", action='store_true',
                       help="Use only the last job in history")
    
    # Output
    parser.add_argument("--output_dir", type=str, 
                       default="results/cpp/decorte/skill_overlap_scores",
                       help="Directory to save skill overlap scores")
    parser.add_argument("--log_dir", type=str,
                       default="src/cpp/decorte/logs",
                       help="Directory for log files")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_dir)
    
    logger.info("=" * 80)
    logger.info("Skill Overlap Scoring for Career Path Prediction (Title-based)")
    logger.info("=" * 80)
    logger.info(f"Configuration: {vars(args)}\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Step 1: Load ESCO Mappings ---
    logger.info("[1/4] Loading ESCO mappings...")
    try:
        title_to_uri, uri_to_skills = load_esco_mappings(args.occupations_file, args.relations_file)
    except Exception as e:
        logger.error(f"Failed to load ESCO mappings: {e}")
        return

    # --- Step 2: Load Data (Train/Val/Test) ---
    logger.info("[2/4] Loading dataset...")
    # We force ONLY_TITLES=True if we want clean title mapping, but the user might pass use_text_description
    # However, if use_text_description is True, the history string will contain descriptions.
    # Our simple split(SEP) parser might fail if descriptions are included.
    # We should warn if use_text_description is True.
    if args.use_text_description:
        logger.warning("Using text descriptions might interfere with title-based skill mapping. Ensure history contains clean titles if possible.")
        
    data = Data(DATA_TYPE=args.data_type, ONLY_TITLES=not args.use_text_description)
    (train_pairs, _), (val_pairs, _), (test_pairs, _) = data.get_data_with_job_ids(stage='transformation_finetuning')
    
    # Collect all targets from all splits to ensure consistent target set
    all_targets = list(set([t for _, t in train_pairs + val_pairs + test_pairs]))
    logger.info(f"  ✓ Total unique targets: {len(all_targets)}")
    
    # Clear memory of train pairs if we are only scoring val/test
    # But usually we might want to score train too? The original code scored val and test.
    # We'll stick to val and test as in the original code.
    del train_pairs
    gc.collect()
    
    # --- Step 3: Process splits sequentially ---
    logger.info("[3/4] Computing skill overlap scores sequentially...")
    
    splits_to_process = ['test', 'val']
    
    for split_name in splits_to_process:
        logger.info(f"\n  > Processing {split_name} split...")
        
        if split_name == 'test':
            current_pairs = test_pairs
        elif split_name == 'val':
            current_pairs = val_pairs
        else:
            continue
        
        # Filter if needed
        if args.last_job_only:
            # Keep only the last job in history (assuming SEP_TOKEN separates them)
            filtered_pairs = []
            for h, t in current_pairs:
                parts = h.split(SEP_TOKEN)
                last_job = parts[-1] if parts else h
                filtered_pairs.append((last_job, t))
            current_pairs = filtered_pairs
            
        logger.info(f"    Sample count: {len(current_pairs)}")
        
        # Compute scores
        scores_dict = compute_skill_overlap_scores_for_split(
            current_pairs, all_targets, title_to_uri, uri_to_skills
        )
        scores_dict['split'] = split_name
        
        # Evaluate if test
        if split_name == 'test':
            test_metrics = calculate_ranking_metrics(
                scores_dict['scores'], scores_dict['true_target_indices']
            )
            logger.info(f"  ✓ Test skill overlap metrics: {test_metrics}")
            
        # Save immediately
        output_path = os.path.join(args.output_dir, f"{split_name}_scores_skill_overlap.pkl")
        logger.info(f"    Saving {split_name} scores to {output_path}...")
        with open(output_path, 'wb') as f:
            pickle.dump(scores_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"    Saved. Shape: {scores_dict['scores'].shape}")
        
        # Clear memory
        del scores_dict
        gc.collect()

    logger.info("\n" + "=" * 80)
    logger.info("SKILL OVERLAP SCORING COMPLETE")
    logger.info("=" * 80)
    if 'test_metrics' in locals():
        logger.info(f"Test metrics (skill overlap only):")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
