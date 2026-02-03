"""
Occupation Score Fuser for Career Path Prediction.

Fuses scores from two sources for next occupation prediction:
1. MLP-based text scores (S_text) from train_cpp_enhanced_v2.py
2. Skill overlap scores (S_overlap) from skill_overlap_scoring_v2.py

Fusion modes:
- Linear:   S_hybrid = alpha × S_text + (1 - alpha) × S_overlap
- Bayesian: S_hybrid = S_text × (S_overlap + epsilon)^w

Features:
- Grid search over hyperparameters (alpha/weight, optional normalization)
- Parallel processing for fast grid search
- Comprehensive metrics (MRR, Recall@K)
- Logging with loguru
- Compatible output format for downstream analysis

Usage:
    python -m cpp.decorte.occupation_score_fuser \
        --text_scores_dir results/cpp/decorte_static/job_titles_only/scores \
        --overlap_scores_dir results/cpp/decorte/skill_overlap_scores \
        --output_dir results/cpp/decorte/fused_scores \
        --fusion_mode linear \
        --alphas 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
        --splits test \
        --save_scores
"""

import argparse
import gc
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Tuple, Literal

import numpy as np
from joblib import Parallel, delayed
from loguru import logger


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_dir: str, log_file: str = "occupation_score_fuser.log"):
    """Configure logging to both file and stdout."""
    os.makedirs(log_dir, exist_ok=True)
    
    logger.remove()
    logger.add(
        os.path.join(log_dir, log_file),
        format="{time} | {level} | {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        enqueue=True
    )
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>",
        level="INFO"
    )


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class FusionConfig:
    """Configuration for fusion hyperparameters."""
    fusion_mode: Literal["bayesian", "linear"] = "linear"
    alpha: float = 0.5         # For linear: weight on text scores
    weight: float = 1.0        # For Bayesian: exponent on overlap scores
    epsilon: float = 1e-6      # For Bayesian: small value to avoid zero
    normalize_text: bool = False    # Whether to normalize text scores
    normalize_overlap: bool = False # Whether to normalize overlap scores
    norm_method: Literal["minmax", "zscore"] = "minmax" # Normalization method


@dataclass
class ScoreData:
    """Container for score data from a single source."""
    scores: np.ndarray           # [n_samples, n_targets]
    target_labels: List[str]     # [n_targets] - normalized (stripped)
    true_target_indices: List[int]  # [n_samples]
    histories: Optional[List[str]] = None
    true_targets: Optional[List[str]] = None
    split: str = ""
    
    # Mapping for reordering
    label_to_idx: Optional[Dict[str, int]] = None
    
    def __post_init__(self):
        """Build label to index mapping."""
        if self.label_to_idx is None:
            self.label_to_idx = {label: i for i, label in enumerate(self.target_labels)}


# ============================================================================
# DATA LOADING
# ============================================================================

def extract_occupation_name(label: str) -> str:
    """
    Extract occupation name from a label that may contain description.
    
    Handles formats like:
    - "esco role: cook \n description: ..." -> "cook"
    - "role: cook \n ..." -> "cook"
    - "cook" -> "cook"
    - "cook " -> "cook"
    
    Args:
        label: Raw label string
        
    Returns:
        Extracted and normalized occupation name
    """
    import re
    
    label = label.strip()
    
    # Try to extract from "esco role: <name> \n" pattern
    match = re.match(r"esco role:\s*(.+?)\s*\n", label, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    
    # Try to extract from "role: <name> \n" pattern
    match = re.match(r"role:\s*(.+?)\s*\n", label, re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    
    # No pattern matched, return normalized label
    return label.strip().lower()


def load_scores(pkl_path: str) -> ScoreData:
    """
    Load scores from pickle file and normalize labels.
    
    Args:
        pkl_path: Path to pickle file with scores
        
    Returns:
        ScoreData with normalized labels
    """
    logger.debug(f"Loading scores from {pkl_path}")
    
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # Normalize target labels (strip whitespace)
    target_labels = [label.strip() for label in data['target_labels']]
    
    # Normalize true_targets if present
    true_targets = None
    if 'true_targets' in data:
        true_targets = [t.strip() for t in data['true_targets']]
    
    return ScoreData(
        scores=data['scores'],
        target_labels=target_labels,
        true_target_indices=data['true_target_indices'],
        histories=data.get('histories'),
        true_targets=true_targets,
        split=data.get('split', '')
    )


def align_scores(text_data: ScoreData, overlap_data: ScoreData) -> Tuple[np.ndarray, np.ndarray, List[str], List[int]]:
    """
    Align score matrices to have the same target label ordering.
    
    Handles cases where labels have different formats (e.g., one has descriptions,
    one has just names) by extracting occupation names for matching.
    
    Uses text_data's label ordering as the canonical order and reorders
    overlap_data's scores to match.
    
    Args:
        text_data: ScoreData from text-based MLP
        overlap_data: ScoreData from skill overlap scoring
        
    Returns:
        Tuple of (aligned_text_scores, aligned_overlap_scores, target_labels, true_target_indices)
    """
    # Verify same number of samples
    if text_data.scores.shape[0] != overlap_data.scores.shape[0]:
        raise ValueError(
            f"Sample count mismatch: text has {text_data.scores.shape[0]}, "
            f"overlap has {overlap_data.scores.shape[0]}"
        )
    
    # Extract normalized occupation names for matching
    text_names = [extract_occupation_name(label) for label in text_data.target_labels]
    overlap_names = [extract_occupation_name(label) for label in overlap_data.target_labels]
    
    # Build mapping from normalized name to index for overlap data
    overlap_name_to_idx = {name: i for i, name in enumerate(overlap_names)}
    
    # Check for duplicates (shouldn't happen but let's be safe)
    if len(overlap_name_to_idx) != len(overlap_names):
        logger.warning(f"  ⚠️  Duplicate occupation names detected in overlap labels")
    
    # Verify same target labels (as sets of normalized names)
    text_names_set = set(text_names)
    overlap_names_set = set(overlap_names)
    
    if text_names_set != overlap_names_set:
        missing_in_overlap = text_names_set - overlap_names_set
        missing_in_text = overlap_names_set - text_names_set
        
        # Log some examples for debugging
        if missing_in_overlap:
            examples = list(missing_in_overlap)[:5]
            logger.error(f"  Examples missing in overlap: {examples}")
        if missing_in_text:
            examples = list(missing_in_text)[:5]
            logger.error(f"  Examples missing in text: {examples}")
        
        raise ValueError(
            f"Target label mismatch! "
            f"Missing in overlap: {len(missing_in_overlap)}, "
            f"Missing in text: {len(missing_in_text)}"
        )
    
    # Use text_data's ordering as canonical
    canonical_labels = text_data.target_labels
    
    # Build reordering index for overlap scores using normalized names
    # overlap_reorder[i] = j means overlap's column j should go to position i
    overlap_reorder = []
    for text_name in text_names:
        overlap_idx = overlap_name_to_idx[text_name]
        overlap_reorder.append(overlap_idx)
    
    overlap_reorder = np.array(overlap_reorder)
    
    # Reorder overlap scores to match text ordering
    aligned_overlap_scores = overlap_data.scores[:, overlap_reorder]
    
    # Verify true targets match (after alignment, using normalized names)
    if text_data.true_targets and overlap_data.true_targets:
        text_true_names = [extract_occupation_name(t) for t in text_data.true_targets]
        overlap_true_names = [extract_occupation_name(t) for t in overlap_data.true_targets]
        mismatches = sum(
            1 for t, o in zip(text_true_names, overlap_true_names)
            if t != o
        )
        if mismatches > 0:
            logger.warning(f"  ⚠️  {mismatches} true_targets mismatches after alignment")
    
    logger.info(f"  ✓ Aligned {text_data.scores.shape[1]} targets, {text_data.scores.shape[0]} samples")
    
    return (
        text_data.scores,
        aligned_overlap_scores,
        canonical_labels,
        text_data.true_target_indices
    )


# ============================================================================
# NORMALIZATION
# ============================================================================

def normalize_scores(scores: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize scores to [0, 1] range.
    
    Args:
        scores: Score matrix [n_samples, n_targets]
        method: Normalization method ('minmax' or 'zscore')
        
    Returns:
        Normalized scores
    """
    if method == "minmax":
        # Per-row min-max normalization
        row_min = scores.min(axis=1, keepdims=True)
        row_max = scores.max(axis=1, keepdims=True)
        denom = row_max - row_min
        denom = np.where(denom == 0, 1, denom)  # Avoid division by zero
        return (scores - row_min) / denom
    elif method == "zscore":
        # Per-row z-score normalization
        row_mean = scores.mean(axis=1, keepdims=True)
        row_std = scores.std(axis=1, keepdims=True)
        row_std = np.where(row_std == 0, 1, row_std)
        return (scores - row_mean) / row_std
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# ============================================================================
# FUSION
# ============================================================================

def fuse_scores(
    text_scores: np.ndarray,
    overlap_scores: np.ndarray,
    config: FusionConfig
) -> np.ndarray:
    """
    Fuse text and overlap scores according to configuration.
    
    Args:
        text_scores: [n_samples, n_targets] from MLP
        overlap_scores: [n_samples, n_targets] from skill overlap
        config: FusionConfig with fusion parameters
        
    Returns:
        Fused scores [n_samples, n_targets]
    """
    # Apply normalization if requested
    if config.normalize_text:
        text_scores = normalize_scores(text_scores, method=config.norm_method)
    if config.normalize_overlap:
        overlap_scores = normalize_scores(overlap_scores, method=config.norm_method)
    
    if config.fusion_mode == "linear":
        # Linear: S_hybrid = alpha × S_text + (1 - alpha) × S_overlap
        fused = config.alpha * text_scores + (1 - config.alpha) * overlap_scores
    elif config.fusion_mode == "bayesian":
        # Bayesian: S_hybrid = S_text × (S_overlap + epsilon)^w
        # Add epsilon to avoid zero values
        overlap_safe = overlap_scores + config.epsilon
        fused = text_scores * (overlap_safe ** config.weight)
    else:
        raise ValueError(f"Unknown fusion mode: {config.fusion_mode}")
    
    return fused


# ============================================================================
# METRICS
# ============================================================================

def calculate_ranking_metrics(
    scores: np.ndarray,
    true_target_indices: List[int],
    k_values: List[int] = [1, 5, 10, 20]
) -> Dict[str, float]:
    """
    Calculate ranking metrics from score matrix.
    
    Args:
        scores: [n_samples, n_targets] score matrix
        true_target_indices: List of true target indices for each sample
        k_values: List of K values for Recall@K
        
    Returns:
        Dictionary with MRR and Recall@K metrics
    """
    n_samples = len(true_target_indices)
    
    # Sort indices in descending order of score
    sorted_indices = np.argsort(-scores, axis=1)
    
    # Calculate MRR
    reciprocal_ranks = []
    valid_samples = 0
    
    for i, true_idx in enumerate(true_target_indices):
        if true_idx < 0:
            reciprocal_ranks.append(0.0)
            continue
        
        valid_samples += 1
        rank_list = list(sorted_indices[i])
        try:
            rank = rank_list.index(true_idx) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)
    
    mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    # Calculate Recall@K
    recall_at_k = {}
    for k in k_values:
        hits = 0
        for i, true_idx in enumerate(true_target_indices):
            if true_idx < 0:
                continue
            if true_idx in sorted_indices[i, :k]:
                hits += 1
        recall_at_k[f'R@{k}'] = hits / valid_samples if valid_samples > 0 else 0.0
    
    metrics = {'MRR': mrr}
    metrics.update(recall_at_k)
    metrics['valid_samples'] = valid_samples
    
    return metrics


# ============================================================================
# GRID SEARCH
# ============================================================================

def make_config_key(fusion_mode: str, config_tuple: Tuple) -> str:
    """Create a string key from config tuple for JSON serialization."""
    if fusion_mode == "linear":
        alpha, norm_text, norm_overlap, norm_method = config_tuple
        return f"alpha{alpha:.2f}_normT{int(norm_text)}_normO{int(norm_overlap)}_{norm_method}"
    else:  # bayesian
        weight, epsilon, norm_text, norm_overlap, norm_method = config_tuple
        return f"w{weight:.2f}_eps{epsilon:.0e}_normT{int(norm_text)}_normO{int(norm_overlap)}_{norm_method}"


def process_config(
    config_tuple: Tuple,
    fusion_mode: str,
    text_scores: np.ndarray,
    overlap_scores: np.ndarray,
    true_target_indices: List[int],
    k_values: List[int]
) -> Tuple[Tuple, str, Dict[str, float], np.ndarray]:
    """
    Process a single configuration and return metrics.
    
    Args:
        config_tuple: Tuple of hyperparameters
        fusion_mode: 'linear' or 'bayesian'
        text_scores: Aligned text scores
        overlap_scores: Aligned overlap scores
        true_target_indices: True target indices
        k_values: K values for metrics
        
    Returns:
        Tuple of (config_tuple, config_key, metrics, fused_scores)
    """
    if fusion_mode == "linear":
        alpha, norm_text, norm_overlap, norm_method = config_tuple
        config = FusionConfig(
            fusion_mode="linear",
            alpha=alpha,
            normalize_text=norm_text,
            normalize_overlap=norm_overlap,
            norm_method=norm_method
        )
    else:  # bayesian
        weight, epsilon, norm_text, norm_overlap, norm_method = config_tuple
        config = FusionConfig(
            fusion_mode="bayesian",
            weight=weight,
            epsilon=epsilon,
            normalize_text=norm_text,
            normalize_overlap=norm_overlap,
            norm_method=norm_method
        )
    
    config_key = make_config_key(fusion_mode, config_tuple)
    
    # Fuse scores
    fused_scores = fuse_scores(text_scores, overlap_scores, config)
    
    # Calculate metrics
    metrics = calculate_ranking_metrics(fused_scores, true_target_indices, k_values)
    
    return config_tuple, config_key, metrics, fused_scores


def process_config_chunk(
    config_chunk: List[Tuple],
    fusion_mode: str,
    text_scores: np.ndarray,
    overlap_scores: np.ndarray,
    true_target_indices: List[int],
    k_values: List[int]
) -> List[Tuple[Tuple, str, Dict[str, float], None]]:
    """
    Process a chunk of configurations in a single worker.
    Returns None for fused_scores to save memory during parallel processing.
    """
    results = []
    for config_tuple in config_chunk:
        config_tuple, config_key, metrics, _ = process_config(
            config_tuple, fusion_mode, text_scores, overlap_scores,
            true_target_indices, k_values
        )
        # Don't return fused_scores to save memory
        results.append((config_tuple, config_key, metrics, None))
    return results


def run_grid_search(
    text_scores: np.ndarray,
    overlap_scores: np.ndarray,
    true_target_indices: List[int],
    fusion_mode: str,
    alphas: List[float],
    weights: List[float],
    epsilons: List[float],
    normalize_options: List[Tuple[bool, bool]],
    norm_methods: List[str] = ["minmax"],
    k_values: List[int] = [1, 5, 10, 20],
    num_workers: int = 1,
    metric_for_best: str = "MRR"
) -> Tuple[Dict[Tuple, Dict], Tuple, Dict[str, float]]:
    """
    Run grid search over hyperparameters.
    
    Args:
        text_scores: Aligned text scores [n_samples, n_targets]
        overlap_scores: Aligned overlap scores [n_samples, n_targets]
        true_target_indices: True target indices
        fusion_mode: 'linear' or 'bayesian'
        alphas: List of alpha values (for linear)
        weights: List of weight values (for bayesian)
        epsilons: List of epsilon values (for bayesian)
        normalize_options: List of (norm_text, norm_overlap) tuples
        k_values: K values for metrics
        num_workers: Number of parallel workers
        metric_for_best: Metric to use for selecting best config
        
    Returns:
        Tuple of (all_metrics, best_config, best_metrics)
    """
    # Build parameter grid
    if fusion_mode == "linear":
        param_grid = list(product(alphas, normalize_options, norm_methods))
        # Flatten normalize_options into config tuple
        param_grid = [(alpha, norm[0], norm[1], method) for alpha, norm, method in param_grid]
        logger.info(f"Grid search: {len(alphas)} alphas × {len(normalize_options)} norm options × {len(norm_methods)} methods = {len(param_grid)} configs")
    else:  # bayesian
        param_grid = list(product(weights, epsilons, normalize_options, norm_methods))
        param_grid = [(w, eps, norm[0], norm[1], method) for w, eps, norm, method in param_grid]
        logger.info(f"Grid search: {len(weights)} weights × {len(epsilons)} epsilons × {len(normalize_options)} norm options × {len(norm_methods)} methods = {len(param_grid)} configs")
    
    all_metrics = {}
    best_config = None
    best_metric_value = -float('inf')
    
    start_time = time.time()
    
    if num_workers != 1:
        # Parallel execution
        import math
        chunk_size = math.ceil(len(param_grid) / abs(num_workers))
        chunks = [param_grid[i:i + chunk_size] for i in range(0, len(param_grid), chunk_size)]
        
        logger.info(f"Running PARALLEL grid search with {abs(num_workers)} workers ({len(chunks)} chunks)")
        
        results_nested = Parallel(n_jobs=num_workers, verbose=1)(
            delayed(process_config_chunk)(
                chunk, fusion_mode, text_scores, overlap_scores,
                true_target_indices, k_values
            )
            for chunk in chunks
        )
        
        # Flatten results
        for chunk_results in results_nested:
            for config_tuple, config_key, metrics, _ in chunk_results:
                all_metrics[config_tuple] = metrics
                
                if metrics[metric_for_best] > best_metric_value:
                    best_metric_value = metrics[metric_for_best]
                    best_config = config_tuple
    else:
        # Sequential execution with progress logging
        for idx, config_tuple in enumerate(param_grid):
            config_tuple, config_key, metrics, _ = process_config(
                config_tuple, fusion_mode, text_scores, overlap_scores,
                true_target_indices, k_values
            )
            all_metrics[config_tuple] = metrics
            
            if metrics[metric_for_best] > best_metric_value:
                best_metric_value = metrics[metric_for_best]
                best_config = config_tuple
            
            # Progress logging
            if (idx + 1) % 10 == 0 or idx == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (idx + 1)
                remaining = avg_time * (len(param_grid) - idx - 1)
                logger.info(
                    f"  [{idx+1}/{len(param_grid)}] {config_key}: "
                    f"MRR={metrics['MRR']:.4f}, R@1={metrics['R@1']:.4f} "
                    f"({avg_time:.2f}s/config, ~{remaining:.0f}s remaining)"
                )
    
    total_time = time.time() - start_time
    logger.info(f"Grid search completed in {total_time:.1f}s ({total_time/len(param_grid):.3f}s per config)")
    
    best_metrics = all_metrics[best_config]
    
    return all_metrics, best_config, best_metrics


def print_metrics_table(
    all_metrics: Dict[Tuple, Dict[str, float]],
    fusion_mode: str,
    k_values: List[int] = [1, 5, 10, 20]
):
    """Print a formatted comparison table of metrics."""
    logger.info("=" * 100)
    logger.info(f"FUSION RESULTS - Grid Search ({fusion_mode.upper()} mode)")
    logger.info("=" * 100)
    
    # Sort by MRR descending
    sorted_configs = sorted(
        all_metrics.items(),
        key=lambda x: x[1].get("MRR", 0.0),
        reverse=True
    )
    
    # Build header
    if fusion_mode == "linear":
        header = f"{'alpha':<8} {'nT':<4} {'nO':<4} {'method':<10}"
    else:
        header = f"{'weight':<8} {'eps':<10} {'nT':<4} {'nO':<4} {'method':<10}"
    
    header += f" {'MRR':<8}"
    for k in k_values:
        header += f" {'R@'+str(k):<8}"
    
    logger.info(header)
    logger.info("-" * 100)
    
    # Show top 20 configurations
    for config_tuple, m in sorted_configs[:20]:
        if fusion_mode == "linear":
            alpha, norm_text, norm_overlap, norm_method = config_tuple
            row = f"{alpha:<8.2f} {int(norm_text):<4} {int(norm_overlap):<4} {norm_method:<10}"
        else:
            weight, epsilon, norm_text, norm_overlap, norm_method = config_tuple
            row = f"{weight:<8.2f} {epsilon:<10.0e} {int(norm_text):<4} {int(norm_overlap):<4} {norm_method:<10}"
        
        row += f" {m.get('MRR', 0.0):<8.4f}"
        for k in k_values:
            row += f" {m.get(f'R@{k}', 0.0):<8.4f}"
        
        logger.info(row)
    
    if len(sorted_configs) > 20:
        logger.info(f"... ({len(sorted_configs) - 20} more configurations not shown)")
    
    logger.info("=" * 100)
    
    # Report best configuration
    best_config, best_metrics = sorted_configs[0]
    if fusion_mode == "linear":
        alpha, norm_text, norm_overlap, norm_method = best_config
        logger.info(f"BEST CONFIG: alpha={alpha:.2f}, normalize_text={norm_text}, normalize_overlap={norm_overlap}, method={norm_method}")
    else:
        weight, epsilon, norm_text, norm_overlap, norm_method = best_config
        logger.info(f"BEST CONFIG: weight={weight:.2f}, epsilon={epsilon:.0e}, normalize_text={norm_text}, normalize_overlap={norm_overlap}, method={norm_method}")
    
    logger.info(f"  MRR: {best_metrics['MRR']:.4f}")
    for k in k_values:
        logger.info(f"  R@{k}: {best_metrics.get(f'R@{k}', 0.0):.4f}")


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuse MLP text scores with skill overlap scores for next occupation prediction"
    )
    
    # Input paths
    parser.add_argument("--text_scores_dir", type=str, required=True,
                        help="Directory containing *_scores_text.pkl files from train_cpp_enhanced_v2.py")
    parser.add_argument("--overlap_scores_dir", type=str, required=True,
                        help="Directory containing *_scores_skill_overlap.pkl files from skill_overlap_scoring_v2.py")
    
    # Output
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save fused scores and metrics")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Directory for log files (defaults to output_dir)")
    
    # Fusion mode
    parser.add_argument("--fusion_mode", type=str, choices=["linear", "bayesian"],
                        default="linear",
                        help="Fusion mode: 'linear' (alpha blend) or 'bayesian' (multiplicative)")
    
    # Linear fusion parameters
    parser.add_argument("--alphas", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                        help="Comma-separated alpha values for linear fusion (weight on text scores)")
    
    # Bayesian fusion parameters
    parser.add_argument("--weights", type=str, default="0.0,0.5,1.0,1.5,2.0",
                        help="Comma-separated weight values for Bayesian fusion")
    parser.add_argument("--epsilons", type=str, default="1e-6",
                        help="Comma-separated epsilon values for Bayesian fusion")
    
    # Normalization options
    parser.add_argument("--normalize_text", action='store_true',
                        help="Normalize text scores to [0,1] (single config mode)")
    parser.add_argument("--normalize_overlap", action='store_true',
                        help="Normalize overlap scores to [0,1] (single config mode)")
    parser.add_argument("--grid_normalize", action='store_true',
                        help="Include normalization options (enabled/disabled) in grid search")
    parser.add_argument("--norm_methods", type=str, default="minmax",
                        help="Comma-separated normalization methods to include in grid search (minmax,zscore)")
    
    # Splits to process
    parser.add_argument("--splits", type=str, nargs='+', default=['val'],
                        help="Which splits to process for tuning (default: val)")
    parser.add_argument("--eval_splits", type=str, nargs='+', default=['test', 'clean_test'],
                        help="Which splits to evaluate best config on (default: test clean_test)")
    
    # Parallelization
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel workers for grid search. Use -1 for all cores.")
    
    # Output options
    parser.add_argument("--save_scores", action='store_true',
                        help="Save fused scores to pickle file")
    parser.add_argument("--metric_for_best", type=str, default="MRR",
                        help="Metric to use for selecting best configuration")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    log_dir = args.log_dir if args.log_dir else args.output_dir
    setup_logging(log_dir)
    
    logger.info("=" * 80)
    logger.info("Occupation Score Fuser for Career Path Prediction")
    logger.info("=" * 80)
    logger.info(f"Configuration: {vars(args)}\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    config_path = os.path.join(args.output_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    logger.info(f"Saved configuration to: {config_path}")
    
    # Parse hyperparameters
    alphas = [float(a.strip()) for a in args.alphas.split(",")]
    weights = [float(w.strip()) for w in args.weights.split(",")]
    epsilons = [float(e.strip()) for e in args.epsilons.split(",")]
    
    # Determine normalization options for grid search
    if args.grid_normalize:
        normalize_options = [(False, False), (True, False), (False, True), (True, True)]
    else:
        normalize_options = [(args.normalize_text, args.normalize_overlap)]
    
    logger.info(f"Fusion mode: {args.fusion_mode}")
    if args.fusion_mode == "linear":
        logger.info(f"Alphas: {alphas}")
    else:
        logger.info(f"Weights: {weights}")
        logger.info(f"Epsilons: {epsilons}")
    
    # Parse normalization methods
    norm_methods = [m.strip() for m in args.norm_methods.split(",")]
    
    logger.info(f"Normalization options: {normalize_options}")
    logger.info(f"Normalization methods: {norm_methods}")
    logger.info(f"Splits to process: {args.splits}")
    
    # K values for metrics
    k_values = [1, 5, 10, 20]
    
    # Process each tuning split
    tuning_split_results: Dict[str, Dict] = {}
    best_overall_config = None
    best_overall_metrics = None
    best_overall_split = None
    best_metric_value = -float('inf')
    
    for split in args.splits:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing split: {split}")
        logger.info(f"{'='*60}")
        
        # Construct file paths
        text_file = os.path.join(args.text_scores_dir, f"{split}_scores_text.pkl")
        overlap_file = os.path.join(args.overlap_scores_dir, f"{split}_scores_skill_overlap.pkl")

        # Check files exist
        if not os.path.exists(text_file):
            logger.warning(f"  ⚠️  Text scores file not found: {text_file}")
            continue
        if not os.path.exists(overlap_file):
            logger.warning(f"  ⚠️  Overlap scores file not found: {overlap_file}")
            continue
        
        # Load scores
        logger.info(f"  Loading text scores from {text_file}")
        text_data = load_scores(text_file)
        logger.info(f"    Shape: {text_data.scores.shape}")
        
        logger.info(f"  Loading overlap scores from {overlap_file}")
        overlap_data = load_scores(overlap_file)
        logger.info(f"    Shape: {overlap_data.scores.shape}")
        
        # Align scores
        logger.info("  Aligning score matrices...")
        text_scores, overlap_scores, target_labels, true_target_indices = align_scores(
            text_data, overlap_data
        )
        
        # Log score statistics
        logger.info(f"  Text scores - min: {text_scores.min():.4f}, max: {text_scores.max():.4f}, mean: {text_scores.mean():.4f}")
        logger.info(f"  Overlap scores - min: {overlap_scores.min():.4f}, max: {overlap_scores.max():.4f}, mean: {overlap_scores.mean():.4f}")
        
        # Run grid search
        logger.info(f"  Running grid search...")
        all_metrics, best_config, best_metrics = run_grid_search(
            text_scores=text_scores,
            overlap_scores=overlap_scores,
            true_target_indices=true_target_indices,
            fusion_mode=args.fusion_mode,
            alphas=alphas,
            weights=weights,
            epsilons=epsilons,
            normalize_options=normalize_options,
            norm_methods=norm_methods,
            k_values=k_values,
            num_workers=args.num_workers,
            metric_for_best=args.metric_for_best
        )
        
        # Print results table
        print_metrics_table(all_metrics, args.fusion_mode, k_values)
        
        # Store results
        tuning_split_results[split] = {
            'all_metrics': all_metrics,
            'best_config': best_config,
            'best_metrics': best_metrics,
            'target_labels': target_labels,
            'true_target_indices': true_target_indices,
        }
        
        # Track best config across tuning splits
        metric_value = best_metrics.get(args.metric_for_best, -float('inf'))
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_overall_config = best_config
            best_overall_metrics = best_metrics
            best_overall_split = split
        
        # Save fused scores if requested
        if args.save_scores:
            # Recompute best fused scores for saving
            _, _, _, best_fused_scores = process_config(
                best_config, args.fusion_mode, text_scores, overlap_scores,
                true_target_indices, k_values
            )
            
            # Build output dict matching expected format
            output_dict = {
                'scores': best_fused_scores,
                'target_labels': target_labels,
                'true_target_indices': true_target_indices,
                'histories': text_data.histories,
                'true_targets': text_data.true_targets,
                'split': split,
                'fusion_config': {
                    'fusion_mode': args.fusion_mode,
                    'config': best_config,
                    'config_key': make_config_key(args.fusion_mode, best_config),
                }
            }
            
            output_path = os.path.join(args.output_dir, f"{split}_scores_fused.pkl")
            with open(output_path, 'wb') as f:
                pickle.dump(output_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"  ✓ Saved fused scores to {output_path}")
        
        # Clean up memory
        del text_data, overlap_data, text_scores, overlap_scores
        gc.collect()
    
    if best_overall_config is None:
        logger.error("No tuning splits produced a best configuration; exiting.")
        return
    
    best_config_key = make_config_key(args.fusion_mode, best_overall_config)
    logger.info(f"\nBest tuning split determined: {best_overall_split} → {best_config_key}")

    # Evaluate best config on designated evaluation splits
    evaluation_results: Dict[str, Dict] = {}
    for eval_split in args.eval_splits:
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating best config on split: {eval_split}")
        logger.info(f"{'='*60}")

        text_file = os.path.join(args.text_scores_dir, f"{eval_split}_scores_text.pkl")
        overlap_file = os.path.join(args.overlap_scores_dir, f"{eval_split}_scores_skill_overlap.pkl")
        if not os.path.exists(text_file):
            logger.warning(f"  ⚠️  Text scores file not found: {text_file}")
            continue
        if not os.path.exists(overlap_file):
            logger.warning(f"  ⚠️  Overlap scores file not found: {overlap_file}")
            continue

        eval_text_data = load_scores(text_file)
        eval_overlap_data = load_scores(overlap_file)
        eval_text_scores, eval_overlap_scores, eval_target_labels, eval_true_target_indices = align_scores(
            eval_text_data, eval_overlap_data
        )
        logger.info(f"  Evaluation text scores shape: {eval_text_scores.shape}")
        logger.info(f"  Evaluation overlap scores shape: {eval_overlap_scores.shape}")

        _, _, eval_metrics, eval_fused_scores = process_config(
            best_overall_config,
            args.fusion_mode,
            eval_text_scores,
            eval_overlap_scores,
            eval_true_target_indices,
            k_values
        )

        evaluation_results[eval_split] = {
            'metrics': eval_metrics,
            'target_labels': eval_target_labels,
            'true_target_indices': eval_true_target_indices,
        }

        logger.info(f"  Evaluation metrics: MRR={eval_metrics['MRR']:.4f}")
        for k in k_values:
            logger.info(f"    R@{k}: {eval_metrics.get(f'R@{k}', 0.0):.4f}")

        if args.save_scores:
            eval_output = {
                'scores': eval_fused_scores,
                'target_labels': eval_target_labels,
                'true_target_indices': eval_true_target_indices,
                'histories': eval_text_data.histories,
                'true_targets': eval_text_data.true_targets,
                'split': eval_split,
                'fusion_config': {
                    'fusion_mode': args.fusion_mode,
                    'config': best_overall_config,
                    'config_key': best_config_key,
                }
            }
            output_path = os.path.join(args.output_dir, f"{eval_split}_scores_fused.pkl")
            with open(output_path, 'wb') as f:
                pickle.dump(eval_output, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"  ✓ Saved evaluation fused scores to {output_path}")

        del eval_text_data, eval_overlap_data, eval_text_scores, eval_overlap_scores
        gc.collect()
    
    # Save grid search metrics to JSON
    metrics_output = {
        'fusion_mode': args.fusion_mode,
        'metric_for_best': args.metric_for_best,
        'config': {
            'alphas': alphas if args.fusion_mode == 'linear' else None,
            'weights': weights if args.fusion_mode == 'bayesian' else None,
            'epsilons': epsilons if args.fusion_mode == 'bayesian' else None,
            'normalize_options': normalize_options,
            'norm_methods': norm_methods,
        },
        'best_tuning_split': best_overall_split,
        'best_config': {
            'key': best_config_key,
            'params': list(best_overall_config),
        },
        'tuning_splits': {},
        'evaluation_splits': {}
    }
    
    for split, results in tuning_split_results.items():
        best_config = results['best_config']
        config_key = make_config_key(args.fusion_mode, best_config)
        
        # Convert all_metrics to JSON-serializable format
        all_metrics_json = {}
        for cfg, m in results['all_metrics'].items():
            cfg_key = make_config_key(args.fusion_mode, cfg)
            all_metrics_json[cfg_key] = m
        
        metrics_output['tuning_splits'][split] = {
            'best_config': {
                'key': config_key,
                'params': list(best_config),
            },
            'best_metrics': results['best_metrics'],
            'all_results': [
                {
                    'config_key': make_config_key(args.fusion_mode, cfg),
                    'params': list(cfg),
                    'metrics': m,
                }
                for cfg, m in sorted(
                    results['all_metrics'].items(),
                    key=lambda x: x[1].get('MRR', 0.0),
                    reverse=True
                )
            ]
        }

    for split, eval_result in evaluation_results.items():
        metrics_output['evaluation_splits'][split] = {
            'metrics': eval_result['metrics'],
            'config_key': best_config_key,
        }
    
    metrics_path = os.path.join(args.output_dir, "grid_search_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    logger.info(f"\n✓ Saved grid search metrics to {metrics_path}")
    
    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("FUSION COMPLETE")
    logger.info("=" * 80)
    
    logger.info("\n" + "=" * 80)
    logger.info("FUSION COMPLETE")
    logger.info("=" * 80)
    
    logger.info(f"\nBest tuning split: {best_overall_split}")
    logger.info(f"  Best config: {best_config_key}")
    logger.info(f"  {args.metric_for_best}: {best_overall_metrics[args.metric_for_best]:.4f}")
    for k in k_values:
        logger.info(f"  R@{k}: {best_overall_metrics.get(f'R@{k}', 0.0):.4f}")

    for split, results in tuning_split_results.items():
        best_config = results['best_config']
        best_metrics = results['best_metrics']
        config_key = make_config_key(args.fusion_mode, best_config)
        
        logger.info(f"\n{split.upper()}:")
        logger.info(f"  Best config: {config_key}")
        logger.info(f"  MRR: {best_metrics['MRR']:.4f}")
        for k in k_values:
            logger.info(f"  R@{k}: {best_metrics.get(f'R@{k}', 0.0):.4f}")

    if evaluation_results:
        logger.info("\nEVALUATION SPLITS:")
        for split, result in evaluation_results.items():
            metrics = result['metrics']
            logger.info(f"  {split.upper()}:")
            logger.info(f"    Config: {best_config_key}")
            logger.info(f"    MRR: {metrics['MRR']:.4f}")
            for k in k_values:
                logger.info(f"    R@{k}: {metrics.get(f'R@{k}', 0.0):.4f}")
    
    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    main()

