"""
Score Fusion for Career Path Prediction.

Combines MLP text-based scores (S_text) with skill overlap scores (S_skills):
    S_final = alpha * S_text + (1-alpha) * S_skills

Optimizes alpha using train+val data, then evaluates on test set.
"""

import argparse
import os
import sys
import pickle
from typing import Dict, List, Tuple, Optional
from loguru import logger
import numpy as np
from scipy.optimize import minimize_scalar
import pandas as pd

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(log_dir: str):
    """Configure logging to both file and stdout."""
    os.makedirs(log_dir, exist_ok=True)
    
    logger.remove()
    logger.add(
        os.path.join(log_dir, "score_fusion.log"),
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
# SCORE LOADING & ALIGNMENT
# ============================================================================

def load_scores(score_path: str) -> Dict:
    """Load score dictionary from pickle file."""
    with open(score_path, 'rb') as f:
        return pickle.load(f)


def align_scores(text_scores: Dict, skill_scores: Dict) -> Tuple[np.ndarray, np.ndarray, List[int], List[str]]:
    """
    Align text scores and skill scores to ensure they have the same sample/target ordering.
    
    Both score files should have been generated from the same data pairs, but we verify
    the alignment and handle any differences (including prompt formatting).
    """
    # Check if target labels are in the same order
    text_targets = text_scores['target_labels']
    skill_targets = skill_scores['target_labels']
    
    if text_targets == skill_targets:
        logger.info("  ✓ Target labels are aligned")
        return (
            text_scores['scores'],
            skill_scores['scores'],
            text_scores['true_target_indices'],
            text_targets
        )
    
    # Need to reorder skill scores to match text scores
    logger.warning("  ⚠️  Target labels differ, realigning scores...")
    
    # Build mapping from skill target order to text target order
    skill_target_to_idx = {t: i for i, t in enumerate(skill_targets)}
    
    # Reorder skill scores columns
    n_samples = skill_scores['scores'].shape[0]
    n_text_targets = len(text_targets)
    aligned_skill_scores = np.zeros((n_samples, n_text_targets), dtype=np.float32)
    
    for i, target in enumerate(text_targets):
        match_idx = -1
        
        # 1. Try direct match
        if target in skill_target_to_idx:
            match_idx = skill_target_to_idx[target]
        
        # 2. Try extracting title from "esco role: <title>\n..." format
        else:
            clean_target = target
            if "role: " in target:
                # Handle "esco role: ..." or just "role: ..."
                # Split by "role: ", take the part after, then split by newline to get title
                try:
                    clean_target = target.split("role: ", 1)[1].split("\n", 1)[0].strip()
                    clean_target = clean_target + " "
                except IndexError:
                    pass
            
            if clean_target in skill_target_to_idx:
                match_idx = skill_target_to_idx[clean_target]
        
        if match_idx != -1:
            aligned_skill_scores[:, i] = skill_scores['scores'][:, match_idx]
        else:
            # Log first few characters to debug (limit to first 5 unmatched to avoid spam)
            if i < 5: 
                logger.warning(f"    Target '{target[:30]}...' not found in skill scores (tried clean: '{clean_target if 'clean_target' in locals() else target}')")
    
    return (
        text_scores['scores'],
        aligned_skill_scores,
        text_scores['true_target_indices'],
        text_targets
    )


# ============================================================================
# EVALUATION
# ============================================================================

def calculate_ranking_metrics(scores: np.ndarray, true_target_indices: List[int],
                              k_values: List[int] = [1, 5, 10, 20],
                              batch_size: int = 1000) -> Dict[str, float]:
    """
    Calculate ranking metrics from score matrix using batched processing to save memory.
    
    Args:
        scores: Score matrix [n_samples, n_targets]
        true_target_indices: Index of true target for each sample
        k_values: List of K values for Recall@K
        batch_size: Number of samples to process at a time
        
    Returns:
        Dictionary with MRR and Recall@K metrics
    """
    n_samples = len(true_target_indices)
    
    reciprocal_ranks = []
    hits_at_k = {k: 0 for k in k_values}
    
    # Process in batches
    # If scores is small (e.g. < 5000 samples), we can just process all at once
    # but for safety with large 100k+ sets, we batch
    if n_samples == 0:
        return {'MRR': 0.0, **{f'R@{k}': 0.0 for k in k_values}}

    # Use tqdm only if substantial number of batches
    iterator = range(0, n_samples, batch_size)
    if n_samples > batch_size * 5:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc="  Calculating metrics")

    for start_idx in iterator:
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
        recall_at_k[f'R@{k}'] = hits_at_k[k] / n_samples
    
    metrics = {'MRR': mrr}
    metrics.update(recall_at_k)
    
    return metrics


def fuse_scores(s_text: np.ndarray, s_skill: np.ndarray, alpha: float) -> np.ndarray:
    """
    Fuse text and skill scores.
    
    S_final = alpha * S_text + (1-alpha) * S_skills
    
    Args:
        s_text: Text-based scores [n_samples, n_targets]
        s_skill: Skill overlap scores [n_samples, n_targets]
        alpha: Fusion weight (0 = skill only, 1 = text only)
        
    Returns:
        Fused scores [n_samples, n_targets]
    """
    return alpha * s_text + (1 - alpha) * s_skill


# ============================================================================
# ALPHA OPTIMIZATION
# ============================================================================

def optimize_alpha_grid(
    s_text_train: np.ndarray,
    s_skill_train: np.ndarray,
    true_indices_train: List[int],
    s_text_val: np.ndarray,
    s_skill_val: np.ndarray,
    true_indices_val: List[int],
    metric: str = 'MRR',
    n_points: int = 101,
) -> Tuple[float, float]:
    """
    Optimize alpha using grid search on combined train+val set.
    
    Args:
        s_text_train: Text scores for training split
        s_skill_train: Skill scores for training split
        true_indices_train: True target indices for training split
        s_text_val: Text scores for validation split
        s_skill_val: Skill scores for validation split
        true_indices_val: True target indices for validation split
        metric: Metric to optimize ('MRR', 'R@1', 'R@5', etc.)
        n_points: Number of alpha values to try
        
    Returns:
        Tuple of (best_alpha, best_metric_value)
    """
    # Combine train + val
    s_text_combined = np.concatenate([s_text_train, s_text_val], axis=0)
    s_skill_combined = np.concatenate([s_skill_train, s_skill_val], axis=0)
    true_indices_combined = true_indices_train + true_indices_val
    
    logger.info(f"  > Optimizing alpha on train+val ({len(true_indices_combined)} samples)")
    
    alphas = np.linspace(0, 1, n_points)
    best_alpha = 0.5
    best_metric = -1
    
    results = []
    for alpha in alphas:
        fused = fuse_scores(s_text_combined, s_skill_combined, alpha)
        metrics = calculate_ranking_metrics(fused, true_indices_combined)
        
        results.append({
            'alpha': alpha,
            **metrics
        })
        
        if metrics[metric] > best_metric:
            best_metric = metrics[metric]
            best_alpha = alpha
    
    logger.info(f"  > Best alpha: {best_alpha:.3f} (train+val {metric}: {best_metric:.4f})")
    
    return best_alpha, best_metric, results


def optimize_alpha_scipy(
    s_text_combined: np.ndarray,
    s_skill_combined: np.ndarray,
    true_indices_combined: List[int],
    metric: str = 'MRR',
) -> Tuple[float, float]:
    """
    Optimize alpha using scipy's bounded minimization.
    
    Args:
        s_text_combined: Text scores for combined train+val
        s_skill_combined: Skill scores for combined train+val
        true_indices_combined: True target indices
        metric: Metric to optimize
        
    Returns:
        Tuple of (best_alpha, best_metric_value)
    """
    def neg_metric(alpha):
        fused = fuse_scores(s_text_combined, s_skill_combined, alpha)
        metrics = calculate_ranking_metrics(fused, true_indices_combined)
        return -metrics[metric]
    
    result = minimize_scalar(neg_metric, bounds=(0, 1), method='bounded')
    
    best_alpha = result.x
    best_metric = -result.fun
    
    return best_alpha, best_metric


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Score Fusion for CPP")
    
    # Input paths
    parser.add_argument("--text_scores_dir", type=str, required=True,
                       help="Directory containing MLP text scores (*_scores_text.pkl)")
    parser.add_argument("--skill_scores_dir", type=str, required=True,
                       help="Directory containing skill overlap scores (*_scores_skill_overlap.pkl)")
    
    # Optimization
    parser.add_argument("--optimize_metric", type=str, default="MRR",
                       choices=["MRR", "R@1", "R@5", "R@10", "R@20"],
                       help="Metric to optimize alpha for")
    parser.add_argument("--n_alpha_points", type=int, default=101,
                       help="Number of alpha values to try in grid search")
    parser.add_argument("--fixed_alpha", type=float, default=None,
                       help="Use a fixed alpha instead of optimizing (for ablation)")
    
    # Output
    parser.add_argument("--output_dir", type=str,
                       default="results/cpp/decorte/fusion_results",
                       help="Directory to save fusion results")
    parser.add_argument("--log_dir", type=str,
                       default="src/cpp/decorte/logs",
                       help="Directory for log files")
    parser.add_argument("--results_csv", type=str, default=None,
                       help="Path to append results to CSV (default: output_dir/fusion_results.csv)")
    parser.add_argument("--run_name", type=str, default="fusion",
                       help="Name for this fusion run")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_dir)
    
    logger.info("=" * 80)
    logger.info("Score Fusion for Career Path Prediction")
    logger.info("=" * 80)
    logger.info(f"Configuration: {vars(args)}\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- Step 1: Load scores ---
    logger.info("[1/4] Loading scores...")
    
    # Load text scores
    # text_train = load_scores(os.path.join(args.text_scores_dir, "train_scores_text.pkl"))
    text_val = load_scores(os.path.join(args.text_scores_dir, "val_scores_text.pkl"))
    text_test = load_scores(os.path.join(args.text_scores_dir, "test_scores_text.pkl"))
    logger.info(f"  ✓ Loaded text scores: val={text_val['scores'].shape}, test={text_test['scores'].shape}")
    
    # Load skill overlap scores
    # skill_train = load_scores(os.path.join(args.skill_scores_dir, "train_scores_skill_overlap.pkl"))
    skill_val = load_scores(os.path.join(args.skill_scores_dir, "val_scores_skill_overlap.pkl"))
    skill_test = load_scores(os.path.join(args.skill_scores_dir, "test_scores_skill_overlap.pkl"))
    logger.info(f"  ✓ Loaded skill scores: val={skill_val['scores'].shape}, test={skill_test['scores'].shape}\n")
    
    # --- Step 2: Align scores ---
    logger.info("[2/4] Aligning scores...")
    
    # s_text_train, s_skill_train, idx_train, _ = align_scores(text_train, skill_train)
    s_text_val, s_skill_val, idx_val, _ = align_scores(text_val, skill_val)
    s_text_test, s_skill_test, idx_test, target_labels = align_scores(text_test, skill_test)
    logger.info("")
    
    # --- Step 3: Evaluate individual methods ---
    logger.info("[3/4] Evaluating individual methods on test set...")
    
    text_only_metrics = calculate_ranking_metrics(s_text_test, idx_test)
    skill_only_metrics = calculate_ranking_metrics(s_skill_test, idx_test)
    
    logger.info(f"  > Text only (MLP):       MRR={text_only_metrics['MRR']:.4f}, R@5={text_only_metrics['R@5']:.4f}, R@10={text_only_metrics['R@10']:.4f}")
    logger.info(f"  > Skill only (overlap):  MRR={skill_only_metrics['MRR']:.4f}, R@5={skill_only_metrics['R@5']:.4f}, R@10={skill_only_metrics['R@10']:.4f}")
    logger.info("")
    
    # --- Step 4: Optimize alpha (or use fixed) ---
    if args.fixed_alpha is not None:
        best_alpha = args.fixed_alpha
        logger.info(f"[4/4] Using fixed alpha: {best_alpha:.3f}")
        alpha_results = None
    else:
        logger.info(f"[4/4] Optimizing alpha for {args.optimize_metric} on validation set...")
        # Use validation set for both "train" and "val" arguments to effectively
        # optimize only on validation data (skipping the huge training set)
        best_alpha, best_trainval_metric, alpha_results = optimize_alpha_grid(
            s_text_val, s_skill_val, idx_val,
            s_text_val, s_skill_val, idx_val,
            metric=args.optimize_metric,
            n_points=args.n_alpha_points,
        )
    
    # --- Step 5: Evaluate fused scores on test ---
    logger.info("\n  > Evaluating fused scores on test set...")
    
    fused_test = fuse_scores(s_text_test, s_skill_test, best_alpha)
    fused_metrics = calculate_ranking_metrics(fused_test, idx_test)
    
    logger.info("\n" + "=" * 80)
    logger.info("FINAL TEST SET RESULTS")
    logger.info("=" * 80)
    logger.info(f"Best alpha: {best_alpha:.4f}")
    logger.info("")
    logger.info("Method                  | MRR    | R@1    | R@5    | R@10   | R@20")
    logger.info("-" * 70)
    logger.info(f"Text only (alpha=1.0)   | {text_only_metrics['MRR']:.4f} | {text_only_metrics['R@1']:.4f} | {text_only_metrics['R@5']:.4f} | {text_only_metrics['R@10']:.4f} | {text_only_metrics['R@20']:.4f}")
    logger.info(f"Skill only (alpha=0.0)  | {skill_only_metrics['MRR']:.4f} | {skill_only_metrics['R@1']:.4f} | {skill_only_metrics['R@5']:.4f} | {skill_only_metrics['R@10']:.4f} | {skill_only_metrics['R@20']:.4f}")
    logger.info(f"Fused (alpha={best_alpha:.2f})      | {fused_metrics['MRR']:.4f} | {fused_metrics['R@1']:.4f} | {fused_metrics['R@5']:.4f} | {fused_metrics['R@10']:.4f} | {fused_metrics['R@20']:.4f}")
    logger.info("=" * 80)
    
    # Calculate improvement
    mrr_improvement = fused_metrics['MRR'] - text_only_metrics['MRR']
    r5_improvement = fused_metrics['R@5'] - text_only_metrics['R@5']
    logger.info(f"\nImprovement over text-only:")
    logger.info(f"  MRR: {mrr_improvement:+.4f} ({100*mrr_improvement/text_only_metrics['MRR']:+.2f}%)")
    logger.info(f"  R@5: {r5_improvement:+.4f} ({100*r5_improvement/text_only_metrics['R@5']:+.2f}%)")
    
    # --- Step 6: Save results ---
    logger.info("\n  > Saving results...")
    
    # Save fused scores
    fused_scores_dict = {
        'scores': fused_test,
        'target_labels': target_labels,
        'true_target_indices': idx_test,
        'alpha': best_alpha,
        'text_scores_dir': args.text_scores_dir,
        'skill_scores_dir': args.skill_scores_dir,
    }
    fused_path = os.path.join(args.output_dir, "test_scores_fused.pkl")
    with open(fused_path, 'wb') as f:
        pickle.dump(fused_scores_dict, f)
    logger.info(f"  ✓ Saved fused scores to {fused_path}")
    
    # Save alpha optimization results
    if alpha_results is not None:
        alpha_df = pd.DataFrame(alpha_results)
        alpha_path = os.path.join(args.output_dir, "alpha_optimization.csv")
        alpha_df.to_csv(alpha_path, index=False)
        logger.info(f"  ✓ Saved alpha optimization results to {alpha_path}")
    
    # Append to results CSV
    results_csv = args.results_csv if args.results_csv else os.path.join(args.output_dir, "fusion_results.csv")
    results_row = {
        'run_name': args.run_name,
        'best_alpha': best_alpha,
        'optimize_metric': args.optimize_metric,
        'text_MRR': text_only_metrics['MRR'],
        'text_R@5': text_only_metrics['R@5'],
        'text_R@10': text_only_metrics['R@10'],
        'skill_MRR': skill_only_metrics['MRR'],
        'skill_R@5': skill_only_metrics['R@5'],
        'skill_R@10': skill_only_metrics['R@10'],
        'fused_MRR': fused_metrics['MRR'],
        'fused_R@1': fused_metrics['R@1'],
        'fused_R@5': fused_metrics['R@5'],
        'fused_R@10': fused_metrics['R@10'],
        'fused_R@20': fused_metrics['R@20'],
        'MRR_improvement': mrr_improvement,
        'R@5_improvement': r5_improvement,
    }
    
    results_df = pd.DataFrame([results_row])
    if os.path.exists(results_csv):
        results_df.to_csv(results_csv, mode='a', header=False, index=False)
    else:
        results_df.to_csv(results_csv, mode='w', header=True, index=False)
    logger.info(f"  ✓ Appended results to {results_csv}")


if __name__ == "__main__":
    main()


