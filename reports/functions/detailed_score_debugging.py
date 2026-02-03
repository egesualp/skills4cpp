"""
Detailed Score Debugging Analysis
==================================
This script provides deeper insights into why the enhanced model fails.

Usage:
    python detailed_score_debugging.py <baseline.pkl> <enhanced.pkl> [output_dir]
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def load_scores(filepath):
    """Load score data from pickle file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded: {filepath}")
    print(f"  Shape: {data['scores'].shape}, Split: {data.get('split', 'unknown')}")
    return data


def analyze_score_distributions(baseline, enhanced, output_dir=None):
    """Analyze and compare score distributions."""
    scores_b = baseline['scores']
    scores_e = enhanced['scores']
    
    print("\n" + "="*60)
    print("SCORE DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Overall statistics
    print("\nOverall Score Statistics:")
    print(f"  Baseline - Mean: {scores_b.mean():.6f}, Std: {scores_b.std():.6f}")
    print(f"  Baseline - Min: {scores_b.min():.6f}, Max: {scores_b.max():.6f}")
    print(f"  Enhanced - Mean: {scores_e.mean():.6f}, Std: {scores_e.std():.6f}")
    print(f"  Enhanced - Min: {scores_e.min():.6f}, Max: {scores_e.max():.6f}")
    
    # Per-sample max scores
    max_scores_b = scores_b.max(axis=1)
    max_scores_e = scores_e.max(axis=1)
    
    print("\nPer-Sample Maximum Scores:")
    print(f"  Baseline - Mean: {max_scores_b.mean():.6f}, Std: {max_scores_b.std():.6f}")
    print(f"  Enhanced - Mean: {max_scores_e.mean():.6f}, Std: {max_scores_e.std():.6f}")
    
    # Score range (max - min) per sample
    range_b = scores_b.max(axis=1) - scores_b.min(axis=1)
    range_e = scores_e.max(axis=1) - scores_e.min(axis=1)
    
    print("\nScore Range per Sample:")
    print(f"  Baseline - Mean: {range_b.mean():.6f}, Std: {range_b.std():.6f}")
    print(f"  Enhanced - Mean: {range_e.mean():.6f}, Std: {range_e.std():.6f}")
    
    # Check for zero or constant scores
    zero_rows_b = (scores_b == 0).all(axis=1).sum()
    zero_rows_e = (scores_e == 0).all(axis=1).sum()
    
    print("\nZero Score Rows:")
    print(f"  Baseline: {zero_rows_b} samples")
    print(f"  Enhanced: {zero_rows_e} samples")
    
    # Check for NaN or Inf
    nan_b = np.isnan(scores_b).sum()
    nan_e = np.isnan(scores_e).sum()
    inf_b = np.isinf(scores_b).sum()
    inf_e = np.isinf(scores_e).sum()
    
    print("\nInvalid Values:")
    print(f"  Baseline - NaN: {nan_b}, Inf: {inf_b}")
    print(f"  Enhanced - NaN: {nan_e}, Inf: {inf_e}")
    
    # Visualization
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Score distribution histograms
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].hist(scores_b.flatten(), bins=100, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Baseline: All Scores')
        axes[0, 0].set_xlabel('Score')
        axes[0, 0].set_ylabel('Frequency')
        
        axes[0, 1].hist(scores_e.flatten(), bins=100, alpha=0.7, edgecolor='black', color='orange')
        axes[0, 1].set_title('Enhanced: All Scores')
        axes[0, 1].set_xlabel('Score')
        axes[0, 1].set_ylabel('Frequency')
        
        axes[1, 0].hist(max_scores_b, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Baseline: Max Scores per Sample')
        axes[1, 0].set_xlabel('Max Score')
        axes[1, 0].set_ylabel('Frequency')
        
        axes[1, 1].hist(max_scores_e, bins=50, alpha=0.7, edgecolor='black', color='orange')
        axes[1, 1].set_title('Enhanced: Max Scores per Sample')
        axes[1, 1].set_xlabel('Max Score')
        axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'score_distributions.png', dpi=150, bbox_inches='tight')
        print(f"\n  Saved: {output_dir / 'score_distributions.png'}")
        plt.close()


def analyze_rank_changes(baseline, enhanced, output_dir=None):
    """Analyze how ranks change for true targets."""
    scores_b = baseline['scores']
    scores_e = enhanced['scores']
    true_indices = np.array(baseline['true_target_indices'])
    
    print("\n" + "="*60)
    print("TRUE TARGET RANK ANALYSIS")
    print("="*60)
    
    ranks_b = []
    ranks_e = []
    rank_changes = []
    
    for i in range(len(true_indices)):
        true_idx = true_indices[i]
        
        # Get ranks (0-indexed, 0 = top rank)
        rank_b = (scores_b[i] > scores_b[i, true_idx]).sum()
        rank_e = (scores_e[i] > scores_e[i, true_idx]).sum()
        
        ranks_b.append(rank_b)
        ranks_e.append(rank_e)
        rank_changes.append(rank_e - rank_b)
    
    ranks_b = np.array(ranks_b)
    ranks_e = np.array(ranks_e)
    rank_changes = np.array(rank_changes)
    
    print("\nTrue Target Ranks:")
    print(f"  Baseline - Mean: {ranks_b.mean():.1f}, Median: {np.median(ranks_b):.1f}")
    print(f"  Baseline - Min: {ranks_b.min()}, Max: {ranks_b.max()}")
    print(f"  Enhanced - Mean: {ranks_e.mean():.1f}, Median: {np.median(ranks_e):.1f}")
    print(f"  Enhanced - Min: {ranks_e.min()}, Max: {ranks_e.max()}")
    
    print("\nRank Changes (Enhanced - Baseline):")
    print(f"  Mean: {rank_changes.mean():.1f}")
    print(f"  Median: {np.median(rank_changes):.1f}")
    print(f"  Improved (negative): {(rank_changes < 0).sum()} samples ({100*(rank_changes < 0).mean():.1f}%)")
    print(f"  Degraded (positive): {(rank_changes > 0).sum()} samples ({100*(rank_changes > 0).mean():.1f}%)")
    print(f"  Unchanged: {(rank_changes == 0).sum()} samples ({100*(rank_changes == 0).mean():.1f}%)")
    
    # Top-k accuracy comparison
    print("\nTop-k Accuracy:")
    for k in [1, 5, 10, 20, 50]:
        acc_b = (ranks_b < k).mean()
        acc_e = (ranks_e < k).mean()
        print(f"  Top-{k:>2}: Baseline {100*acc_b:5.1f}% | Enhanced {100*acc_e:5.1f}% | Δ {100*(acc_e-acc_b):+5.1f}%")
    
    # Visualization
    if output_dir:
        output_dir = Path(output_dir)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Rank comparison scatter
        axes[0].scatter(ranks_b, ranks_e, alpha=0.3, s=10)
        axes[0].plot([0, max(ranks_b.max(), ranks_e.max())], 
                     [0, max(ranks_b.max(), ranks_e.max())], 
                     'r--', label='No change')
        axes[0].set_xlabel('Baseline Rank')
        axes[0].set_ylabel('Enhanced Rank')
        axes[0].set_title('True Target Rank Comparison')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Rank change histogram
        axes[1].hist(rank_changes, bins=100, edgecolor='black', alpha=0.7)
        axes[1].axvline(0, color='red', linestyle='--', label='No change')
        axes[1].set_xlabel('Rank Change (Enhanced - Baseline)')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Distribution of Rank Changes')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'rank_analysis.png', dpi=150, bbox_inches='tight')
        print(f"\n  Saved: {output_dir / 'rank_analysis.png'}")
        plt.close()
    
    return ranks_b, ranks_e, rank_changes


def analyze_failure_cases(baseline, enhanced, n_cases=10):
    """Analyze specific cases where baseline succeeds but enhanced fails."""
    scores_b = baseline['scores']
    scores_e = enhanced['scores']
    true_indices = np.array(baseline['true_target_indices'])
    
    # Find cases where baseline is correct (top-1)
    pred_b = np.argmax(scores_b, axis=1)
    correct_baseline = np.where(pred_b == true_indices)[0]
    
    if len(correct_baseline) == 0:
        print("\n⚠️  No cases where baseline is correct!")
        return
    
    print("\n" + "="*60)
    print(f"FAILURE CASE ANALYSIS (First {n_cases} cases)")
    print("="*60)
    
    for idx in correct_baseline[:n_cases]:
        true_idx = true_indices[idx]
        
        # Get predictions
        pred_b_idx = np.argmax(scores_b[idx])
        pred_e_idx = np.argmax(scores_e[idx])
        
        # Get ranks
        rank_b = (scores_b[idx] > scores_b[idx, true_idx]).sum()
        rank_e = (scores_e[idx] > scores_e[idx, true_idx]).sum()
        
        # Get scores for true target
        score_b_true = scores_b[idx, true_idx]
        score_e_true = scores_e[idx, true_idx]
        
        # Get scores for top predictions
        score_b_pred = scores_b[idx, pred_b_idx]
        score_e_pred = scores_e[idx, pred_e_idx]
        
        print(f"\nSample {idx}:")
        print(f"  True target index: {true_idx}")
        print(f"  Baseline:")
        print(f"    Predicted: {pred_b_idx} (rank 0) | Score: {score_b_pred:.6f}")
        print(f"    True target rank: {rank_b} | Score: {score_b_true:.6f}")
        print(f"  Enhanced:")
        print(f"    Predicted: {pred_e_idx} (rank 0) | Score: {score_e_pred:.6f}")
        print(f"    True target rank: {rank_e} | Score: {score_e_true:.6f}")
        print(f"  Rank degradation: {rank_e - rank_b:+d}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python detailed_score_debugging.py <baseline.pkl> <enhanced.pkl> [output_dir]")
        sys.exit(1)
    
    baseline_path = sys.argv[1]
    enhanced_path = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    # Load data
    baseline = load_scores(baseline_path)
    enhanced = load_scores(enhanced_path)
    
    # Run analyses
    analyze_score_distributions(baseline, enhanced, output_dir)
    analyze_rank_changes(baseline, enhanced, output_dir)
    analyze_failure_cases(baseline, enhanced, n_cases=10)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    if output_dir:
        print(f"Visualizations saved to: {output_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
