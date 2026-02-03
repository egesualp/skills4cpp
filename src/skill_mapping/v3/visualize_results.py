"""
visualize_results.py - Create visualizations for LLM re-ranking results

Generates plots to analyze:
- Tier distribution across jobs
- Rank shifts (original vs final)
- Top-K improvement analysis
- Per-tier precision/recall curves

Usage:
    python -m skill_mapping.v3.visualize_results \
        --reranked_scores ./outputs/llm_reranking/llm_reranked_scores_compact.json \
        --ground_truth ./data/processed/ground_truth.json \
        --output_dir ./outputs/llm_reranking/visualizations
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
from loguru import logger

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib/Seaborn not available. Install with: pip install matplotlib seaborn")
    PLOTTING_AVAILABLE = False


def load_data(reranked_scores_path: Path, ground_truth_path: Path):
    """Load re-ranked scores and ground truth."""
    with open(reranked_scores_path, 'r') as f:
        data = json.load(f)
        reranked_scores = data['scores']
    
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
        ground_truth = {k: set(v) for k, v in ground_truth.items()}
    
    return reranked_scores, ground_truth


def analyze_tier_distribution(reranked_scores: Dict) -> pd.DataFrame:
    """Analyze tier distribution across all jobs."""
    tier_counts = Counter()
    
    for job_id, skills in reranked_scores.items():
        for skill in skills:
            tier = skill.get('tier', 'Unknown')
            tier_counts[tier] += 1
    
    df = pd.DataFrame([
        {'Tier': tier, 'Count': count}
        for tier, count in tier_counts.items()
    ])
    
    return df


def analyze_rank_shifts(reranked_scores: Dict, ground_truth: Dict) -> pd.DataFrame:
    """Analyze how ground truth skills shifted in ranking."""
    shifts = []
    
    for job_id, skills in reranked_scores.items():
        if job_id not in ground_truth:
            continue
        
        true_skills = ground_truth[job_id]
        
        for skill in skills:
            if skill['skill_uri'] in true_skills:
                original_rank = skill['original_rank']
                final_rank = skill['rank']
                shift = original_rank - final_rank  # Positive = moved up
                tier = skill.get('tier', 'Unknown')
                
                shifts.append({
                    'job_id': job_id,
                    'skill_uri': skill['skill_uri'],
                    'original_rank': original_rank,
                    'final_rank': final_rank,
                    'shift': shift,
                    'tier': tier,
                })
    
    return pd.DataFrame(shifts)


def analyze_topk_improvements(reranked_scores: Dict, ground_truth: Dict) -> pd.DataFrame:
    """Analyze improvements in top-K coverage."""
    k_values = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
    results = []
    
    for k in k_values:
        original_hits = 0
        final_hits = 0
        total_possible = 0
        
        for job_id, skills in reranked_scores.items():
            if job_id not in ground_truth:
                continue
            
            true_skills = ground_truth[job_id]
            total_possible += len(true_skills)
            
            # Original top-K
            original_topk = {
                s['skill_uri'] for s in sorted(skills, key=lambda x: x['original_rank'])[:k]
            }
            original_hits += len(original_topk & true_skills)
            
            # Final top-K
            final_topk = {s['skill_uri'] for s in skills[:k]}
            final_hits += len(final_topk & true_skills)
        
        original_recall = original_hits / total_possible if total_possible > 0 else 0
        final_recall = final_hits / total_possible if total_possible > 0 else 0
        
        results.append({
            'K': k,
            'Original_Recall': original_recall,
            'Final_Recall': final_recall,
            'Improvement': final_recall - original_recall,
        })
    
    return pd.DataFrame(results)


def plot_tier_distribution(df: pd.DataFrame, output_path: Path):
    """Plot tier distribution."""
    if not PLOTTING_AVAILABLE:
        return
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='Tier', y='Count', palette='viridis')
    plt.title('Distribution of Skills Across Tiers', fontsize=14, fontweight='bold')
    plt.xlabel('Tier', fontsize=12)
    plt.ylabel('Number of Skills', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved tier distribution plot to {output_path}")


def plot_rank_shifts(df: pd.DataFrame, output_path: Path):
    """Plot rank shift distributions by tier."""
    if not PLOTTING_AVAILABLE:
        return
    
    plt.figure(figsize=(12, 6))
    
    # Filter to meaningful shifts
    df_filtered = df[df['shift'].abs() > 0]
    
    sns.boxplot(data=df_filtered, x='tier', y='shift', palette='Set2')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    plt.title('Rank Shifts by Tier (Ground Truth Skills Only)', fontsize=14, fontweight='bold')
    plt.xlabel('Tier', fontsize=12)
    plt.ylabel('Rank Shift (positive = moved up)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved rank shifts plot to {output_path}")


def plot_topk_improvements(df: pd.DataFrame, output_path: Path):
    """Plot top-K recall improvements."""
    if not PLOTTING_AVAILABLE:
        return
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(df['K'], df['Original_Recall'], marker='o', label='Original (Linear Fusion)', linewidth=2)
    plt.plot(df['K'], df['Final_Recall'], marker='s', label='LLM Re-ranked', linewidth=2)
    
    plt.title('Recall@K: Original vs LLM Re-ranked', fontsize=14, fontweight='bold')
    plt.xlabel('K', fontsize=12)
    plt.ylabel('Recall', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved top-K improvements plot to {output_path}")


def plot_improvement_bars(df: pd.DataFrame, output_path: Path):
    """Plot improvement bars."""
    if not PLOTTING_AVAILABLE:
        return
    
    plt.figure(figsize=(12, 6))
    
    # Focus on key K values
    df_key = df[df['K'].isin([5, 10, 20, 50, 100])]
    
    colors = ['green' if x > 0 else 'red' for x in df_key['Improvement']]
    plt.bar(df_key['K'].astype(str), df_key['Improvement'] * 100, color=colors, alpha=0.7)
    
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    plt.title('Recall Improvement by K (percentage points)', fontsize=14, fontweight='bold')
    plt.xlabel('K', fontsize=12)
    plt.ylabel('Improvement (%)', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved improvement bars plot to {output_path}")


def generate_summary_report(
    tier_dist: pd.DataFrame,
    rank_shifts: pd.DataFrame,
    topk_improvements: pd.DataFrame,
    output_path: Path,
):
    """Generate a text summary report."""
    lines = []
    lines.append("=" * 80)
    lines.append("LLM RE-RANKING ANALYSIS SUMMARY")
    lines.append("=" * 80)
    lines.append("")
    
    # Tier distribution
    lines.append("TIER DISTRIBUTION")
    lines.append("-" * 40)
    for _, row in tier_dist.iterrows():
        pct = (row['Count'] / tier_dist['Count'].sum()) * 100
        lines.append(f"  {row['Tier']}: {row['Count']:,} ({pct:.1f}%)")
    lines.append("")
    
    # Rank shifts
    lines.append("RANK SHIFTS (Ground Truth Skills)")
    lines.append("-" * 40)
    for tier in ['Essential', 'Optional', 'Irrelevant']:
        tier_shifts = rank_shifts[rank_shifts['tier'] == tier]
        if len(tier_shifts) > 0:
            mean_shift = tier_shifts['shift'].mean()
            median_shift = tier_shifts['shift'].median()
            moved_up = (tier_shifts['shift'] > 0).sum()
            moved_down = (tier_shifts['shift'] < 0).sum()
            lines.append(f"  {tier}:")
            lines.append(f"    Mean shift: {mean_shift:+.1f} ranks")
            lines.append(f"    Median shift: {median_shift:+.1f} ranks")
            lines.append(f"    Moved up: {moved_up} ({moved_up/len(tier_shifts)*100:.1f}%)")
            lines.append(f"    Moved down: {moved_down} ({moved_down/len(tier_shifts)*100:.1f}%)")
    lines.append("")
    
    # Top-K improvements
    lines.append("TOP-K RECALL IMPROVEMENTS")
    lines.append("-" * 40)
    for _, row in topk_improvements.iterrows():
        if row['K'] in [5, 10, 20, 50, 100]:
            improvement = row['Improvement'] * 100
            lines.append(f"  Recall@{row['K']:3d}: {row['Original_Recall']:.4f} → {row['Final_Recall']:.4f} ({improvement:+.2f}%)")
    lines.append("")
    
    lines.append("=" * 80)
    
    report = "\n".join(lines)
    
    # Print to console
    print(report)
    
    # Save to file
    with open(output_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Saved summary report to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Visualize LLM re-ranking results"
    )
    
    parser.add_argument(
        "--reranked_scores",
        type=Path,
        required=True,
        help="Path to LLM re-ranked scores JSON (compact format)"
    )
    parser.add_argument(
        "--ground_truth",
        type=Path,
        required=True,
        help="Path to ground truth JSON"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for visualizations"
    )
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    reranked_scores, ground_truth = load_data(args.reranked_scores, args.ground_truth)
    logger.info(f"Loaded {len(reranked_scores)} jobs with re-ranked scores")
    logger.info(f"Loaded {len(ground_truth)} jobs with ground truth")
    
    # Analyze tier distribution
    logger.info("Analyzing tier distribution...")
    tier_dist = analyze_tier_distribution(reranked_scores)
    
    # Analyze rank shifts
    logger.info("Analyzing rank shifts...")
    rank_shifts = analyze_rank_shifts(reranked_scores, ground_truth)
    
    # Analyze top-K improvements
    logger.info("Analyzing top-K improvements...")
    topk_improvements = analyze_topk_improvements(reranked_scores, ground_truth)
    
    # Generate visualizations
    if PLOTTING_AVAILABLE:
        logger.info("Generating plots...")
        plot_tier_distribution(tier_dist, args.output_dir / "tier_distribution.png")
        plot_rank_shifts(rank_shifts, args.output_dir / "rank_shifts.png")
        plot_topk_improvements(topk_improvements, args.output_dir / "topk_improvements.png")
        plot_improvement_bars(topk_improvements, args.output_dir / "improvement_bars.png")
    
    # Save data tables
    logger.info("Saving data tables...")
    tier_dist.to_csv(args.output_dir / "tier_distribution.csv", index=False)
    rank_shifts.to_csv(args.output_dir / "rank_shifts.csv", index=False)
    topk_improvements.to_csv(args.output_dir / "topk_improvements.csv", index=False)
    
    # Generate summary report
    logger.info("Generating summary report...")
    generate_summary_report(
        tier_dist,
        rank_shifts,
        topk_improvements,
        args.output_dir / "summary_report.txt"
    )
    
    logger.success(f"Analysis complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()







