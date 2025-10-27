#!/usr/bin/env python3
"""
EDA script to analyze the sweet spot for min_samples_per_group parameter.

Shows how different thresholds affect:
- Number of skill groups retained
- Amount of training data retained
- Distribution of samples per group
"""
import argparse
import json
from pathlib import Path
import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not installed. Plots will be skipped.")

def main():
    p = argparse.ArgumentParser(
        description="Analyze min_samples_per_group threshold effects"
    )
    p.add_argument(
        "--pairs_glob", 
        required=True,
        help="Glob pattern to raw↔ESCO CSVs"
    )
    p.add_argument(
        "--skills_per_occ",
        default="data/skills_per_occupations.csv",
        help="Path to skills_per_occupations.csv"
    )
    p.add_argument(
        "--processed_dir",
        default="data/processed",
        help="Path containing skill2group.json"
    )
    p.add_argument(
        "--min_groups_per_title",
        type=int,
        default=1,
        help="Minimum groups per title (for filtering)"
    )
    p.add_argument(
        "--output",
        help="Optional: save plot to file"
    )
    args = p.parse_args()

    print("=" * 80)
    print("EDA: Finding Sweet Spot for min_samples_per_group")
    print("=" * 80)

    # Load mappings
    print("\n[1/5] Loading skill2group mapping...")
    proc = Path(args.processed_dir)
    with open(proc / "skill2group.json") as f:
        skill2group = json.load(f)
    print(f"      Loaded {len(skill2group)} skill→group mappings")

    # Load pairs
    print("\n[2/5] Loading pairs...")
    pairs_files = list(Path().glob(args.pairs_glob))
    if not pairs_files:
        raise FileNotFoundError(f"No files found for {args.pairs_glob}")
    pairs = pd.concat([pd.read_csv(f) for f in pairs_files], ignore_index=True)
    print(f"      Loaded {len(pairs)} pairs from {len(pairs_files)} files")

    # Load occupation-skill relations
    print("\n[3/5] Loading occupation-skill relations...")
    rel = pd.read_csv(args.skills_per_occ, usecols=["occupationUri", "skillUri"])
    rel = rel.rename(columns={"occupationUri": "esco_id", "skillUri": "skill_id"})
    print(f"      Loaded {len(rel)} occupation-skill relations")

    # Map to groups
    print("\n[4/5] Mapping to skill groups...")
    df = pairs.merge(rel, on="esco_id", how="left")
    
    # Diagnostic: check unique skills before mapping
    unique_skills_before = df["skill_id"].nunique()
    print(f"      Unique skills in data: {unique_skills_before}")
    print(f"      Skills in skill2group mapping: {len(skill2group)}")
    
    df["group_id"] = df["skill_id"].map(skill2group)
    
    # Diagnostic: check mapping success
    mapped_count = df["group_id"].notna().sum()
    total_count = len(df)
    print(f"      Successfully mapped: {mapped_count}/{total_count} ({100*mapped_count/total_count:.1f}%)")
    
    df = df.dropna(subset=["group_id"])
    print(f"      After mapping: {len(df)} rows")

    # Aggregate groups per title
    agg = (
        df.groupby(["raw_title", "raw_description"])["group_id"]
        .apply(lambda x: sorted(set(x)))
        .reset_index()
    )
    
    # Filter by min_groups_per_title
    agg = agg[agg["group_id"].map(len) >= args.min_groups_per_title]
    print(f"      After filtering (min_groups_per_title={args.min_groups_per_title}): {len(agg)} titles")

    # Count group frequencies
    print("\n[5/5] Analyzing group frequencies...")
    group_counts = {}
    for arr in agg["group_id"]:
        for g in arr:
            group_counts[g] = group_counts.get(g, 0) + 1
    
    if len(group_counts) == 0:
        print("\n" + "=" * 80)
        print("ERROR: No skill groups found after mapping!")
        print("=" * 80)
        print("\nPossible causes:")
        print("1. skill2group.json doesn't contain the skills from skills_per_occupations.csv")
        print("2. The esco_id values in pairs don't match occupationUri in skills_per_occupations.csv")
        print("3. build_hierarchy.py needs to be run with correct ESCO datasets")
        print("\nTroubleshooting steps:")
        print("- Check if build_hierarchy.py was run successfully")
        print("- Verify that skill2group.json contains mappings for skills in your data")
        print("- Check the first few rows of your pairs CSV to see esco_id format")
        return
    
    print(f"      Total unique groups: {len(group_counts)}")
    print(f"      Group frequency range: {min(group_counts.values())} - {max(group_counts.values())}")

    # Analyze different thresholds
    print("\n" + "=" * 80)
    print("ANALYSIS: Impact of Different min_samples_per_group Thresholds")
    print("=" * 80)
    
    thresholds = [1, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]
    results = []
    
    for threshold in thresholds:
        # Filter groups
        keep_groups = {g for g, c in group_counts.items() if c >= threshold}
        
        # Filter titles
        filtered_agg = agg.copy()
        filtered_agg["group_id"] = filtered_agg["group_id"].apply(
            lambda arr: [g for g in arr if g in keep_groups]
        )
        filtered_agg = filtered_agg[filtered_agg["group_id"].map(len) >= args.min_groups_per_title]
        
        results.append({
            "threshold": threshold,
            "n_groups": len(keep_groups),
            "n_titles": len(filtered_agg),
            "pct_groups": 100 * len(keep_groups) / len(group_counts),
            "pct_titles": 100 * len(filtered_agg) / len(agg)
        })
    
    results_df = pd.DataFrame(results)
    
    print("\n" + results_df.to_string(index=False))
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("GROUP FREQUENCY DISTRIBUTION")
    print("=" * 80)
    freq_series = pd.Series(list(group_counts.values()))
    print(freq_series.describe())
    
    print("\nPercentiles:")
    for pct in [10, 25, 50, 75, 90, 95, 99]:
        val = freq_series.quantile(pct / 100)
        print(f"  {pct:2d}th percentile: {val:.1f} samples")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    # Find threshold that keeps 80% of data
    keep_80_data = results_df[results_df["pct_titles"] >= 80.0]
    if not keep_80_data.empty:
        rec_80 = keep_80_data.iloc[-1]
        print(f"\n✓ To keep ≥80% of data:")
        print(f"  min_samples_per_group = {rec_80['threshold']}")
        print(f"  → Retains {rec_80['n_titles']} titles ({rec_80['pct_titles']:.1f}%)")
        print(f"  → Retains {rec_80['n_groups']} groups ({rec_80['pct_groups']:.1f}%)")
    
    # Find threshold that keeps 90% of groups
    keep_90_groups = results_df[results_df["pct_groups"] >= 90.0]
    if not keep_90_groups.empty:
        rec_90 = keep_90_groups.iloc[-1]
        print(f"\n✓ To keep ≥90% of groups:")
        print(f"  min_samples_per_group = {rec_90['threshold']}")
        print(f"  → Retains {rec_90['n_titles']} titles ({rec_90['pct_titles']:.1f}%)")
        print(f"  → Retains {rec_90['n_groups']} groups ({rec_90['pct_groups']:.1f}%)")
    
    # Balanced recommendation
    median_threshold = int(freq_series.quantile(0.25))
    rec_bal = results_df[results_df["threshold"] == median_threshold]
    if rec_bal.empty:
        # Find closest
        rec_bal = results_df.iloc[(results_df["threshold"] - median_threshold).abs().argmin()]
    else:
        rec_bal = rec_bal.iloc[0]
    
    print(f"\n✓ Balanced (25th percentile of group frequencies):")
    print(f"  min_samples_per_group = {rec_bal['threshold']}")
    print(f"  → Retains {rec_bal['n_titles']} titles ({rec_bal['pct_titles']:.1f}%)")
    print(f"  → Retains {rec_bal['n_groups']} groups ({rec_bal['pct_groups']:.1f}%)")
    
    # Create visualization
    if not HAS_MATPLOTLIB:
        print("\n✗ Skipping plots (matplotlib not installed)")
        print("\n" + "=" * 80)
        print("Analysis complete!")
        print("=" * 80)
        return
    
    print("\n[Plotting results...]")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Groups vs Threshold
    ax1 = axes[0, 0]
    ax1.plot(results_df["threshold"], results_df["n_groups"], marker='o', linewidth=2, markersize=6)
    ax1.set_xlabel("min_samples_per_group", fontsize=11)
    ax1.set_ylabel("Number of Groups Retained", fontsize=11)
    ax1.set_title("Groups Retained vs Threshold", fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Titles vs Threshold
    ax2 = axes[0, 1]
    ax2.plot(results_df["threshold"], results_df["n_titles"], marker='s', color='orange', linewidth=2, markersize=6)
    ax2.set_xlabel("min_samples_per_group", fontsize=11)
    ax2.set_ylabel("Number of Titles Retained", fontsize=11)
    ax2.set_title("Titles Retained vs Threshold", fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Group frequency distribution
    ax3 = axes[1, 0]
    freq_values = list(group_counts.values())
    ax3.hist(freq_values, bins=50, edgecolor='black', alpha=0.7)
    ax3.set_xlabel("Samples per Group", fontsize=11)
    ax3.set_ylabel("Number of Groups", fontsize=11)
    ax3.set_title("Group Frequency Distribution", fontsize=12, fontweight='bold')
    ax3.axvline(freq_series.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {freq_series.median():.0f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Combined retention percentages
    ax4 = axes[1, 1]
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(results_df["threshold"], results_df["pct_groups"], marker='o', color='steelblue', 
                     linewidth=2, markersize=6, label='% Groups')
    line2 = ax4_twin.plot(results_df["threshold"], results_df["pct_titles"], marker='s', color='darkorange',
                          linewidth=2, markersize=6, label='% Titles')
    
    ax4.set_xlabel("min_samples_per_group", fontsize=11)
    ax4.set_ylabel("% Groups Retained", fontsize=11, color='steelblue')
    ax4_twin.set_ylabel("% Titles Retained", fontsize=11, color='darkorange')
    ax4.set_title("Retention Percentages vs Threshold", fontsize=12, fontweight='bold')
    ax4.tick_params(axis='y', labelcolor='steelblue')
    ax4_twin.tick_params(axis='y', labelcolor='darkorange')
    ax4.grid(True, alpha=0.3)
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax4.legend(lines, labels, loc='upper right')
    
    plt.tight_layout()
    
    if args.output:
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plot saved to: {args.output}")
    else:
        plt.savefig("min_samples_analysis.png", dpi=150, bbox_inches='tight')
        print(f"\n✓ Plot saved to: min_samples_analysis.png")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()

