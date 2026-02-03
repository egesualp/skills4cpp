#!/usr/bin/env python3
"""
ISCO Skill Coverage Analysis Script

Analyzes whether the ISCO group approach is viable for skill prediction by:
1. Loading pre-trained ISCO classifier and predicting ISCO groups for test data
2. Calculating skill coverage at top-1, top-2, top-3, top-10 ISCO predictions
3. Providing descriptive statistics for skills per ISCO group

Usage:
    python -m skill_mapping.v2.helpers.isco_skill_coverage_analysis
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ==============================================================================
# Model Definition (mirror from isco_trainer.py)
# ==============================================================================
class ISCOClassifier(nn.Module):
    """Deep MLP for single-label ISCO prediction."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = [512],
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class SingleLabelEncoder:
    """Load/decode ISCO labels."""

    def __init__(self):
        self.str2idx = {}
        self.idx2str = {}

    @classmethod
    def load(cls, path: str) -> "SingleLabelEncoder":
        with open(path, "r") as f:
            data = json.load(f)
        encoder = cls()
        encoder.str2idx = data["str2idx"]
        encoder.idx2str = {int(k): v for k, v in data["idx2str"].items()}
        return encoder

    def decode(self, idx: int) -> str:
        return self.idx2str[idx]

    def __len__(self):
        return len(self.str2idx)


# ==============================================================================
# Data Loading
# ==============================================================================
def load_test_data(test_csv_path: str) -> pd.DataFrame:
    """Load test data with occupation mapping."""
    print(f"Loading test data from {test_csv_path}...")
    df = pd.read_csv(test_csv_path)
    print(f"  Loaded {len(df)} test samples")
    return df


def load_occupations(occupations_path: str) -> pd.DataFrame:
    """Load ESCO occupations with ISCO group mapping."""
    print(f"Loading occupations from {occupations_path}...")
    df = pd.read_csv(occupations_path)
    print(f"  Loaded {len(df)} occupations")
    return df


def load_occupation_skill_relations(relations_path: str) -> pd.DataFrame:
    """Load occupation-skill relations."""
    print(f"Loading occupation-skill relations from {relations_path}...")
    df = pd.read_csv(relations_path)
    print(f"  Loaded {len(df)} relations")
    return df


def load_embeddings(embeddings_path: str) -> np.ndarray:
    """Load pre-computed embeddings."""
    print(f"Loading embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path)
    print(f"  Loaded embeddings shape: {embeddings.shape}")
    return embeddings


def load_model(
    model_path: str,
    config_path: str,
    label_encoder_path: str,
    device: str = "cuda",
) -> Tuple[ISCOClassifier, SingleLabelEncoder, dict]:
    """Load pre-trained ISCO classifier."""
    print(f"Loading model from {model_path}...")

    # Load config
    with open(config_path, "r") as f:
        results = json.load(f)

    config = results["config"]
    best_params = results["best_params"]

    # Extract hidden dims
    hidden_dims = config.get("hidden_dims", [512])
    use_batchnorm = config.get("use_batchnorm", True)

    # Create model
    model = ISCOClassifier(
        input_dim=config["input_dim"],
        num_classes=config["num_classes"],
        hidden_dims=hidden_dims,
        dropout=best_params.get("dropout", 0.1),
        use_batchnorm=use_batchnorm,
    )

    # Load weights
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load label encoder
    label_encoder = SingleLabelEncoder.load(label_encoder_path)

    print(f"  Model loaded: input_dim={config['input_dim']}, num_classes={config['num_classes']}")
    print(f"  Hidden dims: {hidden_dims}, use_batchnorm: {use_batchnorm}")

    return model, label_encoder, config


# ==============================================================================
# ISCO Group -> Skills Mapping
# ==============================================================================
def build_isco_to_skills_mapping(
    occupations_df: pd.DataFrame,
    relations_df: pd.DataFrame,
    isco_level: int = 2,
) -> Dict[str, Set[str]]:
    """
    Build mapping from ISCO group -> set of skill URIs.

    For each ISCO group, aggregate all skills from all occupations in that group.
    """
    print(f"Building ISCO group (level {isco_level}) to skills mapping...")

    # Create occupation -> ISCO group mapping
    occ_to_isco = {}
    for _, row in occupations_df.iterrows():
        occ_uri = row["conceptUri"]
        isco_code = str(row.get("iscoGroup", "")).strip()
        if isco_code and isco_code.lower() != "nan":
            # Truncate to desired level (e.g., level 2 = first 2 digits)
            isco_truncated = isco_code[:isco_level].zfill(isco_level)
            occ_to_isco[occ_uri] = isco_truncated

    # Build ISCO -> skills mapping
    isco_to_skills: Dict[str, Set[str]] = defaultdict(set)
    for _, row in relations_df.iterrows():
        occ_uri = row["occupationUri"]
        skill_uri = row["skillUri"]
        if occ_uri in occ_to_isco:
            isco_group = occ_to_isco[occ_uri]
            isco_to_skills[isco_group].add(skill_uri)

    print(f"  Built mapping for {len(isco_to_skills)} ISCO groups")
    return dict(isco_to_skills)


def build_occupation_to_skills_mapping(relations_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """Build mapping from occupation URI -> set of skill URIs."""
    print("Building occupation to skills mapping...")
    occ_to_skills: Dict[str, Set[str]] = defaultdict(set)
    for _, row in relations_df.iterrows():
        occ_uri = row["occupationUri"]
        skill_uri = row["skillUri"]
        occ_to_skills[occ_uri].add(skill_uri)
    print(f"  Built mapping for {len(occ_to_skills)} occupations")
    return dict(occ_to_skills)


# ==============================================================================
# Prediction and Coverage Calculation
# ==============================================================================
@torch.no_grad()
def predict_topk_isco(
    model: ISCOClassifier,
    embeddings: np.ndarray,
    label_encoder: SingleLabelEncoder,
    k: int = 10,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict top-k ISCO groups for each embedding.

    Returns:
        top_k_indices: (n_samples, k) array of predicted ISCO indices
        top_k_probs: (n_samples, k) array of prediction probabilities
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model.eval()

    embeddings_tensor = torch.from_numpy(embeddings).float().to(device)
    logits = model(embeddings_tensor)
    probs = torch.softmax(logits, dim=-1)

    top_k = torch.topk(probs, k=min(k, probs.shape[-1]), dim=-1)
    top_k_indices = top_k.indices.cpu().numpy()
    top_k_probs = top_k.values.cpu().numpy()

    return top_k_indices, top_k_probs


def calculate_skill_coverage(
    test_df: pd.DataFrame,
    top_k_indices: np.ndarray,
    label_encoder: SingleLabelEncoder,
    isco_to_skills: Dict[str, Set[str]],
    occ_to_skills: Dict[str, Set[str]],
    k_values: List[int] = [1, 2, 3, 5, 10],
) -> Dict[str, Dict]:
    """
    Calculate skill coverage for different top-k predictions.

    For each test sample:
    - Get the ground truth skills from the occupation
    - Get the predicted ISCO groups (top-k)
    - Aggregate skills from predicted ISCO groups
    - Calculate what fraction of ground truth skills are covered
    """
    print("Calculating skill coverage at different k values...")

    coverage_results = {k: {"coverage_ratios": [], "covered_counts": [], "total_counts": []} for k in k_values}

    # Determine the URI column
    if "esco_id" in test_df.columns:
        uri_col = "esco_id"
    elif "conceptUri" in test_df.columns:
        uri_col = "conceptUri"
    else:
        raise ValueError("No URI column found in test data")

    skipped_no_gt = 0
    for i, row in test_df.iterrows():
        occ_uri = row[uri_col]

        # Get ground truth skills for this occupation
        gt_skills = occ_to_skills.get(occ_uri, set())
        if not gt_skills:
            skipped_no_gt += 1
            continue

        # Calculate coverage at each k
        for k in k_values:
            # Get top-k predicted ISCO groups
            predicted_isco_indices = top_k_indices[i, :k]
            predicted_iscos = [label_encoder.decode(idx) for idx in predicted_isco_indices]

            # Aggregate skills from predicted ISCO groups
            predicted_skills = set()
            for isco in predicted_iscos:
                predicted_skills.update(isco_to_skills.get(isco, set()))

            # Calculate coverage
            covered = gt_skills & predicted_skills
            coverage_ratio = len(covered) / len(gt_skills) if gt_skills else 0

            coverage_results[k]["coverage_ratios"].append(coverage_ratio)
            coverage_results[k]["covered_counts"].append(len(covered))
            coverage_results[k]["total_counts"].append(len(gt_skills))

    print(f"  Processed {len(test_df) - skipped_no_gt} samples (skipped {skipped_no_gt} with no ground truth)")

    # Calculate summary statistics
    summary = {}
    for k in k_values:
        ratios = coverage_results[k]["coverage_ratios"]
        if ratios:
            summary[k] = {
                "mean_coverage": np.mean(ratios),
                "median_coverage": np.median(ratios),
                "std_coverage": np.std(ratios),
                "min_coverage": np.min(ratios),
                "max_coverage": np.max(ratios),
                "p25_coverage": np.percentile(ratios, 25),
                "p75_coverage": np.percentile(ratios, 75),
                "n_samples": len(ratios),
                "perfect_coverage_pct": np.mean([r >= 1.0 for r in ratios]) * 100,
                "above_50pct_coverage": np.mean([r >= 0.5 for r in ratios]) * 100,
                "above_80pct_coverage": np.mean([r >= 0.8 for r in ratios]) * 100,
            }
        else:
            summary[k] = {"error": "No valid samples"}

    return summary


# ==============================================================================
# Descriptive Statistics for ISCO Groups
# ==============================================================================
def calculate_isco_skill_stats(isco_to_skills: Dict[str, Set[str]]) -> pd.DataFrame:
    """Calculate descriptive statistics for skills per ISCO group."""
    print("Calculating descriptive statistics for skills per ISCO group...")

    stats = []
    for isco_group, skills in isco_to_skills.items():
        stats.append({"isco_group": isco_group, "n_skills": len(skills)})

    stats_df = pd.DataFrame(stats)
    stats_df = stats_df.sort_values("isco_group")

    return stats_df


def calculate_isco_occupation_stats(
    occupations_df: pd.DataFrame,
    isco_level: int = 2,
) -> pd.DataFrame:
    """Calculate number of occupations per ISCO group."""
    print("Calculating occupations per ISCO group...")

    isco_counts = defaultdict(int)
    for _, row in occupations_df.iterrows():
        isco_code = str(row.get("iscoGroup", "")).strip()
        if isco_code and isco_code.lower() != "nan":
            isco_truncated = isco_code[:isco_level].zfill(isco_level)
            isco_counts[isco_truncated] += 1

    stats = [{"isco_group": k, "n_occupations": v} for k, v in isco_counts.items()]
    stats_df = pd.DataFrame(stats).sort_values("isco_group")

    return stats_df


# ==============================================================================
# Report Generation
# ==============================================================================
ISCO_NAMES = {
    "01": "Commissioned Armed Forces Officers",
    "02": "Non-commissioned Armed Forces Officers",
    "03": "Armed Forces Occupations, Other Ranks",
    "11": "Chief Executives, Senior Officials and Legislators",
    "12": "Administrative and Commercial Managers",
    "13": "Production and Specialized Services Managers",
    "14": "Hospitality, Retail and Other Services Managers",
    "21": "Science and Engineering Professionals",
    "22": "Health Professionals",
    "23": "Teaching Professionals",
    "24": "Business and Administration Professionals",
    "25": "Information and Communications Technology Professionals",
    "26": "Legal, Social and Cultural Professionals",
    "31": "Science and Engineering Associate Professionals",
    "32": "Health Associate Professionals",
    "33": "Business and Administration Associate Professionals",
    "34": "Legal, Social, Cultural and Related Associate Professionals",
    "35": "Information and Communications Technicians",
    "41": "General and Keyboard Clerks",
    "42": "Customer Services Clerks",
    "43": "Numerical and Material Recording Clerks",
    "44": "Other Clerical Support Workers",
    "51": "Personal Service Workers",
    "52": "Sales Workers",
    "53": "Personal Care Workers",
    "54": "Protective Services Workers",
    "61": "Market-oriented Skilled Agricultural Workers",
    "62": "Market-oriented Skilled Forestry, Fishery and Hunting Workers",
    "63": "Subsistence Farmers, Fishers, Hunters and Gatherers",
    "71": "Building and Related Trades Workers",
    "72": "Metal, Machinery and Related Trades Workers",
    "73": "Handicraft and Printing Workers",
    "74": "Electrical and Electronic Trades Workers",
    "75": "Food Processing, Wood Working, Garment and Other Craft Workers",
    "81": "Stationary Plant and Machine Operators",
    "82": "Assemblers",
    "83": "Drivers and Mobile Plant Operators",
    "91": "Cleaners and Helpers",
    "92": "Agricultural, Forestry and Fishery Labourers",
    "93": "Labourers in Mining, Construction, Manufacturing and Transport",
    "94": "Food Preparation Assistants",
    "95": "Street and Related Sales and Service Workers",
    "96": "Refuse Workers and Other Elementary Workers",
}


def generate_report(
    coverage_summary: Dict,
    isco_skill_stats: pd.DataFrame,
    isco_occ_stats: pd.DataFrame,
    isco_to_skills: Dict[str, Set[str]],
    model_config: dict,
) -> str:
    """Generate comprehensive analysis report."""
    report = []
    report.append("=" * 100)
    report.append("ISCO GROUP SKILL COVERAGE ANALYSIS REPORT")
    report.append("=" * 100)
    report.append("")

    # Model info
    report.append("MODEL INFORMATION")
    report.append("-" * 100)
    report.append(f"  ISCO Level: {model_config.get('isco_level', 'N/A')}")
    report.append(f"  Number of Classes: {model_config.get('num_classes', 'N/A')}")
    report.append(f"  Input Dimension: {model_config.get('input_dim', 'N/A')}")
    report.append("")

    # 1. Skill Coverage Analysis
    report.append("=" * 100)
    report.append("1. SKILL COVERAGE BY TOP-K ISCO PREDICTIONS")
    report.append("=" * 100)
    report.append("")
    report.append("This shows what fraction of ground truth skills are covered when using")
    report.append("skills from the top-k predicted ISCO groups.")
    report.append("")

    report.append(f"{'Top-K':<10} {'Mean':>10} {'Median':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'P25':>10} {'P75':>10}")
    report.append("-" * 100)

    for k in sorted(coverage_summary.keys()):
        stats = coverage_summary[k]
        if "error" not in stats:
            report.append(
                f"Top-{k:<5} "
                f"{stats['mean_coverage']:>10.4f} "
                f"{stats['median_coverage']:>10.4f} "
                f"{stats['std_coverage']:>10.4f} "
                f"{stats['min_coverage']:>10.4f} "
                f"{stats['max_coverage']:>10.4f} "
                f"{stats['p25_coverage']:>10.4f} "
                f"{stats['p75_coverage']:>10.4f}"
            )

    report.append("")
    report.append("Coverage thresholds (% of samples reaching threshold):")
    report.append(f"{'Top-K':<10} {'Perfect (100%)':>15} {'≥80%':>15} {'≥50%':>15}")
    report.append("-" * 60)

    for k in sorted(coverage_summary.keys()):
        stats = coverage_summary[k]
        if "error" not in stats:
            report.append(
                f"Top-{k:<5} "
                f"{stats['perfect_coverage_pct']:>14.1f}% "
                f"{stats['above_80pct_coverage']:>14.1f}% "
                f"{stats['above_50pct_coverage']:>14.1f}%"
            )

    report.append("")

    # 2. Skills per ISCO Group
    report.append("=" * 100)
    report.append("2. SKILLS PER ISCO GROUP - DESCRIPTIVE STATISTICS")
    report.append("=" * 100)
    report.append("")

    n_skills_array = isco_skill_stats["n_skills"].values
    report.append("Overall Statistics:")
    report.append(f"  Number of ISCO groups: {len(isco_skill_stats)}")
    report.append(f"  Mean skills per group: {np.mean(n_skills_array):.1f}")
    report.append(f"  Median skills per group: {np.median(n_skills_array):.1f}")
    report.append(f"  Std: {np.std(n_skills_array):.1f}")
    report.append(f"  Min: {np.min(n_skills_array)}")
    report.append(f"  Max: {np.max(n_skills_array)}")
    report.append(f"  P10: {np.percentile(n_skills_array, 10):.0f}")
    report.append(f"  P25: {np.percentile(n_skills_array, 25):.0f}")
    report.append(f"  P75: {np.percentile(n_skills_array, 75):.0f}")
    report.append(f"  P90: {np.percentile(n_skills_array, 90):.0f}")
    report.append("")

    # Merge stats for detailed view
    merged_stats = isco_skill_stats.merge(isco_occ_stats, on="isco_group", how="outer").fillna(0)
    merged_stats["n_occupations"] = merged_stats["n_occupations"].astype(int)
    merged_stats["n_skills"] = merged_stats["n_skills"].astype(int)
    merged_stats = merged_stats.sort_values("isco_group")

    report.append("Detailed breakdown by ISCO group:")
    report.append(f"{'ISCO':<6} {'Skills':>8} {'Occupations':>12} {'Description':<60}")
    report.append("-" * 100)

    for _, row in merged_stats.iterrows():
        isco = row["isco_group"]
        desc = ISCO_NAMES.get(isco, "Unknown")
        report.append(f"{isco:<6} {row['n_skills']:>8} {row['n_occupations']:>12} {desc:<60}")

    report.append("")

    # 3. Key Insights
    report.append("=" * 100)
    report.append("3. KEY INSIGHTS AND RECOMMENDATIONS")
    report.append("=" * 100)
    report.append("")

    # Calculate insights
    top1_coverage = coverage_summary.get(1, {}).get("mean_coverage", 0)
    top3_coverage = coverage_summary.get(3, {}).get("mean_coverage", 0)
    top10_coverage = coverage_summary.get(10, {}).get("mean_coverage", 0)
    top1_80pct = coverage_summary.get(1, {}).get("above_80pct_coverage", 0)
    top3_80pct = coverage_summary.get(3, {}).get("above_80pct_coverage", 0)

    avg_skills_per_group = np.mean(n_skills_array)
    total_unique_skills = len(set().union(*isco_to_skills.values()))

    report.append("Coverage Analysis:")
    report.append(f"  - Top-1 ISCO prediction covers {top1_coverage:.1%} of ground truth skills on average")
    report.append(f"  - Top-3 ISCO predictions cover {top3_coverage:.1%} of ground truth skills on average")
    report.append(f"  - Top-10 ISCO predictions cover {top10_coverage:.1%} of ground truth skills on average")
    report.append("")
    report.append(f"  - {top1_80pct:.1f}% of samples achieve ≥80% coverage with top-1 prediction")
    report.append(f"  - {top3_80pct:.1f}% of samples achieve ≥80% coverage with top-3 predictions")
    report.append("")

    report.append("ISCO Group Statistics:")
    report.append(f"  - Average: {avg_skills_per_group:.1f} skills per ISCO group")
    report.append(f"  - Total unique skills across all ISCO groups: {total_unique_skills}")
    report.append("")

    # Viability assessment
    report.append("Viability Assessment:")
    if top3_coverage >= 0.7:
        report.append("  ✓ PROMISING: Top-3 ISCO predictions cover ≥70% of skills")
        report.append("    → ISCO-based skill suggestion is a viable approach")
    elif top3_coverage >= 0.5:
        report.append("  ~ MODERATE: Top-3 ISCO predictions cover 50-70% of skills")
        report.append("    → ISCO approach provides reasonable coverage, may need supplementation")
    else:
        report.append("  ✗ LIMITED: Top-3 ISCO predictions cover <50% of skills")
        report.append("    → ISCO approach alone may not provide sufficient coverage")

    report.append("")

    if avg_skills_per_group > 500:
        report.append(f"  ⚠ CAUTION: High avg skills per group ({avg_skills_per_group:.0f})")
        report.append("    → May lead to low precision (many irrelevant skills)")
    elif avg_skills_per_group < 100:
        report.append(f"  ⚠ NOTE: Low avg skills per group ({avg_skills_per_group:.0f})")
        report.append("    → Good precision potential but may miss skills")

    report.append("")
    report.append("=" * 100)
    report.append("END OF REPORT")
    report.append("=" * 100)

    return "\n".join(report)


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="ISCO Skill Coverage Analysis")

    # Paths
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h2_soft_deep_larger_val_mpnet_kw_cp_decorte",
        help="Directory containing ISCO classifier and config",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/category_test_split_isco.csv",
        help="Path to test CSV",
    )
    parser.add_argument(
        "--occupations_csv",
        type=str,
        default="/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/occupations_en.csv",
        help="Path to ESCO occupations CSV",
    )
    parser.add_argument(
        "--relations_csv",
        type=str,
        default="/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/occupationSkillRelations_en.csv",
        help="Path to occupation-skill relations CSV",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for report (defaults to model_dir)",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir) if args.output_dir else model_dir

    print("\n" + "=" * 100)
    print("ISCO SKILL COVERAGE ANALYSIS")
    print("=" * 100 + "\n")

    # Load all data
    test_df = load_test_data(args.test_csv)
    occupations_df = load_occupations(args.occupations_csv)
    relations_df = load_occupation_skill_relations(args.relations_csv)

    # Load embeddings
    cache_dir = model_dir / "cache"
    embedding_files = list(cache_dir.glob("*_test.npy"))
    if not embedding_files:
        raise FileNotFoundError(f"No test embedding files found in {cache_dir}")
    embeddings_path = embedding_files[0]
    embeddings = load_embeddings(str(embeddings_path))

    # Verify embedding count matches test data
    if len(embeddings) != len(test_df):
        print(f"  WARNING: Embedding count ({len(embeddings)}) != test samples ({len(test_df)})")
        print("  Adjusting test_df to match embeddings...")
        test_df = test_df.iloc[: len(embeddings)]

    # Load model
    model, label_encoder, config = load_model(
        model_path=str(model_dir / "isco_classifier.pt"),
        config_path=str(model_dir / "results.json"),
        label_encoder_path=str(model_dir / "label_encoder.json"),
        device=args.device,
    )

    isco_level = config.get("isco_level", 2)

    # Build mappings
    isco_to_skills = build_isco_to_skills_mapping(occupations_df, relations_df, isco_level=isco_level)
    occ_to_skills = build_occupation_to_skills_mapping(relations_df)

    # Get predictions
    print("\nGenerating ISCO predictions...")
    top_k_indices, top_k_probs = predict_topk_isco(model, embeddings, label_encoder, k=10, device=args.device)
    print(f"  Generated top-10 predictions for {len(embeddings)} samples")

    # Calculate coverage
    k_values = [1, 2, 3, 5, 10]
    coverage_summary = calculate_skill_coverage(
        test_df, top_k_indices, label_encoder, isco_to_skills, occ_to_skills, k_values=k_values
    )

    # Calculate ISCO stats
    isco_skill_stats = calculate_isco_skill_stats(isco_to_skills)
    isco_occ_stats = calculate_isco_occupation_stats(occupations_df, isco_level=isco_level)

    # Generate report
    print("\nGenerating report...")
    report = generate_report(coverage_summary, isco_skill_stats, isco_occ_stats, isco_to_skills, config)

    # Print report
    print("\n" + report)

    # Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "isco_skill_coverage_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    # Save coverage data as JSON
    coverage_path = output_dir / "isco_skill_coverage_data.json"
    with open(coverage_path, "w") as f:
        json.dump(coverage_summary, f, indent=2)
    print(f"Coverage data saved to: {coverage_path}")

    # Save ISCO stats as CSV
    stats_path = output_dir / "isco_group_stats.csv"
    merged = isco_skill_stats.merge(isco_occ_stats, on="isco_group", how="outer").fillna(0)
    merged["description"] = merged["isco_group"].map(ISCO_NAMES)
    merged.to_csv(stats_path, index=False)
    print(f"ISCO group stats saved to: {stats_path}")

    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE!")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()



