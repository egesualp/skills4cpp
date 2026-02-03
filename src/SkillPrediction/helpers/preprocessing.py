# src/skill_mapping/preprocessing.py
"""
Prepares raw_title/raw_description/esco_id pairs into training data for hierarchical skill category prediction.

Reads CSVs with raw job data, cleans text, maps to ESCO skill groups using pre-processed files,
and outputs train/dev/test splits for both title-only and title+description variants.
"""
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List
import pandas as pd
from sklearn.model_selection import train_test_split

BULLET_RE = re.compile(r"^\s*[-•*\u2022]\s*")
WS_RE = re.compile(r"\s+")

def clean_text(text: str, lowercase=False, max_chars=None) -> str:
    """
    Clean text by removing bullet points, collapsing whitespace, and optionally truncating.
    
    Args:
        text: Raw text to clean
        lowercase: Whether to convert to lowercase
        max_chars: Maximum character length (truncates at word boundary if exceeded)
    
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Remove bullet points from line starts and strip empty lines
    lines = [BULLET_RE.sub("", ln.strip()) for ln in text.splitlines() if ln.strip()]
    
    # Join lines with ". " and collapse whitespace
    s = ". ".join(lines)
    s = WS_RE.sub(" ", s).strip()
    
    if lowercase:
        s = s.lower()
    
    if max_chars and len(s) > max_chars:
        # Truncate at word boundary
        s = s[:max_chars].rsplit(" ", 1)[0]
    
    return s

def main():
    p = argparse.ArgumentParser(
        description="Prepare raw job data for hierarchical skill category prediction"
    )
    p.add_argument(
        "--pairs_glob", 
        required=True, 
        help="Glob pattern to raw↔ESCO CSVs (e.g. data/title_pairs_desc/*.csv)"
    )
    p.add_argument(
        "--skills_per_occ", 
        default="data/skills_per_occupations.csv",
        help="Path to skills_per_occupations.csv"
    )
    p.add_argument(
        "--processed_dir", 
        default="data/processed", 
        help="Path containing skill2group.json & group2label.json"
    )
    p.add_argument(
        "--out_dir", 
        required=True, 
        help="Output directory for cat_* files"
    )
    p.add_argument(
        "--lowercase", 
        action="store_true",
        help="Convert text to lowercase"
    )
    p.add_argument(
        "--max_chars", 
        type=int, 
        default=600,
        help="Maximum characters per text field"
    )
    p.add_argument(
        "--min_groups_per_title", 
        type=int, 
        default=1,
        help="Minimum number of skill groups required per title"
    )
    p.add_argument(
        "--min_samples_per_group", 
        type=int, 
        default=10,
        help="Minimum samples required to keep a skill group"
    )
    p.add_argument(
        "--test_size", 
        type=float, 
        default=0.1,
        help="Proportion of data for test set"
    )
    p.add_argument(
        "--dev_size", 
        type=float, 
        default=0.1,
        help="Proportion of training data for dev set"
    )
    p.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducible splits"
    )
    args = p.parse_args()

    # --- Load mappings ---
    print(f"Loading skill mappings from {args.processed_dir}...")
    proc = Path(args.processed_dir)
    with open(proc / "skill2group.json") as f:
        skill2group = json.load(f)
    with open(proc / "group2label.json") as f:
        group2label = json.load(f)
    print(f"Loaded {len(skill2group)} skill→group mappings and {len(group2label)} group labels")

    # --- Load DECORTE-style pairs ---
    print(f"Loading pairs from {args.pairs_glob}...")
    pairs_files = list(Path().glob(args.pairs_glob))
    if not pairs_files:
        raise FileNotFoundError(f"No files found for {args.pairs_glob}")
    print(f"Found {len(pairs_files)} files")
    pairs = pd.concat([pd.read_csv(f) for f in pairs_files], ignore_index=True)
    print(f"Loaded {len(pairs)} raw pairs")

    # --- Clean text ---
    print("Cleaning text fields...")
    pairs["raw_description"] = pairs["raw_description"].fillna("").apply(
        lambda x: clean_text(x, lowercase=args.lowercase, max_chars=args.max_chars)
    )
    pairs["text_title_only"] = pairs["raw_title"].astype(str)
    pairs["text_with_desc"] = pairs.apply(
        lambda r: f"{r['raw_title']}. {r['raw_description']}" if r['raw_description'] else r['raw_title'],
        axis=1
    )

    # --- Join with ESCO skills ---
    print(f"Loading occupation-skill relations from {args.skills_per_occ}...")
    rel = pd.read_csv(args.skills_per_occ, usecols=["occupationUri", "skillUri"])
    rel = rel.rename(columns={"occupationUri": "esco_id", "skillUri": "skill_id"})
    print(f"Loaded {len(rel)} occupation-skill relations")
    
    df = pairs.merge(rel, on="esco_id", how="left")
    
    # Diagnostic: check mapping success
    unique_skills = df["skill_id"].nunique()
    print(f"Unique skills in merged data: {unique_skills}")
    print(f"Skills in skill2group mapping: {len(skill2group)}")
    
    df["group_id"] = df["skill_id"].map(skill2group)
    
    mapped_count = df["group_id"].notna().sum()
    total_count = len(df)
    print(f"Successfully mapped: {mapped_count}/{total_count} ({100*mapped_count/total_count:.1f}%)")
    
    if mapped_count == 0:
        raise ValueError(
            f"No skills were mapped to groups! "
            f"skill2group.json has {len(skill2group)} mappings but none match the {unique_skills} unique skills in your data. "
            f"This likely means build_hierarchy.py needs to be re-run or there's a mismatch in skill URIs."
        )
    
    df = df.dropna(subset=["group_id"])
    print(f"After mapping to skill groups: {len(df)} rows")

    # --- Aggregate groups per title ---
    print("Aggregating skill groups per title...")
    agg = (
        df.groupby(["raw_title", "raw_description", "text_title_only", "text_with_desc"])["group_id"]
        .apply(lambda x: sorted(set(x)))
        .reset_index()
    )
    print(f"Aggregated to {len(agg)} unique titles")

    # Filter by # groups per title
    agg = agg[agg["group_id"].map(len) >= args.min_groups_per_title]
    print(f"After filtering by min_groups_per_title={args.min_groups_per_title}: {len(agg)} titles")

    # --- Drop rare groups ---
    print(f"Filtering rare groups (min_samples_per_group={args.min_samples_per_group})...")
    counts = {}
    for arr in agg["group_id"]:
        for g in arr:
            counts[g] = counts.get(g, 0) + 1
    
    keep = {g for g, c in counts.items() if c >= args.min_samples_per_group}
    print(f"Keeping {len(keep)}/{len(counts)} groups")
    
    agg["group_id"] = agg["group_id"].apply(lambda arr: [g for g in arr if g in keep])
    agg = agg[agg["group_id"].map(len) >= args.min_groups_per_title]
    print(f"After filtering: {len(agg)} titles remain")

    # Validate we have enough data
    if len(agg) == 0:
        raise ValueError(
            "No samples remaining after filtering! "
            "Try reducing --min_samples_per_group or --min_groups_per_title"
        )
    
    min_samples_needed = int(1 / min(args.test_size, args.dev_size / (1.0 - args.test_size))) + 1
    if len(agg) < min_samples_needed:
        raise ValueError(
            f"Not enough samples ({len(agg)}) for splitting with test_size={args.test_size} "
            f"and dev_size={args.dev_size}. Need at least {min_samples_needed} samples. "
            "Try reducing --min_samples_per_group or --min_groups_per_title"
        )

    # Create binary columns for each group
    groups = sorted({g for arr in agg["group_id"] for g in arr})
    print(f"Creating binary columns for {len(groups)} groups...")
    for g in groups:
        agg[g] = agg["group_id"].apply(lambda arr: int(g in arr))

    # --- Save splits ---
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    
    def split_save(df, variant):
        """Split data and save train/dev/test CSVs"""
        tr, te = train_test_split(df, test_size=args.test_size, random_state=args.seed)
        tr, dv = train_test_split(tr, test_size=args.dev_size/(1.0-args.test_size), random_state=args.seed)
        
        tr.to_csv(out / f"cat_{variant}_train.csv", index=False)
        dv.to_csv(out / f"cat_{variant}_dev.csv", index=False)
        te.to_csv(out / f"cat_{variant}_test.csv", index=False)
        
        print(f"Saved cat_{variant}_[train/dev/test].csv → {len(tr)}/{len(dv)}/{len(te)} samples")

    print("\nSaving splits...")
    # Title-only variant
    split_save(
        agg[["text_title_only"] + groups].rename(columns={"text_title_only": "title"}), 
        "title"
    )
    
    # Title + description variant
    split_save(
        agg[["text_with_desc"] + groups].rename(columns={"text_with_desc": "title"}), 
        "desc"
    )

    # Save metadata
    meta = {
        "groups": groups,
        "group_labels": {g: group2label.get(g, g) for g in groups}
    }
    with open(out / "cat_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"\n[✓] Successfully wrote metadata + {len(groups)} groups to {out}")
    print(f"[✓] Output files: cat_title_*.csv, cat_desc_*.csv, cat_meta.json")

if __name__ == "__main__":
    main()

