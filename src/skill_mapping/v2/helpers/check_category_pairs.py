import argparse
import pandas as pd
import sys
import random
from pathlib import Path
from loguru import logger

# Ensure we can import from skill_mapping
sys.path.append(str(Path(__file__).resolve().parents[3]))

from skill_mapping.v2.category_trainer import (
    load_esco_data,
    build_occupation_samples,
    build_validation_samples,
    HIER_COL_MAP
)

class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
        self.stdout.flush()

def print_sample(sample, label="Sample", soft_labels=False):
    print(f"\n--- {label} ---")
    if "uri" in sample:
        print(f"URI: {sample['uri']}")
    if "job_id" in sample:
        print(f"Job ID: {sample['job_id']}")
    
    print(f"Text: {sample['text']}")
    
    if soft_labels and "category_counts" in sample:
        print("Categories (Soft Labels - Counts):")
        total = sum(sample["category_counts"].values())
        sorted_cats = sorted(sample["category_counts"].items(), key=lambda x: x[1], reverse=True)
        for cat, count in sorted_cats:
            percentage = (count / total) * 100
            print(f"  - {cat}: {count} ({percentage:.1f}%)")
    elif "categories" in sample:
        print(f"Categories: {sample['categories']}")

def main():
    parser = argparse.ArgumentParser(description="Check Category Pairs for Sanity Check")
    
    # Data paths (subset of original args)
    parser.add_argument("--esco_path", type=str, required=True, help="Path to ESCO master CSV")
    parser.add_argument("--val_path", type=str, required=True, help="Path to validation CSV")
    parser.add_argument(
        "--augmented_path", type=str, default=None,
        help="Path to augmented occupation CSV"
    )
    
    parser.add_argument("--target_level", type=int, default=1, choices=[0, 1, 2, 3])
    parser.add_argument(
        "--soft_labels", action="store_true",
        help="Use soft labels"
    )
    
    parser.add_argument("--num_examples", type=int, default=5, help="Number of examples to print")
    parser.add_argument("--output_file", type=str, default="sanity_check_results.txt", help="Output file name (saved in script directory)")

    args = parser.parse_args()

    # Redirect output to file
    output_path = Path(__file__).parent / args.output_file
    print(f"Saving output to: {output_path}")
    tee = Tee(output_path, "w")
    
    print(f"Target Level: {args.target_level} ({HIER_COL_MAP[args.target_level]})")
    print(f"Soft Labels: {args.soft_labels}")
    
    # 1. Load ESCO Data
    print("\n[1] Loading ESCO Data...")
    esco_df, cat_col = load_esco_data(args.esco_path, args.target_level)
    
    # 2. Load Augmented Data
    augmented_df = None
    if args.augmented_path:
        print(f"\n[2] Loading Augmented Data from {args.augmented_path}...")
        augmented_df = pd.read_csv(args.augmented_path)
    
    # 3. Build Training Samples (Occupation -> Categories)
    print("\n[3] Building Training Samples (Occupations)...")
    train_samples = build_occupation_samples(
        esco_df, cat_col, augmented_df=augmented_df, soft_labels=args.soft_labels
    )
    print(f"Found {len(train_samples)} training samples.")
    
    print(f"\n[3.1] Random Training Examples ({args.num_examples}):")
    if train_samples:
        for _ in range(args.num_examples):
            sample = random.choice(train_samples)
            print_sample(sample, label="Training Sample", soft_labels=args.soft_labels)

    # Validate Stratified Sampling
    print("\n[3.2] Validating Stratified Sampling...")
    from skill_mapping.v2.category_trainer import stratified_split_occupations
    from collections import Counter
    import numpy as np
    
    # Perform split
    train_split, val_split = stratified_split_occupations(train_samples, test_size=0.1)
    
    # Extract unique URIs and their ISCO groups
    train_uris = {s['uri']: s.get('isco_group', 'UNKNOWN') for s in train_split}
    val_uris = {s['uri']: s.get('isco_group', 'UNKNOWN') for s in val_split}
    
    # Check for leakage
    intersection = set(train_uris.keys()) & set(val_uris.keys())
    print(f"  - Train Unique Occupations: {len(train_uris)}")
    print(f"  - Val Unique Occupations: {len(val_uris)}")
    print(f"  - Intersection (Leakage Check): {len(intersection)} (Should be 0)")
    if intersection:
        print(f"    WARNING: DATA LEAKAGE DETECTED! Shared URIs: {list(intersection)[:5]}...")
    else:
        print("    PASS: No data leakage detected (occupations disjoint).")
        
    # Check distribution of ISCO groups
    train_counts = Counter(train_uris.values())
    val_counts = Counter(val_uris.values())
    
    print("\n  - Stratification Check (Top 5 ISCO Groups):")
    all_groups = set(train_counts.keys()) | set(val_counts.keys())
    sorted_groups = sorted(all_groups, key=lambda g: train_counts[g] + val_counts[g], reverse=True)
    
    print(f"    {'ISCO Group':<15} {'Train':<10} {'Val':<10} {'Val Ratio':<10}")
    print(f"    {'-'*15} {'-'*10} {'-'*10} {'-'*10}")
    
    for group in sorted_groups[:10]:
        t = train_counts.get(group, 0)
        v = val_counts.get(group, 0)
        total = t + v
        ratio = v / total if total > 0 else 0
        print(f"    {group:<15} {t:<10} {v:<10} {ratio:.1%}")

    # 4. Build Validation Samples (Job Description -> Categories)
    print("\n[4] Building Validation Samples (External Validation)...")
    val_df = pd.read_csv(args.val_path)
    val_samples = build_validation_samples(val_df, esco_df, cat_col, soft_labels=args.soft_labels)
    print(f"Found {len(val_samples)} validation samples.")
    
    print(f"\n[4.1] Random Validation Examples ({args.num_examples}):")
    if val_samples:
        for _ in range(args.num_examples):
            sample = random.choice(val_samples)
            print_sample(sample, label="Validation Sample", soft_labels=args.soft_labels)
    else:
        print("No validation samples found!")

if __name__ == "__main__":
    main()

