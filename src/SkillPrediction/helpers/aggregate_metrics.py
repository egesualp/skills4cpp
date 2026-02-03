#!/usr/bin/env python3
"""
Aggregate metrics JSON files under an outputs directory into a single pandas DataFrame.

Searches recursively for files named `grid_each_metrics.json`, `metrics.json`, or `results.json`,
flattens nested JSON objects into dot-separated keys, aligns columns across all files, and
adds a `folder` column containing the relative folder path that contains each metrics file.
Missing values are represented as `np.nan`.

Usage:
    python aggregate_metrics.py /path/to/outputs --out aggregated.csv
"""
from pathlib import Path
import json
import os
from typing import Dict, Any, Iterable
import pandas as pd
import numpy as np


def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """Recursively flatten a nested dict into a single-level dict with dot-separated keys."""
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def find_metric_files(root: str, names: Iterable[str] = None):
    """Yield Path objects for metric files under `root` matching known filenames."""
    if names is None:
        names = {
            "grid_each_metrics.json",
            "grid_search_metrics.json",
            "metrics.json",
            "results.json",
        }
    root_path = Path(root)
    for p in root_path.rglob("*"):
        if p.is_file() and p.name in names:
            yield p


def aggregate_metrics(root: str, save_csv: str = None) -> pd.DataFrame:
    """
    Crawl `root` for known metric JSON files, load and flatten them, then return a DataFrame.

    - `root`: path to the outputs directory to crawl.
    - `save_csv`: optional path to write the aggregated CSV.
    """
    rows = []
    all_keys = set()

    for file_path in find_metric_files(root):
        try:
            with open(file_path, "r") as fh:
                payload = json.load(fh)
        except Exception:
            # skip files that cannot be read or parsed
            continue

        if isinstance(payload, dict):
            flat = flatten_dict(payload)
        else:
            # non-dict JSON (e.g., a list or scalar) — store under 'value'
            flat = {"value": payload}

        all_keys.update(flat.keys())
        folder = os.path.relpath(str(file_path.parent), start=root)
        row = {"folder": folder, "file_name": file_path.name}
        row.update(flat)
        rows.append(row)

    # Create DataFrame; pandas will fill missing entries with NaN
    df = pd.DataFrame(rows)

    # Ensure all discovered metric keys exist as columns (so missing ones become NaN)
    for k in sorted(all_keys):
        if k not in df.columns:
            df[k] = np.nan

    # Reorder columns: folder, file_name, then sorted metric keys
    metric_cols = [c for c in df.columns if c not in ("folder", "file_name")]
    ordered_cols = ["folder", "file_name"] + sorted(metric_cols)
    df = df.loc[:, ordered_cols]

    if save_csv:
        df.to_csv(save_csv, index=False)

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Aggregate metrics JSON files under a root outputs directory.")
    parser.add_argument("root", nargs="?", default="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs", help="Root outputs directory to crawl.")
    parser.add_argument("--out", "-o", default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    df = aggregate_metrics(args.root, save_csv=args.out)
    print(f"Found {len(df)} metric files. DataFrame shape: {df.shape}")
    if args.out:
        print(f"Wrote aggregated CSV to {args.out}")


if __name__ == "__main__":
    main()


