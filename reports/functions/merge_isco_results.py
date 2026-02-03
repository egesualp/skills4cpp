#!/usr/bin/env python3
"""
Scan an outputs directory for folders with "isco_model_" in their name,
read each folder's results.json and write a combined JSON array with the
folder name and its results.

Usage:
  python merge_isco_results.py \
    --outputs-dir /path/to/outputs \
    --out-file /path/to/combined_isco_model_results.json
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger("merge_isco_results")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def find_isco_dirs(outputs_dir: Path, pattern: str = "isco_model_") -> List[Path]:
    if not outputs_dir.exists() or not outputs_dir.is_dir():
        raise FileNotFoundError(f"outputs directory not found: {outputs_dir}")
    dirs = [p for p in sorted(outputs_dir.iterdir()) if p.is_dir() and pattern in p.name]
    logger.info("Found %d matching directories", len(dirs))
    return dirs


def load_results_file(dir_path: Path) -> Any:
    results_path = dir_path / "results.json"
    if not results_path.exists():
        logger.warning("No results.json in %s — skipping", dir_path)
        return None
    try:
        with results_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        logger.warning("Failed to read/parse %s: %s", results_path, e)
        return None


def merge(outputs_dir: Path, out_file: Path, pattern: str = "isco_model_") -> None:
    dirs = find_isco_dirs(outputs_dir, pattern=pattern)
    combined: List[Dict[str, Any]] = []
    for d in dirs:
        data = load_results_file(d)
        if data is None:
            continue
        combined.append({"folder": d.name, "results": data})

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as fh:
        json.dump(combined, fh, indent=2, ensure_ascii=False)
    logger.info("Wrote %d entries to %s", len(combined), out_file)


def main() -> None:
    p = argparse.ArgumentParser(description="Merge isco_model_* results.json files")
    p.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs"),
        help="Path to outputs directory (default: project outputs)",
    )
    p.add_argument(
        "--out-file",
        type=Path,
        default=Path("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/combined_isco_model_results.json"),
        help="Output combined JSON file path",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default="isco_model_",
        help="Directory name pattern to match (default: 'isco_model_')",
    )
    args = p.parse_args()

    try:
        merge(args.outputs_dir, args.out_file, pattern=args.pattern)
    except Exception as e:
        logger.error("Error: %s", e)
        raise


if __name__ == "__main__":
    main()

