#!/usr/bin/env python3
"""
Evaluate skill coverage from a predictions.jsonl file.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from .metrics import compute_skill_coverage, load_skills_per_occupation
from .utils import load_esco_titles
from .evaluate import pretty_print_metrics


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Evaluate skill coverage from a predictions.jsonl file."
    )
    parser.add_argument(
        "--predictions-path",
        type=str,
        required=True,
        help="Path to the predictions.jsonl file.",
    )
    parser.add_argument(
        "--esco-path",
        type=str,
        default="data/esco_datasets/skills_en_v1.1.1.csv",
        help="Path to the ESCO skills CSV file.",
    )
    parser.add_argument(
        "--skills-path",
        type=str,
        default="data/skills_per_occupations.csv",
        help="Path to the skills_per_occupations.csv file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save the skill coverage metrics JSON file. "
             "Defaults to the directory of the predictions file.",
    )
    args = parser.parse_args()

    # 1. Load ESCO data
    logger.info(f"Loading ESCO data from {args.esco_path}...")
    esco_ids, esco_titles = load_esco_titles(args.esco_path)
    id_to_row = {uri: i for i, uri in enumerate(esco_ids)}
    title_to_row = {title: i for i, title in enumerate(esco_titles)}

    # 2. Load skills per occupation
    logger.info(f"Loading skills data from {args.skills_path}...")
    skills_by_occupation = load_skills_per_occupation(args.skills_path)

    # 3. Load predictions
    logger.info(f"Loading predictions from {args.predictions_path}...")
    predictions = []
    with open(args.predictions_path, "r") as f:
        for line in f:
            # Check for blank lines and skip them
            if line.strip():
                try:
                    predictions.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line: {line.strip()}")
            else:
                logger.warning("Skipping blank line in predictions file.")
    
    if not predictions:
        logger.error("No valid predictions found in the file. Exiting.")
        return

    # Dynamically determine the ground truth column key
    first_pred = predictions[0]
    gold_key = next((key for key in first_pred if key.startswith("gold_")), None)
    
    if not gold_key:
        logger.error("Could not determine ground truth column (e.g., 'gold_esco_id') from predictions file. Exiting.")
        return
        
    logger.info(f"Using '{gold_key}' as the ground truth key.")

    # 4. Prepare data for metric computation
    gold_values = [p[gold_key] for p in predictions]
    predicted_titles_list = [p["predicted_esco_ids"] for p in predictions]
    
    # Determine if gold values are URIs or titles
    is_uri = gold_values[0].startswith("http") if gold_values else False
    
    if is_uri:
        logger.info("Ground truth appears to be in URI format.")
        gold_rows = [id_to_row.get(val, -1) for val in gold_values]
    else:
        logger.info("Ground truth appears to be in title format.")
        gold_rows = [title_to_row.get(val, -1) for val in gold_values]

    n_preds = len(predictions)
    # Use the number of predictions from the first entry to define the shape, assuming it's consistent
    k = len(predicted_titles_list[0]) if predicted_titles_list else 0
    I = np.full((n_preds, k), -1, dtype=int)

    # Determine if predicted values are URIs or titles
    pred_is_uri = predicted_titles_list[0][0].startswith("http") if predicted_titles_list and predicted_titles_list[0] else False

    if pred_is_uri:
        logger.info("Predicted values appear to be in URI format.")
        lookup_dict = id_to_row
    else:
        logger.info("Predicted values appear to be in title format.")
        lookup_dict = title_to_row

    for i, titles in enumerate(predicted_titles_list):
        for j, title in enumerate(titles):
            if j < k:
                I[i, j] = lookup_dict.get(title, -1)

    # Filter out predictions where gold standard was not found in ESCO data
    valid_indices = [i for i, row in enumerate(gold_rows) if row != -1]
    if len(valid_indices) < n_preds:
        logger.warning(
            f"{n_preds - len(valid_indices)} out of {n_preds} gold values were not found "
            "in the ESCO dataset and will be excluded from the evaluation."
        )

        # Filter out predictions where predicted values were not found in ESCO data
        I = I[valid_indices]
        gold_rows = [gold_rows[i] for i in valid_indices]


    # 5. Compute skill coverage
    logger.info("Computing skill coverage...")
    skill_coverage_metrics = compute_skill_coverage(
        I, gold_rows, esco_ids, skills_by_occupation
    )

    # 6. Print and save results
    logger.info("Skill Coverage Results:")
    pretty_print_metrics(skill_coverage_metrics)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.predictions_path).parent

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "skill_coverage_metrics.json"
    
    with open(output_path, "w") as f:
        json.dump(skill_coverage_metrics, f, indent=2)
    
    logger.info(f"Skill coverage metrics saved to {output_path}")


if __name__ == "__main__":
    main()
