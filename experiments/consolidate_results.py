#!/usr/bin/env python3
"""
Experiment Results Consolidation Script

This script scans all experiment results in the results/ directory,
consolidates metrics from metrics.json and skill_coverage_metrics.json files,
and outputs the data to both CSV and JSON formats.

Usage:
    python experiments/consolidate_results.py

Output:
    - consolidated_results.csv: CSV format for Excel/Google Sheets
    - consolidated_results.json: JSON format for the HTML dashboard
"""

import json
import os
from pathlib import Path
import csv
from typing import Dict, List, Any


def load_json_file(file_path: Path) -> Dict[str, Any]:
    """Load a JSON file and return its contents."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def extract_dataset_name(data_paths: List[str]) -> str:
    """Extract a readable dataset name from data paths."""
    if not data_paths:
        return "unknown"
    
    # Get the first data path and extract meaningful parts
    first_path = data_paths[0]
    if "decorte" in first_path:
        return "decorte"
    elif "talent_clef" in first_path:
        return "talent_clef"
    elif "kw_cp" in first_path:
        return "kw_cp"
    elif "kw_occ" in first_path:
        return "kw_occ"
    else:
        return Path(first_path).stem


def consolidate_results(results_dir: Path) -> List[Dict[str, Any]]:
    """
    Scan all experiment directories and consolidate results.
    
    Args:
        results_dir: Path to the results directory
        
    Returns:
        List of dictionaries containing consolidated experiment data
    """
    consolidated = []
    
    # Iterate through all subdirectories
    for exp_dir in sorted(results_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
            
        exp_name = exp_dir.name
        metrics_file = exp_dir / "metrics.json"
        skill_coverage_file = exp_dir / "skill_coverage_metrics.json"
        
        # Skip if no metrics.json exists
        if not metrics_file.exists():
            print(f"Warning: {exp_name} has no metrics.json, skipping...")
            continue
        
        # Load metrics
        metrics = load_json_file(metrics_file)
        skill_coverage = load_json_file(skill_coverage_file)
        
        # Build consolidated entry
        entry = {
            "experiment_name": exp_name,
        }
        
        # Add run details if available
        run_details = metrics.get("run_details", {})
        entry["model_id"] = run_details.get("model_id", "unknown")
        entry["dataset"] = extract_dataset_name(run_details.get("data_path", []))
        
        # Add main metrics
        entry["recall@1"] = metrics.get("recall@1")
        entry["recall@5"] = metrics.get("recall@5")
        entry["recall@10"] = metrics.get("recall@10")
        entry["map@10"] = metrics.get("map@10")
        entry["mrr@10"] = metrics.get("mrr@10")
        entry["map_full"] = metrics.get("map_full")
        entry["mrr_full"] = metrics.get("mrr_full")
        entry["coverage"] = metrics.get("coverage")
        entry["N_eval"] = metrics.get("N_eval")
        entry["encode_ms_per_query"] = metrics.get("encode_ms_per_query")
        
        # Add skill coverage metrics from both sources
        # Check metrics.json first (newer format)
        for k in ["skill_coverage@1", "skill_coverage@3", "skill_coverage@5", "skill_coverage@10"]:
            if k in metrics:
                entry[k] = metrics[k]
            elif k in skill_coverage:
                entry[k] = skill_coverage[k]
            else:
                entry[k] = None
        
        # Add other details
        entry["topk"] = run_details.get("topk")
        entry["use_faiss"] = run_details.get("use_faiss")
        entry["normalize_embeddings"] = run_details.get("normalize_embeddings")
        
        consolidated.append(entry)
        print(f"✓ Processed {exp_name}")
    
    return consolidated


def write_csv(data: List[Dict[str, Any]], output_file: Path):
    """Write consolidated data to CSV file."""
    if not data:
        print("No data to write to CSV")
        return
    
    # Get all unique keys across all entries
    fieldnames = []
    seen = set()
    for entry in data:
        for key in entry.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"\n✓ CSV written to {output_file}")


def write_json(data: List[Dict[str, Any]], output_file: Path):
    """Write consolidated data to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ JSON written to {output_file}")


def main():
    """Main function to consolidate results."""
    # Get script directory and navigate to results
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"
    
    if not results_dir.exists():
        print(f"Error: Results directory not found at {results_dir}")
        return
    
    print(f"Scanning experiments in {results_dir}...\n")
    
    # Consolidate results
    consolidated = consolidate_results(results_dir)
    
    if not consolidated:
        print("\nNo experiments found with metrics.json files.")
        return
    
    print(f"\n{len(consolidated)} experiments consolidated.")
    
    # Write outputs
    csv_output = script_dir / "consolidated_results.csv"
    json_output = script_dir / "consolidated_results.json"
    
    write_csv(consolidated, csv_output)
    write_json(consolidated, json_output)
    
    print(f"\n✓ Consolidation complete! Open dashboard.html to view results.")


if __name__ == "__main__":
    main()





