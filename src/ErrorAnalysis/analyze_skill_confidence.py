import ijson
import numpy as np
import argparse
from typing import Dict, List, Tuple
from tqdm import tqdm
import os

def analyze_file(file_path: str, top_k_values: List[int] = [10, 50, 100]) -> Dict[int, List[float]]:
    """
    Analyzes a single JSON file and computes mean confidence scores for each job at different top-k cutoffs.
    
    Args:
        file_path: Path to the JSON file.
        top_k_values: List of k values to calculate top-k statistics for.
        
    Returns:
        A dictionary where keys are k values and values are lists of mean scores (one per job).
    """
    print(f"Analyzing {file_path}...")
    
    # Initialize storage for each k
    # We store the list of mean scores across all jobs
    results = {k: [] for k in top_k_values}
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return results

    try:
        if file_path.endswith('.jsonl'):
            import json
            with open(file_path, 'r') as f:
                for line in tqdm(f, desc="Processing jobs"):
                    try:
                        item = json.loads(line)
                        # Try 'predictions' first (common in jsonl output), then 'skills'
                        skills = item.get('predictions', item.get('skills', []))
                    except json.JSONDecodeError:
                        continue
                        
                    if not skills:
                        continue
                        
                    # Extract and sort scores
                    scores = [float(s.get('score', 0.0)) for s in skills]
                    scores.sort(reverse=True)
                    
                    for k in top_k_values:
                        top_k_scores = scores[:k]
                        if not top_k_scores:
                            mean_score = 0.0
                        else:
                            mean_score = sum(top_k_scores) / len(top_k_scores)
                        results[k].append(mean_score)
                        
        else:
            with open(file_path, 'rb') as f:
                # ijson.kvitems returns (key, value) pairs from the root object
                # The file structure is assumed to be {"job_id": [{"score": 0.9, ...}, ...], ...}
                parser = ijson.kvitems(f, "")
                
                for job_id, skills in tqdm(parser, desc="Processing jobs"):
                    if not skills:
                        continue
                    
                    # Extract scores. 
                    scores = [float(s.get('score', 0.0)) for s in skills]
                    scores.sort(reverse=True)
                    
                    for k in top_k_values:
                        # Take top k scores
                        top_k_scores = scores[:k]
                        
                        if not top_k_scores:
                            mean_score = 0.0
                        else:
                            mean_score = sum(top_k_scores) / len(top_k_scores)
                            
                        results[k].append(mean_score)
                    
    except Exception as e:
        print(f"Error processing file: {e}")
        import traceback
        traceback.print_exc()
        
    return results

def compute_statistics(data: List[float]) -> Dict[str, float]:
    """Computes descriptive statistics for a list of values."""
    if not data:
        return {
            "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, 
            "p25": 0.0, "median": 0.0, "p75": 0.0
        }
    
    arr = np.array(data)
    return {
        "mean": np.mean(arr),
        "std": np.std(arr),
        "min": np.min(arr),
        "max": np.max(arr),
        "p25": np.percentile(arr, 25),
        "median": np.median(arr),
        "p75": np.percentile(arr, 75)
    }

def print_comparison(results1: Dict[int, List[float]], name1: str,
                     results2: Dict[int, List[float]], name2: str):
    """Prints a comparison table for the two datasets."""
    
    print("\n" + "="*80)
    print(f"{'STATISTIC':<15} | {'DATASET':<20} | {'TOP-10':<12} | {'TOP-50':<12} | {'TOP-100':<12}")
    print("="*80)
    
    stats_to_show = ["mean", "std", "median", "p25", "p75", "min", "max"]
    
    # Calculate all stats first
    k_values = sorted(results1.keys())
    
    stats1 = {k: compute_statistics(results1[k]) for k in k_values}
    stats2 = {k: compute_statistics(results2[k]) for k in k_values}
    
    for stat in stats_to_show:
        # Row for Dataset 1
        row1 = f"{stat:<15} | {name1:<20}"
        for k in k_values:
            val = stats1[k][stat]
            row1 += f" | {val:<12.4f}"
        print(row1)
        
        # Row for Dataset 2
        row2 = f"{'':<15} | {name2:<20}"
        for k in k_values:
            val = stats2[k][stat]
            row2 += f" | {val:<12.4f}"
        print(row2)
        print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description="Compare skill confidence score statistics between two datasets.")
    parser.add_argument("--file1", required=True, help="Path to the first JSON file (e.g., kw_cp_jobbert)")
    parser.add_argument("--name1", default="Dataset 1", help="Display name for the first dataset")
    parser.add_argument("--file2", required=True, help="Path to the second JSON file (e.g., decorte)")
    parser.add_argument("--name2", default="Dataset 2", help="Display name for the second dataset")
    
    args = parser.parse_args()
    
    print(f"Starting comparison analysis...")
    print(f"Dataset 1: {args.name1} ({args.file1})")
    print(f"Dataset 2: {args.name2} ({args.file2})")
    print("-" * 40)
    
    results1 = analyze_file(args.file1)
    results2 = analyze_file(args.file2)
    
    print_comparison(results1, args.name1, results2, args.name2)

if __name__ == "__main__":
    main()
