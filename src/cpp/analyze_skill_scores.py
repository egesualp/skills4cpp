import json
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from tqdm import tqdm
import os
import sys
from loguru import logger

# Add the project root to sys.path to allow imports from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from src.cpp.data_classes import Data
    from src.cpp.utils import SEP_TOKEN
except ImportError as e:
    logger.error(f"Error: Required modules not found. {e}")
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.cpp.data_classes import Data
    from src.cpp.utils import SEP_TOKEN

def parse_args():
    parser = argparse.ArgumentParser(description="Analyze skill scores per job and career path.")
    parser.add_argument("--skill_scores_file", type=str, required=True,
                        help="Path to JSON file with skill scores keyed by job_id (e.g., fused_predictions.json)")
    parser.add_argument("--data_type", type=str, default="decorte",
                        help="Data type for Data class (default: decorte)")
    parser.add_argument("--thresholds", type=float, nargs="+", 
                        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                        help="Thresholds to apply to skill scores.")
    parser.add_argument("--output_path", type=str, default="skill_score_analysis.csv",
                        help="Path to save the summary statistics.")
    return parser.parse_args()

def load_skill_scores(file_path: str) -> Dict[str, List[Tuple[str, float]]]:
    """Load skill scores from JSON file."""
    logger.info(f"Loading skill scores from {file_path}...")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract scores dictionary
    if 'scores' in data:
        scores_dict = data['scores']
    else:
        scores_dict = data
    
    formatted_scores = {}
    for job_id, skill_list in scores_dict.items():
        skills = []
        for skill_info in skill_list:
            if isinstance(skill_info, (list, tuple)):
                skill_uri = skill_info[0]
                score = float(skill_info[1]) if len(skill_info) > 1 else 1.0
            else:
                skill_uri = skill_info.get('skill_uri') or skill_info.get('skillUri')
                score = float(skill_info.get('score', 1.0))
            skills.append((skill_uri, score))
        formatted_scores[str(job_id)] = skills
    
    logger.info(f"Loaded {len(formatted_scores)} unique job_ids.")
    return formatted_scores

def analyze_scores():
    args = parse_args()
    
    # 1. Load Data
    logger.info(f"Initializing Data for type: {args.data_type}")
    data = Data(DATA_TYPE=args.data_type, ONLY_TITLES=True, consider_subspans=True)
    
    # Get all splits and combine them for full analysis
    splits = data.get_data_with_job_ids(stage='transformation_finetuning')
    all_career_paths_job_ids = []
    for split in splits:
        # split is (pairs, job_ids_list)
        all_career_paths_job_ids.extend(split[1])
    
    logger.info(f"Found {len(all_career_paths_job_ids)} career paths in total.")
    
    # 2. Load Skill Scores
    job_skill_scores = load_skill_scores(args.skill_scores_file)
    
    # 3. Analyze
    all_scores = []
    job_results = []
    path_results = []
    
    thresholds = sorted(args.thresholds)
    
    logger.info("Analyzing skill scores...")
    for path_job_ids in tqdm(all_career_paths_job_ids, desc="Processing paths"):
        path_unique_skills = {} # skillUri -> max_score in this path
        path_counts = {t: 0 for t in thresholds}
        
        jobs_in_path_count = 0
        for job_id in path_job_ids:
            job_id_str = str(job_id)
            if job_id_str in job_skill_scores:
                jobs_in_path_count += 1
                skills = job_skill_scores[job_id_str]
                
                scores = [s[1] for s in skills]
                all_scores.extend(scores)
                
                job_counts = {t: 0 for t in thresholds}
                for _, score in skills:
                    for t in thresholds:
                        if score >= t:
                            job_counts[t] += 1
                
                job_results.append({
                    'job_id': job_id_str,
                    'num_skills_total': len(skills),
                    **{f'cnt_t_{t:.1f}': job_counts[t] for t in thresholds}
                })
                
                # For path analysis (union of skills)
                for uri, score in skills:
                    if uri not in path_unique_skills or score > path_unique_skills[uri]:
                        path_unique_skills[uri] = score
        
        if path_unique_skills:
            for score in path_unique_skills.values():
                for t in thresholds:
                    if score >= t:
                        path_counts[t] += 1
            
            path_results.append({
                'num_jobs': jobs_in_path_count,
                'num_skills_total': len(path_unique_skills),
                **{f'cnt_t_{t:.1f}': path_counts[t] for t in thresholds}
            })

    # 4. Global Stats
    all_scores = np.array(all_scores)
    logger.info("\n--- Global Descriptive Statistics for Skill Scores ---")
    stats = {
        'count': len(all_scores),
        'mean': np.mean(all_scores),
        'std': np.std(all_scores),
        'min': np.min(all_scores),
        'q1': np.percentile(all_scores, 25),
        'median': np.median(all_scores),
        'q3': np.percentile(all_scores, 75),
        'max': np.max(all_scores)
    }
    for k, v in stats.items():
        logger.info(f"{k:8s}: {v:.4f}")

    # 5. Threshold Analysis
    df_jobs = pd.DataFrame(job_results)
    df_paths = pd.DataFrame(path_results)
    
    summary = []
    
    logger.info("\n--- Threshold Impact Analysis ---")
    logger.info(f"{'Threshold':>10} | {'Avg Skills/Job':>15} | {'% Remaining':>12} | {'Avg Skills/Path':>15} | {'% Remaining':>12}")
    logger.info("-" * 75)
    
    avg_total_job = df_jobs['num_skills_total'].mean()
    avg_total_path = df_paths['num_skills_total'].mean()
    
    for t in thresholds:
        avg_job = df_jobs[f'cnt_t_{t:.1f}'].mean()
        pct_job = (avg_job / avg_total_job * 100) if avg_total_job > 0 else 0
        
        avg_path = df_paths[f'cnt_t_{t:.1f}'].mean()
        pct_path = (avg_path / avg_total_path * 100) if avg_total_path > 0 else 0
        
        logger.info(f"{t:10.1f} | {avg_job:15.2f} | {pct_job:11.1f}% | {avg_path:15.2f} | {pct_path:11.1f}%")
        
        summary.append({
            'threshold': t,
            'avg_skills_per_job': avg_job,
            'pct_remaining_job': pct_job,
            'avg_skills_per_path': avg_path,
            'pct_remaining_path': pct_path
        })
    
    # Save summary
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(args.output_path, index=False)
    logger.info(f"\nSummary saved to {args.output_path}")

if __name__ == "__main__":
    analyze_scores()
