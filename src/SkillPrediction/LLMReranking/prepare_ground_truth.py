"""
prepare_ground_truth.py - Prepare ground truth labels for evaluation

Extracts ground truth skill labels from ESCO occupation-skill mappings.

Usage:
    python -m skill_mapping.v3.prepare_ground_truth \
        --jobs_csv ./data/title_pairs_desc/decorte_master.csv \
        --occupations_csv ./data/esco_datasets/occupations_en.csv \
        --occ_skills_csv ./data/esco_datasets/occupationSkillRelations_en.csv \
        --output_json ./data/processed/ground_truth.json \
        --isco_groups 5120,2654
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd
from loguru import logger


def prepare_ground_truth(
    jobs_csv: Path,
    occupations_csv: Path,
    occ_skills_csv: Path,
    output_json: Path,
    isco_groups: Optional[List[str]] = None,
    relation_type: Optional[str] = None,  # 'essential', 'optional', or None for all
):
    """
    Prepare ground truth labels.
    
    Args:
        jobs_csv: Path to decorte_master.csv
        occupations_csv: Path to occupations_en.csv
        occ_skills_csv: Path to occupation-skill relation CSV
        output_json: Output JSON path
        isco_groups: List of ISCO groups to filter
        relation_type: Filter by relation type ('essential', 'optional', or None)
    """
    logger.info("Loading data...")
    
    # Load jobs
    jobs_df = pd.read_csv(jobs_csv)
    logger.info(f"Loaded {len(jobs_df)} jobs")
    
    # Filter by ISCO groups if specified
    if isco_groups:
        occupations_df = pd.read_csv(occupations_csv)
        occupations_df = occupations_df[['conceptUri', 'iscoGroup']]
        occupations_df.columns = ['esco_id', 'iscoGroup']
        
        # Merge to get ISCO groups for jobs
        jobs_df = jobs_df.merge(occupations_df, on='esco_id', how='left')
        
        # Filter by ISCO groups
        jobs_df = jobs_df[jobs_df['iscoGroup'].isin(isco_groups)]
        logger.info(f"Filtered to {len(jobs_df)} jobs with ISCO groups: {isco_groups}")
    
    # Load occupation-skill relations
    occ_skills_df = pd.read_csv(occ_skills_csv)
    logger.info(f"Loaded {len(occ_skills_df)} occupation-skill relations")
    
    # Filter by relation type if specified
    if relation_type:
        if 'relationType' in occ_skills_df.columns:
            occ_skills_df = occ_skills_df[occ_skills_df['relationType'] == relation_type]
            logger.info(f"Filtered to {len(occ_skills_df)} {relation_type} relations")
    
    # Build ground truth
    logger.info("Building ground truth...")
    ground_truth = {}
    
    for _, row in jobs_df.iterrows():
        job_id = str(row['job_id'])
        esco_id = row['esco_id']
        
        # Get skills for this occupation
        occ_skills = occ_skills_df[occ_skills_df['occupationUri'] == esco_id]
        
        if not occ_skills.empty:
            skill_uris = set(occ_skills['skillUri'].values)
            ground_truth[job_id] = list(skill_uris)
    
    logger.info(f"Built ground truth for {len(ground_truth)} jobs")
    
    # Save to JSON
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(ground_truth, f, indent=2)
    
    logger.success(f"Ground truth saved to {output_json}")
    
    # Print statistics
    num_skills_per_job = [len(skills) for skills in ground_truth.values()]
    logger.info(f"Average skills per job: {sum(num_skills_per_job) / len(num_skills_per_job):.2f}")
    logger.info(f"Min skills per job: {min(num_skills_per_job)}")
    logger.info(f"Max skills per job: {max(num_skills_per_job)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare ground truth labels for evaluation"
    )
    
    parser.add_argument(
        "--jobs_csv",
        type=Path,
        required=True,
        help="Path to decorte_master.csv"
    )
    parser.add_argument(
        "--occupations_csv",
        type=Path,
        required=True,
        help="Path to occupations_en.csv"
    )
    parser.add_argument(
        "--occ_skills_csv",
        type=Path,
        required=True,
        help="Path to occupation-skill relation CSV"
    )
    parser.add_argument(
        "--output_json",
        type=Path,
        required=True,
        help="Output JSON path"
    )
    parser.add_argument(
        "--isco_groups",
        type=str,
        default=None,
        help="Comma-separated list of ISCO groups to filter (e.g., '5120,2654')"
    )
    parser.add_argument(
        "--relation_type",
        type=str,
        choices=['essential', 'optional'],
        default=None,
        help="Filter by relation type (essential/optional, or all if not specified)"
    )
    
    args = parser.parse_args()
    
    # Parse ISCO groups
    isco_groups = None
    if args.isco_groups:
        isco_groups = [g.strip() for g in args.isco_groups.split(',')]
    
    prepare_ground_truth(
        jobs_csv=args.jobs_csv,
        occupations_csv=args.occupations_csv,
        occ_skills_csv=args.occ_skills_csv,
        output_json=args.output_json,
        isco_groups=isco_groups,
        relation_type=args.relation_type,
    )


if __name__ == "__main__":
    main()

