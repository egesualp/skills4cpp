import argparse
import os
import sys
import random
from loguru import logger
from tqdm import tqdm

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

try:
    from src.cpp.data_classes import Data
    from src.cpp.utils import SEP_TOKEN
    from src.cpp.data_loaders import load_job_skill_data_by_id
except ImportError as e:
    print(f"Error: Required modules not found. {e}")
    sys.exit(1)

def setup_logging():
    logger.remove()
    logger.add(sys.stdout, format="<green>{time}</green> | <level>{message}</level>", level="INFO")

def main():
    parser = argparse.ArgumentParser(description="Validate Job ID Mapping for Skill Overlap Scoring")
    
    # Data paths (same as skill_overlap_scoring.py)
    parser.add_argument("--data_type", type=str, default="decorte")
    parser.add_argument("--skill_scores_file", type=str, required=True,
                       help="Path to JSON file with skill scores keyed by job_id")
    parser.add_argument("--esco_skills_file", type=str, default="data/esco_datasets/skills_en.csv")
    parser.add_argument("--skill_properties_file", type=str, default="data/processed/master_datasets_2/skill_properties_map.json")
    
    # Feature configuration
    parser.add_argument("--use_text_description", action='store_true',
                       help="Include job descriptions in data loading")
    
    # Validation options
    parser.add_argument("--sample_size", type=int, default=5, help="Number of mapped samples to print")
    
    args = parser.parse_args()
    setup_logging()
    
    logger.info("=" * 80)
    logger.info("Validating Job ID Mapping")
    logger.info("=" * 80)
    logger.info(f"Configuration: {vars(args)}\n")
    
    # --- Step 1: Load Data ---
    logger.info("[1/3] Loading data to extract unique job_ids...")
    data = Data(DATA_TYPE=args.data_type, ONLY_TITLES=not args.use_text_description)
    (train_pairs, train_job_ids), (val_pairs, val_job_ids), (test_pairs, test_job_ids) = data.get_data_with_job_ids(stage='transformation_finetuning')
    
    # Collect all job IDs from data
    all_job_ids_in_data = set()
    for ids in train_job_ids + val_job_ids + test_job_ids:
        all_job_ids_in_data.update(str(jid) for jid in ids)
        
    logger.info(f"  ✓ Found {len(all_job_ids_in_data)} unique job IDs across train/val/test")
    
    # Collect train+val for loader (mimicking original script)
    train_val_job_ids_set = set()
    for ids in train_job_ids + val_job_ids:
        train_val_job_ids_set.update(str(jid) for jid in ids)
    
    # --- Step 2: Load Skill Map ---
    logger.info("[2/3] Loading skill mappings...")
    job_skill_map, _, _ = load_job_skill_data_by_id(
        skill_scores_file=args.skill_scores_file,
        esco_skills_file=args.esco_skills_file,
        skill_properties_file=args.skill_properties_file,
        pooling_strategy="mean",
        train_val_job_ids=train_val_job_ids_set,
    )
    logger.info(f"  ✓ Loaded skill map with {len(job_skill_map)} job IDs")
    
    # --- Step 3: Validate Mapping ---
    logger.info("[3/3] Validating coverage...")
    
    missing_jobs = []
    mapped_jobs = []
    mapped_with_zero_skills = []
    
    for job_id in all_job_ids_in_data:
        if job_id in job_skill_map:
            skills = job_skill_map[job_id]
            mapped_jobs.append(job_id)
            if len(skills) == 0:
                mapped_with_zero_skills.append(job_id)
        else:
            missing_jobs.append(job_id)
            
    coverage = len(mapped_jobs) / len(all_job_ids_in_data) * 100 if all_job_ids_in_data else 0.0
    
    logger.info("\n" + "=" * 40)
    logger.info(f"METRICS")
    logger.info("=" * 40)
    logger.info(f"Total Unique Jobs in Data: {len(all_job_ids_in_data)}")
    logger.info(f"Mapped Jobs:               {len(mapped_jobs)}")
    logger.info(f"Missing Jobs:              {len(missing_jobs)}")
    logger.info(f"Coverage:                  {coverage:.2f}%")
    logger.info(f"Mapped but 0 skills:       {len(mapped_with_zero_skills)}")
    logger.info("=" * 40)
    
    if missing_jobs:
        logger.info("\nSample MISSING Job IDs (first 20):")
        for jid in missing_jobs[:20]:
            logger.info(f"  - {jid}")
            
    if mapped_with_zero_skills:
        logger.info("\nSample MAPPED BUT EMPTY Job IDs (first 20):")
        for jid in mapped_with_zero_skills[:20]:
            logger.info(f"  - {jid}")
            
    logger.info("\nSample SUCCESSFUL Mappings:")
    if mapped_jobs:
        sample_mapped = random.sample(mapped_jobs, min(args.sample_size, len(mapped_jobs)))
        for jid in sample_mapped:
            skills = job_skill_map[jid]
            logger.info(f"Job ID: {jid}")
            logger.info(f"  Num Skills: {len(skills)}")
            for s in skills[:3]:
                logger.info(f"    - {s}")
            if len(skills) > 3:
                logger.info("    ...")
            logger.info("-" * 20)

if __name__ == "__main__":
    main()

