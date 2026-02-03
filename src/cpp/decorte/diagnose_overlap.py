import sys
import os
import random
from typing import Dict, List
import pandas as pd
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.cpp.data_classes import Data
from src.cpp.data_loaders import load_job_and_skill_data
from src.cpp.decorte.skill_overlap_scoring import (
    extract_job_titles_from_history, 
    get_career_skills, 
    get_job_skills
)

def main():
    # 1. Load Data
    print("Loading data...")
    # Use same settings as your run
    data = Data(DATA_TYPE='karrierewege_100k', ONLY_TITLES=True)
    _, _, test_pairs = data.get_data(stage='transformation_finetuning')
    
    # 2. Load Skill Map
    print("Loading skill map...")
    master_skill_file = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/karrierewege_esco_100k_esco_ground_truth/job_title_skills_master.csv"
    job_skill_map, _, _ = load_job_and_skill_data(
        master_skill_file=master_skill_file,
        esco_skills_file="data/esco_datasets/skills_en.csv",
        skill_properties_file="data/processed/master_datasets_2/skill_properties_map.json",
        pooling_strategy="mean"
    )
    
    print(f"\nLoaded {len(job_skill_map)} jobs in skill map.")
    sample_keys = list(job_skill_map.keys())[:5]
    print(f"Sample skill map keys: {sample_keys}")
    
    # 3. Diagnose Overlap
    print("\nDiagnosing 100 random samples...")
    random.seed(42)
    samples = random.sample(test_pairs, 100)
    
    zero_career_skills = 0
    zero_target_skills = 0
    zero_overlap_with_true = 0
    
    print("\n--- Detailed Checks ---")
    for i, (history, target) in enumerate(samples[:5]): # Show detail for first 5
        print(f"\nSample {i+1}:")
        
        # Check History
        titles = extract_job_titles_from_history(history)
        career_skills = get_career_skills(history, job_skill_map)
        
        print(f"  History Titles: {titles}")
        found_titles = [t for t in titles if t in job_skill_map]
        missing_titles = [t for t in titles if t not in job_skill_map]
        
        if missing_titles:
            print(f"  ❌ Missing Titles in Map: {missing_titles}")
        else:
            print(f"  ✓ All titles found in map")
            
        print(f"  Career Skills Count: {len(career_skills)}")
        
        # Check Target
        target_skills = get_job_skills(target, job_skill_map)
        
        # Manually check clean target
        target_clean = target.strip().lower()
        if "esco role: " in target:
             target_clean = target.split("esco role: ")[1].split("\n")[0].strip().lower()
        elif "role: " in target:
             target_clean = target.split("role: ")[1].split("\n")[0].strip().lower()
             
        print(f"  Target Raw: '{target}'")
        print(f"  Target Clean: '{target_clean}'")
        
        if target_clean in job_skill_map:
            print(f"  ✓ Target found in map (Skills: {len(job_skill_map[target_clean])})")
        else:
            print(f"  ❌ Target NOT found in map")
            
        # Overlap
        overlap = len(career_skills & target_skills)
        print(f"  Overlap with True Target: {overlap}")
        
    # Aggregate Stats
    for history, target in samples:
        career_skills = get_career_skills(history, job_skill_map)
        target_skills = get_job_skills(target, job_skill_map)
        overlap = len(career_skills & target_skills)
        
        if len(career_skills) == 0:
            zero_career_skills += 1
        if len(target_skills) == 0:
            zero_target_skills += 1
        if overlap == 0:
            zero_overlap_with_true += 1
            
    print(f"\n--- Aggregate Results on 100 samples ---")
    print(f"  Samples with 0 career skills: {zero_career_skills}")
    print(f"  Samples with 0 target skills: {zero_target_skills}")
    print(f"  Samples with 0 overlap with TRUE target: {zero_overlap_with_true}")

if __name__ == "__main__":
    main()


