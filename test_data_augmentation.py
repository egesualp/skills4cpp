#!/usr/bin/env python3
"""
Test script for src/data_augmentation/data.py
Prints structured input samples from the data augmentation module.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_augmentation.data import get_data_for_augmentation, JOB_AUGMENT_TEMPLATE, SKILL_AUGMENT_TEMPLATE
import json

if __name__ == '__main__':
    print("Testing data_augmentation/data.py...")
    print("="*80)
    
    try:
        jobs_ds, skills_ds = get_data_for_augmentation()
        
        # Print structured input samples
        print("\n" + "="*80)
        print("STRUCTURED INPUT SAMPLES")
        print("="*80)
        
        print("\n--- JOB TITLES (Query Input) ---")
        print(f"Total samples: {len(jobs_ds)}")
        print("\nSample job titles (first 3):")
        for i in range(min(3, len(jobs_ds))):
            sample = dict(jobs_ds[i])
            print(f"\n  Sample {i+1}:")
            print(f"    {json.dumps(sample, indent=6)}")
            print(f"    Template usage: {JOB_AUGMENT_TEMPLATE.format(job_title=sample.get('query_title', ''), llm_description='[LLM_GENERATED_DESCRIPTION]')}")
        
        print("\n--- SKILLS (Corpus Input) ---")
        print(f"Total samples: {len(skills_ds)}")
        print("\nSample skills (first 3):")
        for i in range(min(3, len(skills_ds))):
            sample = dict(skills_ds[i])
            print(f"\n  Sample {i+1}:")
            print(f"    {json.dumps(sample, indent=6)}")
            if 'esco_description' in sample:
                print(f"    Template usage: {SKILL_AUGMENT_TEMPLATE.format(skill_name=sample.get('corpus_name', ''), esco_description=sample.get('esco_description', ''))}")
        
        print("\n" + "="*80)
        print("Dataset Structure:")
        print("="*80)
        print(f"\nJobs Dataset features: {list(jobs_ds.features.keys())}")
        print(f"Skills Dataset features: {list(skills_ds.features.keys())}")
        
        print("\n" + "="*80)
        print("✓ Test completed successfully!")
        print("="*80)
        
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


