import pickle
import numpy as np
import sys

files = [
    "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/job_title_desc/scores/test_scores_text.pkl",
    "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte_static/job_titles_skills_pjmath_desc_weighted_idf_logpool_adv_2/scores/test_scores_text.pkl"
]

for fpath in files:
    print(f"--- Analyzing {fpath} ---")
    try:
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
        print("Keys:", data.keys())
        if 'scores' in data:
            print("Scores shape:", data['scores'].shape)
        if 'true_target_indices' in data:
            print("Target indices shape/len:", len(data['true_target_indices']))
            print("First 5 targets:", data['true_target_indices'][:5])
        
        # Check if there are identifiable keys for samples
        # Often these might be 'job_ids', 'query_ids', or similar if not just index based
        potential_ids = [k for k in data.keys() if 'id' in k.lower() or 'job' in k.lower()]
        print("Potential ID keys:", potential_ids)
        for key in potential_ids:
            val = data[key]
            if hasattr(val, '__len__'):
                print(f"  {key} length: {len(val)}")
                print(f"  {key} first 5: {val[:5]}")
            
    except Exception as e:
        print(f"Error loading {fpath}: {e}")
    print("\n")
