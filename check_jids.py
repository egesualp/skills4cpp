import pickle
import numpy as np
import sys

# Ensure numpy compatibility
if not hasattr(np, '_core'):
    sys.modules['numpy._core'] = np
    sys.modules['numpy._core.numeric'] = np
    sys.modules['numpy._core.multiarray'] = np

file1 = '/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skill_overlap_scores_top100_isco_fused_2/test_clean_scores_skill_overlap.pkl'

with open(file1, 'rb') as f:
    d = pickle.load(f)

print("First 5 job_id lists in fused_2:")
for i, jids in enumerate(d['job_ids'][:5]):
    print(f"Sample {i}: {jids}")

print("\nLengths of first 5 job_id lists:")
print([len(x) for x in d['job_ids'][:5]])

# Check if there are any samples with empty job_ids
empty_job_ids = sum(1 for x in d['job_ids'] if not x)
print(f"\nSamples with empty job_ids: {empty_job_ids}/{len(d['job_ids'])}")
