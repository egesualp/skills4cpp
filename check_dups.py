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

labels = [l.strip().lower() for l in d['target_labels']]
unique_labels = set(labels)
print(f"Total labels: {len(labels)}")
print(f"Unique normalized labels: {len(unique_labels)}")

if len(labels) != len(unique_labels):
    print("DUPLICATES FOUND!")
    counts = {}
    for l in labels:
        counts[l] = counts.get(l, 0) + 1
    dups = [l for l, c in counts.items() if c > 1]
    print(f"Duplicate strings: {dups[:10]}")
else:
    print("No duplicates.")
