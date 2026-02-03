import pickle
import numpy as np
import sys

# Ensure numpy compatibility
if not hasattr(np, '_core'):
    sys.modules['numpy._core'] = np
    sys.modules['numpy._core.numeric'] = np
    sys.modules['numpy._core.multiarray'] = np

file1 = '/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skill_overlap_scores_top100_isco_fused_2/test_clean_scores_skill_overlap.pkl'
file2 = '/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skill_overlap_scores_top100_isco_fused/test_clean_scores_skill_overlap.pkl'

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

d1 = load_pkl(file1)
d2 = load_pkl(file2)

s1 = d1['scores']
s2 = d2['scores']

print("Sample scores from fused_2 (first 5 targets of first sample):")
print(s1[0, :5])

print("\nSample scores from fused (first 5 targets of first sample):")
print(s2[0, :5])

# Find where they differ most
diffs = np.abs(s1 - s2)
max_idx = np.unravel_index(np.argmax(diffs), s1.shape)
print(f"\nMax diff at index {max_idx}:")
print(f"  fused_2: {s1[max_idx]}")
print(f"  fused:   {s2[max_idx]}")

# Check if there's a pattern, e.g. one is just zeros?
print(f"\nStats for fused_2: non-zeros: {np.count_nonzero(s1)}, mean: {np.mean(s1):.4f}, max: {np.max(s1):.4f}")
print(f"Stats for fused:   non-zeros: {np.count_nonzero(s2)}, mean: {np.mean(s2):.4f}, max: {np.max(s2):.4f}")

# Check if maybe the histories are the same but indices are shuffled?
# Histories are identical lists, so they should correspond 1-to-1.
