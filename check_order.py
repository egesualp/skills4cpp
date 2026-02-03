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

t1 = d1['target_labels']
t2 = d2['target_labels']

print(f"Length T1: {len(t1)}, T2: {len(t2)}")
print(f"Target labels identical in order: {t1 == t2}")

if t1 != t2:
    s1 = set(t1)
    s2 = set(t2)
    print(f"Sets identical: {s1 == s2}")
    if s1 != s2:
        print(f"Diff 1-2: {s1-s2}")
        print(f"Diff 2-1: {s2-s1}")
    else:
        # Check first different index
        for i in range(min(len(t1), len(t2))):
            if t1[i] != t2[i]:
                print(f"First diff at {i}: '{t1[i]}' vs '{t2[i]}'")
                break

# Check True Target Indices
i1 = d1['true_target_indices']
i2 = d2['true_target_indices']
print(f"True target indices identical: {i1 == i2}")
if i1 != i2:
    print(f"Index diff count: {sum(1 for a, b in zip(i1, i2) if a != b)}")
    for i in range(len(i1)):
        if i1[i] != i2[i]:
            print(f"First index diff at {i}: {i1[i]} vs {i2[i]}")
            print(f"  Target label in d1 at i1[i]: {t1[i1[i]] if i1[i] >= 0 else 'N/A'}")
            print(f"  Target label in d2 at i2[i]: {t2[i2[i]] if i2[i] >= 0 else 'N/A'}")
            print(f"  Actual target in data: {d1['true_targets'][i]}")
            break
