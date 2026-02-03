import pandas as pd
import pickle
import numpy as np
import sys

# Ensure numpy compatibility if needed, though thesis env seems fine
if not hasattr(np, '_core'):
    sys.modules['numpy._core'] = np
    sys.modules['numpy._core.numeric'] = np
    sys.modules['numpy._core.multiarray'] = np

file1 = '/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skill_overlap_scores_top100_isco_fused_2/test_clean_scores_skill_overlap.pkl'
file2 = '/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skill_overlap_scores_top100_isco_fused/test_clean_scores_skill_overlap.pkl'

def analyze_file(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

data1 = analyze_file(file1)
data2 = analyze_file(file2)

keys1 = set(data1.keys())
keys2 = set(data2.keys())

print(f"Keys in fused_2: {sorted(list(keys1))}")
print(f"Keys in fused:   {sorted(list(keys2))}")

diff_keys = keys1.symmetric_difference(keys2)
print(f"\nKeys present in only one: {diff_keys}")

common_keys = keys1.intersection(keys2)
print(f"\nComparing common keys: {sorted(list(common_keys))}")

for k in sorted(list(common_keys)):
    val1 = data1[k]
    val2 = data2[k]
    
    print(f"\nChecking key: '{k}'")
    
    # Handle numpy arrays (scores)
    if isinstance(val1, np.ndarray):
        print(f"  Type: numpy array, Shape: {val1.shape}")
        if val1.shape != val2.shape:
            print(f"  SHAPES DIFFER: {val1.shape} vs {val2.shape}")
        else:
            if np.array_equal(val1, val2):
                print("  Values are identical.")
            else:
                # Check how much they differ
                # Check for NaNs
                nan1 = np.isnan(val1).sum()
                nan2 = np.isnan(val2).sum()
                print(f"  NaNs: {nan1} vs {nan2}")
                
                diff = val1 != val2
                num_diff = np.sum(diff)
                print(f"  PERCENT DIFFERENT: {100 * num_diff / val1.size:.4f}% ({num_diff}/{val1.size})")
                
                # Check max difference
                max_abs_diff = np.max(np.abs(val1 - val2))
                print(f"  MAX ABS DIFF: {max_abs_diff}")
                
    # Handle lists (histories, true_targets, etc.)
    elif isinstance(val1, list):
        print(f"  Type: list, Length: {len(val1)}")
        if len(val1) != len(val2):
            print(f"  LENGTHS DIFFER: {len(val1)} vs {len(val2)}")
        else:
            matches = [v1 == v2 for v1, v2 in zip(val1, val2)]
            num_matches = sum(matches)
            if num_matches == len(val1):
                print("  Values are identical.")
            else:
                print(f"  PERCENT DIFFERENT: {100 * (len(val1) - num_matches) / len(val1):.4f}% ({len(val1) - num_matches}/{len(val1)})")
                # Show first diff
                for i, (v1, v2) in enumerate(zip(val1, val2)):
                    if v1 != v2:
                        print(f"  First diff at index {i}:")
                        print(f"    1: {v1}")
                        print(f"    2: {v2}")
                        break
    else:
        print(f"  Type: {type(val1)}")
        if val1 == val2:
            print("  Values are identical.")
        else:
            print("  VALUES DIFFER.")
