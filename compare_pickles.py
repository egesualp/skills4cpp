import pandas as pd
import pickle
import numpy as np

file1 = '/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skill_overlap_scores_top100_isco_fused_2/test_clean_scores_skill_overlap.pkl'
file2 = '/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skill_overlap_scores_top100_isco_fused/test_clean_scores_skill_overlap.pkl'

def analyze_file(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    info = {
        'type': type(data),
    }
    
    if isinstance(data, pd.DataFrame):
        info['shape'] = data.shape
        info['columns'] = list(data.columns)
        info['index_sample'] = list(data.index[:5])
    elif isinstance(data, (list, tuple)):
        info['length'] = len(data)
    elif isinstance(data, dict):
        info['keys_count'] = len(data.keys())
        info['keys_sample'] = list(data.keys())[:5]
    
    return data, info

print(f"Analyzing {file1}...")
data1, info1 = analyze_file(file1)
print(f"Info 1: {info1}")

print(f"\nAnalyzing {file2}...")
data2, info2 = analyze_file(file2)
print(f"Info 2: {info2}")

print("\n--- Comparison ---")
if info1['type'] == info2['type']:
    if isinstance(data1, pd.DataFrame):
        print(f"Shapes match: {data1.shape == data2.shape}")
        if data1.shape == data2.shape:
            equals = data1.equals(data2)
            print(f"Exact equality: {equals}")
            if not equals:
                diff = (data1 != data2).sum().sum()
                print(f"Number of differing elements: {diff}")
                mask = (data1 != data2).any(axis=1)
                idx_diff = data1.index[mask]
                print(f"Indices with differences (first 5): {list(idx_diff[:5])}")
    elif isinstance(data1, dict):
        k1 = set(data1.keys())
        k2 = set(data2.keys())
        print(f"Shapes match (keys count): {len(k1) == len(k2)}")
        print(f"Common keys: {len(k1.intersection(k2))}")
        print(f"Keys in 1 not in 2: {len(k1 - k2)}")
        print(f"Keys in 2 not in 1: {len(k2 - k1)}")
        
        common_keys = list(k1.intersection(k2))
        if common_keys:
            diff_count = 0
            for k in common_keys:
                val1 = data1[k]
                val2 = data2[k]
                
                try:
                    if isinstance(val1, np.ndarray) or isinstance(val2, np.ndarray):
                        eq = np.array_equal(val1, val2)
                    elif isinstance(val1, pd.DataFrame) or isinstance(val2, pd.DataFrame):
                        eq = val1.equals(val2)
                    else:
                        eq = (val1 == val2)
                    
                    if not eq:
                        diff_count += 1
                except:
                    diff_count += 1
            print(f"Keys with different values: {diff_count}")
    else:
        equals = data1 == data2
        print(f"Exact equality: {equals}")

else:
    print("Types are different!")
