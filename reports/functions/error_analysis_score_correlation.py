import pickle
import numpy as np
from scipy import stats
import sys

def load_scores(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"\n[Load] {filepath}")
    return data

def align_target_space(data_ref, data_to_fix):
    """
    Reorders the columns of data_to_fix['scores'] and updates its 
    true_target_indices to match the target order in data_ref.
    """
    labels_ref = data_ref['target_labels']
    labels_fix = data_to_fix['target_labels']
    
    # Map label string to its index in the 'to_fix' dataset
    fix_label_to_idx = {label: i for i, label in enumerate(labels_fix)}
    
    # Create the column reordering index
    # (Which index in 'fix' corresponds to index 'i' in 'ref'?)
    try:
        reorder_idx = [fix_label_to_idx[label] for label in labels_ref]
    except KeyError as e:
        sys.exit(f"Error: Target label {e} found in reference but not in the second file.")

    # 1. Reorder scores columns
    data_to_fix['scores'] = data_to_fix['scores'][:, reorder_idx]
    
    # 2. Update target labels to match ref
    data_to_fix['target_labels'] = labels_ref
    
    # 3. Update true_target_indices
    # Since we reordered the columns to match ref, 
    # the true_target_indices should now be identical to ref's indices 
    # *for the same samples*. We'll handle sample alignment next.
    
    print(f"[Align] Target space (columns) aligned across {len(labels_ref)} labels.")
    return data_to_fix

def align_samples(data_a, data_b):
    """Aligns rows based on career path job_ids."""
    def make_hashable(ids_list):
        return [tuple(path) if isinstance(path, list) else (path,) for path in ids_list]

    keys_a = make_hashable(data_a['job_ids'])
    keys_b = make_hashable(data_b['job_ids'])
    
    b_map = {key: i for i, key in enumerate(keys_b)}
    valid_a, valid_b = [], []
    
    for i, key in enumerate(keys_a):
        if key in b_map:
            valid_a.append(i)
            valid_b.append(b_map[key])
            
    def slice_it(data, idx):
        return {
            'scores': data['scores'][idx],
            'true_idx': np.array(data['true_target_indices'])[idx],
            'histories': [data['histories'][i] for i in idx] if 'histories' in data else None,
            'true_labels': [data['true_targets'][i] for i in idx] if 'true_targets' in data else None
        }

    return slice_it(data_a, valid_a), slice_it(data_b, valid_b)

def analyze(d1, d2, name1, name2):
    # Step 1: Align columns
    d2 = align_target_space(d1, d2)
    
    # Step 2: Align rows
    a, b = align_samples(d1, d2)
    
    scores_a, scores_b = a['scores'], b['scores']
    true_idx = a['true_idx'] # Now safe to use A's indices for both
    
    n = scores_a.shape[0]
    
    # Metrics
    spearman = np.array([stats.spearmanr(scores_a[i], scores_b[i])[0] for i in range(n)])
    
    def get_ranks(scores, targets):
        sorted_indices = np.argsort(-scores, axis=1)
        return np.array([np.where(sorted_indices[i] == targets[i])[0][0] + 1 for i in range(n)])

    ranks_a = get_ranks(scores_a, true_idx)
    ranks_b = get_ranks(scores_b, true_idx)
    
    acc_a = (ranks_a == 1).mean()
    acc_b = (ranks_b == 1).mean()
    
    print(f"\n{'='*70}\nRESULTS: {name1} vs {name2}\n{'='*70}")
    print(f"Mean Spearman: {np.nanmean(spearman):.4f}")
    print(f"Accuracy (R@1) {name1}: {100*acc_a:.2f}%")
    print(f"Accuracy (R@1) {name2}: {100*acc_b:.2f}%")
    
    # Improved Analysis
    rank_diff = ranks_a - ranks_b
    top_improved = np.argsort(-rank_diff)[:10]

    print(f"\nTOP 10 IMPROVEMENTS (Rank change)")
    for idx in top_improved:
        target = a['true_labels'][idx] if a['true_labels'] else f"Idx {true_idx[idx]}"
        print(f"Rank {ranks_a[idx]:>4} -> {ranks_b[idx]:>4} | Target: {target}")

if __name__ == '__main__':
    if len(sys.argv) < 3: sys.exit("Usage: python script.py text.pkl skill.pkl")
    data_text = load_scores(sys.argv[1])
    data_skill = load_scores(sys.argv[2])
    analyze(data_text, data_skill, "Text-Only", "Text-Skill")