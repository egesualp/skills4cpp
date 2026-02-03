import pickle
import numpy as np
from scipy import stats
import sys

def load_scores(filepath):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"[Load] {filepath}")
    return data

def align_all(d1, d2):
    """Aligns both target columns and sample rows."""
    # Column alignment
    labels_ref = d1['target_labels']
    fix_map = {lbl: i for i, lbl in enumerate(d2['target_labels'])}
    reorder_idx = [fix_map[lbl] for lbl in labels_ref]
    d2['scores'] = d2['scores'][:, reorder_idx]
    
    # Row alignment
    def mk_hash(ids): return [tuple(p) if isinstance(p, list) else (p,) for p in ids]
    keys1, keys2 = mk_hash(d1['job_ids']), mk_hash(d2['job_ids'])
    b_map = {k: i for i, k in enumerate(keys2)}
    idx1 = [i for i, k in enumerate(keys1) if k in b_map]
    idx2 = [b_map[keys1[i]] for i in idx1]
    
    def slice_data(d, idx):
        return {
            'scores': d['scores'][idx],
            'true_idx': np.array(d['true_target_indices'])[idx],
            'labels': d['target_labels'],
            'histories': [d['histories'][i] for i in idx] if 'histories' in d else None
        }
    
    return slice_data(d1, idx1), slice_data(d2, idx2)

def perform_detailed_analysis(a, b, name_a, name_b):
    scores_a, scores_b = a['scores'], b['scores']
    true_idx = a['true_idx']
    labels = a['labels']
    n = scores_a.shape[0]

    # 1. Rank & Margin Calculation
    def get_metrics(scores, targets):
        sorted_indices = np.argsort(-scores, axis=1)
        ranks = np.array([np.where(sorted_indices[i] == targets[i])[0][0] + 1 for i in range(n)])
        # Margin: Difference between Top 1 and Top 2 scores
        sorted_scores = np.sort(scores, axis=1)[:, ::-1]
        margins = sorted_scores[:, 0] - sorted_scores[:, 1]
        return ranks, sorted_indices, margins

    ranks_a, sorted_idx_a, margins_a = get_metrics(scores_a, true_idx)
    ranks_b, sorted_idx_b, margins_b = get_metrics(scores_b, true_idx)

    # 2. Decoy Identification (The jobs at Rank 1 when model is wrong)
    def get_decoys(ranks, sorted_idx, targets):
        decoys = []
        for i in range(n):
            if ranks[i] > 1:
                decoys.append(labels[sorted_idx[i, 0]])
            else:
                decoys.append(None) # Correct prediction
        return decoys

    decoys_a = get_decoys(ranks_a, sorted_idx_a, true_idx)
    decoys_b = get_decoys(ranks_b, sorted_idx_b, true_idx)

    print(f"\n{'='*80}\nDETAILED DECOY & MARGIN ANALYSIS\n{'='*80}")
    
    # --- INSIGHT 1: Margin Robustness ---
    # Are we more 'confident' when we are right?
    correct_a = ranks_a == 1
    correct_b = ranks_b == 1
    
    print(f"Average Margin (Confidence Gap):")
    print(f"  {name_a:<12} | Correct samples: {margins_a[correct_a].mean():.4f} | Wrong samples: {margins_a[~correct_a].mean():.4f}")
    print(f"  {name_b:<12} | Correct samples: {margins_b[correct_b].mean():.4f} | Wrong samples: {margins_b[~correct_b].mean():.4f}")

    # --- INSIGHT 2: Decoy Overlap ---
    # Do both models fail on the same distractor?
    shared_failures = (~correct_a & ~correct_b)
    same_decoy = sum(1 for i in range(n) if shared_failures[i] and decoys_a[i] == decoys_b[i])
    
    print(f"\nDecoy Analysis (on {shared_failures.sum()} shared failures):")
    print(f"  Samples with the EXACT same Decoy at Rank 1: {same_decoy}")
    print(f"  Percentage of Redundant Failures: {100*same_decoy/shared_failures.sum():.1f}%")

    # --- INSIGHT 3: Surgical Strikes vs. Cleaning the Tail ---
    # Did we move the target closer even if we didn't hit Rank 1?
    improved_ranks = ranks_a > ranks_b
    significant_improvement = (ranks_a - ranks_b) > 10
    
    print(f"\nPositional Movement:")
    print(f"  Samples where {name_b} improved the Rank: {improved_ranks.sum()}")
    print(f"  Samples with 'Significant' improvement (>10 spots): {significant_improvement.sum()}")

    # --- INSIGHT 4: The "Why it failed" Table ---
    print(f"\n{'='*80}\nSAMPLES WHERE {name_b} IMPROVED RANK BUT STILL FAILED R@1\n{'='*80}")
    print(f"{'True Target':<30} | {'Rank A':<8} -> {'Rank B':<8} | {'Decoy B (Blocking Job)'}")
    
    # Filter for samples where rank improved but still not rank 1
    failure_improvement = (ranks_b > 1) & (ranks_b < ranks_a)
    fail_indices = np.where(failure_improvement)[0]
    # Sort by how much they improved to find the most interesting cases
    fail_indices = fail_indices[np.argsort(ranks_a[fail_indices] - ranks_b[fail_indices])[::-1][:10]]

    for i in fail_indices:
        target_lbl = labels[true_idx[i]]
        print(f"{target_lbl[:30]:<30} | {ranks_a[i]:<8} -> {ranks_b[i]:<8} | {decoys_b[i][:40]}")

if __name__ == '__main__':
    if len(sys.argv) < 3: sys.exit("Usage: python script.py text.pkl skill.pkl")
    d1, d2 = align_all(load_scores(sys.argv[1]), load_scores(sys.argv[2]))
    perform_detailed_analysis(d1, d2, "Text-Only", "Text-Skill")