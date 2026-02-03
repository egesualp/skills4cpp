import pickle
import numpy as np
import sys

def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"[Internal] Loaded {path} with {len(data['job_ids'])} samples.")
    return data

def robust_alignment(d1, d2):
    """
    Standardizes everything based on File 1's target order and unique job_ids.
    """
    # 1. Standardize Target Labels (Columns)
    # We use File 1's labels as the master order
    master_labels = [str(lbl).strip() for lbl in d1['target_labels']]
    d1_label_to_idx = {lbl: i for i, lbl in enumerate(master_labels)}
    
    d2_labels_orig = [str(lbl).strip() for lbl in d2['target_labels']]
    d2_label_to_orig_idx = {lbl: i for i, lbl in enumerate(d2_labels_orig)}
    
    # Create a mapping for d2 columns to match master_labels
    try:
        reorder_idx = [d2_label_to_orig_idx[lbl] for lbl in master_labels]
    except KeyError as e:
        sys.exit(f"Error: Label {e} exists in File 1 but not in File 2.")
    
    # 2. Re-map Scores for File 2
    d2_scores_aligned = d2['scores'][:, reorder_idx]
    
    # 3. Align Rows (Samples) via job_ids
    def get_keys(ids): return [tuple(p) if isinstance(p, list) else (p,) for p in ids]
    keys1, keys2 = get_keys(d1['job_ids']), get_keys(d2['job_ids'])
    
    k2_map = {k: i for i, k in enumerate(keys2)}
    
    # We only keep samples found in both
    common_indices_1 = [i for i, k in enumerate(keys1) if k in k2_map]
    common_indices_2 = [k2_map[keys1[i]] for i in common_indices_1]
    
    # 4. Final Ground Truth Check
    # We ignore the provided 'true_target_indices' for d2 and re-calculate 
    # them based on master_labels to ensure 100% accuracy.
    final_true_indices = np.array(d1['true_target_indices'])[common_indices_1]
    
    data_a = {
        'scores': d1['scores'][common_indices_1],
        'targets': final_true_indices
    }
    data_b = {
        'scores': d2_scores_aligned[common_indices_2],
        'targets': final_true_indices
    }
    
    print(f"[Internal] Alignment complete. {len(common_indices_1)} samples matched.")
    return data_a, data_b

def get_ranking_metrics(scores, targets):
    n_samples = scores.shape[0]
    # Rank is the position of the true target score among all scores (1-indexed)
    # We use argsort twice or a search to find the rank
    ranks = []
    for i in range(n_samples):
        row_scores = scores[i]
        true_score = row_scores[targets[i]]
        # How many scores are strictly greater than our true score?
        rank = np.sum(row_scores > true_score) + 1
        ranks.append(rank)
    
    ranks = np.array(ranks)
    mrr = np.mean(1.0 / ranks)
    recall_at = {k: np.mean(ranks <= k) for k in [1, 5, 10, 50]}
    
    return {
        'mrr': mrr,
        'r1': recall_at[1],
        'r5': recall_at[5],
        'r10': recall_at[10],
        'r50': recall_at[50],
        'median': np.median(ranks),
        'raw_ranks': ranks
    }

def print_report(m1, m2, name1, name2):
    print(f"\n{'='*70}")
    print(f"RANKING PERFORMANCE: {name1} vs {name2}")
    print(f"{'='*70}")
    print(f"{'Metric':<15} | {name1:<15} | {name2:<15} | {'Delta':<10}")
    print("-" * 70)
    
    metrics = [('MRR', 'mrr'), ('Recall@1', 'r1'), ('Recall@5', 'r5'), 
               ('Recall@10', 'r10'), ('Recall@50', 'r50'), ('Median Rank', 'median')]
    
    for label, key in metrics:
        v1, v2 = m1[key], m2[key]
        delta = v2 - v1
        print(f"{label:<15} | {v1:15.4f} | {v2:15.4f} | {delta:+10.4f}")

    print(f"\n{'Distribution':<15} | {name1:<15} | {name2:<15}")
    print("-" * 50)
    for low, high in [(1, 1), (2, 5), (6, 10), (11, 50), (51, 1022)]:
        l_str = f"Top {low}" if low == high else f"{low}-{high}"
        c1 = np.sum((m1['raw_ranks'] >= low) & (m1['raw_ranks'] <= high))
        c2 = np.sum((m2['raw_ranks'] >= low) & (m2['raw_ranks'] <= high))
        print(f"{l_str:<15} | {c1:15} | {c2:15}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("Usage: python script.py file_text.pkl file_skill.pkl")
    
    d1_raw = load_pkl(sys.argv[1])
    d2_raw = load_pkl(sys.argv[2])
    
    a, b = robust_alignment(d1_raw, d2_raw)
    
    metrics_a = get_ranking_metrics(a['scores'], a['targets'])
    metrics_b = get_ranking_metrics(b['scores'], b['targets'])
    
    print_report(metrics_a, metrics_b, "Text-Only", "Text-Skill")