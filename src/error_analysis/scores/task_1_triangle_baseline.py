import pickle
import numpy as np
import argparse
import sys
from typing import List, Dict

# Similarity hesaplama adımı zaten pkl içinde olduğu için 
# kullanıcı fonksiyonunu "Score" odaklı hale getirdik.
def calculate_ranking_metrics(scores_matrix: np.ndarray, 
                              true_target_indices: np.ndarray, 
                              k_values: List[int] = [1, 5, 10, 20]) -> Dict[str, float]:
    """
    Calculate ranking metrics: MRR, Recall@K, and Median Rank based on pre-calculated scores.
    """
    n_samples = scores_matrix.shape[0]
    
    # Sort indices in descending order of scores
    # (Higher score = better rank)
    sorted_indices = np.argsort(scores_matrix, axis=1)[:, ::-1]
    
    # Calculate MRR and Ranks
    reciprocal_ranks = []
    ranks = []
    
    for i, true_idx in enumerate(true_target_indices):
        # Find the rank of the true target
        # .index() mantığını numpy ile daha hızlı yapıyoruz
        rank = np.where(sorted_indices[i] == true_idx)[0][0] + 1
        ranks.append(rank)
        reciprocal_ranks.append(1.0 / rank)
    
    mrr = np.mean(reciprocal_ranks)
    median_rank = np.median(ranks)
    
    # Calculate Recall@K
    recall_at_k = {}
    for k in k_values:
        hits = 0
        for i, true_idx in enumerate(true_target_indices):
            if true_idx in sorted_indices[i, :k]:
                hits += 1
        recall_at_k[f'R@{k}'] = hits / n_samples
    
    metrics = {'MRR': mrr, 'MedianRank': median_rank}
    metrics.update(recall_at_k)
    
    return metrics

def align_datasets(d_text, d_skill, d_hybrid):
    """Aligns all datasets on targets (columns) and samples (rows)."""
    # 1. Target Alignment
    master_labels = d_text['target_labels']
    def get_reordered_scores(data, master_labels):
        label_to_idx = {lbl: i for i, lbl in enumerate(data['target_labels'])}
        reorder_idx = [label_to_idx[lbl] for lbl in master_labels]
        return data['scores'][:, reorder_idx]

    # 2. Row Alignment (job_ids)
    def mk_hash(ids): return [tuple(p) if isinstance(p, list) else (p,) for p in ids]
    k_t, k_s, k_h = mk_hash(d_text['job_ids']), mk_hash(d_skill['job_ids']), mk_hash(d_hybrid['job_ids'])
    
    common_keys = set(k_t) & set(k_s) & set(k_h)
    ordered_keys = [k for k in k_t if k in common_keys]
    
    def get_final(data, orig_keys, final_keys, scores):
        key_map = {k: i for i, k in enumerate(orig_keys)}
        idx = [key_map[k] for k in final_keys]
        return scores[idx], np.array(data['true_target_indices'])[idx]

    s_t, t_t = get_final(d_text, k_t, ordered_keys, d_text['scores'])
    s_s, _   = get_final(d_skill, k_s, ordered_keys, get_reordered_scores(d_skill, master_labels))
    s_h, _   = get_final(d_hybrid, k_h, ordered_keys, get_reordered_scores(d_hybrid, master_labels))
    
    return (s_t, s_s, s_h), t_t

def main():
    parser = argparse.ArgumentParser(description="Task 1: Triangle Baseline Analysis")
    parser.add_argument("--text", required=True, help="Path to text-only scores pkl")
    parser.add_argument("--skill", required=True, help="Path to skill-only scores pkl")
    parser.add_argument("--hybrid", required=True, help="Path to hybrid (text+skill) scores pkl")
    args = parser.parse_args()

    # Data loading
    with open(args.text, 'rb') as f: d_t = pickle.load(f)
    with open(args.skill, 'rb') as f: d_s = pickle.load(f)
    with open(args.hybrid, 'rb') as f: d_h = pickle.load(f)

    # Alignment
    (s_t, s_s, s_h), targets = align_datasets(d_t, d_s, d_h)

    # Metrics
    m_t = calculate_ranking_metrics(s_t, targets)
    m_s = calculate_ranking_metrics(s_s, targets)
    m_h = calculate_ranking_metrics(s_h, targets)

    # Reporting
    print(f"\n{'='*85}")
    print(f"{'METRIC':<15} | {'TEXT-ONLY':<15} | {'SKILL-ONLY':<15} | {'HYBRID (T+S)':<15}")
    print(f"{'='*85}")
    
    for metric in ['MRR', 'R@1', 'R@5', 'R@10', 'R@20', 'MedianRank']:
        print(f"{metric:<15} | {m_t[metric]:15.4f} | {m_s[metric]:15.4f} | {m_h[metric]:15.4f}")
    print(f"{'='*85}")

if __name__ == "__main__":
    main()