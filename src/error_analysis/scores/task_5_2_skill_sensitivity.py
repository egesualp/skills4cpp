import pickle
import json
import numpy as np
import argparse
import pandas as pd

def load_pkl(path):
    with open(path, 'rb') as f: return pickle.load(f)

def load_json(path):
    with open(path, 'r') as f: return json.load(f)

def align_and_extract(d_text, d_hybrid, fused_preds):
    # Standard Row Alignment using job_ids
    def mk_hash(ids): return [tuple(p) if isinstance(p, list) else (p,) for p in ids]
    k_t, k_h = mk_hash(d_text['job_ids']), mk_hash(d_hybrid['job_ids'])
    h_map = {k: i for i, k in enumerate(k_h)}
    
    # Matching aligned indices
    matched_data = []
    targets = np.array(d_text['true_target_indices'])
    
    for i, key in enumerate(k_t):
        if key in h_map:
            h_idx = h_map[key]
            orig_idx_str = str(i) # fused_predictions keys are original indices
            
            if orig_idx_str in fused_preds:
                # 1. Get average confidence score of top predicted skills
                skill_scores = [s[1] for s in fused_preds[orig_idx_str]]
                avg_skill_conf = np.mean(skill_scores) if skill_scores else 0
                
                # 2. Get the Rank of the hybrid model
                # (Reordering columns for hybrid score consistency)
                # Assuming hybrid scores were already aligned in Task 1/2 logic
                # For brevity, we calculate rank directly from scores here
                target_idx = targets[i]
                h_score_row = d_hybrid['scores'][h_idx]
                rank_h = np.sum(h_score_row > h_score_row[target_idx]) + 1
                
                matched_data.append({
                    'avg_skill_confidence': avg_skill_conf,
                    'rank': rank_h
                })
                
    return pd.DataFrame(matched_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--hybrid", required=True)
    parser.add_argument("--fused_json", required=True)
    args = parser.parse_args()

    dt, dh = load_pkl(args.text), load_pkl(args.hybrid)
    fused_preds = load_json(args.fused_json)

    df = align_and_extract(dt, dh, fused_preds)

    # Bucketing by Rank Performance
    bins = [0, 1, 5, 20, 100, 1022]
    labels = ['Rank 1', 'Rank 2-5', 'Rank 6-20', 'Rank 21-100', 'Rank 100+']
    df['performance_group'] = pd.cut(df['rank'], bins=bins, labels=labels)

    summary = df.groupby('performance_group')['avg_skill_confidence'].agg(['mean', 'std', 'count']).reset_index()

    print(f"\n{'='*70}")
    print(f"ANALYSIS: SKILL PREDICTION CONFIDENCE VS. CPP RANKING")
    print(f"{'='*70}")
    print(summary.to_string(index=False))
    print(f"{'='*70}")

    # Correlation
    corr = df['avg_skill_confidence'].corr(1.0/df['rank'])
    print(f"Correlation (Confidence Score vs. 1/Rank): {corr:.4f}")

if __name__ == "__main__":
    main()