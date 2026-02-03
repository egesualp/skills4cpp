import pickle
import json
import pandas as pd
import numpy as np
import argparse
from collections import Counter

def load_pkl(path):
    with open(path, 'rb') as f: return pickle.load(f)

def load_json(path):
    with open(path, 'r') as f: return json.load(f)

def align_datasets(d_t, d_s, d_h):
    # Target Alignment (Standard: Text-Only order)
    master_labels = d_t['target_labels']
    def reorder(data, master):
        l_map = {lbl: i for i, lbl in enumerate(data['target_labels'])}
        return data['scores'][:, [l_map[lbl] for lbl in master]]

    scores_s = reorder(d_s, master_labels)
    scores_h = reorder(d_h, master_labels)

    # Row Alignment (job_ids)
    def mk_hash(ids): return [tuple(p) if isinstance(p, list) else (p,) for p in ids]
    k_t, k_s, k_h = mk_hash(d_t['job_ids']), mk_hash(d_s['job_ids']), mk_hash(d_h['job_ids'])
    s_map, h_map = {k: i for i, k in enumerate(k_s)}, {k: i for i, k in enumerate(k_h)}
    
    # We need the original index 'i' to map to fused_predictions
    common = [(i, s_map[k], h_map[k], k) for i, k in enumerate(k_t) if k in s_map and k in h_map]
    idx_t, idx_s, idx_h, job_ids = zip(*common)

    return (d_t['scores'][list(idx_t)], scores_s[list(idx_s)], scores_h[list(idx_h)]), \
           np.array(d_t['true_target_indices'])[list(idx_t)], list(idx_t)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--skill", required=True)
    parser.add_argument("--hybrid", required=True)
    parser.add_argument("--fused_json", required=True, help="Path to fused_predictions.json")
    parser.add_argument("--skills_csv", default="data/esco_datasets/skills_en.csv")
    args = parser.parse_args()

    # 1. Load Data
    d_text, d_skill, d_hybrid = load_pkl(args.text), load_pkl(args.skill), load_pkl(args.hybrid)
    fused_preds = load_json(args.fused_json)
    df_skills = pd.read_csv(args.skills_csv)
    uri_to_label = dict(zip(df_skills['conceptUri'], df_skills['preferredLabel']))

    # 2. Align
    (s_t, s_s, s_h), targets, original_indices = align_datasets(d_text, d_skill, d_hybrid)

    # 3. Calculate Rank Shifts
    def get_ranks(scores, targets):
        ranks = []
        for i in range(len(targets)):
            ranks.append(np.sum(scores[i] > scores[i][targets[i]]) + 1)
        return np.array(ranks)

    ranks_t = get_ranks(s_t, targets)
    ranks_h = get_ranks(s_h, targets)
    
    # Rank Difference: Positive = Improvement, Negative = Regression
    rank_diff = ranks_t - ranks_h 

    # 4. Group Samples: Improvements vs Regressions
    improved_idx = np.where(rank_diff > 10)[0] # Improved by >10 positions
    regressed_idx = np.where(rank_diff < -10)[0] # Regressed by >10 positions

    def get_top_predicted_skills(sample_idx_in_aligned):
        orig_idx = str(original_indices[sample_idx_in_aligned])
        if orig_idx in fused_preds:
            # Top 10 predicted skills for this sample
            return [uri_to_label.get(s[0], s[0]) for s in fused_preds[orig_idx][:10]]
        return []

    # 5. Aggregate Skill Counts
    improved_skills = Counter()
    for idx in improved_idx:
        improved_skills.update(get_top_predicted_skills(idx))

    regressed_skills = Counter()
    for idx in regressed_idx:
        regressed_skills.update(get_top_predicted_skills(idx))

    # 6. Noise Index Calculation
    # A skill is "Noisy" if it appears much more in regressions than improvements
    all_skills = set(improved_skills.keys()) | set(regressed_skills.keys())
    skill_analysis = []
    
    for skill in all_skills:
        imp_count = improved_skills[skill]
        reg_count = regressed_skills[skill]
        total = imp_count + reg_count
        if total < 5: continue # Ignore rare skills
        
        # Noise Score: Ratio of regression association
        noise_score = reg_count / total
        skill_analysis.append({'skill': skill, 'imp': imp_count, 'reg': reg_count, 'noise_score': noise_score})

    df_noise = pd.DataFrame(skill_analysis)

    print(f"\n{'='*80}")
    print(f"SKILL SENSITIVITY & NOISE ANALYSIS")
    print(f"{'='*80}")
    print(f"Samples Improved (>10 ranks): {len(improved_idx)}")
    print(f"Samples Regressed (>10 ranks): {len(regressed_idx)}")
    
    print(f"\nTOP 10 'HELPFUL' SKILLS (High association with Improvements):")
    print(df_noise.sort_values(by=['imp', 'noise_score'], ascending=[False, True]).head(10)[['skill', 'imp', 'reg']])

    print(f"\nTOP 10 'NOISY' SKILLS (High association with Regressions):")
    # noise_score high means it pushes the model to a wrong answer
    print(df_noise.sort_values(by=['noise_score', 'reg'], ascending=[False, False]).head(10)[['skill', 'imp', 'reg', 'noise_score']])

    # Kıyaslama: Skill-Only Rank vs Skill-Only Confidence
    data = []
    for i in range(len(original_indices)):
        orig_idx = str(original_indices[i])
        if orig_idx in fused_preds:
            conf = np.mean([s[1] for s in fused_preds[orig_idx]])
            rank_s = np.sum(ds_aligned[i] > ds_aligned[i][targets[i]]) + 1
            rank_h = np.sum(dh_aligned[i] > dh_aligned[i][targets[i]]) + 1
            data.append({'conf': conf, 'rank_s': rank_s, 'rank_h': rank_h})

    df = pd.DataFrame(data)

    print(f"\n{'='*70}")
    print(f"SKILL-ONLY INTERNAL CONSISTENCY")
    print(f"{'='*70}")
    # Skill-only kendi içinde ne kadar tutarlı?
    print(f"Correlation (Confidence vs 1/Skill-Rank): {df['conf'].corr(1.0/df['rank_s']):.4f}")
    
    # Hibrit model Skill modelini ne kadar dinliyor?
    print(f"Correlation (Skill-Rank vs Hybrid-Rank): {df['rank_s'].corr(df['rank_h']):.4f}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()