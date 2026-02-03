import pickle
import numpy as np
import pandas as pd
import argparse
import sys
from typing import List, Dict

def parse_target_label(label: str) -> str:
    """Extracts [NAME] from 'esco role: [NAME] \n description: ...'"""
    try:
        # 'esco role: ' kısmını at, ilk satırı al (newline öncesi)
        name_part = label.split('\n')[0].replace('esco role:', '').strip()
        return name_part
    except:
        return label

def calculate_metrics(ranks: np.ndarray) -> Dict[str, float]:
    """Calculates MRR, R@5, R@10, R@50, and Median Rank."""
    return {
        'MRR': np.mean(1.0 / ranks),
        'R@5': np.mean(ranks <= 5),
        'R@10': np.mean(ranks <= 10),
        'R@50': np.mean(ranks <= 50),
        'Median': np.median(ranks)
    }

def align_three(d_t, d_s, d_h):
    # Standard Target Alignment (Master: Text-Only)
    master_labels = d_t['target_labels']
    def reorder(data, master):
        l_map = {lbl: i for i, lbl in enumerate(data['target_labels'])}
        return data['scores'][:, [l_map[lbl] for lbl in master]]

    scores_s = reorder(d_skill, master_labels)
    scores_h = reorder(d_hybrid, master_labels)

    # Row Alignment (job_ids)
    def mk_hash(ids): return [tuple(p) if isinstance(p, list) else (p,) for p in ids]
    k_t, k_s, k_h = mk_hash(d_t['job_ids']), mk_hash(d_s['job_ids']), mk_hash(d_h['job_ids'])
    s_map, h_map = {k: i for i, k in enumerate(k_s)}, {k: i for i, k in enumerate(k_h)}
    
    common = [(i, s_map[k], h_map[k]) for i, k in enumerate(k_t) if k in s_map and k in h_map]
    idx_t, idx_s, idx_h = zip(*common)

    return (d_t['scores'][list(idx_t)], scores_s[list(idx_s)], scores_h[list(idx_h)]), \
           np.array(d_t['true_target_indices'])[list(idx_t)], master_labels

def get_ranks(scores, targets):
    n = scores.shape[0]
    ranks = []
    for i in range(n):
        row = scores[i]
        true_score = row[targets[i]]
        ranks.append(np.sum(row > true_score) + 1)
    return np.array(ranks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--skill", required=True)
    parser.add_argument("--hybrid", required=True)
    parser.add_argument("--esco_csv", default="data/occupations_en.csv")
    parser.add_argument("--output", default="/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/src/error_analysis/task4_isco_performance_report.txt")
    args = parser.parse_args()

    # Load ESCO Data
    df_esco = pd.read_csv(args.esco_csv)
    df_esco['clean_label'] = df_esco['preferredLabel'].str.lower().str.strip() if 'preferredLabel' in df_esco else ""

    # Load Pickles
    d_text, d_skill, d_hybrid = pickle.load(open(args.text, 'rb')), pickle.load(open(args.skill, 'rb')), pickle.load(open(args.hybrid, 'rb'))

    # Align
    (s_t, s_s, s_h), targets, master_labels = align_three(d_text, d_skill, d_hybrid)
    
    # Calculate Ranks
    r_t, r_s, r_h = get_ranks(s_t, targets), get_ranks(s_s, targets), get_ranks(s_h, targets)

    # Prepare DataFrame for Analysis
    results_df = pd.DataFrame({
        'target_idx': targets,
        'label': [master_labels[i] for i in targets],
        'rank_text': r_t,
        'rank_skill': r_s,
        'rank_hybrid': r_h
    })
    results_df['clean_name'] = results_df['label'].apply(parse_target_label).str.lower()

    # Merge with ISCO Codes
    results_df = results_df.merge(df_esco[['preferredLabel', 'code']], left_on='clean_name', right_on=df_esco['preferredLabel'].str.lower(), how='left')
    
    # Create ISCO Levels
    results_df['code'] = results_df['code'].astype(str)
    for level in range(1, 5):
        results_df[f'isco_level{level}'] = results_df['code'].str[:level]
        n_level = results_df[f'isco_level{level}'].nunique()
        print(f"Number of distinct groups in level {level}: {n_level}")

    # REPORT GENERATION
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("HIERARCHICAL ERROR ANALYSIS REPORT (ISCO LEVELS)\n")
        f.write("="*80 + "\n")
        f.write(f"Source Text-Only  : {args.text}\n")
        f.write(f"Source Skill-Only : {args.skill}\n")
        f.write(f"Source Hybrid     : {args.hybrid}\n")
        f.write(f"ESCO Reference    : {args.esco_csv}\n\n")

        for lvl in range(1, 5):
            f.write(f"\n--- ISCO LEVEL {lvl} ANALYSIS ---\n")
            col = f'isco_level{lvl}'
            
            group_stats = []
            for name, group in results_df.groupby(col):
                if name == "nan" or len(group) < 5: continue
                m_t = calculate_metrics(group['rank_text'])
                m_s = calculate_metrics(group['rank_skill'])
                m_h = calculate_metrics(group['rank_hybrid'])
                
                group_stats.append({
                    'Code': name,
                    'Count': len(group),
                    'MRR_T': m_t['MRR'], 'MRR_S': m_s['MRR'], 'MRR_H': m_h['MRR'],
                    'Med_T': m_t['Median'], 'Med_H': m_h['Median'],
                    'Diff_H_T': m_h['MRR'] - m_t['MRR']
                })
            
            df_lvl = pd.DataFrame(group_stats)
            if df_lvl.empty: continue

            # Level 1 & 2: Full List | Level 3 & 4: Top/Worst 10
            if lvl <= 2:
                f.write(df_lvl.sort_values('Count', ascending=False).to_string(index=False))
            else:
                f.write("Top 5 Improved (MRR):\n")
                f.write(df_lvl.sort_values('Diff_H_T', ascending=False).head(5).to_string(index=False))
                f.write("\nWorst 5 Regressed (MRR):\n")
                f.write(df_lvl.sort_values('Diff_H_T', ascending=True).head(5).to_string(index=False))
            f.write("\n" + "-"*80 + "\n")

    print(f"Report successfully saved to: {args.output}")