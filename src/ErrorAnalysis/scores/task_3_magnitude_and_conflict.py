import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--skill", required=True)
    args = parser.parse_args()

    dt, ds = load_pkl(args.text), load_pkl(args.skill)
    
    # Global score stats
    st_flat = dt['scores'].flatten()
    ss_flat = ds['scores'].flatten()

    print(f"\n{'='*60}")
    print(f"SCORE MAGNITUDE & VARIANCE ANALYSIS")
    print(f"{'='*60}")
    print(f"{'Statistic':<15} | {'Text-Only':<15} | {'Skill-Only':<15}")
    print("-" * 60)
    print(f"{'Mean Score':<15} | {np.mean(st_flat):15.4f} | {np.mean(ss_flat):15.4f}")
    print(f"{'Std Dev':<15} | {np.std(st_flat):15.4f} | {np.std(ss_flat):15.4f}")
    print(f"{'Max Score':<15} | {np.max(st_flat):15.4f} | {np.max(ss_flat):15.4f}")
    print(f"{'Min Score':<15} | {np.min(st_flat):15.4f} | {np.min(ss_flat):15.4f}")
    
    # Margin Analysis (Confidence)
    def get_margins(scores):
        sorted_scores = np.sort(scores, axis=1)[:, ::-1]
        return sorted_scores[:, 0] - sorted_scores[:, 1]

    m_t = get_margins(dt['scores'])
    m_s = get_margins(ds['scores'])

    print(f"-" * 60)
    print(f"{'Avg Margin':<15} | {np.mean(m_t):15.4f} | {np.mean(m_s):15.4f}")
    print(f"{'Median Margin':<15} | {np.median(m_t):15.4f} | {np.median(m_s):15.4f}")
    print(f"{'='*60}")
    print("Note: Margin is the difference between Top 1 and Top 2 scores.")

if __name__ == "__main__":
    main()