import pickle
import numpy as np
import argparse
import sys

def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def align_three(d_t, d_s, d_h):
    # 1. Target Alignment (Align all to Text order)
    master_labels = d_t['target_labels']
    def get_reordered(data, ref_labels):
        l_map = {lbl: i for i, lbl in enumerate(data['target_labels'])}
        idx = [l_map[lbl] for lbl in ref_labels]
        return data['scores'][:, idx]

    scores_s = get_reordered(d_s, master_labels)
    scores_h = get_reordered(d_h, master_labels)

    # 2. Row Alignment (job_ids)
    def mk_hash(ids): return [tuple(p) if isinstance(p, list) else (p,) for p in ids]
    k_t, k_s, k_h = mk_hash(d_t['job_ids']), mk_hash(d_s['job_ids']), mk_hash(d_h['job_ids'])
    
    s_map = {k: i for i, k in enumerate(k_s)}
    h_map = {k: i for i, k in enumerate(k_h)}
    
    common_idx = []
    for i, k in enumerate(k_t):
        if k in s_map and k in h_map:
            common_idx.append((i, s_map[k], h_map[k]))
            
    return common_idx, d_t['scores'], scores_s, scores_h, np.array(d_t['true_target_indices'])

def main():
    parser = argparse.ArgumentParser(description="Task 2: Triple Synergy Analysis")
    parser.add_argument("--text", required=True)
    parser.add_argument("--skill", required=True)
    parser.add_argument("--hybrid", required=True)
    args = parser.parse_args()

    dt, ds, dh = load_pkl(args.text), load_pkl(args.skill), load_pkl(args.hybrid)
    indices, st, ss, sh, targets = align_three(dt, ds, dh)

    # State counters
    stats = {
        'Synergy': 0,     # T&S Wrong -> H Correct (GOLD!)
        'T_Recovered': 0, # T Wrong, S Correct -> H Correct
        'S_Recovered': 0, # S Wrong, T Correct -> H Correct
        'Degraded': 0,    # T or S Correct -> H Wrong (BAD)
        'All_Correct': 0, 
        'All_Wrong': 0,
        'T_Only_Correct': 0,
        'S_Only_Correct': 0
    }

    for it, isk, ih in indices:
        target = targets[it]
        pt = np.argmax(st[it])
        ps = np.argmax(ss[isk])
        ph = np.argmax(sh[ih])

        ct, cs, ch = (pt == target), (ps == target), (ph == target)

        if not ct and not cs and ch: stats['Synergy'] += 1
        if not ct and cs and ch: stats['T_Recovered'] += 1
        if ct and not cs and ch: stats['S_Recovered'] += 1
        if (ct or cs) and not ch: stats['Degraded'] += 1
        if ct and cs and ch: stats['All_Correct'] += 1
        if not ct and not cs and not ch: stats['All_Wrong'] += 1
        if ct and not cs and not ch: stats['T_Only_Correct'] += 1
        if not ct and cs and not ch: stats['S_Only_Correct'] += 1

    total = len(indices)
    print(f"\n{'='*60}")
    print(f"TRIPLE SUCCESS & SYNERGY ANALYSIS")
    print(f"{'='*60}")
    print(f"Total Common Samples : {total}")
    print(f"-"*60)
    print(f"1. ALL CORRECT       : {stats['All_Correct']} ({100*stats['All_Correct']/total:.1f}%)")
    print(f"2. ALL WRONG         : {stats['All_Wrong']} ({100*stats['All_Wrong']/total:.1f}%)")
    print(f"-"*60)
    print(f"3. SYNERGY (T&S ❌ -> H ✅) : {stats['Synergy']} cases")
    print(f"4. T-RECOVERED (S help)     : {stats['T_Recovered']} cases")
    print(f"5. S-RECOVERED (T help)     : {stats['S_Recovered']} cases")
    print(f"6. DEGRADED (T or S ✅ -> H ❌): {stats['Degraded']} cases")
    print(f"-"*60)
    print(f"T-Only Success (S & H failed): {stats['T_Only_Correct']}")
    print(f"S-Only Success (T & H failed): {stats['S_Only_Correct']}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()