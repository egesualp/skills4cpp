
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.getcwd() + "/src")

from skill_mapping.v5.fused_scorer import FusedScorer as OriginalScorer
from skill_mapping.v5.fused_scorer_chunked import FusedScorer as ChunkedScorer

def run_sanity_check():
    # Paths (using those from command.sh)
    esco_dir = Path("data/esco_datasets")
    label_encoder = Path("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h1_soft_deep_larger_val_mpnet/label_encoder.json")
    task_a_path = Path("/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/experiments/results/infer_decorte_all_jobbert_final_2/predictions.jsonl")
    task_b_path = Path("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/decorte_w_desc_3/similarity_scores.json")
    isco_path = Path("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h1_soft_deep_larger_val_mpnet_expanded/inference_results_soft_labels_deep_larger_val_mpnet_expanded/isco_predictions.json")
    decorte_map_path = Path("/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/decorte_master_3.csv")

    print("Initializing Scorers...")
    orig_scorer = OriginalScorer(esco_dir, label_encoder, isco_level=1)
    chunk_scorer = ChunkedScorer(esco_dir, label_encoder, isco_level=1)

    orig_scorer.build_lookup_tables()
    chunk_scorer.build_lookup_tables()

    orig_scorer.build_affinity_matrix(mode='uniform')
    chunk_scorer.build_affinity_matrix(mode='uniform')

    # Load 10 samples
    print("Loading 10 samples...")
    job_ids = [str(i) for i in range(10)]
    
    # Task A
    task_a_data = {}
    with open(task_a_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 10: break
            task_a_data[str(i)] = json.loads(line).get('predicted_esco_ids', [])
    
    # Task B
    with open(task_b_path, 'r') as f:
        # Since it's 40GB, we need to be careful. But reading the first 10 items from a dict might be hard if it's large.
        # Actually, since it's a JSON dict, we can't easily partial load without ijson.
        # Let's use ijson to get the first 10.
        import ijson
        task_b_data = {}
        with open(task_b_path, 'rb') as bf:
            parser = ijson.kvitems(bf, '')
            count = 0
            for k, v in parser:
                if k in job_ids:
                    # Convert list to dict if needed
                    scores = {}
                    if isinstance(v, list):
                        for item in v:
                            scores[item['skill_uri']] = float(item['score'])
                    elif isinstance(v, dict):
                        scores = {sk: float(sc) for sk, sc in v.items()}
                    task_b_data[k] = scores
                    count += 1
                if count >= 10: break

    # ISCO
    with open(isco_path, 'r') as f:
        raw_isco = json.load(f)
    isco_data = {}
    label_to_idx = {lbl: i for i, lbl in enumerate(orig_scorer.isco_index)}
    n_classes = len(orig_scorer.isco_index)
    for jid in job_ids:
        data = raw_isco[jid]
        prob_vec = np.zeros(n_classes, dtype=np.float32)
        if "topk" in data:
            for item in data["topk"]:
                lbl = item["label"]
                score = item["score"]
                if lbl in label_to_idx:
                    prob_vec[label_to_idx[lbl]] = score
        s = np.sum(prob_vec)
        if s > 0: prob_vec /= s
        isco_data[jid] = prob_vec

    # GT
    decorte_df = pd.read_csv(decorte_map_path)
    decorte_df['job_id'] = decorte_df['job_id'].astype(str)
    gt_data = {}
    for jid in job_ids:
        occ_uri = decorte_df[decorte_df['job_id'] == jid]['esco_id'].values[0]
        gt_data[jid] = orig_scorer.occ_to_skills.get(occ_uri, set())

    # Performance comparison
    print("\nComparing score_job for job_id '0'...")
    jid = '0'
    res_orig, _ = orig_scorer.score_job(task_a_data[jid][:5], task_b_data[jid], isco_data[jid])
    res_chunk, _ = chunk_scorer.score_job(task_a_data[jid][:5], task_b_data[jid], isco_data[jid])

    print(f"Top 3 Original: {res_orig[:3]}")
    print(f"Top 3 Chunked:  {res_chunk[:3]}")
    
    match = True
    for o, c in zip(res_orig[:10], res_chunk[:10]):
        if o[0] != c[0] or abs(o[1] - c[1]) > 1e-6:
            match = False
            break
    print(f"Score Job Match: {match}")

    # Evaluate batch comparison
    print("\nComparing evaluation results for 10 jobs...")
    eval_orig = orig_scorer.evaluate_batch(task_a_data, task_b_data, isco_data, gt_data, task_a_k=5)
    
    # For chunked, we need to mock the iterator
    task_b_iter = [task_b_data] 
    eval_chunk = chunk_scorer.evaluate_chunked_streaming(task_a_data, iter(task_b_iter), isco_data, gt_data, task_a_k=5)

    print(f"Original Eval: {eval_orig}")
    print(f"Chunked  Eval: {eval_chunk}")

if __name__ == "__main__":
    run_sanity_check()
