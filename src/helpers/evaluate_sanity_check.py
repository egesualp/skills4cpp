# evaluate_talentclef_csv.py
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from collections import Counter


# --- Config ---
MODE = "max"  # Choose: "pool" (average pooling) or "max" (max at inference)
NAME="decorte_test_pjmath_expanded"
PAIRS_PATH = "/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs/decorte_test_pairs.csv"          # cols: raw_title, esco_title, esco_id
ESCO_POOL_PATH = "/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/occupations_en_expanded.csv"  # cols: esco_id, esco_title
MODEL_ID = "pj-mathematician/JobGTE-7b-Lora"
BATCH_SIZE = 128
TOP_K = 10
TOP_K_RETRIEVAL = 200  # Used for max: retrieve more candidates before collapsing by ESCO ID
DEVICE = "cuda"

def _pool_esco_embeddings(esco_ids, esco_titles, esco_embs, renorm=True):
    """
    Simple pooling: if an `esco_id` appears multiple times, average its embeddings
    and (optionally) re-normalize. Keep the first title as representative.

    Returns (pooled_ids, pooled_titles, pooled_embs, did_pool)
    """
    from collections import defaultdict
    groups = defaultdict(list)
    for i, eid in enumerate(esco_ids):
        groups[str(eid)].append(i)

    n_rows = len(esco_ids)
    n_unique = len(groups)
    if n_unique == n_rows:
        print("No duplicate esco_id found. Skipping pooling.")
        # Nothing to pool
        return list(esco_ids), list(esco_titles), esco_embs.astype("float32", copy=False), False

    pooled_ids, pooled_titles, pooled_vecs = [], [], []
    for eid, idxs in groups.items():
        vecs = esco_embs[idxs]
        v = vecs.mean(axis=0)
        if renorm:
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
        pooled_ids.append(eid)
        pooled_titles.append(esco_titles[idxs[0]])
        pooled_vecs.append(v.astype("float32", copy=False))

    emb_out = np.vstack(pooled_vecs).astype("float32", copy=False)
    print(f"Pooled {n_rows} rows -> {len(pooled_ids)} unique")
    return pooled_ids, pooled_titles, emb_out, True

def _collapse_by_max(retrieved_ids, retrieved_titles, retrieved_scores):
    """
    max: collapse retrieved candidates by ESCO ID, taking the max score.
    
    Args:
        retrieved_ids: [N, K_retrieval] array of ESCO IDs (strings)
        retrieved_titles: [N, K_retrieval] array of titles (strings)
        retrieved_scores: [N, K_retrieval] array of scores (floats)
    
    Returns:
        collapsed_ids: [N, K_final] where K_final <= K_retrieval (unique ESCO IDs)
        collapsed_titles: [N, K_final]
        collapsed_scores: [N, K_final]
    """
    from collections import defaultdict
    
    N = retrieved_ids.shape[0]
    collapsed_ids_list = []
    collapsed_titles_list = []
    collapsed_scores_list = []
    
    for i in range(N):
        # Build dict: esco_id -> (max_score, title)
        id_to_best = {}
        for j in range(retrieved_ids.shape[1]):
            eid = retrieved_ids[i, j]
            score = retrieved_scores[i, j]
            title = retrieved_titles[i, j]
            
            if eid not in id_to_best or score > id_to_best[eid][0]:
                id_to_best[eid] = (score, title)
        
        # Sort by score descending
        sorted_items = sorted(id_to_best.items(), key=lambda x: x[1][0], reverse=True)
        
        ids = [item[0] for item in sorted_items]
        scores = [item[1][0] for item in sorted_items]
        titles = [item[1][1] for item in sorted_items]
        
        collapsed_ids_list.append(ids)
        collapsed_scores_list.append(scores)
        collapsed_titles_list.append(titles)
    
    return collapsed_ids_list, collapsed_titles_list, collapsed_scores_list

def compute_metrics(pred_ids, gold_ids, ks=(1,5,10)):
    N = len(gold_ids)
    ranks = np.full(N, np.inf, dtype=float)
    for i in range(N):
        row = pred_ids[i]
        if isinstance(row, np.ndarray):
            row = row.tolist()
        try:
            r = row.index(gold_ids[i]) + 1
            ranks[i] = r
        except ValueError:
            pass

    # Recall@k
    recalls = {}
    for k in ks:
        recalls[f"R@{k}"] = float(np.mean([1.0 if (r <= k) else 0.0 for r in ranks]))

    # MRR
    rr = [0.0 if np.isinf(r) else 1.0/r for r in ranks]
    mrr = float(np.mean(rr))

    # MAP@10 (with single gold per query -> equals RR truncated at 10)
    ap10 = [0.0 if np.isinf(r) or r > 10 else 1.0/r for r in ranks]
    map10 = float(np.mean(ap10))

    return recalls, mrr, map10, ranks

def main():
    # --- Load data ---
    pairs_df = pd.read_csv(PAIRS_PATH)
    esco_df  = pd.read_csv(ESCO_POOL_PATH)

    esco_df = esco_df.rename(columns={"preferredLabel": "esco_title", "conceptUri": "esco_id"})

    job_titles = pairs_df["raw_title"].astype(str).tolist()
    gold_ids   = pairs_df["esco_id"].astype(str).tolist()

    esco_titles = esco_df["esco_title"].astype(str).tolist()
    esco_ids    = esco_df["esco_id"].astype(str).tolist()

    dup_ct = sum(1 for c in Counter(esco_ids).values() if c > 1)
    print(f"ESCO rows: {len(esco_ids)} | unique: {len(set(esco_ids))} | duplicated ids: {dup_ct}")


    # --- Load model ---
    print(f"Loading model: {MODEL_ID}")
    model = SentenceTransformer(MODEL_ID, device=DEVICE)

    # --- Encode ESCO pool (batched) ---
    print("Encoding ESCO pool...")
    esco_embs = model.encode(
        esco_titles, batch_size=BATCH_SIZE,
        convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True
    ).astype("float32")

    print(f"\n=== MODE: {MODE} ===")
    
    if MODE == "pool":
        # --- Average pooling: merge duplicates by ESCO ID ---
        pooled_ids, pooled_titles, pooled_embs, did_pool = _pool_esco_embeddings(
            esco_ids, esco_titles, esco_embs, renorm=True
        )
        
        # --- Build FAISS index with pooled vectors ---
        dim = pooled_embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(pooled_embs)
        
        # --- Encode all job titles (batched) ---
        print("Encoding job titles...")
        job_embs = model.encode(
            job_titles, batch_size=BATCH_SIZE,
            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True
        ).astype("float32")
        
        # --- Search top-K directly ---
        print(f"Searching top-{TOP_K}...")
        scores, idxs = index.search(job_embs, TOP_K)
        
        # Map indices to ids/titles
        pred_ids = np.array([[pooled_ids[j] for j in row] for row in idxs], dtype=object)
        pred_titles = np.array([[pooled_titles[j] for j in row] for row in idxs], dtype=object)
    
    elif MODE == "max":
        # --- NO POOLING: Use max at inference instead ---
        # Keep all alternative vectors (rows) for each ESCO ID
        print(f"Total ESCO vectors (with alts): {len(esco_ids)}")
        
        # --- Build FAISS index with ALL vectors (including alts) ---
        dim = esco_embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(esco_embs)
        
        # --- Encode all job titles (batched) ---
        print("Encoding job titles...")
        job_embs = model.encode(
            job_titles, batch_size=BATCH_SIZE,
            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True
        ).astype("float32")
        
        # --- Search top-K_RETRIEVAL candidates (before collapsing) ---
        print(f"Searching top-{TOP_K_RETRIEVAL} candidates per query...")
        M = min(TOP_K_RETRIEVAL, len(esco_ids))  # and usually use max(M, TOP_K)
        scores, idxs = index.search(job_embs, max(M, TOP_K))
        
        # Map indices to ids/titles
        retrieved_ids = np.array([[esco_ids[j] for j in row] for row in idxs], dtype=object)
        retrieved_titles = np.array([[esco_titles[j] for j in row] for row in idxs], dtype=object)
        
        # --- Collapse by ESCO ID using MAX score ---
        print("Collapsing by ESCO ID (max)...")
        collapsed_ids, collapsed_titles, collapsed_scores = _collapse_by_max(
            retrieved_ids, retrieved_titles, scores
        )
        
        # --- Take top-K from collapsed results ---
        pred_ids = []
        pred_titles = []
        final_scores = []
        for i in range(len(collapsed_ids)):
            # Already sorted by score, just take top TOP_K
            pred_ids.append(collapsed_ids[i][:TOP_K])
            pred_titles.append(collapsed_titles[i][:TOP_K])
            final_scores.append(collapsed_scores[i][:TOP_K])
        
        pred_ids = np.array(pred_ids, dtype=object)
        pred_titles = np.array(pred_titles, dtype=object)
        scores = np.array(final_scores, dtype=float)
    
    else:
        raise ValueError(f"Unknown MODE: {MODE}. Choose 'pool' or 'max'.")

    # --- Metrics ---
    recalls, mrr, map10, ranks = compute_metrics(pred_ids, gold_ids, ks=(1,5,10))

    print("\n=== Metrics ===")
    for k, v in recalls.items():
        print(f"{k}: {v:.4f}")
    print(f"MRR: {mrr:.4f}")
    print(f"MAP@10: {map10:.4f}  (≈ MRR when one gold per query)")

    # --- Save detailed results ---
    out = pd.DataFrame({
        "job_title": job_titles,
        "gold_esco_id": gold_ids,
        "rank": ranks,
        "pred_esco_ids_top10": list(map(list, pred_ids)),
        "pred_esco_titles_top10": list(map(list, pred_titles)),
        "scores_top10": list(map(lambda x: list(map(float, x)), scores))
    })
    out_path = f"results_{NAME}_{MODE}.csv"
    out.to_csv(out_path, index=False)
    print(f"\n✅ Saved per-query results to {out_path}")

if __name__ == "__main__":
    main()
