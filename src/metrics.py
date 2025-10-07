import numpy as np
from typing import Dict, List, Tuple


def map_esco_id_to_row(
    gold_ids: List[str], all_esco_ids: List[str]
) -> Tuple[List[int], float]:
    """
    Maps gold ESCO IDs to row indices and computes coverage.
    -1 is used for missing IDs.
    """
    id_to_row = {esco_id: i for i, esco_id in enumerate(all_esco_ids)}
    gold_rows = [id_to_row.get(gold_id, -1) for gold_id in gold_ids]

    n_found = sum(1 for row in gold_rows if row != -1)
    coverage = n_found / len(gold_rows) if gold_rows else 0.0

    return gold_rows, coverage


def compute_recall_at_k(
    I: np.ndarray, gold_rows: List[int], ks: Tuple[int, ...] = (1, 5, 10)
) -> Dict[str, float]:
    """
    Computes recall at various k values.

    Args:
        I: A 2D numpy array of shape (n_queries, n_candidates) containing
           ranked candidate indices for each query.
        gold_rows: A list of gold standard row indices. -1 indicates a missing gold.
        ks: A tuple of integers for which to compute recall.

    Returns:
        A dictionary mapping recall@k to its value.
    """
    if not isinstance(I, np.ndarray) or I.ndim != 2:
        raise TypeError("I must be a 2D numpy array.")
    if not isinstance(gold_rows, list) or len(gold_rows) != I.shape[0]:
        raise ValueError("gold_rows must be a list with length equal to I.shape[0].")
    if not isinstance(ks, tuple):
        raise TypeError("ks must be a tuple of integers.")

    recalls = {}
    valid_gold_rows = [(i, gold_row) for i, gold_row in enumerate(gold_rows) if gold_row != -1]
    n_valid = len(valid_gold_rows)

    if n_valid == 0:
        for k in ks:
            recalls[f"recall@{k}"] = 0.0
        return recalls

    for k in ks:
        hits = 0
        for i, gold_row in valid_gold_rows:
            if gold_row in I[i, :k]:
                hits += 1
        recalls[f"recall@{k}"] = hits / n_valid

    return recalls


def compute_map_mrr_at_10(I: np.ndarray, gold_rows: List[int]) -> Dict[str, float]:
    """
    Computes Mean Average Precision (MAP) and Mean Reciprocal Rank (MRR) at 10.

    Args:
        I: A 2D numpy array of shape (n_queries, n_candidates) containing
           ranked candidate indices for each query.
        gold_rows: A list of gold standard row indices. -1 indicates a missing gold.

    Returns:
        A dictionary with map@10 and mrr@10.
    """
    if not isinstance(I, np.ndarray) or I.ndim != 2:
        raise TypeError("I must be a 2D numpy array.")
    if not isinstance(gold_rows, list) or len(gold_rows) != I.shape[0]:
        raise ValueError("gold_rows must be a list with length equal to I.shape[0].")

    ap_scores = []
    rr_scores = []

    valid_gold_rows = [(i, gold_row) for i, gold_row in enumerate(gold_rows) if gold_row != -1]
    
    if not valid_gold_rows:
        return {"map@10": 0.0, "mrr@10": 0.0}

    for i, gold_row in valid_gold_rows:
        top_10 = I[i, :10]
        try:
            rank = np.where(top_10 == gold_row)[0][0] + 1
            ap_scores.append(1 / rank)
            rr_scores.append(1 / rank)
        except IndexError:
            ap_scores.append(0.0)
            rr_scores.append(0.0)

    map_at_10 = np.mean(ap_scores) if ap_scores else 0.0
    mrr_at_10 = np.mean(rr_scores) if rr_scores else 0.0

    return {"map@10": float(map_at_10), "mrr@10": float(mrr_at_10)}


METRICS = [
    compute_recall_at_k,
    compute_map_mrr_at_10,
]
