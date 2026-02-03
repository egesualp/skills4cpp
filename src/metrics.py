import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional


def map_esco_id_to_row(
    gold_id_lists: List[List[str]], all_esco_ids: List[str]
) -> Tuple[List[List[int]], float]:
    """
    Maps gold ESCO IDs (one or many per query) to row indices and computes coverage.
    -1 is used for missing IDs.
    Returns:
        gold_row_lists: list of lists (per query) of row indices
        coverage: fraction of all gold IDs that were found
    """
    id_to_row = {esco_id: i for i, esco_id in enumerate(all_esco_ids)}

    gold_row_lists: List[List[int]] = []
    total = 0
    found = 0

    for gold_ids in gold_id_lists:
        rows_for_query: List[int] = []
        for gold_id in gold_ids:
            row_idx = id_to_row.get(gold_id, -1)
            rows_for_query.append(row_idx)
            total += 1
            if row_idx != -1:
                found += 1
        gold_row_lists.append(rows_for_query)

    coverage = found / total if total else 0.0

    return gold_row_lists, coverage


def compute_recall_at_k(
    I: np.ndarray, gold_row_lists: List[List[int]], ks: Tuple[int, ...] = (1, 5, 10)
) -> Dict[str, float]:
    """
    Computes recall at various k values with support for multiple golds per query.
    Recall per query = |hits ∩ gold| / |gold|, then averaged over queries with at least one gold.

    Args:
        I: A 2D numpy array of shape (n_queries, n_candidates) containing
           ranked candidate indices for each query.
        gold_row_lists: A list (len n_queries) of lists of gold standard row indices. -1 indicates a missing gold.
        ks: A tuple of integers for which to compute recall.

    Returns:
        A dictionary mapping recall@k to its value.
    """
    if not isinstance(I, np.ndarray) or I.ndim != 2:
        raise TypeError("I must be a 2D numpy array.")
    if not isinstance(gold_row_lists, list) or len(gold_row_lists) != I.shape[0]:
        raise ValueError("gold_row_lists must be a list with length equal to I.shape[0].")
    if not isinstance(ks, tuple):
        raise TypeError("ks must be a tuple of integers.")

    recalls = {}
    valid_gold = []
    for i, row_list in enumerate(gold_row_lists):
        gold_set = {r for r in row_list if r != -1}
        if gold_set:
            valid_gold.append((i, gold_set))

    n_valid = len(valid_gold)

    if n_valid == 0:
        for k in ks:
            recalls[f"recall@{k}"] = 0.0
        return recalls

    for k in ks:
        hits = 0
        for i, gold_set in valid_gold:
            topk = set(I[i, :k].tolist())
            if gold_set:
                # binary: count 1 if any relevant is in top-k
                if topk.intersection(gold_set):
                    hits += 1
        recalls[f"recall@{k}"] = hits / n_valid

    return recalls


def compute_map_mrr(I: np.ndarray, gold_row_lists: List[List[int]]) -> Dict[str, float]:
    """
    Computes MAP and MRR at 10 and for the full ranking with multiple golds per query.
    AP is mean precision at ranks of relevant hits; MRR is 1 / rank of first hit.
    """
    if not isinstance(I, np.ndarray) or I.ndim != 2:
        raise TypeError("I must be a 2D numpy array.")
    if not isinstance(gold_row_lists, list) or len(gold_row_lists) != I.shape[0]:
        raise ValueError("gold_row_lists must be a list with length equal to I.shape[0].")

    ap_scores_10 = []
    rr_scores_10 = []
    ap_scores_full = []
    rr_scores_full = []

    for i, gold_rows in enumerate(gold_row_lists):
        gold_list = [g for g in gold_rows if g != -1]
        if not gold_list:
            continue
        gold_set = set(gold_list)

        def _ap_and_rr(rank_list: np.ndarray) -> Tuple[float, float]:
            hits = 0
            precisions = []
            rr = 0.0
            for rank, candidate in enumerate(rank_list, start=1):
                if candidate in gold_set:
                    hits += 1
                    precisions.append(hits / rank)
                    if rr == 0.0:
                        rr = 1.0 / rank
            if not gold_set:
                return 0.0, rr
            ap = sum(precisions) / len(gold_set) if precisions else 0.0
            return ap, rr

        ap10, rr10 = _ap_and_rr(I[i, :10])
        ap_full, rr_full = _ap_and_rr(I[i, :])

        ap_scores_10.append(ap10)
        rr_scores_10.append(rr10)
        ap_scores_full.append(ap_full)
        rr_scores_full.append(rr_full)

    if not ap_scores_full:
        return {"map@10": 0.0, "mrr@10": 0.0, "map_full": 0.0, "mrr_full": 0.0}

    return {
        "map@10": float(np.mean(ap_scores_10)),
        "mrr@10": float(np.mean(rr_scores_10)),
        "map_full": float(np.mean(ap_scores_full)),
        "mrr_full": float(np.mean(rr_scores_full)),
    }


def load_skills_per_occupation(skills_path: str) -> Dict[str, Set[str]]:
    """
    Loads skills per occupation from CSV file.
    
    Args:
        skills_path: Path to the skills_per_occupations.csv file.
        
    Returns:
        A dictionary mapping occupation URIs to sets of skill URIs.
    """
    df = pd.read_csv(skills_path)
    
    # Group skills by occupation
    skills_by_occupation = {}
    for occupation_uri, group in df.groupby('occupationUri'):
        # Include both essential and optional skills
        skills = set(group['skillUri'].tolist())
        skills_by_occupation[occupation_uri] = skills
    
    return skills_by_occupation


def compute_skill_coverage(
    I: np.ndarray,
    gold_row_lists: List[List[int]],
    esco_ids: List[str],
    skills_by_occupation: Dict[str, Set[str]],
    ks: Tuple[int, ...] = (1, 3, 5, 10)
) -> Dict[str, float]:
    """
    Computes skill coverage at various k values with multiple golds.
    For each query, coverage@k is the max coverage over its gold occupations
    (best-of-gold) to allow 100% when any valid gold is perfectly retrieved.
    """
    if not isinstance(I, np.ndarray) or I.ndim != 2:
        raise TypeError("I must be a 2D numpy array.")
    if not isinstance(gold_row_lists, list) or len(gold_row_lists) != I.shape[0]:
        raise ValueError("gold_row_lists must be a list with length equal to I.shape[0].")
    if not isinstance(ks, tuple):
        raise TypeError("ks must be a tuple of integers.")
    
    coverage_scores = {f"skill_coverage@{k}": [] for k in ks}
    
    for query_idx, gold_rows in enumerate(gold_row_lists):
        gold_rows = [r for r in gold_rows if r != -1]
        if not gold_rows:
            continue

        gold_occupations = []
        for row in gold_rows:
            if 0 <= row < len(esco_ids):
                gold_occupations.append(esco_ids[row])
        if not gold_occupations:
            continue

        for k in ks:
            predicted_rows = I[query_idx, :k]
            predicted_occupations = [esco_ids[row] for row in predicted_rows if row < len(esco_ids)]

            predicted_skills = set()
            for occ_uri in predicted_occupations:
                predicted_skills.update(skills_by_occupation.get(occ_uri, set()))

            per_gold_coverages = []
            for gold_occ in gold_occupations:
                gold_skills = skills_by_occupation.get(gold_occ, set())
                if not gold_skills:
                    continue
                covered_skills = gold_skills.intersection(predicted_skills)
                per_gold_coverages.append(len(covered_skills) / len(gold_skills))

            if per_gold_coverages:
                coverage_scores[f"skill_coverage@{k}"].append(max(per_gold_coverages))
    
    result = {}
    for k in ks:
        scores = coverage_scores[f"skill_coverage@{k}"]
        result[f"skill_coverage@{k}"] = float(np.mean(scores)) if scores else 0.0
    
    return result


METRICS = [
    compute_recall_at_k,
    compute_map_mrr,
]
