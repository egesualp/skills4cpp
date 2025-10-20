import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional


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


def compute_map_mrr(I: np.ndarray, gold_rows: List[int]) -> Dict[str, float]:
    """
    Computes Mean Average Precision (MAP) and Mean Reciprocal Rank (MRR) at 10 and for the full ranking.
    With single ground truth, MAP is equivalent to MRR.

    Args:
        I: A 2D numpy array of shape (n_queries, n_candidates) containing
           ranked candidate indices for each query.
        gold_rows: A list of gold standard row indices. -1 indicates a missing gold.

    Returns:
        A dictionary with map@10, mrr@10, map_full, and mrr_full.
    """
    if not isinstance(I, np.ndarray) or I.ndim != 2:
        raise TypeError("I must be a 2D numpy array.")
    if not isinstance(gold_rows, list) or len(gold_rows) != I.shape[0]:
        raise ValueError("gold_rows must be a list with length equal to I.shape[0].")

    ap_scores_10 = []
    rr_scores_10 = []
    ap_scores_full = []
    rr_scores_full = []

    valid_gold_rows = [(i, gold_row) for i, gold_row in enumerate(gold_rows) if gold_row != -1]
    
    if not valid_gold_rows:
        return {"map@10": 0.0, "mrr@10": 0.0, "map_full": 0.0, "mrr_full": 0.0}

    for i, gold_row in valid_gold_rows:
        # @10
        top_10 = I[i, :10]
        try:
            rank = np.where(top_10 == gold_row)[0][0] + 1
            ap_scores_10.append(1 / rank)
            rr_scores_10.append(1 / rank)
        except IndexError:
            ap_scores_10.append(0.0)
            rr_scores_10.append(0.0)
        
        # full
        full_ranking = I[i, :]
        try:
            rank = np.where(full_ranking == gold_row)[0][0] + 1
            ap_scores_full.append(1 / rank)
            rr_scores_full.append(1 / rank)
        except IndexError:
            ap_scores_full.append(0.0)
            rr_scores_full.append(0.0)

    map_at_10 = np.mean(ap_scores_10) if ap_scores_10 else 0.0
    mrr_at_10 = np.mean(rr_scores_10) if rr_scores_10 else 0.0
    map_full = np.mean(ap_scores_full) if ap_scores_full else 0.0
    mrr_full = np.mean(rr_scores_full) if rr_scores_full else 0.0

    return {
        "map@10": float(map_at_10), 
        "mrr@10": float(mrr_at_10),
        "map_full": float(map_full),
        "mrr_full": float(mrr_full),
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
    gold_rows: List[int],
    esco_ids: List[str],
    skills_by_occupation: Dict[str, Set[str]],
    ks: Tuple[int, ...] = (1, 3, 5, 10)
) -> Dict[str, float]:
    """
    Computes skill coverage at various k values.
    
    Skill coverage measures what proportion of the skills from the gold ESCO occupation
    are covered by the skills from the predicted ESCO occupations.
    
    Args:
        I: A 2D numpy array of shape (n_queries, n_candidates) containing
           ranked candidate indices for each query.
        gold_rows: A list of gold standard row indices. -1 indicates a missing gold.
        esco_ids: List of all ESCO occupation IDs/URIs.
        skills_by_occupation: Dictionary mapping occupation URIs to sets of skill URIs.
        ks: A tuple of integers for which to compute skill coverage.
    
    Returns:
        A dictionary mapping skill_coverage@k to its value.
    """
    if not isinstance(I, np.ndarray) or I.ndim != 2:
        raise TypeError("I must be a 2D numpy array.")
    if not isinstance(gold_rows, list) or len(gold_rows) != I.shape[0]:
        raise ValueError("gold_rows must be a list with length equal to I.shape[0].")
    if not isinstance(ks, tuple):
        raise TypeError("ks must be a tuple of integers.")
    
    coverage_scores = {f"skill_coverage@{k}": [] for k in ks}
    
    valid_gold_rows = [(i, gold_row) for i, gold_row in enumerate(gold_rows) if gold_row != -1]
    
    if not valid_gold_rows:
        return {f"skill_coverage@{k}": 0.0 for k in ks}
    
    for query_idx, gold_row in valid_gold_rows:
        # Get gold occupation URI and its skills
        gold_occupation = esco_ids[gold_row]
        gold_skills = skills_by_occupation.get(gold_occupation, set())
        
        if not gold_skills:
            # If gold occupation has no skills, skip this query
            continue
        
        for k in ks:
            # Get predicted occupation URIs for top-k
            predicted_rows = I[query_idx, :k]
            predicted_occupations = [esco_ids[row] for row in predicted_rows if row < len(esco_ids)]
            
            # Collect all skills from predicted occupations
            predicted_skills = set()
            for occ_uri in predicted_occupations:
                predicted_skills.update(skills_by_occupation.get(occ_uri, set()))
            
            # Calculate coverage: how many gold skills are covered by predicted skills
            covered_skills = gold_skills.intersection(predicted_skills)
            coverage = len(covered_skills) / len(gold_skills)
            
            coverage_scores[f"skill_coverage@{k}"].append(coverage)
    
    # Calculate mean coverage for each k
    result = {}
    for k in ks:
        scores = coverage_scores[f"skill_coverage@{k}"]
        if scores:
            result[f"skill_coverage@{k}"] = float(np.mean(scores))
        else:
            result[f"skill_coverage@{k}"] = 0.0
    
    return result


METRICS = [
    compute_recall_at_k,
    compute_map_mrr,
]
