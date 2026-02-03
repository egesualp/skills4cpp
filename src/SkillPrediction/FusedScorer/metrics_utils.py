import numpy as np
from typing import List, Dict, Set, Tuple

def compute_precision_at_k(
    indices: np.ndarray,
    gold_sets: List[Set[int]],
    k: int
) -> float:
    """
    Compute Mean Precision@k for multi-label retrieval.
    """
    precisions = []
    for i, gold_set in enumerate(gold_sets):
        if not gold_set:
            continue
            
        retrieved = indices[i, :k]
        hits = sum(1 for idx in retrieved if idx in gold_set)
        precisions.append(hits / k)
        
    return float(np.mean(precisions)) if precisions else 0.0

def compute_recall_at_k(
    indices: np.ndarray,
    gold_sets: List[Set[int]],
    k: int
) -> float:
    """
    Compute Mean Recall@k for multi-label retrieval.
    """
    recalls = []
    for i, gold_set in enumerate(gold_sets):
        if not gold_set:
            continue
            
        retrieved = indices[i, :k]
        hits = sum(1 for idx in retrieved if idx in gold_set)
        recalls.append(hits / len(gold_set))
        
    return float(np.mean(recalls)) if recalls else 0.0

def compute_map(
    indices: np.ndarray,
    gold_sets: List[Set[int]],
    k: int | None = None
) -> float:
    """
    Compute Mean Average Precision (MAP).
    If k is provided, computes MAP@k. If k is None, computes MAP on full ranking.
    """
    aps = []
    n_queries = indices.shape[0]
    
    for i in range(n_queries):
        gold_set = gold_sets[i]
        if not gold_set:
            continue
            
        retrieved = indices[i]
        if k is not None:
            retrieved = retrieved[:k]
            
        hits = 0
        sum_precisions = 0.0
        
        for rank, idx in enumerate(retrieved, start=1):
            if idx in gold_set:
                hits += 1
                sum_precisions += hits / rank
                
        # Standard MAP definition divides by min(len(gold_set), k) or len(gold_set)
        # Usually it is divided by the total number of relevant documents.
        aps.append(sum_precisions / len(gold_set))
        
    return float(np.mean(aps)) if aps else 0.0

def compute_mrr(
    indices: np.ndarray,
    gold_sets: List[Set[int]],
    k: int | None = None
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    If k is provided, computes MRR@k. If k is None, computes MRR on full ranking.
    MRR uses the rank of the *first* relevant item found.
    """
    reciprocal_ranks = []
    n_queries = indices.shape[0]
    
    for i in range(n_queries):
        gold_set = gold_sets[i]
        if not gold_set:
            continue
            
        retrieved = indices[i]
        if k is not None:
            retrieved = retrieved[:k]
            
        rr = 0.0
        for rank, idx in enumerate(retrieved, start=1):
            if idx in gold_set:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)
        
    return float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

def compute_ndcg_at_k(
    indices: np.ndarray,
    gold_sets: List[Set[int]],
    k: int
) -> float:
    """
    Compute Mean NDCG@k.
    Assumes binary relevance (1 if in gold_set, 0 otherwise).
    """
    ndcgs = []
    for i, gold_set in enumerate(gold_sets):
        if not gold_set:
            continue
            
        retrieved = indices[i, :k]
        
        # DCG
        dcg = 0.0
        for rank, idx in enumerate(retrieved, start=1):
            if idx in gold_set:
                dcg += 1.0 / np.log2(rank + 1)
                
        # IDCG (Ideal DCG)
        idcg = 0.0
        # In ideal ranking, all relevant items come first
        num_relevant_at_k = min(len(gold_set), k)
        for rank in range(1, num_relevant_at_k + 1):
            idcg += 1.0 / np.log2(rank + 1)
            
        if idcg > 0:
            ndcgs.append(dcg / idcg)
        else:
            ndcgs.append(0.0)
            
    return float(np.mean(ndcgs)) if ndcgs else 0.0

