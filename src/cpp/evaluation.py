import numpy as np

def mrr(predictions):
    """
    Calculate Mean Reciprocal Rank (MRR)
    
    Args:
    predictions (list of tuples): Each tuple contains a true id and a list of predicted ids in order of confidence.
    
    Returns:
    float: MRR score
    """
    reciprocal_ranks = []
    for true_id, preds in predictions:
        preds = preds.tolist() if isinstance(preds, np.ndarray) else preds
        if true_id in preds:
            rank = preds.index(true_id) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0)
    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def r_at_k(predictions, k):
    """
    Calculate Recall at k (R@k)
    
    Args:
    predictions (list of tuples): Each tuple contains a true id and a list of predicted ids in order of confidence.
    k (int): The cut-off rank
    
    Returns:
    float: Recall at k score
    """
    relevant_found = 0
    for true_id, preds in predictions:
        if true_id in preds[:k]:
            relevant_found += 1
    return relevant_found / len(predictions)