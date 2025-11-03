# utils.py

import torch
import numpy as np
import random

def set_seed(seed: int = 42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """
    L2-normalizes a batch of embeddings.
    FAISS index requires normalized embeddings for inner product search.
    """
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    
    # Calculate norm
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Add a small epsilon to avoid division by zero
    epsilon = 1e-12
    
    # Return normalized embeddings
    return embeddings / (norm + epsilon)