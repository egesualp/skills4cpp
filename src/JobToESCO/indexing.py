import faiss
import numpy as np
from pathlib import Path


def build_or_load_faiss_index(emb: np.ndarray, index_path: Path, use_cache: bool = True) -> faiss.Index:
    """
    Builds a FAISS index from embeddings and saves it, or loads it from cache.

    Args:
        emb: A 2D numpy array of embeddings.
        index_path: The path to save/load the index file.
        use_cache: If True, tries to load from index_path and saves the built index.

    Returns:
        The FAISS index.
    """
    if use_cache and index_path.exists():
        return read_index(str(index_path))

    index = build_ip_index(emb)

    if use_cache:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        write_index(index, str(index_path))

    return index


def build_ip_index(emb: np.ndarray) -> faiss.IndexFlatIP:
    """
    Builds a FAISS index for inner product search.

    Args:
        emb: A 2D numpy array of shape (n_vectors, dim) with float32 dtype.
             Each row must be L2-normalized.

    Returns:
        A faiss.IndexFlatIP object with the embeddings added.

    Raises:
        ValueError: If `emb` is not a float32 array or if its rows are not L2-normalized.
    """
    if emb.dtype != np.float32:
        raise ValueError("Embeddings must be of dtype float32. Use emb.astype(np.float32).")
    
    norms = np.linalg.norm(emb, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("Embedding vectors must be L2-normalized. Use sklearn.preprocessing.normalize.")

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index


def search_faiss_index(
    index: faiss.Index, queries: np.ndarray, topk: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Searches a FAISS index with a batch of query vectors.

    Args:
        index: The FAISS index to search.
        queries: A 2D numpy array of shape (n_queries, dim) with float32 dtype.
                 Each row must be L2-normalized.
        topk: The number of nearest neighbors to retrieve.

    Returns:
        A tuple (D, I) where:
        - D: A 2D float32 array of distances (scores).
        - I: A 2D int64 array of indices.
        
    Raises:
        ValueError: If `queries` is not a float32 array or if its rows are not L2-normalized.
    """
    if queries.dtype != np.float32:
        raise ValueError("Query vectors must be of dtype float32. Use queries.astype(np.float32).")

    norms = np.linalg.norm(queries, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("Query vectors must be L2-normalized. Use sklearn.preprocessing.normalize.")

    D, I = index.search(queries, topk)
    return D.astype(np.float32), I.astype(np.int64)


def write_index(index: faiss.Index, path: str):
    """
    Writes a FAISS index to a file.

    Args:
        index: The FAISS index to save.
        path: The path to the file.
    """
    faiss.write_index(index, path)


def read_index(path: str) -> faiss.Index:
    """
    Reads a FAISS index from a file.

    Args:
        path: The path to the file.

    Returns:
        The loaded FAISS index.
    """
    return faiss.read_index(path)



