import faiss
import numpy as np
import pytest
from src.config import ModelConfig
from src.indexing import build_ip_index, search_index
from src.metrics import compute_map_mrr_at_10, compute_recall_at_k
from src.model import BiEncoder


def test_evaluation_smoke():
    """
    A minimal smoke test for the evaluation pipeline.

    - Loads a BiEncoder model.
    - Creates a tiny fake ESCO list and job title pairs.
    - Encodes, builds an in-memory FAISS index, and searches.
    - Asserts R@1 == 1.0 and MAP@10 == 1.0.
    - Asserts dtypes are float32 for embeddings and distances.
    """
    # 1. Load model
    model_config = ModelConfig(
        hf_id="sentence-transformers/all-MiniLM-L6-v2",
        proj_dim=None,
        asymmetric=False,
        normalize_output=True,
    )
    model = BiEncoder(model_config, device="cpu")

    # 2. Create fake data
    esco_titles = ["Data scientist", "Software engineer", "Product manager"]
    job_titles = ["Data scientist"]
    gold_rows = [0]  # The first job title matches the first ESCO title

    # 3. Encode
    esco_embs = model.encode_esco(esco_titles)
    job_embs = model.encode_job(job_titles)

    # Assert dtypes after encoding
    assert esco_embs.dtype == np.float32
    assert job_embs.dtype == np.float32

    # 4. Build FAISS index and search
    index = build_ip_index(esco_embs)
    distances, indices = search_index(index, job_embs, topk=10)

    # Assert dtypes after search
    assert distances.dtype == np.float32

    # 5. Compute and assert metrics
    recall_metrics = compute_recall_at_k(indices, gold_rows, ks=(1,))
    map_metric, _ = compute_map_mrr_at_10(indices, gold_rows)

    assert recall_metrics["recall@1"] == 1.0
    assert map_metric == 1.0
