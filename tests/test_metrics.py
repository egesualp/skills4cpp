import numpy as np
import pytest
from src.metrics import (
    map_esco_id_to_row,
    compute_recall_at_k,
    compute_map_mrr,
    compute_skill_coverage,
)

# Fixtures for test data
@pytest.fixture
def sample_ids():
    all_esco_ids = ["id1", "id2", "id3", "id4", "id5"]
    gold_ids = [["id3"], ["id1"], ["id6"], ["id2", "id5"]]  # id6 missing; last has two golds
    return gold_ids, all_esco_ids

@pytest.fixture
def sample_retrieval_results():
    # 4 queries, top 5 candidates for each
    I = np.array([
        [2, 0, 1, 4, 3],  # gold: 2 (rank 1)
        [0, 2, 1, 3, 4],  # gold: 0 (rank 1)
        [0, 1, 3, 4, 2],  # gold: -1 (missing)
        [1, 0, 3, 4, 2],  # gold: 1 (rank 1) -> but let's change gold_rows for variety
    ])
    # Corresponding gold rows for the queries
    gold_rows = [[2], [0], [-1], [4]] # gold for last query is at index 4
    return I, gold_rows


# Tests for map_esco_id_to_row
def test_map_esco_id_to_row_basic(sample_ids):
    gold_ids, all_esco_ids = sample_ids
    gold_rows, coverage = map_esco_id_to_row(gold_ids, all_esco_ids)
    
    assert gold_rows == [[2], [0], [-1], [1, 4]]
    assert coverage == 4 / 5

def test_map_esco_id_to_row_empty():
    gold_rows, coverage = map_esco_id_to_row([], ["id1", "id2"])
    assert gold_rows == []
    assert coverage == 0.0

    gold_rows, coverage = map_esco_id_to_row([["id1"]], [])
    assert gold_rows == [[-1]]
    assert coverage == 0.0

# Tests for compute_recall_at_k
def test_compute_recall_at_k_perfect_recall():
    I = np.array([[0, 1, 2], [1, 0, 2]])
    gold_rows = [[0], [1]]
    recalls = compute_recall_at_k(I, gold_rows, ks=(1, 2))
    assert recalls["recall@1"] == 1.0
    assert recalls["recall@2"] == 1.0

def test_compute_recall_at_k_partial_recall():
    I = np.array([
        [1, 2, 0], # gold: 0 is at rank 3
        [1, 0, 2], # gold: 1 is at rank 1
        [2, 1, 0], # gold: -1 (missing)
    ])
    gold_rows = [[0], [1], [-1]]
    recalls = compute_recall_at_k(I, gold_rows, ks=(1, 2, 3))
    assert recalls["recall@1"] == 0.5 # 1 of 2 valid queries has a hit at rank 1
    assert recalls["recall@2"] == 0.5 # 1 of 2
    assert recalls["recall@3"] == 1.0 # 2 of 2

# Tests for compute_map_mrr
def test_compute_map_mrr_basic(sample_retrieval_results):
    I, gold_rows_varied = sample_retrieval_results
    # gold_rows_varied = [2, 0, -1, 4]
    # Ranks: 1, 1, missing, 4
    # RR scores: 1, 1, missing, 1/4 -> MRR = (1 + 1 + 0.25) / 3 = 0.75
    # AP scores are the same here.
    
    # Let's modify I to make the last query's gold (4) not in top 3
    I[3] = [1, 0, 3, 4, 2] # gold 4 is at rank 4
    
    metrics = compute_map_mrr(I, gold_rows_varied)
    
    # Valid queries are at index 0, 1, 3. Gold rows are 2, 0, 4.
    # Ranks:
    # Query 0: gold 2 is at rank 1. RR = 1.
    # Query 1: gold 0 is at rank 1. RR = 1.
    # Query 3: gold 4 is at rank 4. RR = 1/4 = 0.25.
    # MRR = (1 + 1 + 0.25) / 3 = 2.25 / 3 = 0.75
    
    # MAP is the same in this case.
    assert pytest.approx(metrics["mrr@10"]) == 0.75
    assert pytest.approx(metrics["map@10"]) == 0.75

def test_compute_map_mrr_no_hits():
    I = np.array([[1, 2, 3], [0, 2, 3]])
    gold_rows = [[0], [1]]
    metrics = compute_map_mrr(I, gold_rows)
    assert metrics["mrr@10"] == 0.0
    assert metrics["map@10"] == 0.0

def test_compute_map_mrr_no_valid_golds():
    I = np.array([[1, 2, 3]])
    gold_rows = [[-1]]
    metrics = compute_map_mrr(I, gold_rows)
    assert metrics["mrr@10"] == 0.0
    assert metrics["map@10"] == 0.0


def test_compute_map_mrr_multiple_golds():
    # One query, two relevant docs at ranks 2 and 4
    I = np.array([[5, 1, 3, 2, 4, 0]])
    gold_rows = [[1, 4]]  # relevant at ranks 2 and 5
    metrics = compute_map_mrr(I, gold_rows)
    # AP: (1/2 + 2/5) / 2 = 0.45; MRR: 1/2
    assert pytest.approx(metrics["map@10"], rel=1e-6) == 0.45
    assert pytest.approx(metrics["mrr@10"], rel=1e-6) == 0.5


def test_compute_skill_coverage_best_of_gold():
    I = np.array([[1, 2, 0]])
    esco_ids = ["a", "b", "c"]
    gold_rows = [[1, 2]]  # gold occupations: b and c
    skills = {
        "a": {"x"},
        "b": {"x"},
        "c": {"y"},
    }
    metrics = compute_skill_coverage(I, gold_rows, esco_ids, skills, ks=(1,))
    # Top-1 predicts "b" which fully covers gold "b"; best-of-gold should be 1.0
    assert metrics["skill_coverage@1"] == 1.0










