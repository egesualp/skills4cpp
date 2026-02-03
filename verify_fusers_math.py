
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.getcwd() + "/src")

from skill_mapping.v5.fused_scorer import FusedScorer as OriginalScorer
from skill_mapping.v5.fused_scorer_chunked import FusedScorer as ChunkedScorer

def run_sanity_check():
    # Paths
    esco_dir = Path("data/esco_datasets")
    label_encoder = Path("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h1_soft_deep_larger_val_mpnet/label_encoder.json")
    
    # Task B data for testing (minimal)
    test_job_id = "0"
    test_task_a_occs = ["http://data.europa.eu/esco/occupation/90f75f67-495d-49fa-ab57-2f320e251d7e"] # cooks
    test_task_b_scores = {
        "http://data.europa.eu/esco/skill/32adf439-5bbc-4759-8f05-fbaacdac60a6": 0.88,
        "http://data.europa.eu/esco/skill/9ccd8da7-733b-4aff-a49f-94c390aeb618": 0.85
    }
    
    print("Initializing Scorers...")
    orig_scorer = OriginalScorer(esco_dir, label_encoder, isco_level=1)
    chunk_scorer = ChunkedScorer(esco_dir, label_encoder, isco_level=1)

    orig_scorer.build_lookup_tables()
    chunk_scorer.build_lookup_tables()

    orig_scorer.build_affinity_matrix(mode='uniform')
    chunk_scorer.build_affinity_matrix(mode='uniform')

    # Mock ISCO probs distribution
    n_classes = len(orig_scorer.isco_index)
    isco_probs = np.zeros(n_classes, dtype=np.float32)
    isco_probs[0] = 1.0 # Assume first class

    print("\n--- Test 1: Direct math comparison for score_job ---")
    alpha, gamma, epsilon = 1.0, 1.0, 0.01
    
    res_orig, _ = orig_scorer.score_job(
        test_task_a_occs, 
        test_task_b_scores, 
        isco_probs, 
        alpha=alpha, 
        gamma=gamma, 
        epsilon=epsilon
    )
    
    res_chunk, _ = chunk_scorer.score_job(
        test_task_a_occs, 
        test_task_b_scores, 
        isco_probs, 
        alpha=alpha, 
        gamma=gamma, 
        epsilon=epsilon
    )

    print(f"Original Result: {res_orig}")
    print(f"Chunked Result:  {res_chunk}")
    
    if len(res_orig) == len(res_chunk):
        diffs = []
        for i in range(len(res_orig)):
            if res_orig[i][0] != res_chunk[i][0]:
                print(f"Mismatch at rank {i}: {res_orig[i][0]} vs {res_chunk[i][0]}")
            diffs.append(abs(res_orig[i][1] - res_chunk[i][1]))
        print(f"Max score difference: {max(diffs)}")
    else:
        print(f"Result length mismatch: {len(res_orig)} vs {len(res_chunk)}")

    print("\n--- Test 2: Checking candidate generation ---")
    # Check if they get the same candidates for a given occupation
    cand_orig = set()
    for occ in test_task_a_occs:
        cand_orig.update(orig_scorer.occ_to_skills.get(occ, set()))
    
    cand_chunk = set()
    for occ in test_task_a_occs:
        cand_chunk.update(chunk_scorer.occ_to_skills.get(occ, set()))
        
    print(f"Orig Candidates count: {len(cand_orig)}")
    print(f"Chunk Candidates count: {len(cand_chunk)}")
    print(f"Intersection count: {len(cand_orig & cand_chunk)}")

    print("\n--- Test 3: Checking affinity matrix values ---")
    skill_uri = list(test_task_b_scores.keys())[0]
    aff_orig = orig_scorer.get_skill_affinity_vector(skill_uri)
    aff_chunk = chunk_scorer.get_skill_affinity_vector(skill_uri)
    
    if aff_orig is not None and aff_chunk is not None:
        print(f"Affinity Match: {np.allclose(aff_orig, aff_chunk)}")
    else:
        print(f"One affinity vector is None: Orig={aff_orig is None}, Chunk={aff_chunk is None}")

if __name__ == "__main__":
    run_sanity_check()
