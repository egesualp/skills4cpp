import sys
import os
import numpy as np
from typing import Dict, List, Set

# Add src to path to allow imports
sys.path.append(os.path.abspath("src"))

from skill_mapping.v2.bayesian_fuser import (
    prepare_vectorized_data, 
    compute_fused_scores_vectorized, 
    FusionConfig,
    VectorizedData
)

def main():
    print("=== Setting up Mock Data ===")
    
    # 1. Mock Similarity Scores (IR Scores)
    # job1 has 3 candidate skills
    similarity_scores = {
        "job1": [
            {"skill_uri": "skill_A", "score": 0.9, "rank": 1}, # High IR score
            {"skill_uri": "skill_B", "score": 0.8, "rank": 2}, # Medium IR score
            {"skill_uri": "skill_C", "score": 0.7, "rank": 3}  # Low IR score
        ]
    }
    
    # 2. Mock Category Scores (Logits for the job)
    # job1 seems to be related to cat_1 and maybe cat_2
    category_scores_raw = {
        "job1": [
            {"category": "cat_1", "score": 2.0}, # High confidence
            {"category": "cat_2", "score": 1.0}, # Medium confidence
            {"category": "cat_3", "score": -1.0} # Low confidence
        ]
    }
    
    # 3. Mock Skill-to-Category Mapping
    # skill_A -> cat_1 (Single)
    # skill_B -> cat_1, cat_2 (Multiple - this is what we want to test)
    # skill_C -> (No category)
    skill_to_categories = {
        "skill_A": {"cat_1"},
        "skill_B": {"cat_1", "cat_2"},
        "skill_C": set()
    }
    
    print("\nJob: job1")
    print("Candidate Skills:")
    for s in similarity_scores["job1"]:
        cats = skill_to_categories.get(s["skill_uri"], set())
        print(f"  - {s['skill_uri']}: IR Score={s['score']}, Categories={cats}")
        
    print("\nJob Category Logits:")
    for c in category_scores_raw["job1"]:
        print(f"  - {c['category']}: {c['score']}")

    # Prepare Vectorized Data
    print("\n=== Preparing Vectorized Data ===")
    vd = prepare_vectorized_data(similarity_scores, category_scores_raw, skill_to_categories)
    
    print(f"Skill URIs: {vd.skill_uris}")
    print(f"Category Names: {vd.category_names}")
    print("Skill-Category Matrix:")
    print(vd.skill_category_matrix)
    
    # Test Aggregation Modes
    print("\n=== Testing Aggregation Modes ===")
    
    # Define a config
    # We use linear fusion to see the effect clearly: score = 0 * sim + 1.0 * cat_prob
    # This way the final score IS the aggregated category probability.
    # Temperature = 1.0 for simple softmax.
    
    # Calculate expected probabilities manually for verification
    logits = np.array([2.0, 1.0, -1.0])
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum()
    print(f"\nManual Probability Calculation (T=1.0):")
    for i, cat in enumerate(vd.category_names):
        # We need to match the order in vectorized data
        # vd.category_names might be sorted or in order of appearance
        # Let's find the index in probs corresponding to the name
        # Note: In prepare_vectorized_data, categories are taken from the first job's list.
        # Our mock list is [cat_1, cat_2, cat_3].
        print(f"  P({cat}) = {probs[i]:.4f}")
        
    p_cat1 = probs[0] # assuming order is preserved
    p_cat2 = probs[1]
    p_cat3 = probs[2]
    
    modes = ["max", "sum"]
    
    for mode in modes:
        print(f"\n--- Testing '{mode}' Aggregation ---")
        
        # We set alpha=1.0 so the result is purely the aggregated category score
        config = FusionConfig(
            fusion_mode="linear", 
            alpha=1.0, 
            temperature=1.0, 
            threshold=0.0,
            aggregate_mode=mode
        )
        
        # Run fusion
        _, top_k_indices, top_k_scores, _ = compute_fused_scores_vectorized(
            vd, config, top_k=3, aggregate_mode=mode
        )
        
        # Get results for job1 (index 0)
        indices = top_k_indices[0]
        scores = top_k_scores[0]
        
        print(f"Results for job1 (alpha=1.0 -> purely aggregated category score):")
        for idx, score in zip(indices, scores):
            skill_name = vd.skill_uris[idx]
            print(f"  Skill: {skill_name}, Aggregated Score: {score:.6f}")
            
            # Verify manual calculation
            if skill_name == "skill_A":
                # Belongs to cat_1
                expected = p_cat1
                print(f"    -> Expected: P(cat_1) = {expected:.6f}")
            elif skill_name == "skill_B":
                # Belongs to cat_1 and cat_2
                if mode == "max":
                    expected = max(p_cat1, p_cat2)
                    print(f"    -> Expected: max(P(cat_1), P(cat_2)) = {expected:.6f}")
                else:
                    expected = p_cat1 + p_cat2
                    print(f"    -> Expected: P(cat_1) + P(cat_2) = {expected:.6f}")
            elif skill_name == "skill_C":
                # No category
                print(f"    -> Expected: 0.000000 (No category)")

if __name__ == "__main__":
    main()









