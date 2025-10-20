#!/usr/bin/env python3
"""
Test script for skill coverage metrics.
"""

import numpy as np
from src.metrics import load_skills_per_occupation, compute_skill_coverage

def test_skill_coverage():
    """Test the skill coverage metric computation."""
    
    print("Loading skills data...")
    skills_path = "data/skills_per_occupations.csv"
    skills_by_occupation = load_skills_per_occupation(skills_path)
    
    print(f"Loaded skills for {len(skills_by_occupation)} occupations")
    
    # Get some sample occupation URIs
    sample_occupations = list(skills_by_occupation.keys())[:10]
    
    # Show skills for a sample occupation
    sample_occ = sample_occupations[0]
    sample_skills = skills_by_occupation[sample_occ]
    print(f"\nSample occupation: {sample_occ}")
    print(f"Number of skills: {len(sample_skills)}")
    
    # Create synthetic test data
    n_queries = 5
    n_candidates = len(sample_occupations)
    
    # Create rankings where first query matches perfectly, 
    # second query has partial match, etc.
    I = np.zeros((n_queries, n_candidates), dtype=int)
    
    for i in range(n_queries):
        # Create a permutation for each query
        I[i] = np.arange(n_candidates)
        # Shuffle to simulate different rankings
        if i > 0:
            np.random.shuffle(I[i])
    
    # Gold rows (ground truth indices)
    gold_rows = [0, 1, 2, 3, 4]  # Each query's gold is at a different position
    
    print("\nComputing skill coverage metrics...")
    coverage_metrics = compute_skill_coverage(
        I, gold_rows, sample_occupations, skills_by_occupation,
        ks=(1, 3, 5, 10)
    )
    
    print("\nSkill Coverage Results:")
    for metric_name, value in coverage_metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    # Verify that coverage@k increases with k (or stays the same)
    coverage_values = [coverage_metrics[f"skill_coverage@{k}"] for k in [1, 3, 5, 10]]
    assert all(coverage_values[i] <= coverage_values[i+1] for i in range(len(coverage_values)-1)), \
        "Coverage should increase or stay the same as k increases"
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    test_skill_coverage()
