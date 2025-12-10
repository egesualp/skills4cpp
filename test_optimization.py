"""
Quick test to verify the skill embedding optimization works correctly.
This compares the old (slow) method with the new (fast) method.
"""

import numpy as np
import time
from src.cpp.data_loaders import (
    extract_unique_skills_from_dataset,
    precompute_skill_embeddings,
    _pooled_skill_vec,
    _pooled_skill_vec_optimized,
    _extract_skill_infos
)

def test_optimization(job_skill_map, esco_skill_text_map, encoder_skill, 
                     data_pairs, use_skill_description=False, 
                     pooling_strategy="mean", alpha=1.0, beta=1.0):
    """
    Test that the optimized version produces the same results as the old version,
    but much faster.
    """
    print("\n" + "="*80)
    print("TESTING SKILL EMBEDDING OPTIMIZATION")
    print("="*80)
    
    # Get a subset of data for testing (to keep test fast)
    test_pairs = data_pairs[:100]  # Test on first 100 samples
    
    embed_dim = encoder_skill.get_sentence_embedding_dimension()
    
    # ============================================================================
    # OLD METHOD (SLOW)
    # ============================================================================
    print("\n1. OLD METHOD (encoding skills repeatedly)...")
    start_time = time.time()
    
    old_results = []
    total_encodings_old = 0
    for h, _ in test_pairs:
        infos = _extract_skill_infos(h, job_skill_map)
        total_encodings_old += len(infos)
        vec = _pooled_skill_vec(
            infos, encoder_skill, esco_skill_text_map, use_skill_description,
            pooling_strategy, alpha, beta, embed_dim
        )
        old_results.append(vec)
    
    old_time = time.time() - start_time
    old_results = np.array(old_results)
    
    print(f"   ⏱️  Time: {old_time:.2f} seconds")
    print(f"   📊 Total encoding operations: {total_encodings_old}")
    print(f"   📐 Result shape: {old_results.shape}")
    
    # ============================================================================
    # NEW METHOD (FAST)
    # ============================================================================
    print("\n2. NEW METHOD (pre-encode unique skills once)...")
    start_time = time.time()
    
    # Step 1: Extract unique skills
    unique_skills = extract_unique_skills_from_dataset(test_pairs, job_skill_map)
    print(f"   🔍 Unique skills found: {len(unique_skills)}")
    
    # Step 2: Pre-encode all unique skills
    skill_embedding_map = precompute_skill_embeddings(
        unique_skills, encoder_skill, esco_skill_text_map, use_skill_description
    )
    
    # Step 3: Process samples using pre-computed embeddings
    new_results = []
    for h, _ in test_pairs:
        infos = _extract_skill_infos(h, job_skill_map)
        vec = _pooled_skill_vec_optimized(
            infos, skill_embedding_map, pooling_strategy, alpha, beta, embed_dim
        )
        new_results.append(vec)
    
    new_time = time.time() - start_time
    new_results = np.array(new_results)
    
    print(f"   ⏱️  Time: {new_time:.2f} seconds")
    print(f"   📊 Unique encoding operations: {len(unique_skills)}")
    print(f"   📐 Result shape: {new_results.shape}")
    
    # ============================================================================
    # COMPARISON
    # ============================================================================
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    # Check if results are the same
    max_diff = np.abs(old_results - new_results).max()
    mean_diff = np.abs(old_results - new_results).mean()
    
    print(f"\n✅ Correctness Check:")
    print(f"   Max difference: {max_diff:.10f}")
    print(f"   Mean difference: {mean_diff:.10f}")
    
    if max_diff < 1e-5:
        print(f"   ✓ Results are identical (difference < 1e-5)")
    else:
        print(f"   ⚠️  Warning: Results differ by {max_diff}")
    
    # Performance comparison
    speedup = old_time / new_time
    efficiency_gain = total_encodings_old / len(unique_skills)
    
    print(f"\n⚡ Performance Comparison:")
    print(f"   Old method: {old_time:.2f}s ({total_encodings_old} encodings)")
    print(f"   New method: {new_time:.2f}s ({len(unique_skills)} encodings)")
    print(f"   Speedup: {speedup:.2f}x faster")
    print(f"   Efficiency gain: {efficiency_gain:.2f}x fewer encodings")
    
    print("\n" + "="*80)
    print(f"✅ OPTIMIZATION TEST {'PASSED' if max_diff < 1e-5 else 'FAILED'}")
    print("="*80 + "\n")
    
    return max_diff < 1e-5


if __name__ == "__main__":
    print("\nThis is a test utility script.")
    print("To run it, you need to:")
    print("1. Load your encoders, job_skill_map, esco_skill_text_map")
    print("2. Load your data_pairs")
    print("3. Call test_optimization() with your data")
    print("\nExample usage:")
    print("""
    from sentence_transformers import SentenceTransformer
    from src.cpp.data_loaders import load_job_and_skill_data
    from src.cpp.data_classes import Data
    
    # Load encoder
    encoder = SentenceTransformer('your-model-name')
    
    # Load data
    job_skill_map, esco_skill_text_map, _ = load_job_and_skill_data(...)
    data = Data(...)
    train_pairs, _, _ = data.get_data(stage='transformation_finetuning')
    
    # Run test
    test_optimization(job_skill_map, esco_skill_text_map, encoder, train_pairs)
    """)








