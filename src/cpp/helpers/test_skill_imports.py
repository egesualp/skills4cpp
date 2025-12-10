"""
Simple validation script to test that skill-based training components import correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.cpp.skill_dataset import (
            SkillBasedCareerPathDataset,
            ISCOGroupBatchSampler,
            collate_skill_batch
        )
        print("  ✓ skill_dataset imports successful")
    except ImportError as e:
        print(f"  ✗ Error importing skill_dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        from src.cpp.train_cpp_skills import (
            load_skill_mappings,
            load_skill_descriptions,
            load_occupation_isco_groups,
            calculate_idf_scores,
            encode_skills,
            pool_skills_with_idf,
            pool_jobs_with_log_decay
        )
        print("  ✓ train_cpp_skills imports successful")
    except ImportError as e:
        print(f"  ✗ Error importing train_cpp_skills: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    try:
        from src.cpp.data_classes import Data
        from src.cpp.utils import SEP_TOKEN
        print("  ✓ Existing modules imports successful")
    except ImportError as e:
        print(f"  ✗ Error importing existing modules: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ All imports successful!")
    return True


def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\nTesting basic functionality...")
    
    try:
        import numpy as np
        from src.cpp.train_cpp_skills import pool_jobs_with_log_decay
        
        # Test logarithmic pooling
        job_vectors = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0])
        ]
        
        result = pool_jobs_with_log_decay(job_vectors, alpha=0.5)
        
        assert result is not None, "Pooling returned None"
        assert result.shape == (3,), f"Expected shape (3,), got {result.shape}"
        
        # Last job should have highest contribution
        # Since weights are log(1 + 0.5*i), the last element should be strongest
        assert result[2] > result[0], "Last job should have higher weight"
        
        print("  ✓ Logarithmic pooling works correctly")
        
    except Exception as e:
        print(f"  ✗ Error in pooling test: {e}")
        return False
    
    try:
        from src.cpp.train_cpp_skills import pool_skills_with_idf
        
        # Test IDF pooling
        skill_embeddings = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ])
        idf_weights = np.array([1.0, 2.0, 1.5])
        
        result = pool_skills_with_idf(skill_embeddings, idf_weights)
        
        assert result is not None, "Pooling returned None"
        assert result.shape == (2,), f"Expected shape (2,), got {result.shape}"
        
        print("  ✓ IDF-weighted pooling works correctly")
        
    except Exception as e:
        print(f"  ✗ Error in IDF pooling test: {e}")
        return False
    
    print("\n✓ All functionality tests passed!")
    return True


def test_dataset_creation():
    """Test that dataset can be created with minimal data."""
    print("\nTesting dataset creation...")
    
    try:
        from src.cpp.skill_dataset import SkillBasedCareerPathDataset
        
        # Create minimal test data
        data_pairs = [
            ("role: software engineer\n description: test", 
             "esco role: data scientist\n description: test desc")
        ]
        
        job_skill_map = {
            "software engineer": [
                {"skill": "python", "skillUri": "http://test/1", "score": 1.0, "idf": 1.5}
            ]
        }
        
        target_occupation_map = {
            "esco role: data scientist\n description: test desc": {
                "title": "data scientist",
                "description": "test desc",
                "isco_group": "2619"
            }
        }
        
        dataset = SkillBasedCareerPathDataset(
            data_pairs=data_pairs,
            job_skill_map=job_skill_map,
            target_occupation_map=target_occupation_map,
            sep_token="<SEP>"
        )
        
        assert len(dataset) == 1, f"Expected 1 sample, got {len(dataset)}"
        
        sample = dataset[0]
        assert 'job_skills_list' in sample, "Missing job_skills_list"
        assert 'target_title' in sample, "Missing target_title"
        assert 'target_isco_group' in sample, "Missing target_isco_group"
        
        print("  ✓ Dataset creation works correctly")
        print(f"    Sample keys: {list(sample.keys())}")
        
    except Exception as e:
        print(f"  ✗ Error in dataset creation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ Dataset creation test passed!")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Skill-Based Training Validation Tests")
    print("=" * 60)
    print()
    
    all_passed = True
    
    if not test_imports():
        all_passed = False
    
    if not test_basic_functionality():
        all_passed = False
    
    if not test_dataset_creation():
        all_passed = False
    
    print()
    print("=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

