"""
Quick test script to verify the CareerPathDataset works correctly.
"""

import sys
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

try:
    from src.cpp.data_classes import Data
    from src.cpp.cpp_dataset import CareerPathDataset, collate_career_path_batch
    from src.cpp.data_loaders import (
        load_all_vocabs,
        load_job_and_skill_data,
        precompute_target_embeddings
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)


def test_dataset():
    """Run a quick test of the dataset functionality."""
    print("=" * 80)
    print("Testing CareerPathDataset")
    print("=" * 80)
    
    # Configuration
    data_type = "decorte"
    master_skill_file = "/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/results/decorte_jobbert_v2_baseline/job_title_skills_master.csv"
    esco_skills_file = "data/esco_datasets/skills_en.csv"
    vocab_dir = "data/processed/master_datasets_2/"
    skill_properties_file = "data/processed/master_datasets_2/skill_properties_map.json"
    encoder_path = "ElenaSenger/career-path-representation-mpnet-decorte"
    
    try:
        # Step 1: Load encoder
        print("\n[1/5] Loading encoder...")
        encoder = SentenceTransformer(encoder_path)
        print(f"  ✓ Loaded encoder with embedding dim: {encoder.get_sentence_embedding_dimension()}")
        
        # Step 2: Load helper maps
        print("\n[2/5] Loading helper maps...")
        all_vocabs = load_all_vocabs(vocab_dir)
        print(f"  ✓ Loaded {len(all_vocabs)} vocabularies")
        
        job_skill_map, esco_skill_text_map, skill_properties_map = load_job_and_skill_data(
            master_skill_file=master_skill_file,
            esco_skills_file=esco_skills_file,
            skill_properties_file=skill_properties_file,
            pooling_strategy="weighted_idf",
            alpha=1.0,
            beta=1.0
        )
        print(f"  ✓ Loaded job_skill_map: {len(job_skill_map)} jobs")
        print(f"  ✓ Loaded esco_skill_text_map: {len(esco_skill_text_map)} skills")
        print(f"  ✓ Loaded skill_properties_map: {len(skill_properties_map)} skills")
        
        # Step 3: Load data
        print("\n[3/5] Loading career path data...")
        data = Data(DATA_TYPE=data_type)
        train_pairs, val_pairs, test_pairs = data.get_data(stage='transformation_finetuning')
        print(f"  ✓ Train pairs: {len(train_pairs)}")
        print(f"  ✓ Val pairs: {len(val_pairs)}")
        print(f"  ✓ Test pairs: {len(test_pairs)}")
        
        # Step 4: Pre-compute targets
        print("\n[4/5] Pre-computing target embeddings...")
        Y_target_dict = precompute_target_embeddings(encoder, list(data.labels), show_progress=False)
        print(f"  ✓ Pre-computed {len(Y_target_dict)} target embeddings")
        
        # Step 5: Create dataset and test it
        print("\n[5/5] Creating dataset and testing...")
        
        # Use a small subset for quick testing
        test_pairs_small = train_pairs[:100]
        
        dataset = CareerPathDataset(
            data_pairs=test_pairs_small,
            encoder=encoder,
            Y_target_dict=Y_target_dict,
            job_skill_map=job_skill_map,
            esco_skill_text_map=esco_skill_text_map,
            skill_properties_map=skill_properties_map,
            all_vocabs=all_vocabs,
            use_skill_description=False,
            pooling_strategy="weighted_idf",
            alpha=1.0,
            beta=1.0,
        )
        print(f"  ✓ Created dataset with {len(dataset)} samples")
        
        # Test __getitem__
        print("\n  Testing __getitem__...")
        sample = dataset[0]
        print(f"    Sample keys: {list(sample.keys())}")
        for key, tensor in sample.items():
            print(f"    {key}: shape={tensor.shape}, dtype={tensor.dtype}")
        
        # Test DataLoader
        print("\n  Testing DataLoader...")
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,  # Use 0 for testing (single process)
            collate_fn=collate_career_path_batch
        )
        
        batch = next(iter(loader))
        print(f"    Batch keys: {list(batch.keys())}")
        for key, tensor in batch.items():
            print(f"    {key}: shape={tensor.shape}, dtype={tensor.dtype}")
        
        # Test with multiple workers
        print("\n  Testing DataLoader with multiple workers...")
        loader_multicore = DataLoader(
            dataset,
            batch_size=8,
            shuffle=True,
            num_workers=2,
            collate_fn=collate_career_path_batch
        )
        
        batch_multicore = next(iter(loader_multicore))
        print(f"    Batch keys: {list(batch_multicore.keys())}")
        for key, tensor in batch_multicore.items():
            print(f"    {key}: shape={tensor.shape}, dtype={tensor.dtype}")
        
        print("\n" + "=" * 80)
        print("✅ All tests passed!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_dataset()
    sys.exit(0 if success else 1)



