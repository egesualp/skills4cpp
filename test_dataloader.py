"""
Diagnostic script to test dataloader functionality for train_cpp_enhanced.py
"""

import sys
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
import numpy as np

try:
    from src.cpp.data_classes import Data
    from src.cpp.utils import SEP_TOKEN
    from src.cpp.cpp_dataset import CareerPathDataset, collate_career_path_batch
    from src.cpp.data_loaders import (
        load_all_vocabs,
        load_job_and_skill_data,
        precompute_target_embeddings
    )
except ImportError as e:
    print(f"Error: Required modules not found. {e}")
    sys.exit(1)


def test_dataloader():
    print("=" * 80)
    print("DATALOADER DIAGNOSTIC TEST")
    print("=" * 80)
    
    # Configuration matching your command
    data_type = "decorte"
    use_text_description = False  # default (titles only)
    batch_size = 32
    encoder_text_name = "ElenaSenger/career-path-representation-mpnet-decorte"
    master_skill_file = "/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/results/decorte_jobbert_v2_baseline/job_title_skills_master.csv"
    esco_skills_file = "data/esco_datasets/skills_en.csv"
    vocab_dir = "data/processed/master_datasets_2/"
    skill_properties_file = "data/processed/master_datasets_2/skill_properties_map.json"
    
    # Step 1: Load encoder
    print("\n[1/6] Loading encoder...")
    encoder_text = SentenceTransformer(encoder_text_name)
    text_dim = encoder_text.get_sentence_embedding_dimension()
    print(f"  ✓ Text encoder dim: {text_dim}")
    
    # Step 2: Load vocabularies
    print("\n[2/6] Loading vocabularies...")
    all_vocabs = load_all_vocabs(vocab_dir)
    structured_dim = sum(len(vocab) for vocab in all_vocabs.values())
    print(f"  ✓ Structured feature dim: {structured_dim}")
    print(f"  ✓ Vocab categories: {list(all_vocabs.keys())}")
    
    # Step 3: Load skill mappings
    print("\n[3/6] Loading skill mappings...")
    job_skill_map, esco_skill_text_map, skill_properties_map = load_job_and_skill_data(
        master_skill_file=master_skill_file,
        esco_skills_file=esco_skills_file,
        skill_properties_file=skill_properties_file,
        pooling_strategy="weighted_idf",
        alpha=1.0,
        beta=1.0
    )
    print(f"  ✓ Loaded {len(job_skill_map)} job-skill mappings")
    print(f"  ✓ Loaded {len(esco_skill_text_map)} skill text mappings")
    print(f"  ✓ Loaded {len(skill_properties_map)} skill properties")
    
    # Step 4: Load data pairs
    print("\n[4/6] Loading career path data...")
    data = Data(DATA_TYPE=data_type, ONLY_TITLES=not use_text_description)
    train_pairs, val_pairs, test_pairs = data.get_data(stage='transformation_finetuning')
    
    print(f"  ✓ Train pairs: {len(train_pairs)}")
    print(f"  ✓ Val pairs: {len(val_pairs)}")
    print(f"  ✓ Test pairs: {len(test_pairs)}")
    
    if len(train_pairs) > 0:
        print(f"\n  Sample train pair:")
        print(f"    Input: {train_pairs[0][0][:100]}...")
        print(f"    Target: {train_pairs[0][1]}")
    
    # Step 5: Pre-compute target embeddings
    print("\n[5/6] Pre-computing target embeddings...")
    actual_labels = list(set([pair[1] for pair in train_pairs + val_pairs + test_pairs]))
    print(f"  ✓ Found {len(actual_labels)} unique target labels")
    
    Y_target_dict = precompute_target_embeddings(encoder_text, actual_labels, show_progress=True)
    Y_target_all = np.array(list(Y_target_dict.values()))
    output_dim = Y_target_all.shape[1]
    print(f"  ✓ Target embedding dim: {output_dim}")
    
    # Step 6: Create datasets and dataloaders
    print("\n[6/6] Creating datasets and dataloaders...")
    
    train_dataset = CareerPathDataset(
        data_pairs=train_pairs,
        encoder=encoder_text,
        Y_target_dict=Y_target_dict,
        job_skill_map=job_skill_map,
        esco_skill_text_map=esco_skill_text_map,
        skill_properties_map=skill_properties_map,
        all_vocabs=all_vocabs,
        use_skill_description=False,
        pooling_strategy="weighted_idf",
        alpha=1.0,
        beta=1.0,
        encoder_skill=encoder_text,
        include_text=True,  # --use_text_history
        include_skill_text=False,
        include_structured=False,
    )
    
    val_dataset = CareerPathDataset(
        data_pairs=val_pairs,
        encoder=encoder_text,
        Y_target_dict=Y_target_dict,
        job_skill_map=job_skill_map,
        esco_skill_text_map=esco_skill_text_map,
        skill_properties_map=skill_properties_map,
        all_vocabs=all_vocabs,
        use_skill_description=False,
        pooling_strategy="weighted_idf",
        alpha=1.0,
        beta=1.0,
        encoder_skill=encoder_text,
        include_text=True,
        include_skill_text=False,
        include_structured=False,
    )
    
    print(f"  ✓ Train dataset size: {len(train_dataset)}")
    print(f"  ✓ Val dataset size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Use 0 for debugging
        collate_fn=collate_career_path_batch,
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size * 2, 
        shuffle=False,
        num_workers=0,
        collate_fn=collate_career_path_batch,
    )
    
    print(f"  ✓ Train batches (expected): {len(train_loader)}")
    print(f"  ✓ Val batches (expected): {len(val_loader)}")
    
    # Test loading a batch
    print("\n" + "=" * 80)
    print("TESTING BATCH LOADING")
    print("=" * 80)
    
    try:
        print("\nLoading first training batch...")
        batch = next(iter(train_loader))
        print(f"  ✓ Batch loaded successfully!")
        print(f"  ✓ Batch keys: {list(batch.keys())}")
        print(f"  ✓ Batch size: {batch['y'].shape[0]}")
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"    - {key}: shape {value.shape}, dtype {value.dtype}")
        
        print("\n✅ SUCCESS: Dataloaders are working correctly!")
        print(f"   - Train dataset: {len(train_dataset)} samples → {len(train_loader)} batches")
        print(f"   - Val dataset: {len(val_dataset)} samples → {len(val_loader)} batches")
        
    except Exception as e:
        print(f"\n❌ ERROR loading batch: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = test_dataloader()
    sys.exit(0 if success else 1)


















