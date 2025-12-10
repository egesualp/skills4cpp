"""
Example script demonstrating how to use the on-the-fly CareerPathDataset
for training models without pre-computing embeddings.

This script shows the exact usage pattern requested by the user.
"""

import argparse
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

try:
    from src.cpp.data_classes import Data
    from src.cpp.utils import SEP_TOKEN
    from src.cpp.cpp_dataset import CareerPathDataset, collate_career_path_batch
    from src.cpp.data_loaders import (
        load_all_vocabs,
        load_job_and_skill_data,
        precompute_target_embeddings
    )
except ImportError:
    print("Error: Required modules not found.")
    exit()


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Example training script using on-the-fly dataset."
    )
    
    # Data paths
    parser.add_argument("--data_type", type=str, default="decorte")
    parser.add_argument(
        "--master_skill_file", 
        type=str, 
        default="/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/results/decorte_jobbert_v2_baseline/job_title_skills_master.csv"
    )
    parser.add_argument(
        "--esco_skills_file", 
        type=str, 
        default="data/esco_datasets/skills_en.csv"
    )
    parser.add_argument(
        "--encoder_path", 
        type=str, 
        default="ElenaSenger/career-path-representation-mpnet-decorte"
    )
    parser.add_argument(
        "--vocab_dir", 
        type=str, 
        default="data/processed/master_datasets_2/"
    )
    parser.add_argument(
        "--skill_properties_file", 
        type=str, 
        default="data/processed/master_datasets_2/skill_properties_map.json"
    )
    
    # Feature configuration
    parser.add_argument(
        "--use_skill_description", 
        action='store_true',
        help="Format skills as 'role: [name] \\n description: [desc]'."
    )
    parser.add_argument(
        "--last_job_only", 
        action='store_true',
        help="Only use 'last job -> next job' pairs."
    )
    
    # Pooling configuration
    parser.add_argument(
        "--pooling_strategy", 
        type=str, 
        default="mean",
        choices=["mean", "weighted_mean", "weighted_idf"]
    )
    parser.add_argument(
        "--alpha", 
        type=float, 
        default=1.0,
        help="Exponent for confidence score (c_i)."
    )
    parser.add_argument(
        "--beta", 
        type=float, 
        default=1.0,
        help="Exponent for IDF score (idf_i)."
    )
    
    # Feature modality selection
    parser.add_argument(
        "--include_text", 
        action='store_true',
        default=True,
        help="Include text history features."
    )
    parser.add_argument(
        "--include_skill_text", 
        action='store_true',
        default=True,
        help="Include skill text features."
    )
    parser.add_argument(
        "--include_structured", 
        action='store_true',
        default=True,
        help="Include structured features."
    )
    
    # DataLoader configuration
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="Batch size for training."
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=4,
        help="Number of workers for DataLoader."
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    print("--- Setting up on-the-fly training dataset ---")
    print(f"Configuration: {vars(args)}")
    
    # --- Step 1: Load the encoder model ---
    print("\n[1/6] Loading encoder model...")
    encoder = SentenceTransformer(args.encoder_path)
    
    # --- Step 2: Load all helper maps ---
    print("\n[2/6] Loading vocabularies and skill mappings...")
    all_vocabs = load_all_vocabs(args.vocab_dir)
    
    job_skill_map, esco_skill_text_map, skill_properties_map = load_job_and_skill_data(
        master_skill_file=args.master_skill_file,
        esco_skills_file=args.esco_skills_file,
        skill_properties_file=args.skill_properties_file,
        pooling_strategy=args.pooling_strategy,
        alpha=args.alpha,
        beta=args.beta
    )
    
    # --- Step 3: Load the raw text pairs ---
    print("\n[3/6] Loading career path data...")
    data = Data(DATA_TYPE=args.data_type)
    train_pairs, val_pairs, test_pairs = data.get_data(stage='transformation_finetuning')
    
    # Filter for "last job only" if requested
    if args.last_job_only:
        print(f"  > Filtering for 'last job only' pairs (no '{SEP_TOKEN}' token in history)...")
        train_pairs = [pair for pair in train_pairs if SEP_TOKEN not in pair[0]]
        val_pairs = [pair for pair in val_pairs if SEP_TOKEN not in pair[0]]
        test_pairs = [pair for pair in test_pairs if SEP_TOKEN not in pair[0]]
        print(f"  > Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}")
    
    # --- Step 4: Pre-compute target embeddings (still efficient!) ---
    print("\n[4/6] Pre-computing target embeddings...")
    Y_target_dict = precompute_target_embeddings(encoder, list(data.labels), show_progress=True)
    
    # --- Step 5: Create the on-the-fly Datasets ---
    print("\n[5/6] Creating PyTorch Datasets...")
    train_dataset = CareerPathDataset(
        data_pairs=train_pairs,
        encoder=encoder,
        Y_target_dict=Y_target_dict,
        job_skill_map=job_skill_map,
        esco_skill_text_map=esco_skill_text_map,
        skill_properties_map=skill_properties_map,
        all_vocabs=all_vocabs,
        use_skill_description=args.use_skill_description,
        pooling_strategy=args.pooling_strategy,
        alpha=args.alpha,
        beta=args.beta,
        include_text=args.include_text,
        include_skill_text=args.include_skill_text,
        include_structured=args.include_structured,
    )
    
    val_dataset = CareerPathDataset(
        data_pairs=val_pairs,
        encoder=encoder,
        Y_target_dict=Y_target_dict,
        job_skill_map=job_skill_map,
        esco_skill_text_map=esco_skill_text_map,
        skill_properties_map=skill_properties_map,
        all_vocabs=all_vocabs,
        use_skill_description=args.use_skill_description,
        pooling_strategy=args.pooling_strategy,
        alpha=args.alpha,
        beta=args.beta,
        include_text=args.include_text,
        include_skill_text=args.include_skill_text,
        include_structured=args.include_structured,
    )
    
    test_dataset = CareerPathDataset(
        data_pairs=test_pairs,
        encoder=encoder,
        Y_target_dict=Y_target_dict,
        job_skill_map=job_skill_map,
        esco_skill_text_map=esco_skill_text_map,
        skill_properties_map=skill_properties_map,
        all_vocabs=all_vocabs,
        use_skill_description=args.use_skill_description,
        pooling_strategy=args.pooling_strategy,
        alpha=args.alpha,
        beta=args.beta,
        include_text=args.include_text,
        include_skill_text=args.include_skill_text,
        include_structured=args.include_structured,
    )
    
    print(f"  > Train dataset: {len(train_dataset)} samples")
    print(f"  > Val dataset: {len(val_dataset)} samples")
    print(f"  > Test dataset: {len(test_dataset)} samples")
    
    # --- Step 6: Wrap in DataLoader for multi-core processing ---
    print("\n[6/6] Creating DataLoaders...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_career_path_batch,
        pin_memory=True  # Faster GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,  # Larger batch for validation
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_career_path_batch,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_career_path_batch,
        pin_memory=True
    )
    
    print(f"  > Train batches: {len(train_loader)}")
    print(f"  > Val batches: {len(val_loader)}")
    print(f"  > Test batches: {len(test_loader)}")
    
    # --- Demo: Iterate through one batch ---
    print("\n--- Demo: Fetching one batch from train_loader ---")
    for batch in train_loader:
        print(f"Batch keys: {list(batch.keys())}")
        for key, tensor in batch.items():
            print(f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}")
        break
    
    print("\n--- Setup Complete! ---")
    print("You can now use train_loader, val_loader, and test_loader in your training loop.")
    print("\nExample training loop:")
    print("""
    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            # batch contains: 'h_text', 'h_skill_text', 'h_structured_*', 'y'
            
            # Move to GPU if available
            # batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            # outputs = model(batch)
            # loss = criterion(outputs, batch['y'])
            
            # Backward pass
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            
            pass
    """)


if __name__ == "__main__":
    main()



