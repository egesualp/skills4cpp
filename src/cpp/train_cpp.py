"""
Training script for Career Path Prediction using on-the-fly dataset.

This script demonstrates how to train a model using the CareerPathDataset
which generates embeddings dynamically during training.
"""

import argparse
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

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


class MultiModalCPPModel(nn.Module):
    """
    Example multi-modal model for Career Path Prediction.
    
    Combines text history, skill text, and structured features.
    """
    
    def __init__(self, text_dim, skill_text_dim, structured_dim, hidden_dim, output_dim):
        super(MultiModalCPPModel, self).__init__()
        
        # Feature projections
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.skill_text_proj = nn.Sequential(
            nn.Linear(skill_text_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.structured_proj = nn.Sequential(
            nn.Linear(structured_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, batch):
        """
        Forward pass.
        
        Args:
            batch: Dictionary with keys 'h_text', 'h_skill_text', 'h_structured_*'
        
        Returns:
            Predicted embeddings [batch_size, output_dim]
        """
        # Project each modality
        text_feat = self.text_proj(batch['h_text'])
        skill_text_feat = self.skill_text_proj(batch['h_skill_text'])
        
        # Handle structured features (could have multiple vocab types)
        structured_keys = [k for k in batch.keys() if k.startswith('h_structured_')]
        if structured_keys:
            # Concatenate all structured features if multiple exist
            structured_concat = torch.cat([batch[k] for k in structured_keys], dim=1)
            structured_feat = self.structured_proj(structured_concat)
        else:
            structured_feat = torch.zeros_like(text_feat)
        
        # Fuse all features
        fused = torch.cat([text_feat, skill_text_feat, structured_feat], dim=1)
        output = self.fusion(fused)
        
        return output


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train CPP model with on-the-fly dataset.")
    
    # Data paths
    parser.add_argument("--data_type", type=str, default="decorte")
    parser.add_argument(
        "--master_skill_file", 
        type=str, 
        default="/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/results/decorte_jobbert_v2_baseline/job_title_skills_master.csv"
    )
    parser.add_argument("--esco_skills_file", type=str, default="data/esco_datasets/skills_en.csv")
    parser.add_argument("--encoder_path", type=str, default="ElenaSenger/career-path-representation-mpnet-decorte")
    parser.add_argument("--vocab_dir", type=str, default="data/processed/master_datasets_2/")
    parser.add_argument("--skill_properties_file", type=str, default="data/processed/master_datasets_2/skill_properties_map.json")
    
    # Feature configuration
    parser.add_argument("--use_skill_description", action='store_true')
    parser.add_argument("--last_job_only", action='store_true')
    parser.add_argument("--pooling_strategy", type=str, default="weighted_idf", choices=["mean", "weighted_mean", "weighted_idf"])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    
    # Model configuration
    parser.add_argument("--hidden_dim", type=int, default=512)
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="results/cpp_training")
    parser.add_argument("--checkpoint_freq", type=int, default=1)
    
    return parser.parse_args()


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Forward pass
        outputs = model(batch)
        loss = criterion(outputs, batch['y'])
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch)
            loss = criterion(outputs, batch['y'])
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    args = parse_args()
    print("=" * 80)
    print("Training Career Path Prediction Model with On-the-Fly Dataset")
    print("=" * 80)
    print(f"Configuration: {vars(args)}\n")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}\n")
    
    # --- Step 1: Load encoder ---
    print("[1/6] Loading encoder model...")
    encoder = SentenceTransformer(args.encoder_path)
    embed_dim = encoder.get_sentence_embedding_dimension()
    print(f"  ✓ Loaded encoder with embedding dim: {embed_dim}\n")
    
    # --- Step 2: Load helper maps ---
    print("[2/6] Loading vocabularies and skill mappings...")
    all_vocabs = load_all_vocabs(args.vocab_dir)
    
    job_skill_map, esco_skill_text_map, skill_properties_map = load_job_and_skill_data(
        master_skill_file=args.master_skill_file,
        esco_skills_file=args.esco_skills_file,
        skill_properties_file=args.skill_properties_file,
        pooling_strategy=args.pooling_strategy,
        alpha=args.alpha,
        beta=args.beta
    )
    print()
    
    # --- Step 3: Load data pairs ---
    print("[3/6] Loading career path data...")
    data = Data(DATA_TYPE=args.data_type)
    train_pairs, val_pairs, test_pairs = data.get_data(stage='transformation_finetuning')
    
    if args.last_job_only:
        print(f"  > Filtering for 'last job only' pairs...")
        train_pairs = [pair for pair in train_pairs if SEP_TOKEN not in pair[0]]
        val_pairs = [pair for pair in val_pairs if SEP_TOKEN not in pair[0]]
        test_pairs = [pair for pair in test_pairs if SEP_TOKEN not in pair[0]]
    
    print(f"  ✓ Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}\n")
    
    # --- Step 4: Pre-compute target embeddings ---
    print("[4/6] Pre-computing target embeddings...")
    Y_target_dict = precompute_target_embeddings(encoder, list(data.labels), show_progress=True)
    print()
    
    # --- Step 5: Create datasets and dataloaders ---
    print("[5/6] Creating datasets and dataloaders...")
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
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_career_path_batch,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_career_path_batch,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f"  ✓ Train batches: {len(train_loader)}, Val batches: {len(val_loader)}\n")
    
    # --- Step 6: Initialize model ---
    print("[6/6] Initializing model...")
    
    # Get structured feature dimension
    structured_dim = sum(len(vocab) for vocab in all_vocabs.values())
    
    model = MultiModalCPPModel(
        text_dim=embed_dim,
        skill_text_dim=embed_dim,
        structured_dim=structured_dim,
        hidden_dim=args.hidden_dim,
        output_dim=embed_dim
    ).to(device)
    
    print(f"  ✓ Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"    - Text dim: {embed_dim}")
    print(f"    - Skill text dim: {embed_dim}")
    print(f"    - Structured dim: {structured_dim}")
    print(f"    - Hidden dim: {args.hidden_dim}")
    print(f"    - Output dim: {embed_dim}\n")
    
    # Initialize optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # --- Training loop ---
    print("=" * 80)
    print("Starting Training")
    print("=" * 80)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")
        print("-" * 40)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        print(f"  Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        print(f"  Val Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if epoch % args.checkpoint_freq == 0 or val_loss < best_val_loss:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'args': vars(args)
            }
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = os.path.join(args.output_dir, 'best_model.pt')
                torch.save(checkpoint, checkpoint_path)
                print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
            
            if epoch % args.checkpoint_freq == 0:
                checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pt')
                torch.save(checkpoint, checkpoint_path)
                print(f"  ✓ Saved checkpoint: {checkpoint_path}")
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()



