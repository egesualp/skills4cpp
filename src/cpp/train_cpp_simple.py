"""
Simple Training Script for Career Path Prediction (MLP Baseline).

Reproduces MLP metrics with:
- Simple concatenation architecture
- Fixed hyperparameters
- Early stopping
- WandB integration
- Ranking metrics (MRR, Recall@K)

Similar in simplicity to vector_transformation.py but for CPP task.
"""

import argparse
import os
import sys
import time
from typing import Dict, List
import yaml

from loguru import logger
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("WandB not available. Install with: pip install wandb")

from src.cpp.data_classes import Data
from src.cpp.cpp_dataset import CareerPathDataset, collate_career_path_batch
from src.cpp.data_loaders import (
    load_all_vocabs,
    load_job_and_skill_data,
    precompute_target_embeddings,
    precompute_input_embeddings,
)
from src.cpp.utils import SEP_TOKEN

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)


# ============================================================================
# MODEL
# ============================================================================

class SimpleCPPModel(nn.Module):
    """Simple concatenation MLP for career path prediction."""
    
    def __init__(self, input_dim, output_dim, hidden_dim=512, n_layers=2, dropout=0.1):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, output_dim))
        self.model = nn.Sequential(*layers)
    
    def forward(self, batch):
        """Forward pass - concatenate all features."""
        features = []
        
        # Text history
        if 'h_text' in batch:
            features.append(batch['h_text'])
        
        # Skill text
        if 'h_skill_text' in batch:
            features.append(batch['h_skill_text'])
        
        # Structured features
        structured_keys = [k for k in batch.keys() if k.startswith('h_structured_')]
        if structured_keys:
            features.extend([batch[k] for k in structured_keys])
        
        x = torch.cat(features, dim=1)
        return self.model(x)


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def calculate_ranking_metrics_gpu(y_pred_vectors, y_true_vectors, Y_target_all, device):
    """
    GPU-optimized ranking metrics calculation.
    
    Returns:
        Dictionary with MRR and Recall@K metrics
    """
    # Move target embeddings to GPU
    Y_target_tensor = torch.from_numpy(Y_target_all).to(device)
    
    # Calculate cosine similarity on GPU
    y_pred_norm = torch.norm(y_pred_vectors, dim=1, keepdim=True)
    y_target_norm = torch.norm(Y_target_tensor, dim=1, keepdim=True).t()
    
    sim_matrix = torch.mm(y_pred_vectors, Y_target_tensor.t()) / (y_pred_norm @ y_target_norm + 1e-8)
    
    # Sort indices in descending order
    sorted_indices = torch.argsort(sim_matrix, dim=1, descending=True)
    
    # Find true target indices
    true_target_indices = []
    for y_true in y_true_vectors:
        distances = torch.norm(Y_target_tensor - y_true.unsqueeze(0), dim=1)
        true_idx = torch.argmin(distances).item()
        true_target_indices.append(true_idx)
    
    # Calculate MRR
    reciprocal_ranks = []
    for i, true_idx in enumerate(true_target_indices):
        rank_list = sorted_indices[i].tolist()
        try:
            rank = rank_list.index(true_idx) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(1.0 / (len(rank_list) + 1))
    
    mrr = np.mean(reciprocal_ranks)
    
    # Calculate Recall@K
    k_values = [1, 5, 10, 20]
    recall_at_k = {}
    sorted_indices_np = sorted_indices.cpu().numpy()
    
    for k in k_values:
        hits = 0
        for i, true_idx in enumerate(true_target_indices):
            if true_idx in sorted_indices_np[i, :k]:
                hits += 1
        recall_at_k[f'R@{k}'] = hits / len(true_target_indices)
    
    metrics = {'MRR': mrr}
    metrics.update(recall_at_k)
    
    return metrics


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        
        y_pred = model(batch)
        target = torch.ones(y_pred.size(0)).to(device)
        loss = criterion(y_pred, batch['y'], target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, Y_target_all, device, criterion=None):
    """Evaluate model and return ranking metrics."""
    model.eval()
    all_y_pred = []
    all_y_true = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            y_pred = model(batch)
            
            if criterion:
                target = torch.ones(y_pred.size(0)).to(device)
                loss = criterion(y_pred, batch['y'], target)
                total_loss += loss.item()
            
            all_y_pred.append(y_pred)
            all_y_true.append(batch['y'])
    
    # Concatenate on GPU
    y_pred_vectors = torch.cat(all_y_pred, dim=0)
    y_true_vectors = torch.cat(all_y_true, dim=0)
    
    # Calculate metrics
    metrics = calculate_ranking_metrics_gpu(y_pred_vectors, y_true_vectors, Y_target_all, device)
    
    if criterion:
        metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


# ============================================================================
# CONFIGURATION
# ============================================================================

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description="Simple CPP Training Script")
    
    # Config file (optional - can override individual params)
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    
    # Data
    parser.add_argument("--data_type", type=str, default="decorte")
    parser.add_argument("--master_skill_file", type=str, 
                       default="/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/results/decorte_jobbert_v2_baseline/job_title_skills_master.csv")
    parser.add_argument("--esco_skills_file", type=str, default="data/esco_datasets/skills_en.csv")
    parser.add_argument("--vocab_dir", type=str, default="data/processed/master_datasets_2/")
    parser.add_argument("--skill_properties_file", type=str, default="data/processed/master_datasets_2/skill_properties_map.json")
    
    # Encoder
    parser.add_argument("--encoder", type=str, 
                       default="ElenaSenger/career-path-representation-mpnet-decorte",
                       help="Sentence transformer model")
    
    # Features
    parser.add_argument("--use_text_description", action='store_true',
                       help="Include job descriptions (default: titles only)")
    parser.add_argument("--use_skill_description", action='store_true')
    parser.add_argument("--pooling_strategy", type=str, default="weighted_idf")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    
    # Model architecture
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    
    # Training
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="results/cpp_simple")
    parser.add_argument("--run_name", type=str, default="cpp_simple")
    parser.add_argument("--save_model", action='store_true')
    
    # WandB
    parser.add_argument("--use_wandb", action='store_true')
    parser.add_argument("--wandb_project", type=str, default="cpp-simple")
    parser.add_argument("--wandb_entity", type=str, default=None)
    
    # Cache
    parser.add_argument("--embeddings_cache_dir", type=str,
                       default="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/embeddings")
    parser.add_argument("--force_recompute", action='store_true')
    
    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
        # Update args with config values (args take precedence)
        for key, value in config.items():
            if not hasattr(args, key) or getattr(args, key) is None:
                setattr(args, key, value)
    
    logger.info("=" * 80)
    logger.info("Simple Career Path Prediction Training")
    logger.info("=" * 80)
    logger.info(f"Run: {args.run_name}")
    logger.info(f"Data: {args.data_type}")
    logger.info(f"Encoder: {args.encoder}")
    logger.info(f"Hidden dim: {args.hidden_dim}, Layers: {args.n_layers}, Dropout: {args.dropout}")
    logger.info(f"Max epochs: {args.max_epochs}, Patience: {args.patience}")
    logger.info(f"Learning rate: {args.lr}, Weight decay: {args.weight_decay}")
    logger.info("=" * 80 + "\n")
    
    # Initialize WandB
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=args.run_name
        )
        logger.info("✓ WandB initialized\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    logger.info(f"Using device: {device}\n")
    
    # --- Step 1: Load encoder ---
    logger.info("[1/5] Loading encoder...")
    encoder = SentenceTransformer(args.encoder)
    text_dim = encoder.get_sentence_embedding_dimension()
    logger.info(f"  ✓ Encoder dimension: {text_dim}\n")
    
    # --- Step 2: Load data ---
    logger.info("[2/5] Loading data...")
    data = Data(
        DATA_TYPE=args.data_type,
        ONLY_TITLES=not args.use_text_description,
        consider_subspans=True,
        LOAD_CLEAN_TEST=False
    )
    
    train_pairs, val_pairs, test_pairs = data.get_data(stage='transformation_finetuning')
    logger.info(f"  ✓ Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}\n")
    
    # --- Step 3: Load vocabs and skill mappings ---
    logger.info("[3/5] Loading vocabularies and skill mappings...")
    
    # Extract train+val job titles for IDF
    from src.cpp.train_cpp_enhanced_v2 import extract_raw_titles_from_doc
    train_val_jobs = set()
    for history_doc, target_doc in train_pairs + val_pairs:
        train_val_jobs.update(extract_raw_titles_from_doc(history_doc))
        train_val_jobs.update(extract_raw_titles_from_doc(target_doc))
    
    all_vocabs = load_all_vocabs(args.vocab_dir)
    structured_dim = sum(len(vocab) for vocab in all_vocabs.values())
    
    job_skill_map, esco_skill_text_map, skill_properties_map = load_job_and_skill_data(
        master_skill_file=args.master_skill_file,
        esco_skills_file=args.esco_skills_file,
        skill_properties_file=args.skill_properties_file,
        pooling_strategy=args.pooling_strategy,
        alpha=args.alpha,
        beta=args.beta,
        train_val_occ=train_val_jobs
    )
    logger.info(f"  ✓ Structured features: {structured_dim} dimensions\n")
    
    # --- Step 4: Precompute embeddings ---
    logger.info("[4/5] Computing embeddings...")
    os.makedirs(args.embeddings_cache_dir, exist_ok=True)
    encoder_name = args.encoder.split('/')[-1]
    
    # Target embeddings
    all_target_labels = list(set([t for _, t in train_pairs + val_pairs + test_pairs]))
    Y_target_dict, Y_target_all = precompute_target_embeddings(
        encoder, all_target_labels,
        show_progress=True,
        cache_dir=args.embeddings_cache_dir,
        encoder_name=encoder_name,
        force_recompute=args.force_recompute
    )
    output_dim = Y_target_all.shape[1]
    logger.info(f"  ✓ Target embeddings: {output_dim} dimensions\n")
    
    # Input embeddings
    logger.info("  Computing train embeddings...")
    train_pairs, train_h_text, train_h_skill = precompute_input_embeddings(
        train_pairs, Y_target_dict, encoder, encoder,
        job_skill_map, esco_skill_text_map,
        use_skill_description=args.use_skill_description,
        pooling_strategy=args.pooling_strategy,
        alpha=args.alpha, beta=args.beta,
        use_text_history=True,
        use_skill_text=True,
        cache_dir=args.embeddings_cache_dir,
        encoder_skill_name=encoder_name,
        force_recompute=args.force_recompute,
        split_name="train"
    )
    
    logger.info("  Computing val embeddings...")
    val_pairs, val_h_text, val_h_skill = precompute_input_embeddings(
        val_pairs, Y_target_dict, encoder, encoder,
        job_skill_map, esco_skill_text_map,
        use_skill_description=args.use_skill_description,
        pooling_strategy=args.pooling_strategy,
        alpha=args.alpha, beta=args.beta,
        use_text_history=True,
        use_skill_text=True,
        cache_dir=args.embeddings_cache_dir,
        encoder_skill_name=encoder_name,
        force_recompute=False,
        split_name="val"
    )
    
    logger.info("  Computing test embeddings...")
    test_pairs, test_h_text, test_h_skill = precompute_input_embeddings(
        test_pairs, Y_target_dict, encoder, encoder,
        job_skill_map, esco_skill_text_map,
        use_skill_description=args.use_skill_description,
        pooling_strategy=args.pooling_strategy,
        alpha=args.alpha, beta=args.beta,
        use_text_history=True,
        use_skill_text=True,
        cache_dir=args.embeddings_cache_dir,
        encoder_skill_name=encoder_name,
        force_recompute=False,
        split_name="test"
    )
    logger.info("  ✓ All embeddings computed\n")
    
    # --- Step 5: Create datasets and dataloaders ---
    logger.info("[5/5] Creating datasets...")
    
    # Combine train+val for final training
    combined_pairs = train_pairs + val_pairs
    combined_h_text = np.concatenate([train_h_text, val_h_text], axis=0) if train_h_text is not None else None
    combined_h_skill = np.concatenate([train_h_skill, val_h_skill], axis=0) if train_h_skill is not None else None
    
    train_dataset = CareerPathDataset(
        data_pairs=combined_pairs,
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
        encoder_skill=encoder,
        include_text=True,
        include_skill_text=True,
        include_structured=True,
        pre_h_text=combined_h_text,
        pre_h_skill_text=combined_h_skill,
        device=device,
        pin_embeddings_to_gpu=False,
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
        encoder_skill=encoder,
        include_text=True,
        include_skill_text=True,
        include_structured=True,
        pre_h_text=test_h_text,
        pre_h_skill_text=test_h_skill,
        device=device,
        pin_embeddings_to_gpu=False,
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_career_path_batch,
        pin_memory=(device.type == 'cuda')
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_career_path_batch,
        pin_memory=(device.type == 'cuda')
    )
    
    logger.info(f"  ✓ Train: {len(train_dataset)}, Test: {len(test_dataset)}\n")
    
    # Calculate input dimension
    input_dim = text_dim + text_dim + structured_dim  # text + skill_text + structured
    
    # --- Initialize model ---
    logger.info("Initializing model...")
    model = SimpleCPPModel(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout
    ).to(device)
    
    logger.info(f"  ✓ Model: {sum(p.numel() for p in model.parameters())} parameters\n")
    
    # --- Training setup ---
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best_val_mrr = 0.0
    epochs_no_improve = 0
    best_model_state = None
    
    logger.info("=" * 80)
    logger.info("TRAINING")
    logger.info("=" * 80 + "\n")
    
    start_time = time.time()
    
    # --- Training loop ---
    for epoch in range(args.max_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate on test set (since we're using train+val for training)
        test_metrics = evaluate(model, test_loader, Y_target_all, device, criterion)
        test_mrr = test_metrics['MRR']
        
        epoch_time = time.time() - epoch_start
        
        # Logging
        logger.info(
            f"Epoch {epoch+1:3d}/{args.max_epochs} | "
            f"Loss: {train_loss:.4f} | "
            f"Test MRR: {test_mrr:.4f} | "
            f"R@1: {test_metrics['R@1']:.4f} | "
            f"R@5: {test_metrics['R@5']:.4f} | "
            f"R@10: {test_metrics['R@10']:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # WandB logging
        if WANDB_AVAILABLE and args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'test_mrr': test_mrr,
                'test_r@1': test_metrics['R@1'],
                'test_r@5': test_metrics['R@5'],
                'test_r@10': test_metrics['R@10'],
                'test_r@20': test_metrics['R@20'],
            })
        
        # Early stopping
        if test_mrr > best_val_mrr + args.min_delta:
            best_val_mrr = test_mrr
            epochs_no_improve = 0
            best_model_state = model.state_dict().copy()
            logger.info(f"  → New best model! MRR: {best_val_mrr:.4f}")
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= args.patience:
            logger.info(f"\nEarly stopping after {epoch+1} epochs (patience: {args.patience})")
            break
    
    total_time = time.time() - start_time
    logger.info(f"\nTraining completed in {total_time/60:.2f} minutes")
    
    # --- Load best model ---
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        logger.info(f"Loaded best model (MRR: {best_val_mrr:.4f})")
    
    # --- Final evaluation ---
    logger.info("\n" + "=" * 80)
    logger.info("FINAL TEST RESULTS")
    logger.info("=" * 80)
    
    final_metrics = evaluate(model, test_loader, Y_target_all, device)
    
    for metric, value in final_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    logger.info("=" * 80 + "\n")
    
    # WandB final metrics
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.log({
            'final_test_mrr': final_metrics['MRR'],
            'final_test_r@1': final_metrics['R@1'],
            'final_test_r@5': final_metrics['R@5'],
            'final_test_r@10': final_metrics['R@10'],
            'final_test_r@20': final_metrics['R@20'],
        })
        wandb.finish()
    
    # --- Save model ---
    if args.save_model:
        model_path = os.path.join(args.output_dir, 'model.pt')
        torch.save({
            'model_state_dict': best_model_state if best_model_state else model.state_dict(),
            'config': vars(args),
            'metrics': final_metrics,
        }, model_path)
        logger.info(f"Model saved to: {model_path}\n")
    
    # --- Save results to CSV ---
    results_path = os.path.join(args.output_dir, 'results.csv')
    results_data = {
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'run_name': args.run_name,
        'data_type': args.data_type,
        'encoder': args.encoder,
        'hidden_dim': args.hidden_dim,
        'n_layers': args.n_layers,
        'dropout': args.dropout,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'epochs_trained': epoch + 1,
        **{f'test_{k}': v for k, v in final_metrics.items()},
    }
    
    results_df = pd.DataFrame([results_data])
    
    if os.path.exists(results_path):
        results_df.to_csv(results_path, mode='a', header=False, index=False)
    else:
        results_df.to_csv(results_path, mode='w', header=True, index=False)
    
    logger.info(f"Results saved to: {results_path}")
    logger.info("\n✓ Done!")


if __name__ == "__main__":
    main()





