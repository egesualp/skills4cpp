"""
Vector Transformation Training Script.

Reproduces the vector transformation approach:
- Transforms career history embeddings → target embeddings
- Simple neural network with input normalization
- Validates on separate validation set during training
- Similar to the original vector_transformation.py but simplified
"""

import argparse
import os
import sys
import time
import copy

from loguru import logger
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("WandB not available. Install with: pip install wandb")

from src.cpp.data_classes import Data

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)


# ============================================================================
# MODEL (same as vector_transformation.py)
# ============================================================================

class VectorTransformModel(nn.Module):
    """
    Neural network model for vector transformation with multiple hidden layers.
    Same as in vector_transformation.py
    """
    def __init__(
        self, input_size, hidden_sizes, output_size, dropout=False, dropout_rate=0.1
    ):
        super(VectorTransformModel, self).__init__()
        self.hidden_layers = nn.ModuleList()
        prev_size = input_size

        # Create hidden layers based on provided hidden sizes
        for hidden_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size

        self.dropout = dropout
        self.dropout_rate = dropout_rate
        if self.dropout:
            self.dropout_layer = nn.Dropout(p=self.dropout_rate)

        # Output layer
        self.output = nn.Linear(prev_size, output_size)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
            if self.dropout:
                x = self.dropout_layer(x)
        x = self.output(x)
        return x


# ============================================================================
# DATA LOADING
# ============================================================================

def create_data_loader(
    pairs,
    encoder,
    batch_size,
    target2idx,
    target_emb_all_cpu,
    shuffle=True,
    name="dataset",
    device="cuda",
):
    """
    Create DataLoader from (career_history, target) pairs.
    Encodes texts and creates tensors.
    """
    logger.info(f"Processing {name}: encoding {len(pairs)} pairs...")
    
    # Separate inputs and targets
    career_texts, target_texts = zip(*pairs)
    
    # Encode career history texts
    logger.info(f"  Encoding career histories...")
    career_embeddings = encoder.encode(
        career_texts,
        show_progress_bar=True,
        convert_to_tensor=True,
        batch_size=256
    )

    # Map targets to unique label ids and reuse cached embeddings
    target_ids = torch.tensor([target2idx[t] for t in target_texts], dtype=torch.long)
    target_embeddings = target_emb_all_cpu[target_ids]
    
    # Create dataset
    dataset = TensorDataset(
        career_embeddings.float().cpu(),
        target_embeddings.float(),
        target_ids
    )
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=(device == "cuda")
    )
    
    logger.info(f"  ✓ {name} loader created: {len(dataset)} samples\n")
    
    return loader, career_embeddings.shape[1], target_embeddings.shape[1]


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch with input normalization."""
    model.train()
    total_loss = 0.0
    
    for inputs, targets, _target_ids in tqdm(dataloader, desc="Training", leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Normalize the input (key difference from train_cpp_simple.py)
        inputs_normalized = inputs / torch.norm(inputs, dim=1, keepdim=True)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs_normalized)
        
        # CosineEmbeddingLoss expects labels of 1 for similar pairs
        labels = torch.ones(inputs.size(0)).to(device)
        loss = criterion(outputs, targets, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(
    model,
    dataloader,
    criterion,
    device,
    candidate_targets_cpu,
    compute_ranking_metrics=False,
    ranking_chunk_size=None,
    max_eval_samples=None,
    k_values=(1, 5, 10, 20),
):
    """
    Evaluate model on validation/test set.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        criterion: Loss function
        device: torch device
        compute_ranking_metrics: If True, compute MRR and R@k metrics
        ranking_chunk_size: Number of query samples per similarity chunk (controls peak RAM)
        max_eval_samples: If set, only evaluate the first N samples (useful for per-epoch validation)
        k_values: Recall@K values to compute
    
    Returns:
        If compute_ranking_metrics=False: average loss
        If compute_ranking_metrics=True: dict with loss and ranking metrics
    """
    model.eval()
    total_loss = 0.0
    n_seen = 0

    # For ranking metrics: store predictions (queries) only; candidates come from the full dataset
    outputs_cpu = []
    true_label_ids_cpu = []

    with torch.no_grad():
        for inputs, targets, target_ids in tqdm(dataloader, desc="Evaluating", leave=False):
            if max_eval_samples is not None and n_seen >= max_eval_samples:
                break

            bs_full = inputs.size(0)
            bs = bs_full
            if max_eval_samples is not None:
                bs = min(bs_full, max_eval_samples - n_seen)
                inputs = inputs[:bs]
                targets = targets[:bs]
                target_ids = target_ids[:bs]

            inputs = inputs.to(device)
            targets = targets.to(device)
            target_ids = target_ids.to(device)

            # Forward pass (no input normalization during eval, matching earlier behavior)
            outputs = model(inputs)

            # Normalize outputs and targets for comparison + loss
            targets_normalized = targets / torch.norm(targets, dim=1, keepdim=True)
            outputs_normalized = outputs / torch.norm(outputs, dim=1, keepdim=True)

            labels = torch.ones(outputs_normalized.size(0)).to(device)
            loss = criterion(outputs_normalized, targets_normalized, labels)

            total_loss += loss.item() * bs
            n_seen += bs

            if compute_ranking_metrics:
                outputs_cpu.append(outputs_normalized.cpu())
                true_label_ids_cpu.append(target_ids.cpu())

    avg_loss = total_loss / max(n_seen, 1)

    if not compute_ranking_metrics:
        return avg_loss

    # Candidate targets: unique label embeddings (CPU), normalized
    candidate_targets = candidate_targets_cpu.float()
    candidate_targets = candidate_targets / torch.norm(candidate_targets, dim=1, keepdim=True)
    n_candidates = candidate_targets.size(0)

    queries = torch.cat(outputs_cpu, dim=0)  # [n_queries, dim] CPU
    true_indices = torch.cat(true_label_ids_cpu, dim=0)  # [n_queries] CPU (label ids)
    n_queries = queries.size(0)

    max_k = max(k_values)

    # Choose query chunk size to cap peak RAM of similarity matrix: chunk_size * n_candidates * 4 bytes
    if ranking_chunk_size is not None:
        chunk_size = ranking_chunk_size
    else:
        if n_candidates >= 100_000:
            chunk_size = 128
        elif n_candidates >= 50_000:
            chunk_size = 256
        elif n_candidates >= 20_000:
            chunk_size = 512
        else:
            chunk_size = 1000
    chunk_size = min(chunk_size, n_queries)

    rr_sum = 0.0
    hits = {k: 0 for k in k_values}

    logger.info(
        f"  Ranking metrics: queries={n_queries}, candidates={n_candidates}, chunk_size={chunk_size}"
    )

    for start in tqdm(range(0, n_queries, chunk_size), desc="Computing similarities", leave=False):
        end = min(start + chunk_size, n_queries)
        chunk_q = queries[start:end]  # [m, dim]
        chunk_true = true_indices[start:end]  # [m]
        m = chunk_q.size(0)

        # Similarity: [m, n_candidates] on CPU
        sim = torch.mm(chunk_q, candidate_targets.t())

        # Top-K for recall@k without sorting full matrix
        topk_idx = torch.topk(sim, k=max_k, dim=1, largest=True).indices  # [m, max_k]

        # Rank for MRR: 1 + number of candidates with higher similarity than the true target
        row_idx = torch.arange(m, dtype=torch.long)
        sim_true = sim[row_idx, chunk_true]  # [m]
        ranks = 1 + (sim > sim_true.unsqueeze(1)).sum(dim=1)  # [m]

        rr_sum += torch.reciprocal(ranks.float()).sum().item()

        for k in k_values:
            hits[k] += (topk_idx[:, :k] == chunk_true.unsqueeze(1)).any(dim=1).sum().item()

        # Encourage prompt freeing of large temporaries
        del sim, topk_idx, sim_true, ranks

    metrics = {
        "loss": avg_loss,
        "MRR": rr_sum / max(n_queries, 1),
    }
    for k in k_values:
        metrics[f"R@{k}"] = hits[k] / max(n_queries, 1)

    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Vector Transformation Training")
    
    # Data
    parser.add_argument("--data_type", type=str, default="decorte")
    parser.add_argument("--use_text_description", action='store_true',
                       help="Include job descriptions (default: titles only)")
    
    # Encoder
    parser.add_argument("--encoder", type=str,
                       default="sentence-transformers/all-mpnet-base-v2",
                       help="Sentence transformer model")
    
    # Model architecture
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[512, 512],
                       help="List of hidden layer sizes")
    parser.add_argument("--dropout", action='store_true')
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    
    # Training
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument(
        "--epoch_eval_frac",
        type=float,
        default=1.0,
        help="Fraction of the validation set to use for per-epoch ranking metrics (1.0 = full val).",
    )
    
    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="results/vector_transformation")
    parser.add_argument("--run_name", type=str, default="vector_transform")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Where to save the RP-compatible transformation checkpoint (.pth). Defaults to <output_dir>/model.pth",
    )
    
    # WandB
    parser.add_argument("--use_wandb", action='store_true')
    parser.add_argument("--wandb_project", type=str, default="vector-transformation")
    parser.add_argument("--wandb_entity", type=str, default=None)
    
    # Performance
    parser.add_argument("--ranking_chunk_size", type=int, default=None,
                       help="Chunk size for batched similarity computation (auto-detected if not set)")
    
    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("Vector Transformation Training")
    logger.info("=" * 80)
    logger.info(f"Run: {args.run_name}")
    logger.info(f"Data: {args.data_type}")
    logger.info(f"Encoder: {args.encoder}")
    logger.info(f"Hidden sizes: {args.hidden_sizes}")
    logger.info(f"Dropout: {args.dropout} (rate: {args.dropout_rate})")
    logger.info(f"Max epochs: {args.max_epochs}, Patience: {args.patience}")
    logger.info(f"Learning rate: {args.lr}, Batch size: {args.batch_size}")
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
    if args.model_path is None:
        args.model_path = os.path.join(args.output_dir, "model.pth")
    device = torch.device(args.device)
    logger.info(f"Using device: {device}\n")
    
    # --- Step 1: Load encoder ---
    logger.info("[1/4] Loading encoder...")
    encoder = SentenceTransformer(args.encoder)
    logger.info(f"  ✓ Encoder loaded\n")
    
    # --- Step 2: Load data ---
    logger.info("[2/4] Loading data...")
    data = Data(
        DATA_TYPE=args.data_type,
        ONLY_TITLES=not args.use_text_description,
        consider_subspans=True,
        LOAD_CLEAN_TEST=False
    )
    
    # Get data for transformation finetuning (history → target pairs)
    train_pairs, val_pairs, test_pairs = data.get_data(stage='transformation_finetuning')
    logger.info(f"  ✓ Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}\n")

    # Build unique target labels across splits (fixes inconsistent MRR vs Recall@K when targets repeat)
    logger.info("Building unique target set across splits...")
    target_set = set()
    for _, t in train_pairs:
        target_set.add(t)
    for _, t in val_pairs:
        target_set.add(t)
    for _, t in test_pairs:
        target_set.add(t)
    all_unique_targets = sorted(target_set)
    logger.info(f"  ✓ Unique targets: {len(all_unique_targets)}\n")

    logger.info("Encoding unique target embeddings (cached)...")
    target_emb_all_cpu = encoder.encode(
        all_unique_targets,
        show_progress_bar=True,
        convert_to_tensor=True,
        batch_size=256,
    ).float().cpu()
    target2idx = {t: i for i, t in enumerate(all_unique_targets)}
    
    # --- Step 3: Create data loaders ---
    logger.info("[3/4] Creating data loaders...")
    
    train_loader, input_size, output_size = create_data_loader(
        train_pairs,
        encoder,
        args.batch_size,
        target2idx=target2idx,
        target_emb_all_cpu=target_emb_all_cpu,
        shuffle=True,
        name="train",
        device=args.device,
    )
    
    val_loader, _, _ = create_data_loader(
        val_pairs,
        encoder,
        args.batch_size,
        target2idx=target2idx,
        target_emb_all_cpu=target_emb_all_cpu,
        shuffle=False,
        name="val",
        device=args.device,
    )
    
    test_loader, _, _ = create_data_loader(
        test_pairs,
        encoder,
        args.batch_size,
        target2idx=target2idx,
        target_emb_all_cpu=target_emb_all_cpu,
        shuffle=False,
        name="test",
        device=args.device,
    )
    
    # --- Step 4: Initialize model ---
    logger.info("[4/4] Initializing model...")
    model = VectorTransformModel(
        input_size=input_size,
        hidden_sizes=args.hidden_sizes,
        output_size=output_size,
        dropout=args.dropout,
        dropout_rate=args.dropout_rate
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  ✓ Model: {n_params:,} parameters")
    logger.info(f"  ✓ Input: {input_size}, Hidden: {args.hidden_sizes}, Output: {output_size}\n")
    
    # --- Training setup ---
    criterion = nn.CosineEmbeddingLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_state = None
    best_epoch = 0
    
    logger.info("=" * 80)
    logger.info("TRAINING")
    logger.info("=" * 80 + "\n")
    
    start_time = time.time()
    
    # --- Training loop ---
    for epoch in range(args.max_epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate with ranking metrics (optionally on a fraction of val to control cost)
        if not (0.0 < args.epoch_eval_frac <= 1.0):
            raise ValueError("--epoch_eval_frac must be in (0.0, 1.0].")
        max_val_samples = int(len(val_loader.dataset) * args.epoch_eval_frac)
        max_val_samples = max(1, min(max_val_samples, len(val_loader.dataset)))

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            candidate_targets_cpu=target_emb_all_cpu,
            compute_ranking_metrics=True,
            ranking_chunk_size=args.ranking_chunk_size,
            max_eval_samples=max_val_samples,
        )
        val_loss = val_metrics['loss']
        val_mrr = val_metrics['MRR']
        
        epoch_time = time.time() - epoch_start
        
        # Logging
        logger.info(
            f"Epoch {epoch+1:3d}/{args.max_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val MRR: {val_mrr:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )
        
        # WandB logging
        if WANDB_AVAILABLE and args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_mrr': val_mrr,
            })
        
        # Early stopping (based on validation loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            logger.info(f"  → New best model! Val Loss: {best_val_loss:.4f}, MRR: {val_mrr:.4f}")
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
        logger.info(f"Loaded best model from epoch {best_epoch} (Val Loss: {best_val_loss:.4f})")
    
    # --- Final evaluation ---
    logger.info("\n" + "=" * 80)
    logger.info("FINAL TEST RESULTS")
    logger.info("=" * 80)
    
    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        candidate_targets_cpu=target_emb_all_cpu,
        compute_ranking_metrics=True,
        ranking_chunk_size=args.ranking_chunk_size,
        max_eval_samples=None,
    )
    
    logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
    logger.info(f"Test MRR: {test_metrics['MRR']:.4f}")
    logger.info(f"Test R@1: {test_metrics['R@1']:.4f}")
    logger.info(f"Test R@5: {test_metrics['R@5']:.4f}")
    logger.info(f"Test R@10: {test_metrics['R@10']:.4f}")
    logger.info(f"Test R@20: {test_metrics['R@20']:.4f}")
    logger.info("=" * 80 + "\n")
    
    # WandB final metrics
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.log({
            'final_test_loss': test_metrics['loss'],
            'final_test_mrr': test_metrics['MRR'],
            'final_test_r@1': test_metrics['R@1'],
            'final_test_r@5': test_metrics['R@5'],
            'final_test_r@10': test_metrics['R@10'],
            'final_test_r@20': test_metrics['R@20'],
            'best_val_loss': best_val_loss,
            'best_epoch': best_epoch,
        })
        wandb.finish()
    
    # --- Save model (RP-compatible checkpoint) ---
    # This format matches realistic-career-path-prediction/src/predictor.py:load_model(...)
    torch.save(
        {
            "input_size": int(input_size),
            "hidden_sizes": list(args.hidden_sizes),
            "output_size": int(output_size),
            "model_state_dict": model.state_dict(),
        },
        args.model_path,
    )
    logger.info(f"Model saved to: {args.model_path}\n")
    
    # --- Save results to CSV ---
    results_path = os.path.join(args.output_dir, 'results.csv')
    results_data = {
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'run_name': args.run_name,
        'data_type': args.data_type,
        'encoder': args.encoder,
        'hidden_sizes': str(args.hidden_sizes),
        'dropout': args.dropout,
        'dropout_rate': args.dropout_rate,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'epochs_trained': epoch + 1,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'test_loss': test_metrics['loss'],
        'test_MRR': test_metrics['MRR'],
        'test_R@1': test_metrics['R@1'],
        'test_R@5': test_metrics['R@5'],
        'test_R@10': test_metrics['R@10'],
        'test_R@20': test_metrics['R@20'],
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

