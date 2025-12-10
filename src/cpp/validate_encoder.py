"""
Simple script to validate encoder performance on career path prediction.

Trains a simple linear model to predict next ESCO occupation from job history.
No skills, no complex architectures - just encoder validation.

Usage:
    python validate_encoder.py --data_type decorte --encoder ElenaSenger/career-path-representation-mpnet-decorte
    python validate_encoder.py --data_type decorte_esco --encoder ElenaSenger/career-path-representation-mpnet-decorte-esco
"""

import argparse
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

try:
    from src.cpp.data_classes import Data
    from src.cpp.utils import SEP_TOKEN
except ImportError as e:
    print(f"Error: Required modules not found. {e}")
    sys.exit(1)


class SimpleDataset(Dataset):
    """Simple dataset that just holds pre-computed embeddings."""
    
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def encode_pairs(encoder, pairs, show_progress=True):
    """
    Encode input-target pairs.
    
    Args:
        encoder: SentenceTransformer model
        pairs: List of (history, target) tuples
        show_progress: Show progress bar
    
    Returns:
        X: Input embeddings [n_samples, embed_dim]
        y: Target embeddings [n_samples, embed_dim]
    """
    histories = [pair[0] for pair in pairs]
    targets = [pair[1] for pair in pairs]
    
    print(f"  Encoding {len(histories)} histories...")
    X = encoder.encode(histories, show_progress_bar=show_progress, convert_to_numpy=True)
    
    print(f"  Encoding {len(targets)} targets...")
    y = encoder.encode(targets, show_progress_bar=show_progress, convert_to_numpy=True)
    
    return X, y


def calculate_ranking_metrics(y_pred, y_true, Y_all, k_values=[1, 5, 10, 20]):
    """
    Calculate MRR and Recall@K.
    
    Args:
        y_pred: Predicted embeddings [n_samples, embed_dim]
        y_true: True target embeddings [n_samples, embed_dim]
        Y_all: All possible target embeddings [n_targets, embed_dim]
        k_values: K values for Recall@K
    
    Returns:
        Dictionary with metrics
    """
    # Calculate cosine similarity between predictions and all targets
    sim_matrix = cosine_similarity(y_pred, Y_all)
    sorted_indices = np.argsort(sim_matrix, axis=1)[:, ::-1]
    
    # Find true target indices by finding nearest neighbor in Y_all
    # (handles floating point precision issues)
    true_indices = []
    sim_true_to_all = cosine_similarity(y_true, Y_all)
    for i in range(len(y_true)):
        # Find the index with highest similarity (should be ~1.0 for the same embedding)
        true_idx = np.argmax(sim_true_to_all[i])
        true_indices.append(true_idx)
    
    # Calculate MRR
    reciprocal_ranks = []
    for i, true_idx in enumerate(true_indices):
        rank = list(sorted_indices[i]).index(true_idx) + 1
        reciprocal_ranks.append(1.0 / rank)
    
    mrr = np.mean(reciprocal_ranks)
    
    # Calculate Recall@K
    metrics = {'MRR': mrr}
    for k in k_values:
        hits = sum(1 for i, true_idx in enumerate(true_indices) 
                   if true_idx in sorted_indices[i, :k])
        metrics[f'R@{k}'] = hits / len(true_indices)
    
    return metrics


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        y_pred = model(X_batch)
        
        # CosineEmbeddingLoss with target=1 (all similar pairs)
        target = torch.ones(y_pred.size(0)).to(device)
        loss = criterion(y_pred, y_batch, target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, X, y, Y_all, device):
    """Evaluate model."""
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_pred = model(X_tensor).cpu().numpy()
    
    metrics = calculate_ranking_metrics(y_pred, y, Y_all)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Validate Encoder Performance")
    parser.add_argument("--data_type", type=str, required=True, 
                       choices=["decorte", "decorte_esco"],
                       help="Dataset to use")
    parser.add_argument("--encoder", type=str, required=True,
                       help="Encoder model name or path")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size")
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use")
    parser.add_argument("--only_titles", action='store_true',
                       help="Use only job titles (no descriptions)")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ENCODER VALIDATION")
    print("=" * 80)
    print(f"Data Type: {args.data_type}")
    print(f"Encoder: {args.encoder}")
    print(f"Only Titles: {args.only_titles}")
    print(f"Device: {args.device}")
    print("=" * 80)
    print()
    
    device = torch.device(args.device)
    
    # Load encoder
    print("[1/6] Loading encoder...")
    encoder = SentenceTransformer(args.encoder)
    embed_dim = encoder.get_sentence_embedding_dimension()
    print(f"  ✓ Embedding dimension: {embed_dim}\n")
    
    # Load data
    print("[2/6] Loading data...")
    data = Data(DATA_TYPE=args.data_type, ONLY_TITLES=args.only_titles)
    train_pairs, val_pairs, test_pairs = data.get_data(stage='transformation_finetuning')
    print(f"  ✓ Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}\n")
    
    # Get unique targets first (for ranking evaluation)
    print("[3/6] Computing unique targets...")
    all_target_texts = list(set([pair[1] for pair in train_pairs + val_pairs + test_pairs]))
    Y_all = encoder.encode(all_target_texts, show_progress_bar=True, convert_to_numpy=True)
    print(f"  ✓ {len(all_target_texts)} unique targets\n")
    
    # Create target text to embedding mapping
    target_to_embedding = {text: Y_all[i] for i, text in enumerate(all_target_texts)}
    
    # Encode inputs and get corresponding target embeddings from the mapping
    print("[4/6] Encoding input data...")
    print("  > Train set:")
    X_train = encoder.encode([pair[0] for pair in train_pairs], 
                             show_progress_bar=True, convert_to_numpy=True)
    y_train = np.array([target_to_embedding[pair[1]] for pair in train_pairs])
    
    print("  > Validation set:")
    X_val = encoder.encode([pair[0] for pair in val_pairs], 
                           show_progress_bar=True, convert_to_numpy=True)
    y_val = np.array([target_to_embedding[pair[1]] for pair in val_pairs])
    
    print("  > Test set:")
    X_test = encoder.encode([pair[0] for pair in test_pairs], 
                            show_progress_bar=True, convert_to_numpy=True)
    y_test = np.array([target_to_embedding[pair[1]] for pair in test_pairs])
    print()
    
    # Baseline: No transformation (just encoder embeddings)
    print("[5/6] Evaluating baseline (no transformation)...")
    baseline_val = calculate_ranking_metrics(X_val, y_val, Y_all)
    baseline_test = calculate_ranking_metrics(X_test, y_test, Y_all)
    
    print("  Validation Metrics:")
    for k, v in baseline_val.items():
        print(f"    {k}: {v:.4f}")
    print("\n  Test Metrics:")
    for k, v in baseline_test.items():
        print(f"    {k}: {v:.4f}")
    print()
    
    # Train simple linear model
    print("[6/6] Training linear transformation...")
    model = nn.Linear(embed_dim, embed_dim, bias=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CosineEmbeddingLoss()
    
    train_dataset = SimpleDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                             shuffle=True, pin_memory=(device.type == 'cuda'))
    
    best_val_mrr = 0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, X_val, y_val, Y_all, device)
        
        if val_metrics['MRR'] > best_val_mrr:
            best_val_mrr = val_metrics['MRR']
            best_epoch = epoch
            # Save best model
            best_model_state = model.state_dict().copy()
        
        print(f"  Epoch {epoch+1}/{args.epochs} | Loss: {train_loss:.4f} | "
              f"Val MRR: {val_metrics['MRR']:.4f} | Val R@5: {val_metrics['R@5']:.4f}")
    
    # Load best model and evaluate on test
    model.load_state_dict(best_model_state)
    test_metrics = evaluate(model, X_test, y_test, Y_all, device)
    
    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(f"Best Epoch: {best_epoch + 1}")
    print()
    print("BASELINE (No Transformation):")
    print("  Validation:")
    for k, v in baseline_val.items():
        print(f"    {k}: {v:.4f}")
    print("  Test:")
    for k, v in baseline_test.items():
        print(f"    {k}: {v:.4f}")
    print()
    print("LINEAR TRANSFORMATION:")
    print("  Validation (Best):")
    val_metrics_best = evaluate(model, X_val, y_val, Y_all, device)
    for k, v in val_metrics_best.items():
        print(f"    {k}: {v:.4f}")
    print("  Test:")
    for k, v in test_metrics.items():
        print(f"    {k}: {v:.4f}")
    print()
    
    # Compute improvements
    print("IMPROVEMENT (Linear over Baseline):")
    print(f"  Val MRR: {val_metrics_best['MRR'] - baseline_val['MRR']:+.4f}")
    print(f"  Test MRR: {test_metrics['MRR'] - baseline_test['MRR']:+.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

