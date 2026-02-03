import argparse
import logging
import os
import random
from datetime import datetime

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from src.seq_transformer.dataset import CareerSequenceDataset, collate_fn
from src.seq_transformer.model import CareerPathAggregator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_ranking_metrics_gpu(y_pred_vectors, y_true_vectors, Y_target_all, device):
    """
    GPU-optimized ranking metrics calculation.

    Args:
        y_pred_vectors: Predicted embeddings [n_samples, embed_dim] on GPU
        y_true_vectors: True target embeddings [n_samples, embed_dim] on GPU
        Y_target_all: All possible target embeddings [n_targets, embed_dim] (Tensor or numpy)
        device: torch device

    Returns:
        Dictionary with MRR and Recall@K metrics
    """
    # Ensure Y_target_all is a tensor on the correct device
    if not isinstance(Y_target_all, torch.Tensor):
        Y_target_tensor = torch.from_numpy(Y_target_all).to(device)
    else:
        Y_target_tensor = Y_target_all.to(device)

    # Calculate cosine similarity on GPU: [n_samples, n_targets]
    # Using matrix multiplication for efficiency: pred @ targets.T / (norms)
    y_pred_norm = torch.norm(y_pred_vectors, dim=1, keepdim=True)
    y_target_norm = torch.norm(Y_target_tensor, dim=1, keepdim=True).t()

    # Cosine similarity matrix
    sim_matrix = torch.mm(y_pred_vectors, Y_target_tensor.t()) / (y_pred_norm @ y_target_norm)

    # Sort indices in descending order of similarity
    sorted_indices = torch.argsort(sim_matrix, dim=1, descending=True)

    # Find true target indices by finding closest match for each true vector
    # This is necessary because y_true_vectors are embeddings, not indices
    true_target_indices = []
    
    # Optimization: Calculate distances to all targets for the true vectors to find their indices in Y_target_tensor
    # This can be expensive if done per sample. 
    # Alternatively, we can assume Y_target_all is the reference and find indices once if possible.
    # But here we just search for the closest embedding in Y_target_tensor for each y_true.
    
    # Since y_true is from the dataset, it SHOULD be identical to one of the rows in Y_target_tensor.
    # We can use dot product to find exact match if normalized, or just minimal distance.
    # Let's use the same similarity logic.
    
    for i, y_true in enumerate(y_true_vectors):
        # We want to find which row in Y_target_tensor corresponds to y_true
        # Calculate sim between y_true and all targets
        # y_true is (dim,)
        # Y_target_tensor is (n_targets, dim)
        
        # We can do this in batch if we want, but let's stick to loop for clarity/safety from original script
        # Original script:
        # distances = torch.norm(Y_target_tensor - y_true.unsqueeze(0), dim=1)
        # true_idx = torch.argmin(distances).item()
        
        diff = Y_target_tensor - y_true.unsqueeze(0)
        dist_sq = torch.sum(diff ** 2, dim=1)
        true_idx = torch.argmin(dist_sq).item()
        
        true_target_indices.append(true_idx)

    # Calculate MRR
    reciprocal_ranks = []
    sorted_indices_cpu = sorted_indices.cpu()
    
    for i, true_idx in enumerate(true_target_indices):
        # rank is where true_idx appears in sorted_indices[i]
        # We can use torch.where or convert to list
        # sorted_indices[i] is a tensor of indices
        
        # Find index of true_idx in sorted_indices[i]
        rank_idx = (sorted_indices_cpu[i] == true_idx).nonzero()
        
        if len(rank_idx) > 0:
            rank = rank_idx.item() + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            # Should not happen if true target is in Y_target_all
            reciprocal_ranks.append(0.0)

    mrr = np.mean(reciprocal_ranks)

    # Calculate Recall@K (using standard K values)
    k_values = [1, 5, 10, 20]
    recall_at_k = {}
    sorted_indices_np = sorted_indices_cpu.numpy()

    for k in k_values:
        hits = 0
        for i, true_idx in enumerate(true_target_indices):
            if true_idx in sorted_indices_np[i, :k]:
                hits += 1
        recall_at_k[f'R@{k}'] = hits / len(true_target_indices)

    metrics = {'MRR': mrr}
    metrics.update(recall_at_k)

    return metrics

def evaluate_ranking(model, dataloader, all_targets_tensor, device):
    model.eval()
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for history, target, lengths in dataloader:
            history = history.to(device)
            target = target.to(device)
            lengths = lengths.to(device)
            
            batch_size, seq_len, _ = history.size()
            mask = torch.arange(seq_len, device=device).expand(batch_size, seq_len) >= lengths.unsqueeze(1)
            
            output = model.forward_with_lengths(history, lengths=lengths, mask=mask)
            
            all_outputs.append(output)
            all_targets.append(target)
            
    # Concatenate
    if not all_outputs:
        return {}
        
    y_pred = torch.cat(all_outputs, dim=0)
    y_true = torch.cat(all_targets, dim=0)
    
    return calculate_ranking_metrics_gpu(y_pred, y_true, all_targets_tensor, device)

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for history, target, lengths in dataloader:
        history = history.to(device)
        target = target.to(device)
        lengths = lengths.to(device)
        
        # Create mask for padding
        # shape: (B, L)
        # mask is True where padded
        batch_size, seq_len, _ = history.size()
        mask = torch.arange(seq_len, device=device).expand(batch_size, seq_len) >= lengths.unsqueeze(1)
        
        optimizer.zero_grad()
        
        # Use forward_with_lengths to correctly select the last valid job state
        output = model.forward_with_lengths(history, lengths=lengths, mask=mask)
        
        # CosineEmbeddingLoss expects target as 1 or -1
        # Here we want positive match, so target is 1
        # output and target are (B, dim)
        loss = criterion(output, target, torch.ones(batch_size, device=device))
        
        # Optional: Add negative sampling logic here if desired
        # The prompt mentioned "make it with a random vector far (-1)"
        # We can shuffle targets to create negatives within batch
        if batch_size > 1:
            neg_target = target[torch.randperm(batch_size)]
            neg_output = output
            loss_neg = criterion(neg_output, neg_target, -torch.ones(batch_size, device=device))
            loss = (loss + loss_neg) / 2
            
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for history, target, lengths in dataloader:
            history = history.to(device)
            target = target.to(device)
            lengths = lengths.to(device)
            
            batch_size, seq_len, _ = history.size()
            mask = torch.arange(seq_len, device=device).expand(batch_size, seq_len) >= lengths.unsqueeze(1)
            
            output = model.forward_with_lengths(history, lengths=lengths, mask=mask)
            
            loss = criterion(output, target, torch.ones(batch_size, device=device))
            
            if batch_size > 1:
                neg_target = target[torch.randperm(batch_size)]
                loss_neg = criterion(output, neg_target, -torch.ones(batch_size, device=device))
                loss = (loss + loss_neg) / 2

            total_loss += loss.item()
            
    return total_loss / len(dataloader)

def objective(trial, args):
    # Hyperparameters to tune
    d_model = trial.suggest_categorical('d_model', [512])
    n_layers = trial.suggest_int('n_layers', 2, 3)
    n_heads = trial.suggest_categorical('n_heads', [4, 8])
    lr = trial.suggest_float('lr', 4e-5, 8e-5, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.2)
    
    # Ensure d_model is divisible by n_heads
    if d_model % n_heads != 0:
        # Force compatible heads
        # raise or adjust. Optuna suggests independent, so we might get invalid configs.
        # Simple fix: adjust d_model to be multiple
        d_model = (d_model // n_heads) * n_heads
        # Or just prune/fail invalid
    
    # Setup WandB for this trial if enabled
    if WANDB_AVAILABLE and args.use_wandb:
        run_name = f"{args.run_name}_trial_{trial.number}" if args.run_name else f"trial_{trial.number}"
        config = {
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "lr": lr,
            "dropout": dropout,
            "input_dim": args.input_dim,
            "data_type": args.data_type,
            "combine_method": args.combine_method,
            "use_skills": (args.skill_embeddings_path is not None)
        }
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=config,
            reinit=True,
            group=args.wandb_group or "optuna_study"
        )
    
    # Dataset
    # We should load dataset once outside, but for simplicity here (and if params affect data)
    # Actually data is static. We can pass it.
    
    # Model
    model = CareerPathAggregator(
        input_dim=args.input_dim, # Fixed by data
        model_dim=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout
    ).to(args.device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CosineEmbeddingLoss()
    
    # Pruning
    # trial.report(val_accuracy, step)
    # if trial.should_prune(): raise optuna.TrialPruned()
    
    val_loss = float('inf')
    
    try:
        # Training Loop
        for epoch in range(args.epochs):
            train_loss = train_one_epoch(model, args.train_loader, optimizer, criterion, args.device)
            val_loss = evaluate(model, args.val_loader, criterion, args.device)
            
            logger.info(f"Trial {trial.number}, Epoch {epoch}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}")
            
            if WANDB_AVAILABLE and args.use_wandb:
                wandb.log({
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "epoch": epoch
                })
            
            trial.report(val_loss, epoch)
            if trial.should_prune():
                if WANDB_AVAILABLE and args.use_wandb:
                    wandb.log({"pruned": True})
                raise optuna.exceptions.TrialPruned()
                
    finally:
        if WANDB_AVAILABLE and args.use_wandb:
            wandb.finish()
            
    return val_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job_embeddings_path", type=str, required=True)
    parser.add_argument("--skill_embeddings_path", type=str, default=None, help="Optional path to skill embeddings")
    parser.add_argument("--data_type", type=str, default="karrierewege_100k")
    parser.add_argument("--occupations_path", type=str, default="data/occupations_en.csv")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="experiments/seq_transformer")
    parser.add_argument("--combine_method", type=str, default="concat")
    parser.add_argument("--use_all_subspans", action="store_true", help="Use all contiguous subsequences for training (augmentation)")
    
    # Model hyperparameters (used if --skip_hpo is set or as defaults)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--skip_hpo", action="store_true", help="Skip hyperparameter optimization and use provided params")
    
    # WandB arguments
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="career-path-transformer", help="WandB project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="WandB entity")
    parser.add_argument("--wandb_group", type=str, default=None, help="WandB group name")
    parser.add_argument("--run_name", type=str, default=None, help="Base name for the run")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(42)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Dataset once
    logger.info("Loading dataset...")
    full_dataset = CareerSequenceDataset(
        job_embeddings_path=args.job_embeddings_path,
        skill_embeddings_path=args.skill_embeddings_path,
        data_type=args.data_type,
        occupations_path=args.occupations_path,
        split="train", # Use training split for HPO
        combine_method=args.combine_method,
        max_len=100,
        use_all_subspans=args.use_all_subspans
    )
    
    # Split for HPO (Train/Val)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    args.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    args.val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Determine input dim from first sample
    if len(full_dataset) > 0:
        sample_hist, sample_target = full_dataset[0]
        args.input_dim = sample_hist.shape[-1]
        logger.info(f"Input dimension determined: {args.input_dim}")
    else:
        logger.error("Dataset is empty!")
        return

    # Optuna or Manual Params
    if not args.skip_hpo:
        study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
        study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
        
        logger.info("Best trial:")
        trial = study.best_trial
        logger.info(f"  Value: {trial.value}")
        logger.info("  Params: ")
        for key, value in trial.params.items():
            logger.info(f"    {key}: {value}")
            
        best_params = study.best_params
    else:
        logger.info("Skipping HPO, using provided parameters...")
        best_params = {
            'd_model': args.d_model,
            'n_layers': args.n_layers,
            'n_heads': args.n_heads,
            'lr': args.lr,
            'dropout': args.dropout
        }
        
    # Retrain with best params and evaluate on Test set
    logger.info("Retraining with best parameters on full training set...")
    
    d_model = best_params['d_model']
    n_heads = best_params['n_heads']
    n_layers = best_params['n_layers']
    dropout = best_params['dropout']
    lr = best_params['lr']
    
    # Ensure d_model is divisible by n_heads (logic from objective function)
    if d_model % n_heads != 0:
        d_model = (d_model // n_heads) * n_heads

    final_model = CareerPathAggregator(
        input_dim=args.input_dim,
        model_dim=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=dropout
    ).to(args.device)
    
    optimizer = optim.Adam(final_model.parameters(), lr=lr)
    criterion = nn.CosineEmbeddingLoss()
    
    # Use full_dataset for training (combine train + val used in HPO)
    final_train_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    final_model.train()
    for epoch in range(args.epochs):
        loss = train_one_epoch(final_model, final_train_loader, optimizer, criterion, args.device)
        logger.info(f"Final Training Epoch {epoch}: Loss {loss:.4f}")
        
    # Evaluate on Test
    logger.info("Loading test set...")
    test_dataset = CareerSequenceDataset(
        job_embeddings_path=args.job_embeddings_path,
        skill_embeddings_path=args.skill_embeddings_path,
        data_type=args.data_type,
        occupations_path=args.occupations_path,
        split="test",
        combine_method=args.combine_method,
        max_len=100,
        use_all_subspans=args.use_all_subspans
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Prepare all targets for ranking
    logger.info("Preparing target embeddings for ranking...")
    target_vectors = []
    
    # Iterate over all available jobs in the embedding file to build target set
    # Using full_dataset as source since it has the loaded embeddings
    for uri, emb in full_dataset.job_embeddings.items():
        if full_dataset.use_skills:
            if uri in full_dataset.skill_embeddings:
                 target_vectors.append(full_dataset._get_combined_vector(uri))
        else:
            target_vectors.append(emb)
            
    if not target_vectors:
        logger.error("No valid target vectors found!")
        return

    all_targets_tensor = torch.stack(target_vectors).to(args.device)
    
    logger.info(f"Evaluating on test set ({len(test_dataset)} samples) against {len(all_targets_tensor)} targets...")
    metrics = evaluate_ranking(final_model, test_loader, all_targets_tensor, args.device)
    
    logger.info("Test Metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
        
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.init(
            project=args.wandb_project, 
            entity=args.wandb_entity,
            name=f"{args.run_name}_final_test" if args.run_name else "final_test", 
            config=best_params, 
            reinit=True,
            group=args.wandb_group or "final_evaluation"
        )
        wandb.log(metrics)
        wandb.finish()
    
if __name__ == "__main__":
    main()
