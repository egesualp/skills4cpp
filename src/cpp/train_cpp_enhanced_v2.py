"""
Enhanced Training Script for Career Path Prediction.

Combines:
- On-the-fly dataset (storage efficient)
- Optuna hyperparameter optimization
- Proper ranking metrics (MRR, Recall@K)
- CosineEmbeddingLoss
- Early stopping
- Multi-modal OR concatenation architecture
- Support for different encoders
"""

import argparse
import os
import re
import sys
import time
from typing import Dict, List, Tuple
import multiprocessing
from loguru import logger
import pandas as pd

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm, trange

try:
    from src.cpp.data_classes import Data
    from src.cpp.utils import SEP_TOKEN
    from src.cpp.cpp_dataset import CareerPathDataset, collate_career_path_batch
    from src.cpp.data_loaders import (
        load_all_vocabs,
        load_job_and_skill_data,
        precompute_target_embeddings,
        precompute_input_embeddings
    )
    from src.cpp.generate_embeddings import get_or_compute_embeddings
except ImportError as e:
    print(f"Error: Required modules not found. {e}")
    sys.exit(1)

# Configure logging
# Remove default handler (optional but cleaner)
logger.remove()

# Add a new handler (stdout + file)
logger.add(
    "logs/debug.log",
    format="{time} | {level} | {message}",
    level="DEBUG",
    rotation="10 MB",      # auto-rotate
    retention="7 days",    # keep logs for 7 days
    enqueue=True           # thread/process safe
)

logger.add(
    sys.stdout,
    format="<green>{time}</green> | <level>{message}</level>",
    level="INFO"
)


# Add Optuna logger
#optuna.logger.set_verbosity(optuna.logger.INFO)

# Or for more detail:
#optuna.logger.set_verbosity(optuna.logger.DEBUG)

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class MultiModalCPPModel(nn.Module):
    """
    Multi-modal model with shared encoder hyperparameters.
    Option 2: Shared hyperparameters + Linear fusion head.
    Supports variable number of modalities for ablation studies.
    """
    
    def __init__(self, text_dim, skill_text_dim, structured_dim, 
                 hidden_dim, n_layers, dropout, output_dim,
                 use_modality_weights=False,
                 use_text=True, use_skill=True, use_struct=True):
        super().__init__()
        
        self.use_text = use_text
        self.use_skill = use_skill
        self.use_struct = use_struct
        
        # Count active modalities
        n_modalities = sum([use_text, use_skill, use_struct])
        if n_modalities == 0:
            raise ValueError("At least one modality must be enabled!")
        
        # Shared encoder architecture for each enabled modality
        if use_text:
            self.text_encoder = self._build_encoder(text_dim, hidden_dim, n_layers, dropout)
        if use_skill:
            self.skill_encoder = self._build_encoder(skill_text_dim, hidden_dim, n_layers, dropout)
        if use_struct:
            self.struct_encoder = self._build_encoder(structured_dim, hidden_dim, n_layers, dropout)
        
        # Optional: Learnable modality weights
        self.use_modality_weights = use_modality_weights
        if use_modality_weights:
            if use_text:
                self.alpha_text = nn.Parameter(torch.tensor(1.0))
            if use_skill:
                self.alpha_skill = nn.Parameter(torch.tensor(1.0))
            if use_struct:
                self.alpha_struct = nn.Parameter(torch.tensor(1.0))
        
        # Simple linear fusion head (size depends on active modalities)
        self.fusion_head = nn.Linear(hidden_dim * n_modalities, output_dim)
    
    def _build_encoder(self, input_dim, hidden_dim, n_layers, dropout):
        """Build encoder with shared architecture."""
        layers = []
        current_dim = input_dim
        
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def forward(self, batch):
        # Encode each enabled modality
        encoded_modalities = []
        
        if self.use_text:
            h_text = self.text_encoder(batch['h_text'])
            if self.use_modality_weights:
                h_text = self.alpha_text * h_text
            encoded_modalities.append(h_text)
        
        if self.use_skill:
            h_skill = self.skill_encoder(batch['h_skill_text'])
            if self.use_modality_weights:
                h_skill = self.alpha_skill * h_skill
            encoded_modalities.append(h_skill)
        
        if self.use_struct:
            # Handle structured (might have multiple keys)
            structured_keys = [k for k in batch.keys() if k.startswith('h_structured_')]
            h_struct_concat = torch.cat([batch[k] for k in structured_keys], dim=1)
            h_struct = self.struct_encoder(h_struct_concat)
            if self.use_modality_weights:
                h_struct = self.alpha_struct * h_struct
            encoded_modalities.append(h_struct)
        
        # Simple fusion: concatenate + linear
        fused = torch.cat(encoded_modalities, dim=1)
        output = self.fusion_head(fused)
        
        return output


class SimpleConcatModel(nn.Module):
    """
    Simple concatenation model (Gemini's approach).
    Early fusion: concatenate all features immediately.
    Supports variable number of modalities for ablation studies.
    """
    
    def __init__(self, input_dim, output_dim, n_layers, hidden_dim, dropout,
                 use_text=True, use_skill=True, use_struct=True):
        super().__init__()
        
        self.use_text = use_text
        self.use_skill = use_skill
        self.use_struct = use_struct
        
        # Validate at least one modality
        if not any([use_text, use_skill, use_struct]):
            raise ValueError("At least one modality must be enabled!")
        
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
        # Concatenate enabled features
        features = []
        
        if self.use_text:
            features.append(batch['h_text'])
        
        if self.use_skill:
            features.append(batch['h_skill_text'])
        
        if self.use_struct:
            # Add structured features
            structured_keys = [k for k in batch.keys() if k.startswith('h_structured_')]
            features.extend([batch[k] for k in structured_keys])
        
        x = torch.cat(features, dim=1)
        return self.model(x)


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def calculate_ranking_metrics(y_pred_vectors: np.ndarray, 
                              y_true_vectors: np.ndarray, 
                              Y_target_all: np.ndarray,
                              k_values: List[int] = [1, 5, 10, 20]) -> Dict[str, float]:
    """
    Calculate ranking metrics: MRR and Recall@K.
    
    Args:
        y_pred_vectors: Predicted embeddings [n_samples, embed_dim]
        y_true_vectors: True target embeddings [n_samples, embed_dim]
        Y_target_all: All possible target embeddings [n_targets, embed_dim]
        k_values: List of K values for Recall@K
        
    Returns:
        Dictionary with MRR and Recall@K metrics
    """
    # Calculate cosine similarity between predictions and all targets
    sim_matrix = cosine_similarity(y_pred_vectors, Y_target_all)
    
    # Sort indices in descending order of similarity
    sorted_indices = np.argsort(sim_matrix, axis=1)[:, ::-1]
    
    # Find true target indices
    true_target_indices = []
    for y_true in y_true_vectors:
        true_index = np.where((Y_target_all == y_true).all(axis=1))[0][0]
        true_target_indices.append(true_index)
    
    # Calculate MRR
    reciprocal_ranks = []
    for i, true_idx in enumerate(true_target_indices):
        rank_list = list(sorted_indices[i])
        rank = rank_list.index(true_idx) + 1
        reciprocal_ranks.append(1.0 / rank)
    
    mrr = np.mean(reciprocal_ranks)
    
    # Calculate Recall@K
    recall_at_k = {}
    for k in k_values:
        hits = 0
        for i, true_idx in enumerate(true_target_indices):
            if true_idx in sorted_indices[i, :k]:
                hits += 1
        recall_at_k[f'R@{k}'] = hits / len(true_target_indices)
    
    metrics = {'MRR': mrr}
    metrics.update(recall_at_k)
    
    return metrics


def calculate_ranking_metrics_gpu(y_pred_vectors, y_true_vectors, Y_target_all, device):
    """
    GPU-optimized ranking metrics calculation.

    Args:
        y_pred_vectors: Predicted embeddings [n_samples, embed_dim] on GPU
        y_true_vectors: True target embeddings [n_samples, embed_dim] on GPU
        Y_target_all: All possible target embeddings [n_targets, embed_dim] as numpy array
        device: torch device

    Returns:
        Dictionary with MRR and Recall@K metrics
    """
    # Move target embeddings to GPU once
    Y_target_tensor = torch.from_numpy(Y_target_all).to(device)

    # Calculate cosine similarity on GPU: [n_samples, n_targets]
    # Using matrix multiplication for efficiency: pred @ targets.T / (norms)
    y_pred_norm = torch.norm(y_pred_vectors, dim=1, keepdim=True)
    y_target_norm = torch.norm(Y_target_tensor, dim=1, keepdim=True).t()

    # Cosine similarity matrix
    sim_matrix = torch.mm(y_pred_vectors, Y_target_tensor.t()) / (y_pred_norm @ y_target_norm)

    # Sort indices in descending order of similarity
    sorted_indices = torch.argsort(sim_matrix, dim=1, descending=True)

    # Find true target indices by finding closest match for each true vector
    true_target_indices = []
    for y_true in y_true_vectors:
        # Find the index of the closest target embedding
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
            # If true target not found in top results, assign rank = n_targets + 1
            reciprocal_ranks.append(1.0 / (len(rank_list) + 1))

    mrr = np.mean(reciprocal_ranks)

    # Calculate Recall@K (using standard K values)
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

def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None, use_mixed_precision=False, gradient_accumulation_steps=1, profile_data_loading=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    accumulation_counter = 0
    data_time = 0
    gpu_time = 0
    batch_start = time.time()

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="In-epoch Training")):
        # Time data loading
        data_end = time.time()
        data_time += data_end - batch_start
        
        gpu_start = time.time()
        batch = {k: v.to(device) for k, v in batch.items()}

        if accumulation_counter == 0:
            optimizer.zero_grad()

        if use_mixed_precision and scaler is not None:
            with torch.cuda.amp.autocast():
                y_pred = model(batch)
                # CosineEmbeddingLoss requires target labels (1 for similar)
                target = torch.ones(y_pred.size(0)).to(device)
                loss = criterion(y_pred, batch['y'], target)

            # Scale loss by accumulation steps and call backward
            loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()
        else:
            y_pred = model(batch)
            # CosineEmbeddingLoss requires target labels (1 for similar)
            target = torch.ones(y_pred.size(0)).to(device)
            loss = criterion(y_pred, batch['y'], target)

            # Scale loss by accumulation steps and call backward
            loss = loss / gradient_accumulation_steps
            loss.backward()

        total_loss += loss.item()
        accumulation_counter += 1

        # Update parameters every gradient_accumulation_steps
        if accumulation_counter >= gradient_accumulation_steps:
            if use_mixed_precision and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            accumulation_counter = 0

        # Time GPU computation
        gpu_time += time.time() - gpu_start
        batch_start = time.time()

    # Handle remaining accumulated gradients at end of epoch
    if accumulation_counter > 0:
        if use_mixed_precision and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

    # Log timing breakdown to diagnose data loading bottlenecks
    avg_loss = total_loss / len(dataloader)
    if len(dataloader) > 0 and profile_data_loading:
        total_time = data_time + gpu_time
        if total_time > 0:
            logger.info(f"⏱️  Epoch timing: Data loading: {data_time:.2f}s ({data_time/total_time*100:.1f}%), "
                       f"GPU compute: {gpu_time:.2f}s ({gpu_time/total_time*100:.1f}%)")
            if data_time / total_time > 0.3:  # More than 30% time in data loading
                logger.warning("⚠️  Data loading bottleneck detected! Consider increasing --num_workers or pre-computing more embeddings.")

    return avg_loss


def evaluate(model, dataloader, Y_target_all, device, criterion=None):
    """Evaluate model and return ranking metrics."""
    model.eval()
    all_y_pred = []
    all_y_true = []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluation"):
            batch = {k: v.to(device) for k, v in batch.items()}
            y_pred = model(batch)

            if criterion:
                target = torch.ones(y_pred.size(0)).to(device)
                loss = criterion(y_pred, batch['y'], target)
                total_loss += loss.item()

            # Keep tensors on GPU for efficiency
            all_y_pred.append(y_pred)
            all_y_true.append(batch['y'])

    # Concatenate on GPU
    y_pred_vectors = torch.cat(all_y_pred, dim=0)
    y_true_vectors = torch.cat(all_y_true, dim=0)

    # Use GPU-optimized ranking metrics
    metrics = calculate_ranking_metrics_gpu(y_pred_vectors, y_true_vectors, Y_target_all, device)

    if criterion:
        metrics['loss'] = total_loss / len(dataloader)

    return metrics


# ============================================================================
# OPTUNA OBJECTIVE
# ============================================================================

def objective(trial, train_loader, val_sample_loader, Y_target_all, args,
              text_dim, skill_text_dim, structured_dim, output_dim, scaler=None):
    """Optuna objective function."""
    
    device = torch.device(args.device)
    
    # Suggest hyperparameters
    if args.use_advanced:
        # Multi-modal architecture
        hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 768, 1024])
        n_layers = trial.suggest_int("n_layers", 1, 5)
        dropout = trial.suggest_float("dropout", 0.01, 0.5)
        use_modality_weights = trial.suggest_categorical("use_modality_weights", [True, False])

        # log suggested hyperparameters
        logger.info(f"  > Suggested hyperparameters: hidden_dim={hidden_dim}, n_layers={n_layers}, dropout={dropout}, use_modality_weights={use_modality_weights}")
        
        model = MultiModalCPPModel(
            text_dim=text_dim,
            skill_text_dim=skill_text_dim,
            structured_dim=structured_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            output_dim=output_dim,
            use_modality_weights=use_modality_weights,
            use_text=args.use_text_history,
            use_skill=args.use_skill_text,
            use_struct=args.use_structured
        ).to(device)
    else:
        # Simple concatenation
        hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 768, 1024])
        n_layers = trial.suggest_int("n_layers", 1, 4)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        
        # log suggested hyperparameters
        logger.info(f"  > Suggested hyperparameters: hidden_dim={hidden_dim}, n_layers={n_layers}, dropout={dropout}")
        
        # Calculate input dimension based on enabled modalities
        input_dim = 0
        if args.use_text_history:
            input_dim += text_dim
        if args.use_skill_text:
            input_dim += skill_text_dim
        if args.use_structured:
            input_dim += structured_dim
        
        model = SimpleConcatModel(
            input_dim=input_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_text=args.use_text_history,
            use_skill=args.use_skill_text,
            use_struct=args.use_structured
        ).to(device)
    
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    
    # Create optimizer based on configuration
    if args.optimizer == "sgd":
        momentum = trial.suggest_float("momentum", 0.0, 0.99)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        nesterov = trial.suggest_categorical("nesterov", [True, False])
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, 
                             weight_decay=weight_decay, nesterov=nesterov)
        logger.info(f"  > Using SGD: lr={lr:.6f}, momentum={momentum:.4f}, weight_decay={weight_decay:.6f}, nesterov={nesterov}")
    else:  # adam
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        logger.info(f"  > Using Adam: lr={lr:.6f}, weight_decay={weight_decay:.6f}")
    
    criterion = nn.CosineEmbeddingLoss()
    
    best_val_mrr = 0.0
    epochs_no_improve = 0
    epochs_total = 0
    
    # Training loop
    for epoch in range(args.max_epochs):
        epoch_start_time = time.time()
        epochs_total += 1
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler, args.mixed_precision, args.gradient_accumulation_steps, args.profile_data_loading)
        val_metrics = evaluate(model, val_sample_loader, Y_target_all, device, criterion)
        val_mrr = val_metrics['MRR']
        epoch_time = time.time() - epoch_start_time
        
        # Prepare metrics for logging
        log_metrics = {
            'train_loss': train_loss,
            'val_mrr': val_mrr,
            'val_loss': val_metrics.get('loss', 0),
        }
        for k, v in val_metrics.items():
            if k not in ['MRR', 'loss']:
                log_metrics[f'val_{k.lower()}'] = v

        logger.info(f"  Trial {trial.number} | Epoch {epoch+1}/{args.max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics.get('loss', 0):.4f} | Val MRR: {val_mrr:.4f} | Time: {epoch_time:.1f}s")
        
        if WANDB_AVAILABLE and args.use_wandb:
            wandb.log(log_metrics, step=epochs_total)

        # Report to Optuna
        trial.report(val_mrr, epoch)
        
        # Pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # Early stopping with minimum delta
        if val_mrr > best_val_mrr + args.min_delta:
            best_val_mrr = val_mrr
            epochs_no_improve = 0
            best_epoch = epochs_total
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= args.optuna_patience:
            logger.info(f"  Early stopping triggered after {epoch+1} epochs (Optuna patience: {args.optuna_patience}).")
            break

    trial.set_user_attr("best_epoch", best_epoch)
    return best_val_mrr

def logger_callback(study, trial):
    logger.info(f"\n✓ Trial {trial.number} finished with MRR: {trial.value:.4f}")
    logger.info(f"  Best so far: Trial {study.best_trial.number} with MRR: {study.best_value:.4f}")
    logger.info(f"  Params: {trial.params}\n")


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced CPP Training with Optuna")
    
    # Data paths
    parser.add_argument("--data_type", type=str, default="decorte")
    parser.add_argument("--master_skill_file", type=str, 
                       default="/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/results/decorte_jobbert_v2_baseline/job_title_skills_master.csv")
    parser.add_argument("--esco_skills_file", type=str, default="data/esco_datasets/skills_en.csv")
    parser.add_argument("--vocab_dir", type=str, default="data/processed/master_datasets_2/")
    parser.add_argument("--skill_properties_file", type=str, default="data/processed/master_datasets_2/skill_properties_map.json")
    
    # Encoder configuration
    parser.add_argument("--encoder_text", type=str, default="ElenaSenger/career-path-representation-mpnet-decorte",
                       help="Encoder for text history")
    parser.add_argument("--encoder_skill", type=str, default="",
                       help="Encoder for skills (if empty, use same as encoder_text)")
    
    # Feature configuration
    parser.add_argument("--use_text_description", action='store_true',
                       help="Include job descriptions in text history (default: titles only)")
    parser.add_argument("--use_skill_description", action='store_true')
    parser.add_argument("--last_job_only", action='store_true')
    parser.add_argument("--pooling_strategy", type=str, default="weighted_idf", 
                       choices=["mean", "weighted_mean", "weighted_idf"])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument(
        "--use_skill_path_log_pooling",
        action='store_true',
        help="Use skills_v2-style per-job skill pooling with logarithmic position weighting across jobs for skill text embeddings.",
    )
    parser.add_argument(
        "--skill_path_alpha_decay",
        type=float,
        default=0.5,
        help="Logarithmic decay parameter for job position weighting when --use_skill_path_log_pooling is enabled (0 = mean over jobs).",
    )
    
    # Modality selection (for ablation studies)
    parser.add_argument("--use_text_history", action='store_true', 
                       help="Include job history text features")
    parser.add_argument("--use_skill_text", action='store_true',
                       help="Include skill text features")
    parser.add_argument("--use_structured", action='store_true',
                       help="Include structured meta-features")
    
    # Architecture
    parser.add_argument("--use_advanced", action='store_true',
                       help="Use multi-modal architecture. Auto-enabled for 2+ modalities if not specified.")
    
    # Optuna configuration
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--max_epochs", type=int, default=10, help="Max epochs per trial")
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience for final training")
    parser.add_argument("--optuna_patience", type=int, default=3, help="Early stopping patience for Optuna trials (stricter)")
    parser.add_argument("--val_sample_ratio", type=float, default=0.1, 
                       help="Ratio of validation set to use during Optuna trials (e.g., 0.1 = 10%)")
    parser.add_argument("--min_delta", type=float, default=0.001, 
                       help="Minimum improvement required to reset patience counter (e.g., 0.001 = 0.1% MRR improvement)")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=4092, 
                       help="Batch size for evaluation (validation/test)")
    parser.add_argument("--num_workers", type=int, default=None,
                       help="Number of DataLoader workers (auto-detects from SLURM_CPUS_PER_TASK if not set, default: min(16, cpu_count))")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="results/cpp")
    parser.add_argument("--study_name", type=str, default="cpp_optuna_study")
    parser.add_argument("--results_csv_path", type=str, default="results/cpp/experiment_results.csv",
                       help="Path to save experiment results CSV file.")
    parser.add_argument("--save_model", action='store_true', help="Save the final model")

    # Wandb logging
    parser.add_argument("--use_wandb", action='store_true', help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="cpp-enhanced", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="Wandb entity name")

    # Static parameters
    parser.add_argument("--optuna", action='store_true', help="Run Optuna optimization")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument("--n_layers", type=int, default=1, help="Number of layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--use_modality_weights", action='store_true', help="Use modality weights")
    
    # Optimizer configuration
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "sgd"],
                       help="Optimizer to use (adam or sgd)")
    parser.add_argument("--momentum", type=float, default=0.9,
                       help="Momentum for SGD optimizer (static mode)")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                       help="Weight decay for optimizer (static mode)")
    parser.add_argument("--nesterov", action='store_true',
                       help="Use Nesterov momentum for SGD")

    # Mixed precision training
    parser.add_argument("--mixed_precision", action='store_true',
                       help="Enable mixed precision training (FP16)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Number of steps to accumulate gradients (simulates larger batch size)")
    
    # Output options
    parser.add_argument("--save_study", action='store_true', 
                       help="Save Optuna study to pickle file")

    parser.add_argument("--run_name", type=str, default="cpp_enhanced", help="Run name")
    parser.add_argument("--profile_data_loading", action='store_true',
                       help="Enable data loading vs GPU compute timing diagnostics")
    
    # Embeddings cache
    parser.add_argument("--embeddings_cache_dir", type=str, 
                       default="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/embeddings",
                       help="Directory to save/load pre-computed embeddings cache")
    parser.add_argument("--force_recompute", action='store_true',
                       help="Force recomputation of embeddings even if cache exists")
    
    return parser.parse_args()

# ============================================================================
# HELPERS
# ============================================================================

def extract_raw_titles_from_doc(doc: str) -> List[str]:
        """Extract raw job titles from a document (handles both formatted and plain text).
        
        Note: Titles are normalized (lowercase + stripped) to match the job_skill_map format.
        """
        # For formatted documents: "esco role: cook\n description: ..."
        titles = re.findall(r"esco role: (.*?)\n", doc)
        if not titles:
            # Fallback: try regular role pattern (for history)
            titles = re.findall(r"role: (.*?)\n", doc)
        if not titles:
            # Fallback: assume plain title(s) with possible SEP_TOKEN
            titles = [t.strip() for t in doc.split(SEP_TOKEN) if t.strip()]
        # Normalize titles to match mapping file format (lowercase + stripped)
        return [t.strip().lower() for t in titles]

# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()
    
    # Fix CUDA multiprocessing issue: use 'spawn' instead of 'fork'
    # This must be done before any CUDA operations (encoder loading)
    #try:
    #    multiprocessing.set_start_method('spawn', force=True)
    #    logger.info("🖥️  CUDA multiprocessing set to 'spawn'")
    #except RuntimeError:
    #    # Already set, which is fine
    #    pass
    
    # Validate modality selection
    n_active_modalities = sum([args.use_text_history, args.use_skill_text, args.use_structured])
    if n_active_modalities == 0:
        raise ValueError("At least one modality must be enabled! Use --use_text_history, --use_skill_text, or --use_structured")
    
    # Auto-enable advanced mode for multiple modalities (unless explicitly set)
    if n_active_modalities >= 2 and not args.use_advanced:
        logger.info("ℹ️  Multiple modalities detected. Using SimpleConcatModel (early fusion).")
        logger.info("   To use MultiModalCPPModel (late fusion), add --use_advanced flag.\n")
        #args.use_advanced = True
    elif n_active_modalities == 1 and args.use_advanced:
        logger.warning("⚠️  Warning: Using multi-modal architecture with only one modality.")
        logger.warning("   Consider using simple mode (without --use_advanced) for single-modality experiments.\n")
    
    logger.info("=" * 80)
    logger.info("Enhanced Career Path Prediction Training")
    logger.info("=" * 80)
    logger.info(f"Architecture: {'Multi-modal (Advanced)' if args.use_advanced else 'Simple Concatenation'}")
    logger.info(f"Active Modalities ({n_active_modalities}):")
    logger.info(f"  - Text History: {'✓' if args.use_text_history else '✗'}" + (f" ({'with descriptions' if args.use_text_description else 'titles only'})" if args.use_text_history else ""))
    logger.info(f"  - Skill Text: {'✓' if args.use_skill_text else '✗'}" + (f" ({'with descriptions' if args.use_skill_description else 'names only'})" if args.use_skill_text else ""))
    logger.info(f"  - Structured Features: {'✓' if args.use_structured else '✗'}")
    logger.info(f"Configuration: {vars(args)}\n")
    
    # Initialize wandb
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args,
            name=args.run_name,
            reinit=True
        )
        logger.info(f"🚀 wandb logging enabled for run: {args.run_name}")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    
    # Auto-detect optimal num_workers if not specified
    if args.num_workers is None:
        # Try to get SLURM allocated CPUs first
        slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
        if slurm_cpus:
            args.num_workers = max(1, int(slurm_cpus) - 2)  # Leave 2 CPUs for main process
            logger.info(f"🖥️  Auto-detected num_workers={args.num_workers} from SLURM_CPUS_PER_TASK={slurm_cpus}")
        else:
            # Fallback: min(16, cpu_count - 1) to avoid overwhelming the system
            args.num_workers = min(16, max(1, multiprocessing.cpu_count() - 1))
            logger.info(f"🖥️  Auto-detected num_workers={args.num_workers} from CPU count")
    else:
        logger.info(f"🖥️  Using specified num_workers={args.num_workers}")
        
    logger.info(f"🖥️  DataLoader configuration: num_workers={args.num_workers}, pin_memory={device.type == 'cuda'}")
    
    # --- Step 1: Load encoders ---
    logger.info("[1/7] Loading encoder models...")
    encoder_text = SentenceTransformer(args.encoder_text)
    
    # Use separate skill encoder if specified
    if args.encoder_skill:
        logger.info(f"  > Using separate skill encoder: {args.encoder_skill}")
        encoder_skill = SentenceTransformer(args.encoder_skill)
        skill_text_dim = encoder_skill.get_sentence_embedding_dimension()
    else:
        logger.info(f"  > Using same encoder for skills")
        encoder_skill = encoder_text
        skill_text_dim = encoder_text.get_sentence_embedding_dimension()
    
    text_dim = encoder_text.get_sentence_embedding_dimension()
    logger.info(f"  ✓ Text encoder dim: {text_dim}, Skill encoder dim: {skill_text_dim}\n")
    
    # --- Step 2: Load data pairs ---
    logger.info("[2/7] Loading career path data...")
    # ONLY_TITLES: Extract just job titles (no job descriptions)
    # This is independent from skill extraction thanks to enhanced _extract_skill_infos()
    # which handles both formatted documents and plain titles
    data = Data(DATA_TYPE=args.data_type, ONLY_TITLES=not args.use_text_description)
    train_pairs, val_pairs, test_pairs = data.get_data(stage='transformation_finetuning')
    
    if args.last_job_only:
        logger.info(f"  > Filtering for 'last job only' pairs...")
        train_pairs = [pair for pair in train_pairs if SEP_TOKEN not in pair[0]]
        val_pairs = [pair for pair in val_pairs if SEP_TOKEN not in pair[0]]
        test_pairs = [pair for pair in test_pairs if SEP_TOKEN not in pair[0]]
    
    logger.info(f"  ✓ Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}\n")

    # --- Step 2b: Extract raw job titles for IDF calculation (avoid test set leakage) ---
    logger.info("[2b/7] Extracting train+val job titles for IDF calculation...")
    train_val_jobs = set()

    for history_doc, target_doc in train_pairs + val_pairs:
        # Extract titles from history
        train_val_jobs.update(extract_raw_titles_from_doc(history_doc))
        # Extract title from target
        train_val_jobs.update(extract_raw_titles_from_doc(target_doc))

    logger.info(f"  ✓ Extracted {len(train_val_jobs)} unique train+val job titles\n")

    # --- Step 3: Load helper maps ---
    logger.info("[3/7] Loading vocabularies and skill mappings...")
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
    logger.info(f"  ✓ Structured feature dim: {structured_dim}\n")

    # --- Step 3b: Check job-to-skill coverage ---
    logger.info("[3b/7] Checking job-to-skill coverage...")
    
    # Check train+val coverage (critical for training)
    train_val_jobs_in_dataset = set()
    for history_doc, target_doc in train_pairs + val_pairs:
        train_val_jobs_in_dataset.update(extract_raw_titles_from_doc(history_doc))
        train_val_jobs_in_dataset.update(extract_raw_titles_from_doc(target_doc))
    
    jobs_with_skills = set(job_skill_map.keys())
    train_val_missing = train_val_jobs_in_dataset - jobs_with_skills
    train_val_coverage = len(train_val_jobs_in_dataset - train_val_missing) / max(len(train_val_jobs_in_dataset), 1)
    
    logger.info(f"  > Train+Val unique jobs: {len(train_val_jobs_in_dataset)}")
    logger.info(f"  > Train+Val jobs with skills: {len(train_val_jobs_in_dataset - train_val_missing)}")
    logger.info(f"  > Train+Val jobs without skills: {len(train_val_missing)} ({100*(1-train_val_coverage):.1f}%)")
    logger.info(f"  > Train+Val coverage: {100*train_val_coverage:.1f}%")
    
    if train_val_missing:
        logger.warning(f"  ⚠️  {len(train_val_missing)} train+val jobs will receive zero skill embeddings")
        if len(train_val_missing) <= 20:
            logger.warning(f"  Missing train+val jobs: {sorted(train_val_missing)}")
        else:
            logger.warning(f"  First 20 missing train+val jobs: {sorted(list(train_val_missing))[:20]}")
    
    # Check test coverage (informational only)
    test_jobs_in_dataset = set()
    for history_doc, target_doc in test_pairs:
        test_jobs_in_dataset.update(extract_raw_titles_from_doc(history_doc))
        test_jobs_in_dataset.update(extract_raw_titles_from_doc(target_doc))
    
    test_missing = test_jobs_in_dataset - jobs_with_skills
    test_coverage = len(test_jobs_in_dataset - test_missing) / max(len(test_jobs_in_dataset), 1)
    
    logger.info(f"  > Test unique jobs: {len(test_jobs_in_dataset)}")
    logger.info(f"  > Test jobs with skills: {len(test_jobs_in_dataset - test_missing)}")
    logger.info(f"  > Test jobs without skills: {len(test_missing)} ({100*(1-test_coverage):.1f}%)")
    logger.info(f"  > Test coverage: {100*test_coverage:.1f}% (informational only)")
    logger.info("")
    
    # --- Step 4: Get or compute embeddings (with caching) ---
    (Y_target_dict, Y_target_all, output_dim,
     train_pairs, train_h_text, train_h_skill,
     val_pairs, val_h_text, val_h_skill,
     test_pairs, test_h_text, test_h_skill) = get_or_compute_embeddings(
        train_pairs, val_pairs, test_pairs,
        encoder_text, encoder_skill,
        job_skill_map, esco_skill_text_map,
        args,
        precompute_target_embeddings,
        precompute_input_embeddings
    )
    
    # --- Step 5: Create datasets ---
    logger.info("[6/7] Creating datasets...")
    
    train_dataset = CareerPathDataset(
        data_pairs=train_pairs,
        encoder=encoder_text,
        Y_target_dict=Y_target_dict,
        job_skill_map=job_skill_map,
        esco_skill_text_map=esco_skill_text_map,
        skill_properties_map=skill_properties_map,
        all_vocabs=all_vocabs,
        use_skill_description=args.use_skill_description,
        pooling_strategy=args.pooling_strategy,
        alpha=args.alpha,
        beta=args.beta,
        encoder_skill=encoder_skill,
        include_text=args.use_text_history,
        include_skill_text=args.use_skill_text,
        include_structured=args.use_structured,
        pre_h_text=train_h_text,
        pre_h_skill_text=train_h_skill,
    )

    val_dataset = CareerPathDataset(
        data_pairs=val_pairs,
        encoder=encoder_text,
        Y_target_dict=Y_target_dict,
        job_skill_map=job_skill_map,
        esco_skill_text_map=esco_skill_text_map,
        skill_properties_map=skill_properties_map,
        all_vocabs=all_vocabs,
        use_skill_description=args.use_skill_description,
        pooling_strategy=args.pooling_strategy,
        alpha=args.alpha,
        beta=args.beta,
        encoder_skill=encoder_skill,
        include_text=args.use_text_history,
        include_skill_text=args.use_skill_text,
        include_structured=args.use_structured,
        pre_h_text=val_h_text,
        pre_h_skill_text=val_h_skill,
    )

    test_dataset = CareerPathDataset(
        data_pairs=test_pairs,
        encoder=encoder_text,
        Y_target_dict=Y_target_dict,
        job_skill_map=job_skill_map,
        esco_skill_text_map=esco_skill_text_map,
        skill_properties_map=skill_properties_map,
        all_vocabs=all_vocabs,
        use_skill_description=args.use_skill_description,
        pooling_strategy=args.pooling_strategy,
        alpha=args.alpha,
        beta=args.beta,
        encoder_skill=encoder_skill,
        include_text=args.use_text_history,
        include_skill_text=args.use_skill_text,
        include_structured=args.use_structured,
        pre_h_text=test_h_text,
        pre_h_skill_text=test_h_skill,
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_career_path_batch,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_career_path_batch,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    # Create sampled validation loader for Optuna trials (for faster hyperparameter tuning)
    # Use a fixed seed for reproducibility of random permutation
    val_sample_size = max(1, int(len(val_dataset) * args.val_sample_ratio))
    # If available, use args.seed, else default to 42
    seed = getattr(args, 'seed', 42)
    generator = torch.Generator()
    generator.manual_seed(seed)
    val_sample_indices = torch.randperm(len(val_dataset), generator=generator)[:val_sample_size].tolist()
    val_sample_dataset = torch.utils.data.Subset(val_dataset, val_sample_indices)
    
    val_sample_loader = DataLoader(
        val_sample_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_career_path_batch,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_career_path_batch,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    logger.info(f"  ✓ Created dataloaders (val: {len(val_dataset)}, val_sample: {len(val_sample_dataset)}, test: {len(test_dataset)})\n")
    
    # --- Step 6: Optuna optimization ---
    if args.optuna:
        logger.info("[6/8] Starting Optuna hyperparameter optimization...")
        logger.info(f"  > Running {args.n_trials} trials with max {args.max_epochs} epochs each")
        logger.info(f"  > Early stopping patience: {args.optuna_patience} epochs (stricter for trials)")
        logger.info(f"  > Minimum improvement threshold: {args.min_delta:.4f} MRR")
        logger.info(f"  > Using {args.val_sample_ratio*100:.0f}% of validation set ({len(val_sample_dataset)} samples) for faster tuning\n")

        # Setup mixed precision scaler if enabled
        scaler = None
        if args.mixed_precision:
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
            logger.info("🔥 Mixed precision training enabled")

        study = optuna.create_study(
            direction="maximize",
            study_name=args.study_name,
            pruner=optuna.pruners.MedianPruner()
        )

        start_time = time.time()

        study.optimize(
            lambda trial: objective(
                trial, train_loader, val_sample_loader, Y_target_all, args,
                text_dim, skill_text_dim, structured_dim, output_dim, scaler
            ),
            n_trials=args.n_trials,
            show_progress_bar=True,
            callbacks=[logger_callback]
        )

        elapsed = time.time() - start_time
        logger.info(f"\n  ✓ Optimization complete in {elapsed/60:.2f} minutes")
        logger.info(f"  ✓ Best trial: {study.best_trial.number}")
        logger.info(f"  ✓ Best validation MRR: {study.best_value:.4f}")
        final_epochs = study.best_trial.user_attrs["best_epoch"] + 1
        logger.info(f"  ✓ Optimal training epochs: {final_epochs}")
        logger.info(f"  ✓ Best hyperparameters:")
        for key, value in study.best_params.items():
            logger.info(f"      {key}: {value}")
        logger.info("")

        # setting best params
        hidden_dim = study.best_params["hidden_dim"]
        n_layers = study.best_params["n_layers"]
        dropout = study.best_params["dropout"]
        lr = study.best_params["lr"]
        weight_decay = study.best_params.get("weight_decay", 0.0)
        if "use_modality_weights" in study.best_params:
            use_modality_weights = study.best_params["use_modality_weights"]
        else:
            use_modality_weights = False
        
        # Optimizer-specific params
        if args.optimizer == "sgd":
            momentum = study.best_params.get("momentum", 0.9)
            nesterov = study.best_params.get("nesterov", False)
        else:
            momentum = None
            nesterov = None
    else:
        logger.info("[6/8] Using static hyperparameters... Training with early stopping to find best epoch.")
        logger.info(f"  > Early stopping patience: {args.patience} epochs, min delta: {args.min_delta:.4f} MRR")

        # Setup mixed precision scaler if enabled
        static_scaler = None
        if args.mixed_precision:
            from torch.cuda.amp import GradScaler
            static_scaler = GradScaler()
            logger.info("🔥 Mixed precision training enabled for static run")

        hidden_dim = args.hidden_dim
        n_layers = args.n_layers
        dropout = args.dropout
        lr = args.lr
        use_modality_weights = args.use_modality_weights
        weight_decay = args.weight_decay
        
        # Optimizer-specific params
        if args.optimizer == "sgd":
            momentum = args.momentum
            nesterov = args.nesterov
        else:
            momentum = None
            nesterov = None

        # Build a temporary model for this run
        if args.use_advanced:
            temp_model = MultiModalCPPModel(
                text_dim=text_dim,
                skill_text_dim=skill_text_dim,
                structured_dim=structured_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                dropout=dropout,
                output_dim=output_dim,
                use_modality_weights=use_modality_weights,
                use_text=args.use_text_history,
                use_skill=args.use_skill_text,
                use_struct=args.use_structured
            ).to(device)
        else:
            input_dim = 0
            if args.use_text_history:
                input_dim += text_dim
            if args.use_skill_text:
                input_dim += skill_text_dim
            if args.use_structured:
                input_dim += structured_dim
            
            temp_model = SimpleConcatModel(
                input_dim=input_dim,
                output_dim=output_dim,
                n_layers=n_layers,
                hidden_dim=hidden_dim,
                dropout=dropout,
                use_text=args.use_text_history,
                use_skill=args.use_skill_text,
                use_struct=args.use_structured
            ).to(device)

        # Create optimizer based on configuration
        if args.optimizer == "sgd":
            optimizer = optim.SGD(temp_model.parameters(), lr=lr, momentum=momentum,
                                 weight_decay=weight_decay, nesterov=nesterov)
            logger.info(f"  > Using SGD: lr={lr}, momentum={momentum}, weight_decay={weight_decay}, nesterov={nesterov}")
        else:  # adam
            optimizer = optim.Adam(temp_model.parameters(), lr=lr, weight_decay=weight_decay)
            logger.info(f"  > Using Adam: lr={lr}, weight_decay={weight_decay}")
        
        criterion = nn.CosineEmbeddingLoss()

        best_val_mrr = 0.0
        epochs_no_improve = 0
        best_epoch = 0


        for epoch in trange(args.max_epochs, desc="Epochs", unit="epoch"):
            epoch_start_time = time.time()
            train_loss = train_epoch(temp_model, train_loader, optimizer, criterion, device, static_scaler, args.mixed_precision, args.gradient_accumulation_steps, args.profile_data_loading)
            val_metrics = evaluate(temp_model, val_loader, Y_target_all, device, criterion)
            val_mrr = val_metrics['MRR']
            epoch_time = time.time() - epoch_start_time
            logger.info(f"  Static HP Run | Epoch {epoch+1}/{args.max_epochs} | Train Loss: {train_loss:.4f} | Val MRR: {val_mrr:.4f} | Time: {epoch_time:.1f}s")

            if val_mrr > best_val_mrr + args.min_delta:
                best_val_mrr = val_mrr
                epochs_no_improve = 0
                best_epoch = epoch + 1
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= args.patience:
                logger.info(f"  Early stopping triggered after {epoch+1} epochs (Static training patience: {args.patience}).")
                break

        final_epochs = best_epoch
        if final_epochs == 0:
            logger.warning("  Validation MRR did not improve over initial state. Training for 1 epoch as a fallback.")
            final_epochs = 1
        
        logger.info(f"  ✓ Optimal training epochs found: {final_epochs}\n")
    
    # --- Step 7: Train final model on train+val ---
    logger.info("[7/8] Training final model on train+val with best hyperparameters...")
    logger.info("  > Using full train+val set for final training (no sampling)")
    
    # Combine train and val and REUSE cached embeddings instead of recomputing
    combined_pairs = train_pairs + val_pairs

    def _concat_optional(a, b):
        """Safely concatenate cached embedding arrays that may be None."""
        if a is None and b is None:
            return None
        if a is None:
            return b
        if b is None:
            return a
        return np.concatenate([a, b], axis=0)

    combined_h_text = _concat_optional(train_h_text, val_h_text)
    combined_h_skill = _concat_optional(train_h_skill, val_h_skill)

    combined_dataset = CareerPathDataset(
        data_pairs=combined_pairs,
        encoder=encoder_text,
        Y_target_dict=Y_target_dict,
        job_skill_map=job_skill_map,
        esco_skill_text_map=esco_skill_text_map,
        skill_properties_map=skill_properties_map,
        all_vocabs=all_vocabs,
        use_skill_description=args.use_skill_description,
        pooling_strategy=args.pooling_strategy,
        alpha=args.alpha,
        beta=args.beta,
        encoder_skill=encoder_skill,
        include_text=args.use_text_history,
        include_skill_text=args.use_skill_text,
        include_structured=args.use_structured,
        pre_h_text=combined_h_text,
        pre_h_skill_text=combined_h_skill,
    )

    final_num_workers = 1
    
    combined_loader = DataLoader(
        combined_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=final_num_workers, collate_fn=collate_career_path_batch,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(final_num_workers > 0),
        prefetch_factor=2 if final_num_workers > 0 else None
    )
    
    # Build final model with best hyperparameters
    if args.use_advanced:
        final_model = MultiModalCPPModel(
            text_dim=text_dim,
            skill_text_dim=skill_text_dim,
            structured_dim=structured_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dropout=dropout,
            output_dim=output_dim,
            use_modality_weights=use_modality_weights,
            use_text=args.use_text_history,
            use_skill=args.use_skill_text,
            use_struct=args.use_structured
        ).to(device)
    else:
        # Calculate input dimension based on enabled modalities
        input_dim = 0
        if args.use_text_history:
            input_dim += text_dim
        if args.use_skill_text:
            input_dim += skill_text_dim
        if args.use_structured:
            input_dim += structured_dim
        
        final_model = SimpleConcatModel(
            input_dim=input_dim,
            output_dim=output_dim,
            n_layers=n_layers,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_text=args.use_text_history,
            use_skill=args.use_skill_text,
            use_struct=args.use_structured
        ).to(device)
    
    # Create optimizer based on configuration
    if args.optimizer == "sgd":
        optimizer = optim.SGD(final_model.parameters(), lr=lr, momentum=momentum,
                             weight_decay=weight_decay, nesterov=nesterov)
    else:  # adam
        optimizer = optim.Adam(final_model.parameters(), lr=lr, weight_decay=weight_decay)
    
    criterion = nn.CosineEmbeddingLoss()

    # Setup mixed precision scaler for final training if enabled
    final_scaler = None
    if args.mixed_precision:
        from torch.cuda.amp import GradScaler
        final_scaler = GradScaler()
        logger.info("🔥 Mixed precision training enabled for final training")

    # Train final model
    logger.info(f"  > Training for {final_epochs} epochs...")
    # log model architecture
    logger.info(f"  > Model architecture: {final_model}")

    final_training_start = time.time()
    for epoch in tqdm(range(final_epochs), desc="Final training"):
        epoch_start_time = time.time()
        train_epoch(final_model, combined_loader, optimizer, criterion, device, final_scaler, args.mixed_precision, args.gradient_accumulation_steps, args.profile_data_loading)
        epoch_time = time.time() - epoch_start_time
        if epoch < 2 or epoch % 5 == 0:  # Log timing for first few and every 5th epoch
            logger.info(f"    Final training epoch {epoch+1}/{final_epochs} completed in {epoch_time:.1f}s")

    final_training_time = time.time() - final_training_start
    logger.info(f"  > Final training completed in {final_training_time:.1f}s ({final_training_time/final_epochs:.1f}s per epoch)")
        
    # Evaluate on test set
    logger.info("\n  > Evaluating on test set...")
    test_metrics = evaluate(final_model, test_loader, Y_target_all, device)
    
    logger.info("\n" + "=" * 80)
    logger.info("FINAL TEST SET RESULTS")
    logger.info("=" * 80)
    for metric, value in test_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    logger.info("=" * 80)
    
    # Log to wandb
    if WANDB_AVAILABLE and args.use_wandb:
        # Prefix test metrics with 'test_'
        wandb_test_metrics = {f'test_{k.lower()}': v for k, v in test_metrics.items()}
        wandb.log(wandb_test_metrics)
        wandb.finish()

    # Save model and results
    if args.save_model:
        logger.info(f"  > Saving model to {args.output_dir}/final_model.pt")
        checkpoint = {
            'model_state_dict': final_model.state_dict(),
            'hidden_dim': hidden_dim,
            'n_layers': n_layers,
            'dropout': dropout,
            'lr': lr,
            'optimizer': args.optimizer,
            'weight_decay': weight_decay,
            'use_modality_weights': use_modality_weights,
            'test_metrics': test_metrics,
            'args': vars(args)
        }
        
        # Add optimizer-specific parameters
        if args.optimizer == "sgd":
            checkpoint['momentum'] = momentum
            checkpoint['nesterov'] = nesterov

        save_path = os.path.join(args.output_dir, 'final_model_.pt')
        torch.save(checkpoint, save_path)
        logger.info(f"\nModel saved to: {save_path}")
    
    # Save study results (optional)
    if args.optuna and args.save_study:
        study_path = os.path.join(args.output_dir, 'optuna_study.pkl')
        import pickle
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)
        logger.info(f"Study saved to: {study_path}")

    # --- Save results to CSV ---
    # Construct optimizer details string
    if args.optimizer == "sgd":
        optimizer_details = f"SGD(lr={lr:.6f}, momentum={momentum:.4f}, weight_decay={weight_decay:.6f}, nesterov={nesterov})"
    else:
        optimizer_details = f"Adam(lr={lr:.6f}, weight_decay={weight_decay:.6f})"
    
    results_data = {
        'timestamp': pd.to_datetime('now').strftime('%Y-%m-%d %H:%M:%S'),
        'run_name': args.run_name,
        'architecture': 'MultiModal' if args.use_advanced else 'SimpleConcat',
        'text_history': args.use_text_history,
        'skill_text': args.use_skill_text,
        'structured': args.use_structured,
        'text_encoder': args.encoder_text,
        'skill_encoder': args.encoder_skill if args.encoder_skill else args.encoder_text,
        'pooling_strategy': args.pooling_strategy,
        'last_job_only': args.last_job_only,
        'final_epochs': final_epochs,
        'lr': lr,
        'hidden_dim': hidden_dim,
        'n_layers': n_layers,
        'dropout': dropout,
        'use_modality_weights': use_modality_weights if args.use_advanced else 'N/A',
        **{f'test_{k}': v for k, v in test_metrics.items()},
        'optimizer': args.optimizer,
        'optimizer_details': optimizer_details
    }

    results_df = pd.DataFrame([results_data])
    
    try:
        if os.path.exists(args.results_csv_path):
            results_df.to_csv(args.results_csv_path, mode='a', header=False, index=False)
        else:
            results_df.to_csv(args.results_csv_path, mode='w', header=True, index=False)
        logger.info(f"📈 Results appended to: {args.results_csv_path}")
    except Exception as e:
        logger.error(f"Error saving results to CSV: {e}")


if __name__ == "__main__":
    main()

