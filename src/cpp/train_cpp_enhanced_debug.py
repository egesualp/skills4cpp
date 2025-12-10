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


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device, debug=False, epoch_num=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="In-epoch Training")):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        y_pred = model(batch)
        
        # CosineEmbeddingLoss requires target labels (1 for similar)
        target = torch.ones(y_pred.size(0)).to(device)
        loss = criterion(y_pred, batch['y'], target)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # DEBUG: Show training details for first batch of first epoch
        if debug and batch_idx == 0 and epoch_num == 0:
            logger.debug("=" * 80)
            logger.debug("DEBUG: Training Step Details (First Batch)")
            logger.debug("=" * 80)
            logger.debug(f"Batch size: {y_pred.shape[0]}")
            logger.debug(f"Prediction shape: {y_pred.shape}")
            logger.debug(f"Target shape: {batch['y'].shape}")
            logger.debug(f"Loss value: {loss.item():.6f}")
            logger.debug(f"\nPrediction stats:")
            logger.debug(f"  Min: {y_pred.min().item():.4f}")
            logger.debug(f"  Max: {y_pred.max().item():.4f}")
            logger.debug(f"  Mean: {y_pred.mean().item():.4f}")
            logger.debug(f"  Std: {y_pred.std().item():.4f}")
            logger.debug(f"\nFirst prediction (first 10 dims): {y_pred[0][:10].detach().cpu().numpy()}")
            logger.debug(f"First target (first 10 dims): {batch['y'][0][:10].detach().cpu().numpy()}")
            
            # Calculate cosine similarity for first sample
            cos_sim = torch.nn.functional.cosine_similarity(
                y_pred[0].unsqueeze(0), 
                batch['y'][0].unsqueeze(0)
            ).item()
            logger.debug(f"\nCosine similarity (pred vs target): {cos_sim:.4f}")
            logger.debug("=" * 80 + "\n")
    
    return total_loss / len(dataloader)


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
            
            all_y_pred.append(y_pred.cpu().numpy())
            all_y_true.append(batch['y'].cpu().numpy())
    
    y_pred_vectors = np.concatenate(all_y_pred)
    y_true_vectors = np.concatenate(all_y_true)
    
    metrics = calculate_ranking_metrics(y_pred_vectors, y_true_vectors, Y_target_all)
    
    if criterion:
        metrics['loss'] = total_loss / len(dataloader)
        
    return metrics


# ============================================================================
# OPTUNA OBJECTIVE
# ============================================================================

def objective(trial, train_loader, val_loader, Y_target_all, args, 
              text_dim, skill_text_dim, structured_dim, output_dim):
    """Optuna objective function."""
    
    device = torch.device(args.device)
    
    # Suggest hyperparameters
    if args.use_advanced:
        # Multi-modal architecture
        hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 768])
        n_layers = trial.suggest_int("n_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        use_modality_weights = trial.suggest_categorical("use_modality_weights", [True, False])
        
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
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CosineEmbeddingLoss()
    
    best_val_mrr = 0.0
    epochs_no_improve = 0
    epochs_total = 0
    
    # Training loop
    for epoch in range(args.max_epochs):
        epochs_total += 1
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, debug=args.debug, epoch_num=epoch)
        val_metrics = evaluate(model, val_loader, Y_target_all, device, criterion)
        val_mrr = val_metrics['MRR']
        
        # Prepare metrics for logging
        log_metrics = {
            'train_loss': train_loss,
            'val_mrr': val_mrr,
            'val_loss': val_metrics.get('loss', 0),
        }
        for k, v in val_metrics.items():
            if k not in ['MRR', 'loss']:
                log_metrics[f'val_{k.lower()}'] = v

        logger.info(f"  Trial {trial.number} | Epoch {epoch+1}/{args.max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics.get('loss', 0):.4f} | Val MRR: {val_mrr:.4f}")
        
        if WANDB_AVAILABLE and args.use_wandb:
            wandb.log(log_metrics, step=epochs_total)

        # Report to Optuna
        trial.report(val_mrr, epoch)
        
        # Pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        # Early stopping
        if val_mrr > best_val_mrr:
            best_val_mrr = val_mrr
            epochs_no_improve = 0
            best_epoch = epochs_total
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= args.patience:
            logger.info(f"  Early stopping triggered after {epoch+1} epochs.")
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
    parser.add_argument("--patience", type=int, default=2, help="Early stopping patience")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0,
                       help="Number of DataLoader workers (auto-detects from SLURM_CPUS_PER_TASK if not set)")
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

    parser.add_argument("--run_name", type=str, default="cpp_enhanced", help="Run name")
    
    # Debugging
    parser.add_argument("--debug", action='store_true', help="Enable detailed debugging output")

    
    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()
    
    # Reconfigure logger if debug mode is enabled
    if args.debug:
        logger.remove()  # Remove all handlers
        # Add file handler with DEBUG level
        logger.add(
            "logs/debug.log",
            format="{time} | {level} | {message}",
            level="DEBUG",
            rotation="10 MB",
            retention="7 days",
            enqueue=True
        )
        # Add stdout handler with DEBUG level
        logger.add(
            sys.stdout,
            format="<green>{time}</green> | <level>{message}</level>",
            level="DEBUG"
        )
        logger.debug("🐛 Debug mode enabled - verbose logging active")
    
    # Fix CUDA multiprocessing issue: use 'spawn' instead of 'fork'
    # This must be done before any CUDA operations (encoder loading)
    try:
        multiprocessing.set_start_method('spawn', force=True)
        logger.info("🖥️  CUDA multiprocessing set to 'spawn'")
    except RuntimeError:
        # Already set, which is fine
        pass
    
    # Validate modality selection
    n_active_modalities = sum([args.use_text_history, args.use_skill_text, args.use_structured])
    if n_active_modalities == 0:
        raise ValueError("At least one modality must be enabled! Use --use_text_history, --use_skill_text, or --use_structured")
    
    # Auto-enable advanced mode for multiple modalities (unless explicitly set)
    if n_active_modalities >= 2 and not args.use_advanced:
        logger.info("📊 Multiple modalities detected - automatically enabling multi-modal architecture")
        logger.info("   (Use simple mode by explicitly setting --use_advanced=False if desired)\n")
        args.use_advanced = True
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
    
    # --- Step 1: Load encoders ---
    logger.info("[1/7] Loading encoder models...")
    
    # DEBUG: Show encoder configuration
    if args.debug:
        logger.debug("=" * 80)
        logger.debug("DEBUG: Encoder Configuration")
        logger.debug("=" * 80)
        logger.debug(f"Text encoder: {args.encoder_text}")
        logger.debug(f"Skill encoder: {args.encoder_skill if args.encoder_skill else 'Same as text encoder'}")
        logger.debug(f"Use text description: {args.use_text_description}")
        logger.debug(f"Use skill description: {args.use_skill_description}")
        logger.debug("=" * 80 + "\n")
    
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
    
    # --- Step 2: Load helper maps ---
    logger.info("[2/7] Loading vocabularies and skill mappings...")
    all_vocabs = load_all_vocabs(args.vocab_dir)
    structured_dim = sum(len(vocab) for vocab in all_vocabs.values())
    
    job_skill_map, esco_skill_text_map, skill_properties_map = load_job_and_skill_data(
        master_skill_file=args.master_skill_file,
        esco_skills_file=args.esco_skills_file,
        skill_properties_file=args.skill_properties_file,
        pooling_strategy=args.pooling_strategy,
        alpha=args.alpha,
        beta=args.beta
    )
    logger.info(f"  ✓ Structured feature dim: {structured_dim}\n")
    
    # DEBUG: Show example job-skill mapping
    if args.debug:
        logger.debug("=" * 80)
        logger.debug("DEBUG: Job-Skill Mapping Example")
        logger.debug("=" * 80)
        # Show first 3 jobs with their skills
        for i, (job_title, skills) in enumerate(list(job_skill_map.items())[:3]):
            logger.debug(f"Job {i+1}: '{job_title}'")
            logger.debug(f"  → Number of skills: {len(skills)}")
            logger.debug(f"  → First 5 skills: {skills[:5]}")
            # Show skill properties for first skill if available
            if skills:
                first_skill = skills[0]
                # Handle case where skill might be a dict or string
                if isinstance(first_skill, dict):
                    skill_key = first_skill.get('preferredLabel', first_skill.get('conceptUri', str(first_skill)))
                    logger.debug(f"  → First skill (dict): {first_skill}")
                else:
                    skill_key = first_skill
                    
                if skill_key in skill_properties_map:
                    props = skill_properties_map[skill_key]
                    logger.debug(f"  → First skill properties: {props}")
        
        # Show ESCO skill text map examples
        logger.debug(f"\n--- ESCO Skill Text Map Examples ---")
        logger.debug(f"Total skills in esco_skill_text_map: {len(esco_skill_text_map)}")
        for i, (skill_key, skill_text) in enumerate(list(esco_skill_text_map.items())[:2]):
            logger.debug(f"\nESCO Skill {i+1}:")
            logger.debug(f"  Key: {skill_key}")
            logger.debug(f"  Text format check:")
            has_newline = '\n' in skill_text
            logger.debug(f"    - Has 'role:': {'role:' in skill_text}")
            logger.debug(f"    - Has 'description:': {'description:' in skill_text}")
            logger.debug(f"    - Has newline: {has_newline}")
            logger.debug(f"  Full text:")
            logger.debug(f"  ---START---")
            logger.debug(f"  {skill_text[:400]}..." if len(skill_text) > 400 else f"  {skill_text}")
            logger.debug(f"  ---END---")
        
        logger.debug("=" * 80 + "\n")
    
    # --- Step 3: Load data pairs ---
    logger.info("[3/7] Loading career path data...")
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

    if args.debug:
        logger.debug("=" * 80)
        logger.debug("DEBUG: Data Pair Examples")
        logger.debug("=" * 80)
        logger.debug(f"Example train pair: {train_pairs[0]}")
        logger.debug(f"Example val pair: {val_pairs[0]}")
        logger.debug(f"Example test pair: {test_pairs[0]}")
        logger.debug("=" * 80 + "\n")
    
    logger.info(f"  ✓ Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}\n")
    
    # DEBUG: Show detailed examples of data pairs
    if args.debug:
        logger.debug("=" * 80)
        logger.debug("DEBUG: Data Pair Examples (Text Content)")
        logger.debug("=" * 80)
        for i, pair in enumerate(train_pairs[:2]):
            input_text, target_text = pair
            logger.debug(f"\nTrain Example {i+1}:")
            logger.debug(f"  Input text (length: {len(input_text)}):")
            logger.debug(f"    {input_text[:500]}..." if len(input_text) > 500 else f"    {input_text}")
            logger.debug(f"  Target text: {target_text}")
            # Check if this is a multi-job history
            if SEP_TOKEN in input_text:
                jobs = input_text.split(SEP_TOKEN)
                logger.debug(f"  → Multi-job history with {len(jobs)} jobs")
            else:
                logger.debug(f"  → Single job")
        logger.debug("=" * 80 + "\n")
    
    # --- Step 4: Pre-compute target embeddings ---
    logger.info("[4/7] Pre-computing target embeddings...")
    # Extract actual target labels from the returned data pairs (not from data.labels which are raw)
    actual_labels = list(set([pair[1] for pair in train_pairs + val_pairs + test_pairs]))
    Y_target_dict = precompute_target_embeddings(encoder_text, actual_labels, show_progress=True)
    Y_target_all = np.array(list(Y_target_dict.values()))
    output_dim = Y_target_all.shape[1]
    logger.info(f"  ✓ Target embedding dim: {output_dim}\n")
    
    # DEBUG: Show target embedding examples
    if args.debug:
        logger.debug("=" * 80)
        logger.debug("DEBUG: Target Embeddings")
        logger.debug("=" * 80)
        logger.debug(f"Total unique target labels: {len(actual_labels)}")
        logger.debug(f"Target embedding dimension: {output_dim}")
        # Show first target and its embedding
        first_label = list(Y_target_dict.keys())[0]
        first_embedding = Y_target_dict[first_label]
        logger.debug(f"\nExample target label: '{first_label}'")
        logger.debug(f"Embedding shape: {first_embedding.shape}")
        logger.debug(f"Embedding (first 10 values): {first_embedding[:10]}")
        logger.debug(f"Embedding stats: min={first_embedding.min():.4f}, max={first_embedding.max():.4f}, mean={first_embedding.mean():.4f}")
        logger.debug("=" * 80 + "\n")

    # --- Step 4b: Pre-compute input embeddings ---
    logger.info("[4b/7] Pre-computing input embeddings...")
    train_pairs, train_h_text, train_h_skill = precompute_input_embeddings(
        train_pairs, Y_target_dict, encoder_text, encoder_skill,
        job_skill_map, esco_skill_text_map,
        use_skill_description=args.use_skill_description,
        pooling_strategy=args.pooling_strategy, alpha=args.alpha, beta=args.beta,
        use_text_history=args.use_text_history, use_skill_text=args.use_skill_text,
        debug=args.debug
    )
    val_pairs, val_h_text, val_h_skill = precompute_input_embeddings(
        val_pairs, Y_target_dict, encoder_text, encoder_skill,
        job_skill_map, esco_skill_text_map,
        use_skill_description=args.use_skill_description,
        pooling_strategy=args.pooling_strategy, alpha=args.alpha, beta=args.beta,
        use_text_history=args.use_text_history, use_skill_text=args.use_skill_text,
        debug=False  # Only debug on train set
    )
    test_pairs, test_h_text, test_h_skill = precompute_input_embeddings(
        test_pairs, Y_target_dict, encoder_text, encoder_skill,
        job_skill_map, esco_skill_text_map,
        use_skill_description=args.use_skill_description,
        pooling_strategy=args.pooling_strategy, alpha=args.alpha, beta=args.beta,
        use_text_history=args.use_text_history, use_skill_text=args.use_skill_text,
        debug=False  # Only debug on train set
    )
    logger.info(f"  ✓ Input embeddings pre-computed\n")
    
    # DEBUG: Show input embedding examples
    if args.debug:
        logger.debug("=" * 80)
        logger.debug("DEBUG: Input Embeddings & Pooling Strategy")
        logger.debug("=" * 80)
        logger.debug(f"Pooling strategy: {args.pooling_strategy}")
        logger.debug(f"Alpha (position weight): {args.alpha}")
        logger.debug(f"Beta (IDF weight): {args.beta}")
        
        if args.use_text_history and train_h_text is not None and len(train_h_text) > 0:
            logger.debug(f"\n--- Text History Embeddings ---")
            logger.debug(f"Number of samples: {len(train_h_text)}")
            first_text_emb = train_h_text[0]
            logger.debug(f"Shape: {first_text_emb.shape}")
            logger.debug(f"First embedding (first 10 values): {first_text_emb[:10]}")
            logger.debug(f"Stats: min={first_text_emb.min():.4f}, max={first_text_emb.max():.4f}, mean={first_text_emb.mean():.4f}")
        
        if args.use_skill_text and train_h_skill is not None and len(train_h_skill) > 0:
            logger.debug(f"\n--- Skill Text Embeddings ---")
            logger.debug(f"Number of samples: {len(train_h_skill)}")
            first_skill_emb = train_h_skill[0]
            logger.debug(f"Shape: {first_skill_emb.shape}")
            logger.debug(f"First embedding (first 10 values): {first_skill_emb[:10]}")
            logger.debug(f"Stats: min={first_skill_emb.min():.4f}, max={first_skill_emb.max():.4f}, mean={first_skill_emb.mean():.4f}")
            
            # Show an example of how skills are aggregated for a job
            example_input = train_pairs[0][0]
            # Extract job title (first job if multi-job history)
            if SEP_TOKEN in example_input:
                example_job = example_input.split(SEP_TOKEN)[0].strip()
            else:
                example_job = example_input.strip()
            
            # Clean job title (remove description if present)
            if ": " in example_job:
                example_job_title = example_job.split(": ")[0]
            else:
                example_job_title = example_job
            
            logger.debug(f"\n--- Skill Aggregation Example ---")
            logger.debug(f"Job: '{example_job_title}'")
            # Normalize title to match mapping file format (lowercase + stripped)
            example_job_title_normalized = example_job_title.strip().lower()
            if example_job_title_normalized in job_skill_map:
                skills = job_skill_map[example_job_title_normalized]
                logger.debug(f"Number of skills found: {len(skills)}")
                logger.debug(f"Skills: {skills[:10]}" + (" ..." if len(skills) > 10 else ""))
                
                # Show how skill text is formatted before encoding
                if skills:
                    logger.debug(f"\n--- Skill Text Formatting Example (from esco_skill_text_map) ---")
                    
                    # Show a few skill text examples (up to 3)
                    for idx, skill in enumerate(skills[:3]):
                        if isinstance(skill, dict):
                            # Skills from job_skill_map have 'skillUri', not 'conceptUri'
                            skill_uri = skill.get('skillUri', skill.get('conceptUri', ''))
                            # Try to get the skill name from esco_skill_text_map
                            if skill_uri in esco_skill_text_map:
                                skill_name = esco_skill_text_map[skill_uri].get('name', '')
                            else:
                                skill_name = skill.get('preferredLabel', '')
                        else:
                            skill_name = skill
                            skill_uri = skill
                        
                        logger.debug(f"\n--- Skill {idx+1} ---")
                        logger.debug(f"Skill name: {skill_name}")
                        logger.debug(f"Skill URI: {skill_uri}")
                        
                        # Show what's actually in esco_skill_text_map
                        if skill_uri in esco_skill_text_map:
                            skill_data = esco_skill_text_map[skill_uri]
                            logger.debug(f"Found in esco_skill_text_map: YES")
                            logger.debug(f"Raw data from map:")
                            logger.debug(f"  - name: {skill_data.get('name', '')}")
                            logger.debug(f"  - desc: {skill_data.get('desc', '')[:100]}..." if len(skill_data.get('desc', '')) > 100 else f"  - desc: {skill_data.get('desc', '')}")
                            
                            # Show what will actually be used based on use_skill_description flag
                            if args.use_skill_description:
                                # This matches cpp_dataset.py line 205
                                actual_text = f"role: {skill_data['name']} \n description: {skill_data['desc']}"
                                logger.debug(f"\nWith --use_skill_description: Using formatted text with description")
                            else:
                                # This matches cpp_dataset.py line 207
                                actual_text = skill_data['name']
                                logger.debug(f"\nWithout --use_skill_description: Using ONLY skill name")
                            
                            logger.debug(f"---START---")
                            logger.debug(actual_text)
                            logger.debug(f"---END---")
                            logger.debug(f"Text length: {len(actual_text)} characters")
                        else:
                            logger.debug(f"Found in esco_skill_text_map: NO")
                            logger.debug(f"Fallback: Would use 'role: {skill_name}'")
                
                # Show weighting if using weighted_idf
                if args.pooling_strategy == "weighted_idf" and skills:
                    logger.debug(f"\n--- Weighting Details (for first skill) ---")
                    first_skill = skills[0]
                    
                    # Handle case where skill might be a dict or string
                    if isinstance(first_skill, dict):
                        skill_key = first_skill.get('preferredLabel', first_skill.get('conceptUri', str(first_skill)))
                        logger.debug(f"First skill (dict): {first_skill}")
                    else:
                        skill_key = first_skill
                    
                    if skill_key in skill_properties_map:
                        props = skill_properties_map[skill_key]
                        logger.debug(f"Skill key: '{skill_key}'")
                        logger.debug(f"Properties: {props}")
                        # The actual weight calculation happens in the dataset class
                        # but we can show the IDF component
                        if 'idf' in props:
                            logger.debug(f"IDF value: {props['idf']:.4f}")
            else:
                logger.debug(f"No skills found for this job title in job_skill_map")
        
        logger.debug("=" * 80 + "\n")

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
        pin_memory=(device.type == 'cuda')
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_career_path_batch,
        pin_memory=(device.type == 'cuda')
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_career_path_batch,
        pin_memory=(device.type == 'cuda')
    )
    
    logger.info(f"  ✓ Created dataloaders\n")
    
    # DEBUG: Show example batch structure
    if args.debug:
        logger.debug("=" * 80)
        logger.debug("DEBUG: Batch Structure Example")
        logger.debug("=" * 80)
        # Get one batch from train loader
        sample_batch = next(iter(train_loader))
        logger.debug(f"Batch keys: {sample_batch.keys()}")
        logger.debug(f"Batch size: {sample_batch['y'].shape[0]}")
        
        for key, value in sample_batch.items():
            logger.debug(f"\n'{key}':")
            logger.debug(f"  Shape: {value.shape}")
            logger.debug(f"  Dtype: {value.dtype}")
            if len(value.shape) > 1 and value.shape[1] <= 10:
                logger.debug(f"  First sample: {value[0]}")
            elif len(value.shape) > 1:
                logger.debug(f"  First sample (first 10 dims): {value[0][:10]}")
                logger.debug(f"  Stats: min={value[0].min():.4f}, max={value[0].max():.4f}, mean={value[0].mean():.4f}")
        
        logger.debug("\n--- Feature Dimensions Summary ---")
        total_input_dim = 0
        if 'h_text' in sample_batch:
            logger.debug(f"Text history dimension: {sample_batch['h_text'].shape[1]}")
            total_input_dim += sample_batch['h_text'].shape[1]
        if 'h_skill_text' in sample_batch:
            logger.debug(f"Skill text dimension: {sample_batch['h_skill_text'].shape[1]}")
            total_input_dim += sample_batch['h_skill_text'].shape[1]
        
        structured_keys = [k for k in sample_batch.keys() if k.startswith('h_structured_')]
        if structured_keys:
            structured_total = sum(sample_batch[k].shape[1] for k in structured_keys)
            logger.debug(f"Structured features dimension: {structured_total}")
            logger.debug(f"  Structured feature keys: {structured_keys}")
            total_input_dim += structured_total
        
        logger.debug(f"Total input dimension (if concatenated): {total_input_dim}")
        logger.debug(f"Target dimension: {sample_batch['y'].shape[1]}")
        logger.debug("=" * 80 + "\n")
    
    # --- Step 6: Optuna optimization ---
    if args.optuna:
        logger.info("[6/8] Starting Optuna hyperparameter optimization...")
        logger.info(f"  > Running {args.n_trials} trials with max {args.max_epochs} epochs each")
        logger.info(f"  > Early stopping patience: {args.patience} epochs\n")

        study = optuna.create_study(
            direction="maximize",
            study_name=args.study_name,
            pruner=optuna.pruners.MedianPruner()
        )

        start_time = time.time()

        study.optimize(
            lambda trial: objective(
                trial, train_loader, val_loader, Y_target_all, args,
                text_dim, skill_text_dim, structured_dim, output_dim
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
        if "use_modality_weights" in study.best_params:
            use_modality_weights = study.best_params["use_modality_weights"]
        else:
            use_modality_weights = False
    else:
        logger.info("[6/8] Using static hyperparameters... Training with early stopping to find best epoch.")
        hidden_dim = args.hidden_dim
        n_layers = args.n_layers
        dropout = args.dropout
        lr = args.lr
        use_modality_weights = args.use_modality_weights

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

        optimizer = optim.Adam(temp_model.parameters(), lr=lr)
        criterion = nn.CosineEmbeddingLoss()

        best_val_mrr = 0.0
        epochs_no_improve = 0
        best_epoch = 0


        for epoch in trange(args.max_epochs, desc="Epochs", unit="epoch"):
            train_loss = train_epoch(temp_model, train_loader, optimizer, criterion, device, debug=args.debug, epoch_num=epoch)
            val_metrics = evaluate(temp_model, val_loader, Y_target_all, device, criterion)
            val_mrr = val_metrics['MRR']
            logger.info(f"  Static HP Run | Epoch {epoch+1}/{args.max_epochs} | Train Loss: {train_loss:.4f} | Val MRR: {val_mrr:.4f}")

            if val_mrr > best_val_mrr:
                best_val_mrr = val_mrr
                epochs_no_improve = 0
                best_epoch = epoch + 1
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= args.patience:
                logger.info(f"  Early stopping triggered after {epoch+1} epochs.")
                break

        final_epochs = best_epoch
        if final_epochs == 0:
            logger.warning("  Validation MRR did not improve over initial state. Training for 1 epoch as a fallback.")
            final_epochs = 1
        
        logger.info(f"  ✓ Optimal training epochs found: {final_epochs}\n")
    
    # --- Step 7: Train final model on train+val ---
    logger.info("[7/8] Training final model on train+val with best hyperparameters...")
    
    # Combine train and val and pre-compute their embeddings
    combined_pairs = train_pairs + val_pairs
    combined_pairs, combined_h_text, combined_h_skill = precompute_input_embeddings(
        combined_pairs, Y_target_dict, encoder_text, encoder_skill,
        job_skill_map, esco_skill_text_map,
        use_skill_description=args.use_skill_description,
        pooling_strategy=args.pooling_strategy, alpha=args.alpha, beta=args.beta,
        use_text_history=args.use_text_history, use_skill_text=args.use_skill_text,
        debug=False  # No need to debug here since we already did during initial loading
    )
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
    
    combined_loader = DataLoader(
        combined_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_career_path_batch,
        pin_memory=(device.type == 'cuda')
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
    
    optimizer = optim.Adam(final_model.parameters(), lr=lr)
    criterion = nn.CosineEmbeddingLoss()
    
    # Train final model
    logger.info(f"  > Training for {final_epochs} epochs...")
    # log model architecture
    logger.info(f"  > Model architecture: {final_model}")
    
    # DEBUG: Show model details
    if args.debug:
        logger.debug("=" * 80)
        logger.debug("DEBUG: Final Model Architecture")
        logger.debug("=" * 80)
        logger.debug(f"Architecture type: {'MultiModal' if args.use_advanced else 'SimpleConcat'}")
        logger.debug(f"Model parameters:")
        total_params = sum(p.numel() for p in final_model.parameters())
        trainable_params = sum(p.numel() for p in final_model.parameters() if p.requires_grad)
        logger.debug(f"  Total parameters: {total_params:,}")
        logger.debug(f"  Trainable parameters: {trainable_params:,}")
        logger.debug(f"\nHyperparameters:")
        logger.debug(f"  Hidden dim: {hidden_dim}")
        logger.debug(f"  Number of layers: {n_layers}")
        logger.debug(f"  Dropout: {dropout}")
        logger.debug(f"  Learning rate: {lr}")
        if args.use_advanced:
            logger.debug(f"  Use modality weights: {use_modality_weights}")
        logger.debug("=" * 80 + "\n")

    for epoch in tqdm(range(final_epochs), desc="Final training"):
        train_epoch(final_model, combined_loader, optimizer, criterion, device, debug=args.debug, epoch_num=epoch)
        
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
            'use_modality_weights': use_modality_weights,
            'test_metrics': test_metrics,
            'args': vars(args)
        }

        save_path = os.path.join(args.output_dir, 'final_model_.pt')
        torch.save(checkpoint, save_path)
        logger.info(f"\nModel saved to: {save_path}")
    
    # Save study results
    if args.optuna:
        study_path = os.path.join(args.output_dir, 'optuna_study.pkl')
        import pickle
        with open(study_path, 'wb') as f:
            pickle.dump(study, f)
        logger.info(f"Study saved to: {study_path}")

    # --- Save results to CSV ---
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
        **{f'test_{k}': v for k, v in test_metrics.items()}
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

