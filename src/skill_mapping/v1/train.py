# train.py

import argparse
import os
from pathlib import Path
from loguru import logger
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
from typing import Dict, List, Callable, Optional
import random

# Import our new model and data classes
from skill_mapping.v1.model import CategoryPredictor, SkillSimilarityModel, TextEncoderWrapper
from skill_mapping.v1.data import CategoryDataset, GlobalSkillDataset, HIER_COL_MAP
# We must import our new utils.py
import skill_mapping.v1.utils as utils

# -----------------------------
# Argument Parser
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train models for Hierarchical Skill Prediction")

    parser.add_argument("--task", type=str, required=True, 
                        choices=["train_category", "train_skill_retrieval"],
                        help="Which model to train: the category classifier or the skill retrieval bi-encoder.")
    
    parser.add_argument("--source", type=str, default="esco", 
                        choices=["esco", "decorte", "kw_occ", "kw_cp", "kw_plus", "all"],
                        help="Which job dataset to use for training.")
    parser.add_argument("--input_variant", type=str, default="title+desc",
                        help="Input text format: 'title', 'title+desc', etc.")
    parser.add_argument("--is_structured", action="store_true",
                        help="Use structured prompts (e.g., 'Job Title: ...')")

    # Category-specific args
    parser.add_argument("--target_level", type=int, default=0,
                        help="Hierarchy level for the category predictor (0-3).")

    # Model args
    parser.add_argument("--encoder_ckpt", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--hidden_dim", type=int, default=None, 
                        help="Embedding dim. If None, it's inferred from the encoder_ckpt.")

    # Training args
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate (for fine-tuning the encoder or just the head).")
    parser.add_argument("--scheduler", type=str, default=None, choices=[None, "cosine"])
    parser.add_argument("--eval_threshold", type=float, default=0.5,
                        help="Probability threshold for multi-label classification metrics.")
    
    # I/O & Logging
    parser.add_argument("--out_dir", type=str, default="checkpoints/")
    parser.add_argument("--run_name", type=str, default=None,
                        help="A name for this specific run (used for .pt and .log files).")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--project", type=str, default="hier-skill")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save", type=str, default=None)
    
    # For now, we'll keep evaluation to every epoch
    # parser.add_argument("--eval_strategy", type=str, default="epoch")
    # parser.add_argument("--eval_steps", type=int, default=500)

    return parser.parse_args()

# -----------------------------
# Collate Function for Retrieval
# -----------------------------
def collate_fn_similarity(batch: List[Dict]) -> Dict[str, List[str]]:
    """
    Collator for the GlobalSkillDataset.
    It takes a batch of samples, and for each sample, it picks *one*
    random positive skill.
    
    Args:
        batch: A list of dicts, e.g., 
               [{"text": "job1", "skill_texts": ["skillA", "skillB"]},
                {"text": "job2", "skill_texts": ["skillC"]}]
    
    Returns:
        A dict with two parallel lists (B_job, B_skill):
        {"job_texts": ["job1", "job2"], "skill_texts": ["skillB", "skillC"]}
    """
    job_texts = []
    skill_texts = []
    
    for item in batch:
        if item['skill_texts']: # Only include if there are positive skills
            job_texts.append(item['text'])
            # Pick one random positive skill to pair with the job
            skill_texts.append(random.choice(item['skill_texts']))
            
    return {"job_texts": job_texts, "skill_texts": skill_texts}

# -----------------------------
# Evaluation Functions
# -----------------------------
@torch.no_grad()
def evaluate_classifier(model: CategoryPredictor, dataloader: DataLoader, device: str, threshold: float) -> Dict[str, float]:
    """Evaluates the CategoryPredictor model."""
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Evaluating (Category)"):
        texts = batch["text"]
        y = batch["y"].to(device)
        
        logits = model(texts)
        loss = model.compute_loss(logits, y)
        total_loss += loss.item()
        
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        
        all_preds.append(preds.cpu().numpy())
        all_targets.append(y.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    
    if not all_preds:
        logger.warning("Evaluation loader was empty.")
        return {'eval_loss': avg_loss}

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    metrics = {
        'eval_loss': avg_loss,
        'f1_micro': f1_score(all_targets, all_preds, average='micro', zero_division=0),
        'f1_macro': f1_score(all_targets, all_preds, average='macro', zero_division=0),
        'precision_micro': precision_score(all_targets, all_preds, average='micro', zero_division=0),
        'recall_micro': recall_score(all_targets, all_preds, average='micro', zero_division=0),
        'hamming_loss': hamming_loss(all_targets, all_preds)
    }
    
    model.train()
    return metrics

@torch.no_grad()
def evaluate_similarity(model: SkillSimilarityModel, dataloader: DataLoader) -> Dict[str, float]:
    """Evaluates the SkillSimilarityModel on the contrastive loss."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Evaluating (Skill Retrieval)"):
        job_texts = batch["job_texts"]
        skill_texts = batch["skill_texts"]
        
        if not job_texts:
            continue
            
        job_emb, skill_emb = model(job_texts, skill_texts)
        loss = model.compute_loss(job_emb, skill_emb)
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    model.train()
    return {'eval_loss': avg_loss}

# -----------------------------
# Solution: Class Imbalance
# -----------------------------

def calculate_pos_weights(dataset: CategoryDataset, device: torch.device) -> torch.Tensor:
    """
    Calculates positive weights for BCEWithLogitsLoss to handle class imbalance.
    The weight for a class is: (Number of Negatives) / (Number of Positives)
    """
    logger.info("Calculating positive weights for class imbalance...")
    
    # Stack all target vectors from the dataset
    all_targets = torch.stack([s['target_vec'] for s in dataset.samples])
    num_samples = len(all_targets)
    
    # Count positive occurrences (class frequency) for each class
    pos_counts = all_targets.sum(dim=0)
    
    # Avoid division by zero for classes that *never* appear in the training data
    pos_counts[pos_counts == 0] = 1 
    
    # Calculate negative counts
    neg_counts = num_samples - pos_counts
    
    # Calculate weights
    # We clamp to prevent extremely large weights (e.g., max weight of 100)
    pos_weight = torch.clamp(neg_counts / pos_counts, min=1.0, max=100.0)
    
    logger.info(f"Class weights calculated. Min: {pos_weight.min():.2f}, Max: {pos_weight.max():.2f}, Mean: {pos_weight.mean():.2f}")
    
    return pos_weight.to(device)


# -----------------------------
# Training Loops
# -----------------------------

def train_classifier(
    model: CategoryPredictor, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    scheduler, 
    args: argparse.Namespace,
    pos_weight: Optional[torch.Tensor] = None
):
    """Main training loop for the CategoryPredictor."""
    model.train()
    global_step = 0
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Category)")
        for step, batch in enumerate(progress_bar):
            texts = batch["text"]
            y = batch["y"].to(args.device)

            logits = model(texts)
            loss = model.compute_loss(logits, y, pos_weight=pos_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            epoch_loss += loss.item()
            global_step += 1
            
            if global_step % args.logging_steps == 0:
                if args.wandb:
                    wandb.log({"train/step_loss": loss.item(), "global_step": global_step})
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # --- End of Epoch ---
        avg_train_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} - Avg Train Loss: {avg_train_loss:.4f}")
        
        eval_metrics = evaluate_classifier(model, val_loader, args.device, args.eval_threshold)
        logger.info(f"Epoch {epoch+1} - Eval Metrics: {eval_metrics}")
        
        if args.wandb:
            log_dict = {f"eval/{k}": v for k, v in eval_metrics.items()}
            log_dict["train/epoch_loss"] = avg_train_loss
            log_dict["epoch"] = epoch + 1
            wandb.log(log_dict)
            
    return model

def train_skill_retrieval(
    model: SkillSimilarityModel, 
    train_loader: DataLoader, 
    val_loader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    scheduler, 
    args: argparse.Namespace
):
    """Main training loop for the SkillSimilarityModel (Bi-Encoder)."""
    model.train()
    global_step = 0
    
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} (Skill Retrieval)")
        for step, batch in enumerate(progress_bar):
            job_texts = batch["job_texts"]
            skill_texts = batch["skill_texts"]

            if not job_texts:
                continue

            job_emb, skill_emb = model(job_texts, skill_texts)
            loss = model.compute_loss(job_emb, skill_emb)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            epoch_loss += loss.item()
            global_step += 1
            
            if global_step % args.logging_steps == 0:
                if args.wandb:
                    wandb.log({"train/step_loss": loss.item(), "global_step": global_step})
            
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # --- End of Epoch ---
        avg_train_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} - Avg Train Loss: {avg_train_loss:.4f}")
        
        eval_metrics = evaluate_similarity(model, val_loader)
        logger.info(f"Epoch {epoch+1} - Eval Metrics: {eval_metrics}")
        
        if args.wandb:
            log_dict = {f"eval/{k}": v for k, v in eval_metrics.items()}
            log_dict["train/epoch_loss"] = avg_train_loss
            log_dict["epoch"] = epoch + 1
            wandb.log(log_dict)
            
    return model

# -----------------------------
# Main Execution
# -----------------------------
def main():
    args = parse_args()
    utils.set_seed(args.seed)
    
    # --- Setup Logging ---
    run_name = args.run_name or f"{args.task}_{args.source}_{args.encoder_ckpt.split('/')[-1]}_lvl{args.target_level}"
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.out_dir) / f"{run_name}.log"
    logger.add(log_path, rotation="1 MB")
    logger.info(f"Starting run: {run_name}")
    logger.info(f"Arguments: {vars(args)}")

    if args.wandb:
        wandb.init(project=args.project, name=run_name, config=vars(args))

    device = torch.device(args.device)

    # --- Load DataFrames ---
    logger.info("Loading DataFrames...")
    esco_df = pd.read_csv("data/processed/master_datasets/master_complete_hierarchy_w_occ.csv")
    
    # Load the correct job dataframes based on the source
    # This logic matches your original script
    try:
        if args.source == "esco":
            job_df_train = None
            job_df_val = None
        elif args.source == "decorte":
            job_df_train = pd.read_csv("data/title_pairs_desc/decorte_train_pairs.csv")
            job_df_val = pd.read_csv("data/title_pairs_desc/decorte_val_pairs.csv")
        elif args.source == "kw_occ":
            job_df_train = pd.read_csv("data/title_pairs_desc/karrierewege_plus_occ_train_pairs.csv")
            job_df_val = pd.read_csv("data/title_pairs_desc/karrierewege_plus_occ_val_pairs.csv")
        elif args.source == "kw_cp":
            job_df_train = pd.read_csv("data/title_pairs_desc/karrierewege_plus_cp_train_pairs.csv")
            job_df_val = pd.read_csv("data/title_pairs_desc/karrierewege_plus_cp_val_pairs.csv")
        elif args.source == "kw_plus":
            job_df_train = pd.concat([pd.read_csv("data/title_pairs_desc/karrierewege_plus_occ_train_pairs.csv"), pd.read_csv("data/title_pairs_desc/karrierewege_plus_cp_train_pairs.csv")])
            job_df_val = pd.concat([pd.read_csv("data/title_pairs_desc/karrierewege_plus_occ_val_pairs.csv"), pd.read_csv("data/title_pairs_desc/karrierewege_plus_cp_val_pairs.csv")])
        elif args.source == "all":
            job_df_train = pd.concat([pd.read_csv("data/title_pairs_desc/decorte_train_pairs.csv"), pd.read_csv("data/title_pairs_desc/karrierewege_plus_occ_train_pairs.csv"), pd.read_csv("data/title_pairs_desc/karrierewege_plus_cp_train_pairs.csv")])
            job_df_val = pd.concat([pd.read_csv("data/title_pairs_desc/decorte_val_pairs.csv"), pd.read_csv("data/title_pairs_desc/karrierewege_plus_occ_val_pairs.csv"), pd.read_csv("data/title_pairs_desc/karrierewege_plus_cp_val_pairs.csv")])
    except FileNotFoundError as e:
        logger.error(f"Failed to load job data for source '{args.source}': {e}")
        logger.error("If source is 'esco', this is OK. Otherwise, check your file paths.")
        if args.source != "esco":
            return

    # --- Setup Model & Data ---
    logger.info("Initializing Model and Datasets...")
    
    # This is the key difference for fine-tuning vs. feature-extraction
    if args.task == "train_skill_retrieval":
        # We are FINE-TUNING the encoder.
        # Load the SBERT model as a native nn.Module.
        encoder = SentenceTransformer(args.encoder_ckpt).to(device)
        
        encoder_wrapper = TextEncoderWrapper(encoder) 
        
        model = SkillSimilarityModel(encoder=encoder_wrapper).to(device)
        
        train_dataset = GlobalSkillDataset(
            esco_df=esco_df,
            job_df=job_df_train,
            text_fields=args.input_variant,
            is_structured=args.is_structured,
            use_canonical_jobs=(args.source == "esco" or args.source == "all")
        )
        val_dataset = GlobalSkillDataset(
            esco_df=esco_df,
            job_df=job_df_val,
            text_fields=args.input_variant,
            is_structured=args.is_structured,
            use_canonical_jobs=(args.source == "esco" or args.source == "all")
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_similarity)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_similarity)

        # We want to fine-tune the encoder, so we pass model.parameters()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        scheduler = None
        if args.scheduler == "cosine":
            logger.info("Using CosineAnnealingLR scheduler.")
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
        
        logger.info(f"Training Skill Retrieval Model. Num params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        train_skill_retrieval(model, train_loader, val_loader, optimizer, scheduler, args)


    elif args.task == "train_category":
        # We are FREEZING the encoder and training the head.
        encoder = SentenceTransformer(args.encoder_ckpt).to(device)

        if args.hidden_dim is None:
            logger.info("hidden_dim not set. Inferring from encoder...")
            args.hidden_dim = encoder.get_sentence_embedding_dimension()
            logger.info(f"Inferred hidden_dim: {args.hidden_dim}")
        
        encoder_wrapper = TextEncoderWrapper(encoder)
        
        train_dataset = CategoryDataset(
            esco_df=esco_df,
            job_df=job_df_train,
            hier_level=args.target_level,
            text_fields=args.input_variant,
            is_structured=args.is_structured,
            use_canonical_jobs=(args.source == "esco" or args.source == "all")
        )
        val_dataset = CategoryDataset(
            esco_df=esco_df,
            job_df=job_df_val,
            hier_level=args.target_level,
            text_fields=args.input_variant,
            is_structured=args.is_structured,
            use_canonical_jobs=(args.source == "esco" or args.source == "all")
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        pos_weight = calculate_pos_weights(train_dataset, device)

        model = CategoryPredictor(
            encoder=encoder_wrapper,
            hidden_dim=args.hidden_dim,
            num_categories=train_dataset.num_classes(),
        ).to(device)

        optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr)

        scheduler = None
        if args.scheduler == "cosine":
            logger.info("Using CosineAnnealingLR scheduler.")
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * len(train_loader))
        # We *only* pass the classifier head's parameters to the optimizer
        
        
        logger.info(f"Training Category Predictor. Num trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        
        train_classifier(model, train_loader, val_loader, optimizer, scheduler, args, pos_weight=pos_weight)

    # --- Save Final Model ---
    ckpt_path = Path(args.out_dir) / f"{run_name}.pt"
    # Save the *entire* model (encoder + head or just head)
    if args.save == 'head':
        torch.save(model.classifier.state_dict(), ckpt_path)
    else:
        torch.save(model.state_dict(), ckpt_path)
    logger.success(f"Model saved at {ckpt_path}")

    if args.wandb:
        wandb.save(str(ckpt_path))
        wandb.finish()


if __name__ == "__main__":
    main()