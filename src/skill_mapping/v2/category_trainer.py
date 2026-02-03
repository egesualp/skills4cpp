"""
category_trainer.py - Category Predictor Training with Optuna HPO

Trains a multi-label classifier that predicts skill categories from job/occupation text.
Uses ESCO occupations as training data (occupation -> skill categories mapping).

Usage:
    # Standard training with train/val split from ESCO data:
    python -m skill_mapping.v2.category_trainer \
        --esco_path data/processed/master_datasets_2/master_complete_hierarchy_w_occ.csv \
        --test_path data/test.csv \
        --model_path pj-mathematician/JobSkillBGE-large-en-v1.5 \
        --target_level 1 \
        --output_dir outputs/category_model \
        --n_trials 20

    # With augmented occupation text (recommended when encoder was finetuned on augmented data):
    python -m skill_mapping.v2.category_trainer \
        --esco_path data/processed/master_datasets_2/master_complete_hierarchy_w_occ.csv \
        --augmented_path data/processed/augmentation/augmented_esco_occupations_2.csv \
        --model_path pj-mathematician/JobSkillBGE-large-en-v1.5 \
        --target_level 1 \
        --output_dir outputs/category_model_aug \
        --n_trials 20
    
    # Final training with full dataset (no train/val split):
    python -m skill_mapping.v2.category_trainer \
        --esco_path data/processed/master_datasets_2/master_complete_hierarchy_w_occ.csv \
        --augmented_path data/processed/augmentation/augmented_esco_occupations_2.csv \
        --test_path data/test.csv \
        --model_path pj-mathematician/JobSkillBGE-large-en-v1.5 \
        --target_level 1 \
        --output_dir outputs/category_model_final \
        --use_full_dataset \
        --n_trials 20
"""

import os
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter

import faiss
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from huggingface_hub import snapshot_download
from scipy.special import softmax as scipy_softmax


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


# -----------------------------
# Constants
# -----------------------------
HIER_COL_MAP = {
    0: "pillar_label",
    1: "level1_label",
    2: "level2_label",
    3: "level3_label",
}


# -----------------------------
# Label Encoder
# -----------------------------
class LabelEncoder:
    """Multi-label encoder for category classification."""

    def __init__(self, categories: List[str]):
        unique = sorted(set(categories))
        self.str2idx = {c: i for i, c in enumerate(unique)}
        self.idx2str = {i: c for c, i in self.str2idx.items()}

    def encode_multi(self, items: List[str]) -> torch.Tensor:
        """Convert list of category strings to multi-hot vector."""
        vec = torch.zeros(len(self.str2idx), dtype=torch.float32)
        for item in items:
            if item in self.str2idx:
                vec[self.str2idx[item]] = 1.0
        return vec

    def encode_soft(self, category_counts: Dict[str, int], epsilon: float = 1e-6) -> torch.Tensor:
        """
        Convert category counts to soft label distribution (Label Distribution Learning).
        
        Adds small epsilon to all categories for numerical stability in KL divergence,
        then renormalizes to ensure the distribution sums to 1.0.
        
        Args:
            category_counts: Dict mapping category names to their skill counts
            epsilon: Small value added to all categories for numerical stability
            
        Returns:
            Tensor with smoothed, normalized distribution (sums to 1.0)
        """
        vec = torch.zeros(len(self.str2idx), dtype=torch.float32)
        total_count = sum(category_counts.values())
        
        if total_count == 0:
            # If no counts, return uniform distribution with epsilon smoothing
            vec = torch.full((len(self.str2idx),), epsilon, dtype=torch.float32)
            return vec / vec.sum()
        
        for cat, count in category_counts.items():
            if cat in self.str2idx:
                vec[self.str2idx[cat]] = count / total_count
        
        # Add epsilon smoothing for numerical stability in KL divergence
        vec = vec + epsilon
        
        # Renormalize to ensure distribution sums to 1.0
        vec = vec / vec.sum()
        
        return vec

    def __len__(self):
        return len(self.str2idx)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"str2idx": self.str2idx, "idx2str": {str(k): v for k, v in self.idx2str.items()}}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "LabelEncoder":
        with open(path, "r") as f:
            data = json.load(f)
        encoder = cls([])
        encoder.str2idx = data["str2idx"]
        encoder.idx2str = {int(k): v for k, v in data["idx2str"].items()}
        return encoder


# -----------------------------
# Datasets
# -----------------------------
class EmbeddingDataset(Dataset):
    """Dataset using pre-computed embeddings."""

    def __init__(self, embeddings: np.ndarray, targets: torch.Tensor):
        self.embeddings = torch.from_numpy(embeddings).float()
        self.targets = targets

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {"embedding": self.embeddings[idx], "target": self.targets[idx]}


class BalancedOccupationDataset(Dataset):
    """
    Dataset with Stochastic Dynamic Resampling for balanced occupation representation.
    
    In each epoch, samples exactly `k` text variants per occupation:
    - If occupation has > k samples: randomly sample k (different each epoch)
    - If occupation has < k samples: oversample with replacement to reach k
    
    This ensures each occupation contributes equally regardless of augmentation count,
    while maintaining full semantic coverage across epochs.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        targets: torch.Tensor,
        occupation_ids: List[str],
        samples_per_occupation: int = 10,
    ):
        self.embeddings = torch.from_numpy(embeddings).float()
        self.targets = targets
        self.occupation_ids = occupation_ids
        self.k = samples_per_occupation
        
        # Build occupation -> sample indices mapping
        self.occ_to_indices = {}
        for idx, occ_id in enumerate(occupation_ids):
            if occ_id not in self.occ_to_indices:
                self.occ_to_indices[occ_id] = []
            self.occ_to_indices[occ_id].append(idx)
        
        self.unique_occupations = list(self.occ_to_indices.keys())
        self.num_occupations = len(self.unique_occupations)
        
        # Resample indices for this epoch
        self._resample()
        
        logger.info(
            f"BalancedOccupationDataset: {len(embeddings)} total samples, "
            f"{self.num_occupations} occupations, {self.k} samples/occ/epoch "
            f"-> {len(self)} samples per epoch"
        )

    def _resample(self):
        """Resample k indices per occupation for a new epoch."""
        self.epoch_indices = []
        
        for occ_id in self.unique_occupations:
            indices = self.occ_to_indices[occ_id]
            n = len(indices)
            
            if n >= self.k:
                # Randomly sample k without replacement
                sampled = np.random.choice(indices, size=self.k, replace=False)
            else:
                # Oversample with replacement to reach k
                sampled = np.random.choice(indices, size=self.k, replace=True)
            
            self.epoch_indices.extend(sampled.tolist())
        
        # Shuffle the epoch indices
        np.random.shuffle(self.epoch_indices)

    def on_epoch_end(self):
        """Call this at the end of each epoch to resample."""
        self._resample()

    def __len__(self):
        return len(self.epoch_indices)

    def __getitem__(self, idx):
        real_idx = self.epoch_indices[idx]
        return {"embedding": self.embeddings[real_idx], "target": self.targets[real_idx]}


# -----------------------------
# Model
# -----------------------------
class CategoryClassifier(nn.Module):
    """
    Deep MLP classifier for multi-label category prediction.
    
    Supports configurable number of hidden layers with BatchNorm, ReLU activation,
    and dropout for each layer.
    
    Example architecture with hidden_dims=[512, 256]:
        Input -> Linear(input_dim, 512) -> BatchNorm -> ReLU -> Dropout
              -> Linear(512, 256) -> BatchNorm -> ReLU -> Dropout
              -> Linear(256, num_classes) -> Output
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: List[int] = [512],
        dropout: float = 0.1,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# -----------------------------
# Data Loading
# -----------------------------
def load_esco_data(
    esco_path: str,
    target_level: int,
) -> Tuple[pd.DataFrame, str]:
    """Load ESCO data and return occupation-level aggregated data."""
    logger.info(f"Loading ESCO data from {esco_path}")
    df = pd.read_csv(esco_path)

    cat_col = HIER_COL_MAP[target_level]
    if cat_col not in df.columns:
        raise ValueError(f"Column '{cat_col}' not found. Available: {list(df.columns)}")

    # Filter rows with valid categories
    df = df[df[cat_col].notna()].copy()
    logger.info(f"Loaded {len(df)} rows with valid '{cat_col}'")

    return df, cat_col


def build_occupation_samples(
    esco_df: pd.DataFrame,
    cat_col: str,
    text_col: str = "occupationLabel",
    desc_col: str = "occupationDescription",
    augmented_df: Optional[pd.DataFrame] = None,
    soft_labels: bool = False,
) -> List[Dict]:
    """
    Build training samples: one sample per unique occupation (or per augmented text variant).
    Each occupation maps to multiple skill categories.

    If augmented_df is provided, uses skill_brief from augmented data instead of
    raw occupation title + description. If multiple skill_brief variants exist per
    occupation, creates one sample per variant (data augmentation).
    
    If soft_labels is True, includes category_counts for Label Distribution Learning
    (density = # skills in category / total # skills for occupation).
    """
    samples = []

    # Build augmented text lookup if provided (list of texts per URI for multiple variants)
    aug_text_map = {}
    aug_isco_map = {}
    if augmented_df is not None:
        has_isco = "iscoGroup" in augmented_df.columns
        for _, row in augmented_df.iterrows():
            uri = row.get("conceptUri", "")
            skill_brief = str(row.get("skill_brief", "")).strip()
            if uri and skill_brief and skill_brief != "nan":
                if uri not in aug_text_map:
                    aug_text_map[uri] = []
                aug_text_map[uri].append(skill_brief)
            
            if has_isco and uri:
                isco = row.get("iscoGroup")
                if pd.notna(isco):
                    # Store as string, handle numeric codes (e.g. 2512.0 -> "2512")
                    aug_isco_map[uri] = str(int(isco)) if isinstance(isco, (int, float)) else str(isco)

        total_texts = sum(len(v) for v in aug_text_map.values())
        logger.info(f"Loaded {total_texts} augmented texts for {len(aug_text_map)} occupations")

    # Group by occupation URI to get unique occupations
    for occ_uri, group in esco_df.groupby("occupationUri"):
        first_row = group.iloc[0]

        # Get category information (same for all text variants of this occupation)
        cat_series = group[cat_col].dropna().astype(str)
        categories = cat_series.unique().tolist()

        if not categories:
            continue

        # Prepare category counts for soft labels (shared across variants)
        category_counts = cat_series.value_counts().to_dict() if soft_labels else None

        # Determine ISCO group for stratification
        # Priority: esco_df column -> augmented_df map -> 'UNKNOWN'
        isco_group = "UNKNOWN"
        if "iscoGroup" in first_row and pd.notna(first_row["iscoGroup"]):
            val = first_row["iscoGroup"]
            isco_group = str(int(val)) if isinstance(val, (int, float)) else str(val)
        elif occ_uri in aug_isco_map:
            isco_group = aug_isco_map[occ_uri]

        # Determine text variants: use augmented texts if available, else fallback
        if occ_uri in aug_text_map:
            text_variants = aug_text_map[occ_uri]
        else:
            title = str(first_row.get(text_col, "")).strip()
            desc = str(first_row.get(desc_col, "")).strip()
            fallback_text = f"{title} [SEP] {desc}" if desc and desc != "nan" else title
            text_variants = [fallback_text] if fallback_text.strip() else []

        # Create one sample per text variant
        for text in text_variants:
            if not text.strip():
                continue

            sample = {
                "uri": occ_uri,
                "text": text,
                "categories": categories,
                "isco_group": isco_group,
            }

            # Add category counts for soft labels (Label Distribution Learning)
            if soft_labels:
                sample["category_counts"] = category_counts

            samples.append(sample)

    # Count unique occupations vs total samples
    unique_occs = len(set(s["uri"] for s in samples))
    logger.info(f"Built {len(samples)} samples from {unique_occs} unique occupations")
    if soft_labels:
        logger.info("  - Using soft labels (Label Distribution Learning)")
    if augmented_df is not None:
        aug_samples = sum(1 for s in samples if s["uri"] in aug_text_map)
        logger.info(f"  - {aug_samples} samples using augmented text")
    return samples


def build_validation_samples(
    val_df: pd.DataFrame,
    esco_df: pd.DataFrame,
    cat_col: str,
    soft_labels: bool = False,
) -> List[Dict]:
    """
    Build validation samples from job-skill pairs.
    Maps skillUri to skill categories (handling multiple categories per skill).
    
    If soft_labels is True, includes category_counts for Label Distribution Learning.
    """
    # Create skill -> categories lookup (one skill can have MULTIPLE categories)
    skill_to_cats = {}
    for skill_uri, group in esco_df.groupby("skillUri"):
        # Get ALL categories for this skill
        cats = group[cat_col].dropna().astype(str).unique().tolist()
        if cats:
            skill_to_cats[skill_uri] = cats

    samples = []
    for job_id, group in val_df.groupby("job_id"):
        first_row = group.iloc[0]
        text = str(first_row.get("skill_brief", "")).strip()

        if not text:
            continue

        # Collect categories from all skills for this job
        all_cats = set()
        category_counts = {}
        
        for skill_uri in group["skillUri"].dropna().unique():
            if skill_uri in skill_to_cats:
                # Add ALL categories for this skill
                for cat in skill_to_cats[skill_uri]:
                    all_cats.add(cat)
                    category_counts[cat] = category_counts.get(cat, 0) + 1

        if all_cats:
            sample = {
                "job_id": job_id,
                "text": text,
                "categories": list(all_cats),
            }
            if soft_labels:
                sample["category_counts"] = category_counts
            samples.append(sample)

    logger.info(f"Built {len(samples)} validation samples")
    return samples


def stratified_split_occupations(
    samples: List[Dict],
    test_size: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split samples into train/val using stratified sampling on occupation URIs.
    Stratifies based on 'isco_group' field (e.g. ISCO-08 group code).
    Ensures no data leakage (same occupation in both sets).
    
    Uses a hybrid approach:
    1. Stratifies large ISCO groups (≥ min_group_size occupations)
    2. Randomly splits small ISCO groups to fill remaining quota
    """
    # Group by URI to prevent data leakage (all text variants of an occupation stay together)
    uri_to_strat_label = {}
    uri_to_samples = {}
    
    for s in samples:
        uri = s['uri']
        if uri not in uri_to_samples:
            uri_to_samples[uri] = []
            # Use ISCO group as the stratification key
            uri_to_strat_label[uri] = s.get('isco_group', 'UNKNOWN')
        uri_to_samples[uri].append(s)
    
    unique_uris = list(uri_to_samples.keys())
    
    if len(unique_uris) < 5:
        logger.warning("Not enough occupations to split. Returning all as train.")
        return samples, []

    # Count occupations per ISCO group
    label_counts = Counter(uri_to_strat_label.values())
    
    # Determine minimum group size for stratification
    # We need at least 2 samples per group for stratification to work
    min_group_size = max(2, int(2 / test_size))  # Ensures at least 1 sample in each split
    
    # Separate URIs into stratifiable and non-stratifiable groups
    stratifiable_uris = []
    stratifiable_labels = []
    non_stratifiable_uris = []
    
    for uri in unique_uris:
        label = uri_to_strat_label[uri]
        if label_counts[label] >= min_group_size:
            stratifiable_uris.append(uri)
            stratifiable_labels.append(label)
        else:
            non_stratifiable_uris.append(uri)
    
    logger.info(
        f"Stratification: {len(stratifiable_uris)} occupations in {len(set(stratifiable_labels))} "
        f"large ISCO groups (≥{min_group_size} occupations), "
        f"{len(non_stratifiable_uris)} in small groups"
    )
    
    # Split stratifiable occupations with stratification
    train_uris = []
    val_uris = []
    
    if len(stratifiable_uris) >= 2 and len(set(stratifiable_labels)) >= 2:
        try:
            # Check if we have enough samples for the test_size ratio
            n_val_stratified = max(1, int(len(stratifiable_uris) * test_size))
            
            # Ensure we have at least one sample per class in both train and val
            if n_val_stratified < len(set(stratifiable_labels)):
                # Adjust test_size for stratifiable subset to accommodate all classes
                adjusted_test_size = max(test_size, len(set(stratifiable_labels)) / len(stratifiable_uris))
                if adjusted_test_size < 1.0:
                    train_strat, val_strat = train_test_split(
                        stratifiable_uris,
                        test_size=adjusted_test_size,
                        stratify=stratifiable_labels,
                        random_state=seed,
                    )
                else:
                    # Too many classes, fall through to random split
                    raise ValueError("Too many unique classes for stratification")
            else:
                train_strat, val_strat = train_test_split(
                    stratifiable_uris,
                    test_size=test_size,
                    stratify=stratifiable_labels,
                    random_state=seed,
                )
            
            train_uris.extend(train_strat)
            val_uris.extend(val_strat)
            
            logger.info(
                f"Stratified split: {len(train_strat)} train, {len(val_strat)} val "
                f"from large ISCO groups"
            )
        except (ValueError, Exception) as e:
            # If stratification still fails, treat as non-stratifiable
            logger.warning(f"Stratification failed for large groups ({e}), treating all as small groups")
            non_stratifiable_uris.extend(stratifiable_uris)
    else:
        # Not enough stratifiable occupations
        non_stratifiable_uris.extend(stratifiable_uris)
    
    # Randomly split small groups and add to train/val to reach target ratio
    if non_stratifiable_uris:
        np.random.seed(seed)
        np.random.shuffle(non_stratifiable_uris)
        
        # Calculate how many more we need in validation
        total_uris = len(unique_uris)
        target_val_count = int(total_uris * test_size)
        current_val_count = len(val_uris)
        needed_val_count = max(0, target_val_count - current_val_count)
        
        # Take what we need for validation from small groups
        val_from_small = non_stratifiable_uris[:needed_val_count]
        train_from_small = non_stratifiable_uris[needed_val_count:]
        
        train_uris.extend(train_from_small)
        val_uris.extend(val_from_small)
        
        logger.info(
            f"Random split: {len(train_from_small)} train, {len(val_from_small)} val "
            f"from small ISCO groups"
        )
    
    # Reconstruct sample lists
    train_out = []
    for u in train_uris:
        train_out.extend(uri_to_samples[u])
        
    val_out = []
    for u in val_uris:
        val_out.extend(uri_to_samples[u])
        
    logger.info(
        f"Final split: "
        f"{len(train_uris)} train occupations ({len(train_out)} samples), "
        f"{len(val_uris)} val occupations ({len(val_out)} samples)"
    )
    
    return train_out, val_out



# -----------------------------
# Embedding Cache
# -----------------------------
def encode_and_cache(
    texts: List[str],
    model: SentenceTransformer,
    cache_path: Optional[Path],
    batch_size: int = 64,
) -> np.ndarray:
    """Encode texts and optionally cache to disk."""
    if cache_path and cache_path.exists():
        logger.info(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    logger.info(f"Encoding {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    embeddings = normalize(embeddings, norm="l2", axis=1).astype(np.float32)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)
        logger.info(f"Cached embeddings to {cache_path}")

    return embeddings


# -----------------------------
# Training & Evaluation
# -----------------------------
def calculate_pos_weights(targets: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Calculate positive weights for BCEWithLogitsLoss to handle class imbalance."""
    pos_counts = targets.sum(dim=0)
    pos_counts = torch.clamp(pos_counts, min=1)  # Avoid div by zero
    neg_counts = len(targets) - pos_counts
    weights = torch.clamp(neg_counts / pos_counts, min=1.0, max=100.0)
    return weights.to(device)


class SoftLabelLoss(nn.Module):
    """
    Loss function for Label Distribution Learning (soft labels).
    
    Uses KL divergence between predicted distribution and target distribution.
    The model outputs logits which are converted to a probability distribution
    via softmax (since soft labels sum to 1).
    
    Note: Targets should already be smoothed with epsilon at encoding time
    (see LabelEncoder.encode_soft) for numerical stability.
    """
    
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply temperature scaling and convert to log-probabilities
        log_probs = nn.functional.log_softmax(logits / self.temperature, dim=-1)
        
        # Targets should already be epsilon-smoothed from encode_soft()
        # Just ensure they're properly normalized (defensive)
        targets = targets / targets.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        
        return self.kl_loss(log_probs, targets)


def train_epoch(
    model: CategoryClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        embeddings = batch["embedding"].to(device)
        targets = batch["target"].to(device)

        optimizer.zero_grad()
        logits = model(embeddings)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model: CategoryClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
    soft_labels: bool = False,
) -> Dict[str, float]:
    """
    Evaluate model, return metrics.
    
    For soft labels, uses top-k accuracy and distribution similarity metrics.
    For hard labels, uses standard multi-label classification metrics.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []
    all_logits = []

    for batch in dataloader:
        embeddings = batch["embedding"].to(device)
        targets = batch["target"].to(device)

        logits = model(embeddings)
        loss = criterion(logits, targets)
        total_loss += loss.item()

        all_logits.append(logits.cpu().numpy())
        all_targets.append(targets.cpu().numpy())
        
        if soft_labels:
            # For soft labels, convert logits to probability distribution
            probs = torch.softmax(logits, dim=-1)
        else:
            probs = torch.sigmoid(logits)
            
        preds = (probs > threshold).float()
        all_preds.append(preds.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    all_logits = np.vstack(all_logits)

    metrics = {
        "loss": total_loss / len(dataloader),
    }
    
    if soft_labels:
        # Soft label metrics: measure how well we predict the distribution
        # Convert logits to probabilities
        #probs = np.exp(all_logits) / np.exp(all_logits).sum(axis=-1, keepdims=True)
        probs = scipy_softmax(all_logits, axis=-1)
        
        # Top-k accuracy: check if top predicted categories match top target categories
        for k in [1, 3, 5]:
            # epsilon 1e-6
            top_k_pred = np.argsort(-probs, axis=-1)[:, :k]
            top_k_target = np.argsort(-all_targets, axis=-1)[:, :k]
            # Count how many of top-k predictions are in top-k targets
            matches = np.array([
                len(set(p) & set(t)) / k 
                for p, t in zip(top_k_pred, top_k_target)
            ])
            metrics[f"top{k}_overlap"] = float(matches.mean())
        
        # Mean Squared Error between predicted and target distributions
        mse = np.mean((probs - all_targets) ** 2)
        metrics["mse"] = float(mse)
        
        # Cosine similarity between predicted and target distributions
        dot = np.sum(probs * all_targets, axis=-1)
        norm_probs = np.linalg.norm(probs, axis=-1)
        norm_targets = np.linalg.norm(all_targets, axis=-1)
        cos_sim = dot / (norm_probs * norm_targets + 1e-8)
        metrics["cosine_similarity"] = float(cos_sim.mean())
        
        # Also compute hard-label metrics by thresholding targets
        dynamic_threshold = 0.15
        logger.info(f'Using dynamic_threshold for soft_labels {dynamic_threshold}')
        hard_targets = (all_targets > 0).astype(float)
        hard_preds = (probs > dynamic_threshold).astype(float)
        metrics["f1_micro"] = f1_score(hard_targets, hard_preds, average="micro", zero_division=0)
        metrics["f1_macro"] = f1_score(hard_targets, hard_preds, average="macro", zero_division=0)
    else:
        # Standard multi-label metrics
        metrics["f1_micro"] = f1_score(all_targets, all_preds, average="micro", zero_division=0)
        metrics["f1_macro"] = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        metrics["precision_micro"] = precision_score(all_targets, all_preds, average="micro", zero_division=0)
        metrics["recall_micro"] = recall_score(all_targets, all_preds, average="micro", zero_division=0)
    
    return metrics


# -----------------------------
# Optuna Objective
# -----------------------------
def create_objective(
    train_embeddings: np.ndarray,
    train_targets: torch.Tensor,
    val_embeddings: np.ndarray,
    val_targets: torch.Tensor,
    input_dim: int,
    num_classes: int,
    device: torch.device,
    max_epochs: int = 50,
    soft_labels: bool = False,
    train_occupation_ids: Optional[List[str]] = None,
    samples_per_occupation: Optional[int] = None,
):
    """Create Optuna objective function with extended search space for deeper networks."""

    def objective(trial: optuna.Trial) -> float:
        # Network architecture hyperparameters
        n_layers = trial.suggest_int("n_layers", 1, 4)
        
        # Build hidden dimensions for each layer (decreasing pattern)
        hidden_dims = []
        layer_dim_choices = [256, 512, 768, 1024, 1536, 2048]
        
        for i in range(n_layers):
            # Each layer can have different dimension, typically decreasing
            dim = trial.suggest_categorical(f"hidden_dim_layer_{i}", layer_dim_choices)
            hidden_dims.append(dim)
        
        # Other hyperparameters
        dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.05)
        use_batchnorm = trial.suggest_categorical("use_batchnorm", [True, False])
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        
        # Scheduler hyperparameters
        scheduler_patience = trial.suggest_int("scheduler_patience", 2, 5)
        scheduler_factor = trial.suggest_float("scheduler_factor", 0.1, 0.5, step=0.1)

        # Create datasets and loaders
        if samples_per_occupation and train_occupation_ids:
            train_dataset = BalancedOccupationDataset(
                train_embeddings, train_targets, train_occupation_ids, samples_per_occupation
            )
        else:
            train_dataset = EmbeddingDataset(train_embeddings, train_targets)
        val_dataset = EmbeddingDataset(val_embeddings, val_targets)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Create model with deeper architecture
        model = CategoryClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        ).to(device)

        # Choose loss function based on label type
        if soft_labels:
            criterion = SoftLabelLoss()
        else:
            pos_weight = calculate_pos_weights(train_targets, device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Add ReduceLROnPlateau scheduler
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=scheduler_factor,
            patience=scheduler_patience,
            verbose=False,
        )

        # Training loop with early stopping
        best_val_loss = float("inf")
        patience = 5
        patience_counter = 0

        for epoch in range(max_epochs):
            train_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate(model, val_loader, criterion, device, soft_labels=soft_labels)
            
            # Step the scheduler
            scheduler.step(val_metrics["loss"])

            # Resample for next epoch if using balanced dataset
            if hasattr(train_dataset, 'on_epoch_end'):
                train_dataset.on_epoch_end()

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

            # Report intermediate value for pruning
            trial.report(val_metrics["loss"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val_loss

    return objective


# -----------------------------
# Main Training
# -----------------------------
def extract_hidden_dims_from_params(best_params: Dict) -> List[int]:
    """Extract hidden dimensions from Optuna best_params."""
    n_layers = best_params.get("n_layers", 1)
    hidden_dims = []
    for i in range(n_layers):
        key = f"hidden_dim_layer_{i}"
        if key in best_params:
            hidden_dims.append(best_params[key])
    
    # Fallback for old-style params with single hidden_dim
    if not hidden_dims and "hidden_dim" in best_params:
        hidden_dims = [best_params["hidden_dim"]]
    
    return hidden_dims if hidden_dims else [512]


def train_final_model(
    train_embeddings: np.ndarray,
    train_targets: torch.Tensor,
    val_embeddings: np.ndarray,
    val_targets: torch.Tensor,
    best_params: Dict,
    input_dim: int,
    num_classes: int,
    device: torch.device,
    max_epochs: int = 100,
    soft_labels: bool = False,
    train_occupation_ids: Optional[List[str]] = None,
    samples_per_occupation: Optional[int] = None,
) -> Tuple[CategoryClassifier, Dict]:
    """Train final model with best hyperparameters."""

    # Create datasets
    if samples_per_occupation and train_occupation_ids:
        train_dataset = BalancedOccupationDataset(
            train_embeddings, train_targets, train_occupation_ids, samples_per_occupation
        )
    else:
        train_dataset = EmbeddingDataset(train_embeddings, train_targets)
    val_dataset = EmbeddingDataset(val_embeddings, val_targets)

    train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params["batch_size"], shuffle=False)

    # Extract hidden dimensions from params
    hidden_dims = extract_hidden_dims_from_params(best_params)
    use_batchnorm = best_params.get("use_batchnorm", True)
    
    logger.info(f"Final model architecture: hidden_dims={hidden_dims}, use_batchnorm={use_batchnorm}")

    model = CategoryClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=best_params["dropout"],
        use_batchnorm=use_batchnorm,
    ).to(device)

    # Choose loss function based on label type
    if soft_labels:
        criterion = SoftLabelLoss()
    else:
        pos_weight = calculate_pos_weights(train_targets, device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
    )
    
    # Add ReduceLROnPlateau scheduler
    scheduler_patience = best_params.get("scheduler_patience", 3)
    scheduler_factor = best_params.get("scheduler_factor", 0.3)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=scheduler_factor,
        patience=scheduler_patience,
        verbose=True,
    )

    best_val_loss = float("inf")
    best_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(max_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device, soft_labels=soft_labels)
        
        # Step the scheduler
        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]['lr']

        # Resample for next epoch if using balanced dataset
        if hasattr(train_dataset, 'on_epoch_end'):
            train_dataset.on_epoch_end()

        # Log appropriate metric based on label type
        if soft_labels:
            logger.info(
                f"Epoch {epoch + 1}/{max_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
                f"Top3 Overlap: {val_metrics['top3_overlap']:.4f}, "
                f"Cosine Sim: {val_metrics['cosine_similarity']:.4f}, LR: {current_lr:.2e}"
            )
        else:
            logger.info(
                f"Epoch {epoch + 1}/{max_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
                f"F1 Micro: {val_metrics['f1_micro']:.4f}, LR: {current_lr:.2e}"
            )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    final_metrics = evaluate(model, val_loader, criterion, device, soft_labels=soft_labels)

    return model, final_metrics


def evaluate_on_test(
    model: CategoryClassifier,
    encoder: SentenceTransformer,
    test_df: pd.DataFrame,
    esco_df: pd.DataFrame,
    cat_col: str,
    label_encoder: LabelEncoder,
    device: torch.device,
    cache_path: Optional[Path] = None,
    soft_labels: bool = False,
) -> Dict[str, float]:
    """Evaluate model on test set."""

    # Build test samples
    test_samples = build_validation_samples(test_df, esco_df, cat_col, soft_labels=soft_labels)

    if not test_samples:
        logger.warning("No test samples found")
        return {}

    # Encode texts
    texts = [s["text"] for s in test_samples]
    embeddings = encode_and_cache(texts, encoder, cache_path)

    # Build targets
    if soft_labels:
        targets = torch.stack([label_encoder.encode_soft(s["category_counts"]) for s in test_samples])
        criterion = SoftLabelLoss()
    else:
        targets = torch.stack([label_encoder.encode_multi(s["categories"]) for s in test_samples])
        criterion = nn.BCEWithLogitsLoss()

    # Create loader
    dataset = EmbeddingDataset(embeddings, targets)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Evaluate
    metrics = evaluate(model, loader, criterion, device, soft_labels=soft_labels)

    return metrics


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Category Predictor with Optuna HPO")

    # Data paths
    parser.add_argument("--esco_path", type=str, required=True, help="Path to ESCO master CSV")
    parser.add_argument("--val_path", type=str, default=None, help="Path to validation CSV (job_id, skill_brief, skillUri)")
    parser.add_argument("--test_path", type=str, default=None, help="Path to test CSV (same format as val)")
    parser.add_argument(
        "--augmented_path", type=str, default=None,
        help="Path to augmented occupation CSV (conceptUri, skill_brief). If provided, uses augmented text instead of raw ESCO titles."
    )

    # Model config
    parser.add_argument("--model_path", type=str, default="pj-mathematician/JobSkillBGE-large-en-v1.5")
    parser.add_argument("--target_level", type=int, default=1, choices=[0, 1, 2, 3])
    parser.add_argument(
        "--checkpoint_subfolder",
        type=str,
        default=None,
        help='Checkpoint of the model.'
    )

    # Training config
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument("--max_epochs", type=int, default=50, help="Max epochs per trial")
    parser.add_argument("--final_epochs", type=int, default=100, help="Max epochs for final training")
    parser.add_argument(
        "--soft_labels", action="store_true",
        help="Use soft labels (Label Distribution Learning). Target = category density per occupation."
    )
    parser.add_argument(
        "--samples_per_occupation", type=int, default=None,
        help="Enable Stochastic Dynamic Resampling with k samples per occupation per epoch. "
             "Balances representation across occupations with varying augmentation counts."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--use_full_dataset", action="store_true",
        help="Use full ESCO dataset for training (no train/val split). "
             "Useful for final model training. Requires --test_path for evaluation."
    )

    # Output
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model and results")
    parser.add_argument("--cache_name", type=str, default="embeddings", help="Base name for embedding cache files")

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load ESCO data
    esco_df, cat_col = load_esco_data(args.esco_path, args.target_level)

    # Load augmented data if provided
    augmented_df = None
    if args.augmented_path:
        logger.info(f"Loading augmented data from {args.augmented_path}")
        augmented_df = pd.read_csv(args.augmented_path)

    # Build label encoder from all categories
    all_cats = esco_df[cat_col].dropna().astype(str).unique().tolist()
    label_encoder = LabelEncoder(all_cats)
    logger.info(f"Number of categories: {len(label_encoder)}")

    # Build training samples (ESCO occupations)
    all_esco_samples = build_occupation_samples(
        esco_df, cat_col, augmented_df=augmented_df, soft_labels=args.soft_labels
    )
    
    # Split or use full dataset
    if args.use_full_dataset:
        logger.info("Using FULL ESCO dataset for training (no split)")
        train_samples = all_esco_samples
        esco_val_samples = []
        
        if not args.test_path:
            logger.warning(
                "Using full dataset for training without test set! "
                "Consider providing --test_path for model evaluation."
            )
    else:
        # Split ESCO samples into train and validation (10% validation)
        train_samples, esco_val_samples = stratified_split_occupations(
            all_esco_samples, test_size=0.1, seed=args.seed
        )

    # Load validation data (external validation set)
    if args.val_path:
        logger.info('External validation set is provided.')
        val_df = pd.read_csv(args.val_path)
        external_val_samples = build_validation_samples(val_df, esco_df, cat_col, soft_labels=args.soft_labels)
        if args.use_full_dataset:
            # When using full dataset, external data becomes the validation set
            val_samples = external_val_samples
        else:
            val_samples = esco_val_samples + external_val_samples
    else:
        val_samples = esco_val_samples
        external_val_samples = []

    # Check if we have validation data for HPO
    if not val_samples:
        raise ValueError(
            "No validation data available! Either:\n"
            "  1. Don't use --use_full_dataset (to create ESCO train/val split), OR\n"
            "  2. Provide --val_path (external validation set)"
        )
    
    # Combine validation sets
    logger.info(
        f"Total validation samples: {len(val_samples)} "
        f"({len(esco_val_samples)} from ESCO split, {len(external_val_samples)} from external file)"
    )
    
    if args.soft_labels:
        logger.info("Using soft labels (Label Distribution Learning mode)")

    # Load encoder
    logger.info(f"Loading encoder: {args.model_path}")

    if args.checkpoint_subfolder:
        checkpoint = args.checkpoint_subfolder
    elif args.model_path == "pj-mathematician/JobSkillBGE-large-en-v1.5-v2":
        checkpoint = "checkpoint-2240"
    elif args.model_path == "pj-mathematician/JobSkillBGE-large-en-v1.5":
        checkpoint = "checkpoint-4480"
    else:
        checkpoint = None

    try:
        if checkpoint:
            snapshot_path = snapshot_download(
                    repo_id=args.model_path,
                    allow_patterns=[f"{checkpoint}/*"]  # This downloads only the checkpoint files
                )
            model_path = os.path.join(snapshot_path, checkpoint)
            encoder = SentenceTransformer(model_path, device=args.device)
        else:
            encoder = SentenceTransformer(args.model_path, device=args.device)
            if 'BERT' in args.model_path:
                encoder = SentenceTransformer(modules=[encoder[0], encoder[1]], device=args.device)
    except Exception as e:
        logger.error(f"Failed to load model {args.model_path}: {e}")
        raise

    # Encode and cache embeddings
    train_texts = [s["text"] for s in train_samples]
    val_texts = [s["text"] for s in val_samples]

    # Use different cache name for augmented data to avoid stale cache
    cache_suffix = "_aug" if augmented_df is not None else ""
    # Also differentiate cache if external validation is included
    if args.val_path:
        cache_suffix += "_ext"
    # Differentiate cache for full dataset mode
    if args.use_full_dataset:
        cache_suffix += "_full"

    train_embeddings = encode_and_cache(
        train_texts, encoder, cache_dir / f"{args.cache_name}{cache_suffix}_train.npy"
    )
    val_embeddings = encode_and_cache(
        val_texts, encoder, cache_dir / f"{args.cache_name}{cache_suffix}_val.npy"
    )

    # Build target tensors
    if args.soft_labels:
        train_targets = torch.stack([label_encoder.encode_soft(s["category_counts"]) for s in train_samples])
        val_targets = torch.stack([label_encoder.encode_soft(s["category_counts"]) for s in val_samples])
    else:
        train_targets = torch.stack([label_encoder.encode_multi(s["categories"]) for s in train_samples])
        val_targets = torch.stack([label_encoder.encode_multi(s["categories"]) for s in val_samples])

    # Extract occupation IDs for balanced resampling
    train_occupation_ids = [s["uri"] for s in train_samples]

    input_dim = train_embeddings.shape[1]
    num_classes = len(label_encoder)

    logger.info(f"Train: {len(train_embeddings)} samples, Val: {len(val_embeddings)} samples")
    logger.info(f"Input dim: {input_dim}, Num classes: {num_classes}")
    
    if args.samples_per_occupation:
        unique_occs = len(set(train_occupation_ids))
        logger.info(
            f"Stochastic Dynamic Resampling enabled: {args.samples_per_occupation} samples/occupation/epoch "
            f"-> {unique_occs * args.samples_per_occupation} samples per epoch"
        )

    # Optuna hyperparameter search
    logger.info(f"Starting Optuna search with {args.n_trials} trials...")

    # Use seed for reproducible Optuna sampling
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )

    objective = create_objective(
        train_embeddings=train_embeddings,
        train_targets=train_targets,
        val_embeddings=val_embeddings,
        val_targets=val_targets,
        input_dim=input_dim,
        num_classes=num_classes,
        device=device,
        max_epochs=args.max_epochs,
        soft_labels=args.soft_labels,
        train_occupation_ids=train_occupation_ids,
        samples_per_occupation=args.samples_per_occupation,
    )

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best params: {study.best_params}")
    logger.info(f"Best val loss: {study.best_value:.4f}")

    # Train final model with best params
    logger.info("Training final model with best hyperparameters...")

    model, final_metrics = train_final_model(
        train_embeddings=train_embeddings,
        train_targets=train_targets,
        val_embeddings=val_embeddings,
        val_targets=val_targets,
        best_params=study.best_params,
        input_dim=input_dim,
        num_classes=num_classes,
        device=device,
        max_epochs=args.final_epochs,
        soft_labels=args.soft_labels,
        train_occupation_ids=train_occupation_ids,
        samples_per_occupation=args.samples_per_occupation,
    )

    logger.info(f"Final validation metrics: {final_metrics}")

    # Evaluate on test set if provided
    test_metrics = {}
    if args.test_path:
        logger.info("Evaluating on test set...")
        test_df = pd.read_csv(args.test_path)
        test_metrics = evaluate_on_test(
            model=model,
            encoder=encoder,
            test_df=test_df,
            esco_df=esco_df,
            cat_col=cat_col,
            label_encoder=label_encoder,
            device=device,
            cache_path=cache_dir / f"{args.cache_name}_test.npy",
            soft_labels=args.soft_labels,
        )
        logger.info(f"Test metrics: {test_metrics}")

    # Save model and artifacts
    torch.save(model.state_dict(), output_dir / "category_classifier.pt")
    label_encoder.save(str(output_dir / "label_encoder.json"))

    # Extract architecture details for config
    hidden_dims = extract_hidden_dims_from_params(study.best_params)
    use_batchnorm = study.best_params.get("use_batchnorm", True)

    # Save config and results
    results = {
        "best_params": study.best_params,
        "best_val_loss": study.best_value,
        "final_val_metrics": final_metrics,
        "test_metrics": test_metrics,
        "config": {
            "model_path": args.model_path,
            "target_level": args.target_level,
            "cat_col": cat_col,
            "num_classes": num_classes,
            "input_dim": input_dim,
            "soft_labels": args.soft_labels,
            "samples_per_occupation": args.samples_per_occupation,
            "seed": args.seed,
            "hidden_dims": hidden_dims,
            "use_batchnorm": use_batchnorm,
        },
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.success(f"Training complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()


