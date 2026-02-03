"""
isco_trainer.py - Train an ISCO group predictor from occupation text.

This mirrors the category_trainer flow but targets a single-label ISCO code
(1–4 digit level). Uses SentenceTransformer embeddings + MLP classifier with
Optuna HPO.

ISCO Label Source:
    Always uses 'data/occupations_en.csv' for (conceptUri, iscoGroup) mappings.
    This ensures clean, deduplicated occupation→ISCO pairs.

Text Configuration Modes:
    1. --use_skill_brief: Use pre-formatted 'skill_brief' column from train_path
    2. Without --use_skill_brief: Combine title + description columns
       - --include_description: Whether to include description (default: False)
       - --prefix: Text format style ('role' for 'role: X description: Y' format)

Examples:
    # Basic training with ESCO occupations (title only)
    python -m skill_mapping.v2.isco_trainer \
        --model_path pj-mathematician/JobSkillBGE-large-en-v1.5 \
        --isco_level 4 \
        --output_dir outputs/isco_model \
        --n_trials 20

    # Training with skill_brief column from augmented data
    python -m skill_mapping.v2.isco_trainer \
        --train_path data/processed/augmentation/augmented_esco_occupations.csv \
        --use_skill_brief \
        --output_dir outputs/isco_model_skillbrief \
        --n_trials 20

    # Training with occupations_en_expanded.csv, testing on decorte data
    # (train uses preferredLabel/description, test uses raw_title/raw_description)
    python -m skill_mapping.v2.isco_trainer \
        --train_path data/occupations_en_expanded.csv \
        --text_col preferredLabel \
        --desc_col description \
        --test_path data/title_pairs_desc/category_test_split_isco.csv \
        --test_text_col raw_title \
        --test_desc_col raw_description \
        --include_description \
        --prefix role \
        --output_dir outputs/isco_model_expanded \
        --n_trials 20

    # With balanced sampling (samples_per_occupation)
    python -m skill_mapping.v2.isco_trainer \
        --train_path data/processed/augmentation/augmented_esco_occupations.csv \
        --use_skill_brief \
        --samples_per_occupation 10 \
        --output_dir outputs/isco_model_balanced \
        --n_trials 20
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import textwrap
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

import warnings
warnings.filterwarnings(
    "ignore", 
    message="The number of unique classes is greater than 50%", 
    category=UserWarning
)


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


# -----------------------------
# Label utilities
# -----------------------------
def clean_isco_code(code: str) -> Optional[str]:
    """Normalize ISCO code to a 4-digit string; returns None if invalid."""
    if code is None or str(code).lower() == "nan":
        return None
    s = str(code).strip()
    # Remove trailing .0 if coming from float CSV
    if s.endswith(".0"):
        s = s[:-2]
    # Keep only digits
    s = "".join(ch for ch in s if ch.isdigit())
    if not s:
        return None
    return s.zfill(4)[:4]


def truncate_isco(code: str, level: int) -> Optional[str]:
    """Return the code truncated to level digits (1-4)."""
    base = clean_isco_code(code)
    if base is None:
        return None
    level = max(1, min(4, level))
    return base[:level]


def _first_existing_col(df: pd.DataFrame, candidates: List[str]) -> str:
    """Return first existing column from candidates; raise with helpful message otherwise."""
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of the columns exist: {candidates}. Available: {list(df.columns)}")


class SingleLabelEncoder:
    """Encode single-label strings to indices and back."""

    def __init__(self, labels: List[str]):
        unique = sorted(set(labels))
        self.str2idx = {c: i for i, c in enumerate(unique)}
        self.idx2str = {i: c for c, i in self.str2idx.items()}

    def encode(self, item: str) -> torch.Tensor:
        return torch.tensor(self.str2idx[item], dtype=torch.long)

    def __len__(self):
        return len(self.str2idx)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"str2idx": self.str2idx, "idx2str": {str(k): v for k, v in self.idx2str.items()}}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "SingleLabelEncoder":
        with open(path, "r") as f:
            data = json.load(f)
        encoder = cls([])
        encoder.str2idx = data["str2idx"]
        encoder.idx2str = {int(k): v for k, v in data["idx2str"].items()}
        return encoder


# -----------------------------
# Dataset
# -----------------------------
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings: np.ndarray, targets: torch.Tensor):
        if len(embeddings) != len(targets):
            raise ValueError(
                "Embedding/target length mismatch. "
                f"len(embeddings)={len(embeddings)} vs len(targets)={len(targets)}. "
                "This usually means you loaded a stale embedding cache built for a different split/dataset. "
                "Delete the cache files for this run (or rerun to regenerate them) and try again."
            )
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
class ISCOClassifier(nn.Module):
    """Deep MLP for single-label ISCO prediction."""

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
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


# -----------------------------
# Data loading & sampling
# -----------------------------
def load_esco_data(esco_path: str, isco_col: str) -> pd.DataFrame:
    logger.info(f"Loading ESCO data from {esco_path}")
    df = pd.read_csv(esco_path)
    if isco_col not in df.columns:
        raise ValueError(f"Column '{isco_col}' not found. Available: {list(df.columns)}")
    df = df[df[isco_col].notna()].copy()
    logger.info(f"Loaded {len(df)} rows with ISCO")
    return df


def format_text(
    title: str,
    desc: str,
    prefix: Optional[str],
    include_description: bool,
) -> str:
    """
    Format title and description into a single text string.
    
    Args:
        title: The occupation/job title
        desc: The description text
        prefix: Optional prefix style ('role' uses 'role: ... description: ...' format)
        include_description: Whether to include description in output
    
    Returns:
        Formatted text string
    """
    title = title.strip()
    desc = desc.strip() if desc else ""
    
    # Check if description is valid
    has_valid_desc = desc and desc.lower() != "nan" and include_description
    
    if prefix == "role":
        if has_valid_desc:
            return f"role: {title} \n description: {desc}"
        else:
            return f"role: {title}"
    else:
        if has_valid_desc:
            return f"{title} </s> {desc}"
        else:
            return title


def build_occupation_samples(
    esco_df: pd.DataFrame,
    isco_col: str,
    isco_level: int,
    prefix: Optional[str] = None,
    text_col: Optional[str] = None,
    desc_col: Optional[str] = None,
    augmented_df: Optional[pd.DataFrame] = None,
    use_skill_brief: bool = False,
    include_description: bool = True,
) -> Tuple[List[Dict], Dict[str, object]]:
    """
    Build samples grouped by occupation URI with explicit configuration control.
    
    This function supports two main modes:
    1. skill_brief mode (use_skill_brief=True): Uses pre-formatted 'skill_brief' column from augmented_df
    2. title+description mode (use_skill_brief=False): Combines title and optionally description columns
    
    Args:
        esco_df: ESCO dataframe with occupation-ISCO mappings
        isco_col: Column containing ISCO codes
        isco_level: ISCO level (1-4 digits)
        prefix: Text prefix style ('role' for 'role: ... description: ...' format)
        text_col: Override for title column name
        desc_col: Override for description column name
        augmented_df: Optional augmented occupation CSV
        use_skill_brief: If True, use 'skill_brief' column from augmented_df
        include_description: If True and use_skill_brief=False, include description in text
    
    Returns:
        Tuple of (samples list, config dict with actual columns used)
    
    Common ESCO schemas:
      - master_complete_hierarchy_w_occ.csv: occupationUri, occupationLabel, occupationDescription
      - occupations_en.csv / occupations_en_expanded.csv: conceptUri, preferredLabel, description
    """
    samples: List[Dict] = []
    
    # Track actual configuration used (for reproducibility/debugging)
    config_used = {
        "use_skill_brief": use_skill_brief,
        "include_description": include_description,
        "prefix": prefix,
        "text_col_used": None,
        "desc_col_used": None,
        "uri_col_used": None,
        "augmented_df_provided": augmented_df is not None,
    }

    # Auto-detect key columns (robust to different ESCO exports)
    uri_col = _first_existing_col(esco_df, ["occupationUri", "conceptUri", "esco_id"])
    config_used["uri_col_used"] = uri_col
    
    # Determine text column
    if text_col is None:
        text_col = _first_existing_col(esco_df, ["occupationLabel", "preferredLabel", "raw_title"])
    config_used["text_col_used"] = text_col
    
    # Determine description column (only if needed)
    actual_desc_col = None
    if include_description and not use_skill_brief:
        if desc_col is not None:
            actual_desc_col = desc_col
        else:
            try:
                actual_desc_col = _first_existing_col(
                    esco_df, ["occupationDescription", "description", "raw_description", "definition", "scopeNote"]
                )
            except ValueError:
                logger.warning("No description column found; proceeding without description.")
                actual_desc_col = None
    config_used["desc_col_used"] = actual_desc_col

    # Build augmented text lookup
    aug_text_map: Dict[str, List[str]] = {}
    
    if augmented_df is not None:
        aug_uri_col = _first_existing_col(augmented_df, ["occupationUri", "conceptUri", "esco_id"])
        config_used["aug_uri_col_used"] = aug_uri_col
        
        if use_skill_brief:
            # Mode 1: Use skill_brief column explicitly
            if "skill_brief" not in augmented_df.columns:
                raise ValueError(
                    "--use_skill_brief is set but 'skill_brief' column not found in augmented_df. "
                    f"Available columns: {list(augmented_df.columns)}"
                )
            
            for _, row in augmented_df.iterrows():
                uri = str(row.get(aug_uri_col, "")).strip()
                text = str(row.get("skill_brief", "")).strip()
                if uri and text and text.lower() != "nan":
                    aug_text_map.setdefault(uri, []).append(text)
            
            logger.info(
                f"[use_skill_brief=True] Loaded {sum(len(v) for v in aug_text_map.values())} texts "
                f"from 'skill_brief' for {len(aug_text_map)} occupations"
            )
        else:
            # Mode 2: Build text from title + description columns
            # Detect columns in augmented_df
            aug_title_col = None
            aug_desc_col = None
            
            # Title column detection
            if text_col and text_col in augmented_df.columns:
                aug_title_col = text_col
            else:
                for candidate in ["raw_title", "occupationLabel", "preferredLabel", "title"]:
                    if candidate in augmented_df.columns:
                        aug_title_col = candidate
                        break
            
            # Description column detection (only if include_description)
            if include_description:
                if actual_desc_col and actual_desc_col in augmented_df.columns:
                    aug_desc_col = actual_desc_col
                else:
                    for candidate in ["raw_description", "description", "occupationDescription", "definition", "scopeNote"]:
                        if candidate in augmented_df.columns:
                            aug_desc_col = candidate
                            break
            
            config_used["aug_title_col_used"] = aug_title_col
            config_used["aug_desc_col_used"] = aug_desc_col
            
            if aug_title_col is None:
                logger.warning(
                    f"No title column found in augmented_df. Available: {list(augmented_df.columns)}. "
                    "Will use ESCO fallback for all occupations."
                )
            else:
                for _, row in augmented_df.iterrows():
                    uri = str(row.get(aug_uri_col, "")).strip()
                    title = str(row.get(aug_title_col, "")).strip() if aug_title_col else ""
                    desc = str(row.get(aug_desc_col, "")).strip() if aug_desc_col else ""
                    
                    if not title:
                        continue
                    
                    text = format_text(title, desc, prefix, include_description)
                    if uri and text.strip():
                        aug_text_map.setdefault(uri, []).append(text)
                
                logger.info(
                    f"[use_skill_brief=False, include_description={include_description}] "
                    f"Loaded {sum(len(v) for v in aug_text_map.values())} texts "
                    f"(title_col={aug_title_col}, desc_col={aug_desc_col}) for {len(aug_text_map)} occupations"
                )

    # Build samples from ESCO dataframe
    empty_text_variants_dropped = 0
    fallback_used_for = 0

    for occ_uri, group in esco_df.groupby(uri_col):
        first = group.iloc[0]
        isco_code = truncate_isco(first.get(isco_col, ""), isco_level)
        if not isco_code:
            continue

        occ_uri = str(occ_uri)
        
        if occ_uri in aug_text_map:
            text_variants = aug_text_map[occ_uri]
        else:
            # Fallback to ESCO data
            fallback_used_for += 1
            title = str(first.get(text_col, "")).strip()
            desc = str(first.get(actual_desc_col, "")).strip() if actual_desc_col else ""
            fallback = format_text(title, desc, prefix, include_description)
            text_variants = [fallback] if fallback.strip() else []

        for text in text_variants:
            if not text.strip():
                empty_text_variants_dropped += 1
                continue
            samples.append(
                {
                    "uri": occ_uri,
                    "text": text,
                    "label": isco_code,
                }
            )

    unique_occs = len(set(s["uri"] for s in samples))
    logger.info(f"Built {len(samples)} samples from {unique_occs} occupations at level {isco_level}")
    logger.info(
        f"Sample construction diagnostics: fallback_used_for={fallback_used_for} occupations, "
        f"empty_text_variants_dropped={empty_text_variants_dropped}"
    )
    
    # Log configuration summary
    logger.info(f"Configuration used: {config_used}")
    
    return samples, config_used


def stratified_split_occupations(
    samples: List[Dict],
    test_size: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split by occupationUri, stratified on label.

    Safety: occupations whose label occurs <2 times (at occupation level) are forced into train,
    to avoid having labels that exist only in validation (which tanks accuracy).
    """
    uri_to_label = {}
    uri_to_samples = {}
    for s in samples:
        uri = s["uri"]
        if uri not in uri_to_samples:
            uri_to_samples[uri] = []
            uri_to_label[uri] = s["label"]
        uri_to_samples[uri].append(s)

    # IMPORTANT: make ordering deterministic. Even with a fixed RNG seed, sklearn's split
    # depends on the input order. Without sorting, upstream nondeterminism (e.g., dataframe
    # group ordering) can change the split between runs.
    unique_uris = sorted(uri_to_samples.keys())
    if len(unique_uris) < 5:
        logger.warning("Not enough occupations to split; using all for train.")
        return samples, []

    labels = [uri_to_label[u] for u in unique_uris]
    label_counts = Counter(labels)
    rare_uris = [u for u in unique_uris if label_counts[uri_to_label[u]] < 2]
    common_uris = [u for u in unique_uris if label_counts[uri_to_label[u]] >= 2]

    if rare_uris:
        logger.info(
            f"Rare-label safety: forcing {len(rare_uris)} occupations into TRAIN "
            f"(labels occurring <2 times at occupation level)"
        )

    try:
        if len(common_uris) < 2:
            raise ValueError("Not enough common occupations to stratify after rare-label filtering")
        common_labels = [uri_to_label[u] for u in common_uris]
        train_common, val_uris = train_test_split(
            common_uris, test_size=test_size, stratify=common_labels, random_state=seed
        )
        train_uris = list(train_common) + list(rare_uris)
    except ValueError as e:
        logger.warning(f"Stratified split failed ({e}); falling back to random split.")
        rng = np.random.default_rng(seed)
        rng.shuffle(common_uris)
        n_val = max(1, int(len(common_uris) * test_size))
        val_uris = common_uris[:n_val]
        train_uris = list(common_uris[n_val:]) + list(rare_uris)

    train_samples = [s for u in train_uris for s in uri_to_samples[u]]
    val_samples = [s for u in val_uris for s in uri_to_samples[u]]
    train_labels = set(uri_to_label[u] for u in train_uris)
    val_labels = set(uri_to_label[u] for u in val_uris)
    unseen_in_train = sorted(val_labels - train_labels)
    if unseen_in_train:
        logger.warning(
            f"Split diagnostics: {len(unseen_in_train)} labels appear in VAL but not TRAIN "
            f"(first 20): {unseen_in_train[:20]}"
        )
    else:
        logger.info("Split diagnostics: all VAL labels are present in TRAIN (good).")
    logger.info(
        f"Final split: {len(train_uris)} train occupations ({len(train_samples)} samples), "
        f"{len(val_uris)} val occupations ({len(val_samples)} samples)"
    )
    return train_samples, val_samples


# -----------------------------
# Embedding cache
# -----------------------------
def _fingerprint_texts(
    texts: List[str],
    *,
    max_items: int = 256,
    sample_items: int = 256,
    seed: int = 0,
) -> Dict[str, object]:
    """
    Create a lightweight, deterministic fingerprint for a text list.

    Why not hash all texts? With 400k+ items, full hashing can add noticeable CPU time. This
    fingerprint is still strong enough to detect almost all practical mismatches:
    - hashes first/last `max_items`
    - hashes a deterministic sample of `sample_items` indices (seeded)
    """
    n = len(texts)
    h = hashlib.sha256()

    def _upd(s: str) -> None:
        # Separator avoids ambiguous concatenations.
        h.update(s.encode("utf-8", errors="replace"))
        h.update(b"\0")

    if n == 0:
        return {"n_texts": 0, "sha256": h.hexdigest(), "strategy": "empty"}

    head_k = min(max_items, n)
    tail_k = min(max_items, n - head_k)

    for t in texts[:head_k]:
        _upd(str(t))
    if tail_k > 0:
        for t in texts[-tail_k:]:
            _upd(str(t))

    # Deterministic sample in the middle (avoid sampling duplicates when n is small).
    if n > head_k + tail_k:
        k = min(sample_items, n - head_k - tail_k)
        rng = random.Random(seed)
        # sample indices from [head_k, n - tail_k)
        idxs = rng.sample(range(head_k, n - tail_k), k=k) if k > 0 else []
        for i in idxs:
            _upd(str(texts[i]))

    return {
        "n_texts": n,
        "sha256": h.hexdigest(),
        "strategy": {
            "head": head_k,
            "tail": tail_k,
            "sample_items": min(sample_items, max(0, n - head_k - tail_k)),
            "seed": seed,
        },
    }


def _cache_meta_path(cache_path: Path) -> Path:
    # e.g. foo_train.npy -> foo_train.meta.json
    return cache_path.with_suffix(".meta.json")


def encode_and_cache(
    texts: List[str],
    model: SentenceTransformer,
    cache_path: Optional[Path],
    batch_size: int = 64,
) -> np.ndarray:
    if cache_path and cache_path.exists():
        # Validate cache before trusting it. Without this, it's easy to accidentally reuse
        # embeddings from a different split/test_size/train_path and silently corrupt training.
        logger.info(f"Loading cached embeddings from {cache_path}")
        embeddings = np.load(cache_path)

        meta_path = _cache_meta_path(cache_path)
        expected_fp = _fingerprint_texts(texts)

        # Always verify length (cheap + catches most issues).
        length_ok = embeddings.shape[0] == len(texts)

        # If we have metadata, verify fingerprint too.
        fp_ok = True
        meta = None
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                fp_ok = (
                    int(meta.get("n_texts", -1)) == int(expected_fp["n_texts"])
                    and str(meta.get("sha256", "")) == str(expected_fp["sha256"])
                )
            except Exception as e:
                logger.warning(f"Failed to read/parse cache metadata at {meta_path}: {e}. Will revalidate by length only.")
                fp_ok = True

        if length_ok and fp_ok:
            return embeddings

        # Mismatch: regenerate and overwrite (but keep a backup of the old cache for debugging).
        logger.warning(
            "Cached embeddings do not match current text list; regenerating cache. "
            f"len(texts)={len(texts)} vs cached_rows={embeddings.shape[0]}. "
            f"fingerprint_match={fp_ok} (meta_present={meta_path.exists()})."
        )
        try:
            backup_path = cache_path.with_suffix(cache_path.suffix + ".bak")
            cache_path.replace(backup_path)
            if meta_path.exists():
                meta_backup_path = meta_path.with_suffix(meta_path.suffix + ".bak")
                meta_path.replace(meta_backup_path)
            logger.info(f"Backed up stale cache to {backup_path}")
        except Exception as e:
            logger.warning(f"Could not back up stale cache at {cache_path}: {e}. Will overwrite in place.")

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
        try:
            meta_path = _cache_meta_path(cache_path)
            meta = _fingerprint_texts(texts)
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            logger.info(f"Cached embedding metadata to {meta_path}")
        except Exception as e:
            logger.warning(f"Failed to write cache metadata for {cache_path}: {e}")

    return embeddings


# -----------------------------
# Training & evaluation
# -----------------------------
def calculate_class_weights(targets: torch.Tensor, num_classes: int, device: torch.device) -> torch.Tensor:
    counts = torch.bincount(targets, minlength=num_classes).float()
    counts = torch.clamp(counts, min=1.0)
    inv_freq = 1.0 / counts
    weights = inv_freq / inv_freq.mean()
    return weights.to(device)


def train_epoch(
    model: ISCOClassifier,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
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
    model: ISCOClassifier,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_topk = []
    all_targets = []

    for batch in dataloader:
        embeddings = batch["embedding"].to(device)
        targets = batch["target"].to(device)
        logits = model(embeddings)
        loss = criterion(logits, targets)
        total_loss += loss.item()

        preds = torch.argmax(logits, dim=-1)
        max_k = min(10, logits.shape[-1])
        topk = torch.topk(logits, k=max_k, dim=-1).indices
        all_preds.append(preds.cpu().numpy())
        all_topk.append(topk.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_topk = np.concatenate(all_topk)
    all_targets = np.concatenate(all_targets)

    # Compute top-k hits/accuracy for k in {1,3,5,10} (bounded by num_classes)
    target_col = all_targets.reshape(-1, 1)
    top1_hits = (all_preds == all_targets)
    top3_hits = (all_topk[:, : min(3, all_topk.shape[1])] == target_col).any(axis=1)
    top5_hits = (all_topk[:, : min(5, all_topk.shape[1])] == target_col).any(axis=1)
    top10_hits = (all_topk[:, : min(10, all_topk.shape[1])] == target_col).any(axis=1)

    metrics = {
        "loss": total_loss / len(dataloader),
        "accuracy": float(accuracy_score(all_targets, all_preds)),
        "top1_hits": int(top1_hits.sum()),
        "accuracy_top1": float(top1_hits.mean()),
        "top3_hits": int(top3_hits.sum()),
        "accuracy_top3": float(top3_hits.mean()),
        "top5_hits": int(top5_hits.sum()),
        "accuracy_top5": float(top5_hits.mean()),
        "top10_hits": int(top10_hits.sum()),
        "accuracy_top10": float(top10_hits.mean()),
        "f1_macro": float(f1_score(all_targets, all_preds, average="macro", zero_division=0)),
        "f1_micro": float(f1_score(all_targets, all_preds, average="micro", zero_division=0)),
    }

    return metrics


def _log_dataset_health(samples: List[Dict], label_name: str, seed: int, n_examples: int = 5) -> None:
    """Log dataset stats + a few sample (text,label) pairs."""
    if not samples:
        logger.warning(f"Dataset health ({label_name}): no samples")
        return

    rng = random.Random(seed)
    label_counts = Counter(s["label"] for s in samples)

    logger.info(
        f"Dataset health ({label_name}): {len(samples)} samples, "
        f"{len(set(s['uri'] for s in samples))} unique occupations, "
        f"{len(label_counts)} unique labels"
    )
    logger.info(f"Label distribution top10 ({label_name}): {label_counts.most_common(10)}")
    logger.info(
        f"Label distribution tail ({label_name}): "
        f"{sum(1 for _, c in label_counts.items() if c == 1)} labels with 1 sample, "
        f"{sum(1 for _, c in label_counts.items() if c < 5)} labels with <5 samples"
    )

    # Majority-class baseline accuracy (quick sanity check)
    majority_label, majority_count = label_counts.most_common(1)[0]
    logger.info(
        f"Majority-label baseline ({label_name}): label={majority_label} "
        f"acc={majority_count / len(samples):.4f} ({majority_count}/{len(samples)})"
    )

    k = min(n_examples, len(samples))
    for i, s in enumerate(rng.sample(samples, k=k), start=1):
        text = " ".join(str(s["text"]).split())
        snippet = textwrap.shorten(text, width=240, placeholder="…")
        logger.info(f"Example {i} ({label_name}): label={s['label']} uri={s['uri']} text='{snippet}'")


# -----------------------------
# Optuna objective
# -----------------------------
def extract_hidden_dims_from_params(best_params: Dict) -> List[int]:
    n_layers = best_params.get("n_layers", 1)
    hidden_dims = []
    for i in range(n_layers):
        key = f"hidden_dim_layer_{i}"
        if key in best_params:
            hidden_dims.append(best_params[key])
    if not hidden_dims and "hidden_dim" in best_params:
        hidden_dims = [best_params["hidden_dim"]]
    return hidden_dims if hidden_dims else [512]


def create_objective(
    train_embeddings: np.ndarray,
    train_targets: torch.Tensor,
    val_embeddings: np.ndarray,
    val_targets: torch.Tensor,
    input_dim: int,
    num_classes: int,
    device: torch.device,
    max_epochs: int = 50,
    train_occupation_ids: Optional[List[str]] = None,
    samples_per_occupation: Optional[int] = None,
):
    def objective(trial: optuna.Trial) -> float:
        n_layers = trial.suggest_int("n_layers", 1, 4)
        hidden_dims = []
        layer_dim_choices = [256, 512, 768, 1024, 1536, 2048]
        for i in range(n_layers):
            dim = trial.suggest_categorical(f"hidden_dim_layer_{i}", layer_dim_choices)
            hidden_dims.append(dim)

        dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.05)
        use_batchnorm = trial.suggest_categorical("use_batchnorm", [True, False])
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        scheduler_patience = trial.suggest_int("scheduler_patience", 2, 5)
        scheduler_factor = trial.suggest_float("scheduler_factor", 0.1, 0.5, step=0.1)

        # Create datasets with optional balanced resampling
        if samples_per_occupation and train_occupation_ids:
            train_dataset = BalancedOccupationDataset(
                train_embeddings, train_targets, train_occupation_ids, samples_per_occupation
            )
        else:
            train_dataset = EmbeddingDataset(train_embeddings, train_targets)
        val_dataset = EmbeddingDataset(val_embeddings, val_targets)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = ISCOClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
            use_batchnorm=use_batchnorm,
        ).to(device)

        class_weights = calculate_class_weights(train_targets, num_classes, device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_factor,
            patience=scheduler_patience,
            verbose=False,
        )

        best_val_loss = float("inf")
        patience = 5
        patience_counter = 0

        for epoch in range(max_epochs):
            train_epoch(model, train_loader, optimizer, criterion, device)
            val_metrics = evaluate(model, val_loader, criterion, device)

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

            trial.report(val_metrics["loss"], epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val_loss

    return objective


# -----------------------------
# Final training
# -----------------------------
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
    train_occupation_ids: Optional[List[str]] = None,
    samples_per_occupation: Optional[int] = None,
):
    # Create datasets with optional balanced resampling
    if samples_per_occupation and train_occupation_ids:
        train_dataset = BalancedOccupationDataset(
            train_embeddings, train_targets, train_occupation_ids, samples_per_occupation
        )
    else:
        train_dataset = EmbeddingDataset(train_embeddings, train_targets)
    val_dataset = EmbeddingDataset(val_embeddings, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=best_params["batch_size"], shuffle=False)

    hidden_dims = extract_hidden_dims_from_params(best_params)
    use_batchnorm = best_params.get("use_batchnorm", True)
    logger.info(f"Final model architecture: hidden_dims={hidden_dims}, use_batchnorm={use_batchnorm}")

    model = ISCOClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=best_params["dropout"],
        use_batchnorm=use_batchnorm,
    ).to(device)

    class_weights = calculate_class_weights(train_targets, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
    )

    scheduler_patience = best_params.get("scheduler_patience", 3)
    scheduler_factor = best_params.get("scheduler_factor", 0.3)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
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
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]
        
        # Resample for next epoch if using balanced dataset
        if hasattr(train_dataset, 'on_epoch_end'):
            train_dataset.on_epoch_end()

        logger.info(
            f"Epoch {epoch + 1}/{max_epochs} - "
            f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
            f"Acc: {val_metrics['accuracy']:.4f}, F1m: {val_metrics['f1_macro']:.4f}, LR: {current_lr:.2e}"
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
    final_metrics = evaluate(model, val_loader, criterion, device)
    return model, final_metrics


def evaluate_on_test(
    model: ISCOClassifier,
    encoder: SentenceTransformer,
    test_df: pd.DataFrame,
    isco_col: str,
    isco_level: int,
    label_encoder: SingleLabelEncoder,
    device: torch.device,
    prefix: Optional[str] = None,
    test_text_col: Optional[str] = None,
    test_desc_col: Optional[str] = None,
    use_skill_brief: bool = False,
    include_description: bool = True,
    cache_path: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Evaluate model on test set using the same text FORMAT as training.
    
    Note: Column names can differ between train and test (e.g., train uses 
    'preferredLabel'/'description', test uses 'raw_title'/'raw_description').
    The important thing is that the TEXT FORMAT matches (prefix, include_description).
    
    Args:
        model: Trained ISCO classifier
        encoder: SentenceTransformer for encoding
        test_df: Test dataframe (can be augmented or ESCO format)
        isco_col: Column containing ISCO codes
        isco_level: ISCO level (1-4)
        label_encoder: Label encoder from training
        device: Torch device
        prefix: Text prefix style (must match training format)
        test_text_col: Text/title column in TEST file (can differ from train)
        test_desc_col: Description column in TEST file (can differ from train)
        use_skill_brief: Whether to use skill_brief column
        include_description: Whether to include description (must match training format)
        cache_path: Optional path for embedding cache
    
    Returns:
        Dict of evaluation metrics
    """
    samples, config_used = build_occupation_samples(
        test_df,
        isco_col,
        isco_level,
        prefix=prefix,
        text_col=test_text_col,
        desc_col=test_desc_col,
        augmented_df=None,  # Test df IS the data source, not augmentation
        use_skill_brief=use_skill_brief,
        include_description=include_description,
    )
    
    if not samples:
        logger.warning("No test samples found")
        return {}

    logger.info(f"Test set configuration: {config_used}")
    
    texts = [s["text"] for s in samples]
    embeddings = encode_and_cache(texts, encoder, cache_path)
    targets = torch.stack([label_encoder.encode(s["label"]) for s in samples])
    dataset = EmbeddingDataset(embeddings, targets)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, loader, criterion, device)
    return metrics


# -----------------------------
# Constants
# -----------------------------
# Fixed path for ISCO label source - clean (conceptUri, iscoGroup) mappings
ESCO_OCCUPATIONS_PATH = "data/occupations_en.csv"
ISCO_COL = "iscoGroup"


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Train ISCO group predictor with Optuna HPO")
    parser.add_argument("--val_path", type=str, default=None, help="(Deprecated/ignored) External validation CSV")
    parser.add_argument("--test_path", type=str, default=None, help="Path to test CSV (optional)")
    parser.add_argument(
        "--train_path",
        type=str,
        default=None,
        help="Training text source CSV (conceptUri + text columns). "
             "If not provided, uses ESCO occupations_en.csv with preferredLabel.",
    )
    parser.add_argument(
        "--test_text_col",
        type=str,
        default=None,
        help="Text/title column in TEST file (can differ from train). "
             "Auto-detects from: raw_title, preferredLabel, occupationLabel.",
    )
    parser.add_argument(
        "--test_desc_col",
        type=str,
        default=None,
        help="Description column in TEST file (can differ from train). "
             "Auto-detects from: raw_description, description, occupationDescription.",
    )
    parser.add_argument("--isco_level", type=int, default=4, choices=[1, 2, 3, 4], help="Digits of ISCO code to predict")
    parser.add_argument("--text_col", type=str, default=None, help="Optional override for text/title column")
    parser.add_argument("--desc_col", type=str, default=None, help="Optional override for description column")
    parser.add_argument("--prefix", type=str, default=None, help="Add prefix tokens before encoding the text.")
    parser.add_argument(
        "--use_skill_brief",
        action="store_true",
        help="Use 'skill_brief' column from train_path for text. If False, uses title+description columns.",
    )
    parser.add_argument(
        "--include_description",
        action="store_true",
        help="Include description in text (title + description). Only used when --use_skill_brief is False.",
    )

    # Model config
    parser.add_argument("--model_path", type=str, default="pj-mathematician/JobSkillBGE-large-en-v1.5")
    parser.add_argument(
        "--checkpoint_subfolder",
        type=str,
        default=None,
        help="Checkpoint subfolder when loading from hub (optional).",
    )

    # Training config
    parser.add_argument("--test_size", type=float, default=0.1, help="Share of validation split.")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    parser.add_argument(
        "--fixed_params",
        type=str,
        default=None,
        help="Path to JSON file with fixed hyperparameters (skip Optuna search)",
    )
    parser.add_argument("--max_epochs", type=int, default=50, help="Max epochs per trial")
    parser.add_argument("--final_epochs", type=int, default=100, help="Max epochs for final training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--use_full_dataset",
        action="store_true",
        help="(Deprecated/ignored) Always uses ESCO train/val split for HPO.",
    )
    parser.add_argument(
        "--samples_per_occupation",
        type=int,
        default=None,
        help="Enable Stochastic Dynamic Resampling with k samples per occupation per epoch. "
             "Balances representation across occupations with varying augmentation counts.",
    )

    # Output
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model and results")
    parser.add_argument("--cache_name", type=str, default="embeddings", help="Base name for embedding cache files")

    # Device
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load ISCO label source (fixed path - clean occupation→ISCO mappings)
    logger.info(f"Loading ISCO labels from: {ESCO_OCCUPATIONS_PATH}")
    esco_df = load_esco_data(ESCO_OCCUPATIONS_PATH, ISCO_COL)
    
    # Load training text source
    augmented_df = None
    if args.train_path:
        logger.info(f"Loading training text from: {args.train_path}")
        augmented_df = pd.read_csv(args.train_path)
    else:
        logger.info("No --train_path provided; using ESCO occupations_en.csv for text (preferredLabel)")

    # Log experiment configuration for reproducibility
    logger.info("=" * 60)
    logger.info("EXPERIMENT CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"  use_skill_brief: {args.use_skill_brief}")
    logger.info(f"  include_description: {args.include_description}")
    logger.info(f"  prefix: {args.prefix}")
    logger.info(f"  [TRAIN] text_col: {args.text_col}, desc_col: {args.desc_col}")
    logger.info(f"  [TEST]  text_col: {args.test_text_col}, desc_col: {args.test_desc_col}")
    logger.info(f"  samples_per_occupation: {args.samples_per_occupation}")
    logger.info(f"  train_path: {args.train_path}")
    logger.info(f"  test_path: {args.test_path}")
    logger.info("=" * 60)

    samples, data_config = build_occupation_samples(
        esco_df,
        ISCO_COL,
        args.isco_level,
        text_col=args.text_col,
        desc_col=args.desc_col,
        prefix=args.prefix,
        augmented_df=augmented_df,
        use_skill_brief=args.use_skill_brief,
        include_description=args.include_description,
    )
    _log_dataset_health(samples, label_name="ALL", seed=args.seed, n_examples=5)

    if args.use_full_dataset:
        logger.warning("--use_full_dataset is deprecated/ignored; using ESCO train/val split for HPO.")
    if args.val_path:
        logger.warning("--val_path is deprecated/ignored; using only ESCO-derived validation split.")

    train_samples, val_samples = stratified_split_occupations(samples, args.test_size, seed=args.seed)
    _log_dataset_health(train_samples, label_name="TRAIN", seed=args.seed, n_examples=3)
    _log_dataset_health(val_samples, label_name="VAL", seed=args.seed + 1, n_examples=3)

    if not val_samples:
        raise ValueError(
            "No validation data available after split. Check your ESCO input and filtering."
        )

    # Label encoder
    all_labels = [s["label"] for s in train_samples + val_samples]
    label_encoder = SingleLabelEncoder(all_labels)
    num_classes = len(label_encoder)
    logger.info(f"Number of classes: {num_classes}")

    # Texts and embeddings
    train_texts = [s["text"] for s in train_samples]
    val_texts = [s["text"] for s in val_samples]

    cache_suffix = f"_isco{args.isco_level}"
    if augmented_df is not None:
        cache_suffix += "_aug"
    if args.use_full_dataset:
        cache_suffix += "_full"
    # Make cache names reflect split-defining config to avoid accidental reuse across runs.
    # This complements (and reduces reliance on) runtime cache validation.
    ts_str = f"{args.test_size:.6f}".rstrip("0").rstrip(".").replace(".", "p")
    cache_suffix += f"_seed{args.seed}_ts{ts_str}"

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
                allow_patterns=[f"{checkpoint}/*"],
            )
            model_path = os.path.join(snapshot_path, checkpoint)
            encoder = SentenceTransformer(model_path, device=args.device)
        else:
            encoder = SentenceTransformer(args.model_path, device=args.device)
            if "BERT" in args.model_path:
                encoder = SentenceTransformer(modules=[encoder[0], encoder[1]], device=args.device)
    except Exception as e:
        logger.error(f"Failed to load model {args.model_path}: {e}")
        raise

    train_embeddings = encode_and_cache(
        train_texts, encoder, cache_dir / f"{args.cache_name}{cache_suffix}_train.npy"
    )
    val_embeddings = encode_and_cache(
        val_texts, encoder, cache_dir / f"{args.cache_name}{cache_suffix}_val.npy"
    )

    train_targets = torch.stack([label_encoder.encode(s["label"]) for s in train_samples])
    val_targets = torch.stack([label_encoder.encode(s["label"]) for s in val_samples])

    # Extract occupation IDs for balanced resampling
    train_occupation_ids = [s["uri"] for s in train_samples]

    input_dim = train_embeddings.shape[1]
    logger.info(f"Train: {len(train_embeddings)} samples, Val: {len(val_embeddings)} samples")
    logger.info(f"Input dim: {input_dim}, Num classes: {num_classes}")
    
    if args.samples_per_occupation:
        unique_occs = len(set(train_occupation_ids))
        logger.info(
            f"Stochastic Dynamic Resampling enabled: {args.samples_per_occupation} samples/occupation/epoch "
            f"-> {unique_occs * args.samples_per_occupation} samples per epoch"
        )

    # Hyperparameter selection: fixed params or Optuna search
    best_params = None
    best_val_loss = None
    study = None

    if args.fixed_params:
        logger.info(f"Loading fixed hyperparameters from {args.fixed_params} (skipping Optuna)")
        with open(args.fixed_params, "r") as f:
            best_params = json.load(f)

        # Ensure required keys exist with sensible defaults
        defaults = {
            "batch_size": 64,
            "lr": 1e-3,
            "dropout": 0.1,
            "weight_decay": 1e-5,
            "n_layers": 1,
            "hidden_dim": 512,
            "scheduler_patience": 3,
            "scheduler_factor": 0.3,
            "use_batchnorm": True,
        }
        for k, v in defaults.items():
            best_params.setdefault(k, v)

        logger.info(f"Using fixed hyperparameters: {best_params}")
    else:
        logger.info(f"Starting Optuna search with {args.n_trials} trials...")
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
            train_occupation_ids=train_occupation_ids,
            samples_per_occupation=args.samples_per_occupation,
        )

        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best params: {study.best_params}")
        logger.info(f"Best val loss: {study.best_value:.4f}")
        best_params = study.best_params
        best_val_loss = study.best_value

    # Train final model
    logger.info("Training final model with best hyperparameters...")
    model, final_metrics = train_final_model(
        train_embeddings=train_embeddings,
        train_targets=train_targets,
        val_embeddings=val_embeddings,
        val_targets=val_targets,
        best_params=best_params,
        input_dim=input_dim,
        num_classes=num_classes,
        device=device,
        max_epochs=args.final_epochs,
        train_occupation_ids=train_occupation_ids,
        samples_per_occupation=args.samples_per_occupation,
    )
    logger.info(f"Final validation metrics: {final_metrics}")

    # Test - use same text FORMAT as training (but column names can differ)
    test_metrics = {}
    if args.test_path:
        logger.info("Evaluating on test set...")
        logger.info(f"  Text format: use_skill_brief={args.use_skill_brief}, include_description={args.include_description}, prefix={args.prefix}")
        logger.info(f"  Test columns: text_col={args.test_text_col}, desc_col={args.test_desc_col}")
        test_df = pd.read_csv(args.test_path)
        test_metrics = evaluate_on_test(
            model=model,
            encoder=encoder,
            test_df=test_df,
            isco_col=ISCO_COL,
            isco_level=args.isco_level,
            label_encoder=label_encoder,
            device=device,
            prefix=args.prefix,
            test_text_col=args.test_text_col,
            test_desc_col=args.test_desc_col,
            use_skill_brief=args.use_skill_brief,
            include_description=args.include_description,
            cache_path=cache_dir / f"{args.cache_name}{cache_suffix}_test.npy",
        )
        logger.info(f"Test metrics: {test_metrics}")

    # Save artifacts
    torch.save(model.state_dict(), output_dir / "isco_classifier.pt")
    label_encoder.save(str(output_dir / "label_encoder.json"))

    hidden_dims = extract_hidden_dims_from_params(best_params)
    use_batchnorm = best_params.get("use_batchnorm", True)
    results = {
        "best_params": best_params,
        "best_val_loss": best_val_loss,
        "final_val_metrics": final_metrics,
        "test_metrics": test_metrics,
        "config": {
            # Model configuration
            "model_path": args.model_path,
            "checkpoint_subfolder": args.checkpoint_subfolder,
            "hidden_dims": hidden_dims,
            "use_batchnorm": use_batchnorm,
            "input_dim": input_dim,
            "num_classes": num_classes,
            
            # ISCO configuration
            "isco_level": args.isco_level,
            "isco_col": ISCO_COL,
            "esco_occupations_path": ESCO_OCCUPATIONS_PATH,
            
            # Text configuration (IMPORTANT for reproducibility)
            "use_skill_brief": args.use_skill_brief,
            "include_description": args.include_description,
            "prefix": args.prefix,
            
            # Train column config
            "text_col": args.text_col,
            "desc_col": args.desc_col,
            
            # Test column config (can differ from train)
            "test_text_col": args.test_text_col,
            "test_desc_col": args.test_desc_col,
            
            # Data paths
            "train_path": args.train_path,
            "test_path": args.test_path,
            
            # Training configuration
            "seed": args.seed,
            "test_size": args.test_size,
            "samples_per_occupation": args.samples_per_occupation,
            "n_trials": args.n_trials,
            "max_epochs": args.max_epochs,
            "final_epochs": args.final_epochs,
            
            # Actual columns used (from data_config)
            "data_config": data_config,
        },
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.success(f"Training complete! Model saved to {output_dir}")


if __name__ == "__main__":
    main()

