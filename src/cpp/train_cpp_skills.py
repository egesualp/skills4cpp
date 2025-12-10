"""
Skill-Based Sentence Transformer Finetuning Script.

Trains a sentence transformer model using skill-based career path representations:
- IDF-weighted skill pooling per job
- Logarithmic position weighting across jobs in career path
- MultipleNegativesRankingLoss for contrastive learning
- ISCO group-aware batch sampling
"""

import argparse
import os
import sys
import re
import time
import multiprocessing
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from loguru import logger
from tqdm import tqdm, trange

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.evaluation import SequentialEvaluator

try:
    from src.cpp.data_classes import Data
    from src.cpp.utils import SEP_TOKEN
    from src.cpp.skill_dataset import (
        SkillBasedCareerPathDataset,
        ISCOGroupBatchSampler,
        collate_skill_batch
    )
except ImportError as e:
    print(f"Error: Required modules not found. {e}")
    sys.exit(1)

# Configure logging
logger.remove()
logger.add(
    "logs/train_cpp_skills.log",
    format="{time} | {level} | {message}",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    enqueue=True
)
logger.add(
    sys.stdout,
    format="<green>{time}</green> | <level>{message}</level>",
    level="INFO"
)


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_skill_mappings(job_title_skills_csv: str) -> Dict[str, List[Dict]]:
    """
    Load job title to skills mapping from CSV.
    
    Args:
        job_title_skills_csv: Path to job_title_skills_master.csv
        
    Returns:
        Dict mapping job_title (normalized) to list of skill dicts
    """
    logger.info(f"Loading skill mappings from {job_title_skills_csv}...")
    df = pd.read_csv(job_title_skills_csv)
    
    # Group by job_title
    job_skill_map = defaultdict(list)
    
    for _, row in df.iterrows():
        job_title = row['job_title'].strip().lower()
        skill_dict = {
            'skill': row['skill'],
            'skillUri': row['skillUri'],
            'score': float(row['score']) if 'score' in row else 1.0
        }
        job_skill_map[job_title].append(skill_dict)
    
    logger.info(f"  ✓ Loaded skills for {len(job_skill_map)} job titles")
    return dict(job_skill_map)


def load_skill_descriptions(skills_csv: str) -> Dict[str, Dict[str, str]]:
    """
    Load skill descriptions from ESCO skills CSV.
    
    Args:
        skills_csv: Path to skills_en.csv
        
    Returns:
        Dict mapping skillUri to {name, description}
    """
    logger.info(f"Loading skill descriptions from {skills_csv}...")
    df = pd.read_csv(skills_csv)
    
    skill_desc_map = {}
    for _, row in df.iterrows():
        skill_uri = row['conceptUri']
        skill_desc_map[skill_uri] = {
            'name': row['preferredLabel'],
            'description': row.get('description', '')
        }
    
    logger.info(f"  ✓ Loaded descriptions for {len(skill_desc_map)} skills")
    return skill_desc_map


def load_occupation_isco_groups(occupations_csv: str) -> Dict[str, str]:
    """
    Load ISCO groups for ESCO occupations.
    
    Args:
        occupations_csv: Path to occupations_en.csv
        
    Returns:
        Dict mapping occupation title (normalized) to ISCO group
    """
    logger.info(f"Loading ISCO groups from {occupations_csv}...")
    df = pd.read_csv(occupations_csv)
    
    isco_map = {}
    for _, row in df.iterrows():
        title = row['preferredLabel'].strip().lower()
        isco_group = str(row['iscoGroup'])
        isco_map[title] = isco_group
    
    logger.info(f"  ✓ Loaded ISCO groups for {len(isco_map)} occupations")
    return isco_map


def calculate_idf_scores(
    job_skill_map: Dict[str, List[Dict]],
) -> Dict[str, List[Dict]]:
    """
    Calculate IDF scores for each skill based on occupation frequency.
    
    IDF = log(total_occupations / occupation_count)
    
    Args:
        job_skill_map: Job title to skills mapping
        
    Returns:
        Updated job_skill_map with 'idf' field added to each skill
    """
    logger.info("Calculating IDF scores for skills...")
    
    # Count number of occupations each skill appears in
    skill_occupation_count = defaultdict(int)
    for job_title, skills in job_skill_map.items():
        seen_skills = set()
        for skill_dict in skills:
            skill_uri = skill_dict['skillUri']
            if skill_uri not in seen_skills:
                skill_occupation_count[skill_uri] += 1
                seen_skills.add(skill_uri)
    
    # Calculate IDF
    total_occupations = len(job_skill_map)
    skill_idf = {}
    for skill_uri, count in skill_occupation_count.items():
        skill_idf[skill_uri] = np.log(total_occupations / count)
    
    # Add IDF to job_skill_map
    updated_map = {}
    for job_title, skills in job_skill_map.items():
        updated_skills = []
        for skill_dict in skills:
            skill_uri = skill_dict['skillUri']
            updated_dict = skill_dict.copy()
            updated_dict['idf'] = skill_idf.get(skill_uri, 0.0)
            updated_skills.append(updated_dict)
        updated_map[job_title] = updated_skills
    
    logger.info(f"  ✓ Calculated IDF for {len(skill_idf)} unique skills")
    return updated_map


def create_target_occupation_map(
    data_pairs: List[Tuple[str, str]],
    isco_map: Dict[str, str]
) -> Dict[str, Dict[str, str]]:
    """
    Create mapping from target_doc to occupation information.
    
    Args:
        data_pairs: List of (history_doc, target_doc) pairs
        isco_map: Occupation title to ISCO group mapping
        
    Returns:
        Dict mapping target_doc to {title, description, isco_group}
    """
    logger.info("Creating target occupation map...")
    
    target_map = {}
    missing_isco = 0
    
    for _, target_doc in data_pairs:
        if target_doc in target_map:
            continue
        
        # Extract title and description from target_doc
        # Format: "esco role: <title> \n description: <description>"
        title_match = re.search(r"esco role: (.*?)\n", target_doc)
        desc_match = re.search(r"description: (.*?)$", target_doc, re.DOTALL)
        
        if title_match and desc_match:
            title = title_match.group(1).strip()
            description = desc_match.group(1).strip()
            
            # Look up ISCO group
            title_normalized = title.lower()
            isco_group = isco_map.get(title_normalized, "unknown")
            
            if isco_group == "unknown":
                missing_isco += 1
            
            target_map[target_doc] = {
                'title': title,
                'description': description,
                'isco_group': isco_group
            }
    
    logger.info(f"  ✓ Created map for {len(target_map)} target occupations")
    if missing_isco > 0:
        logger.warning(f"  ⚠️  {missing_isco} occupations missing ISCO group")
    
    return target_map


# ============================================================================
# SKILL ENCODING AND POOLING
# ============================================================================

def encode_skills(
    skill_info_list: List[Dict],
    skill_desc_map: Dict[str, Dict[str, str]],
    encoder: SentenceTransformer,
    use_description: bool = True,
    device: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode skills and return embeddings with IDF weights.
    
    Args:
        skill_info_list: List of skill dicts with 'skillUri', 'score', 'idf'
        skill_desc_map: Skill URI to description mapping
        encoder: SentenceTransformer model
        use_description: Whether to include skill description
        device: Device to use for encoding (GPU acceleration)
        
    Returns:
        Tuple of (skill_embeddings, idf_weights)
    """
    if not skill_info_list:
        return np.array([]), np.array([])
    
    # Prepare skill texts
    skill_texts = []
    idf_weights = []
    
    for skill_dict in skill_info_list:
        skill_uri = skill_dict['skillUri']
        
        if skill_uri in skill_desc_map:
            skill_info = skill_desc_map[skill_uri]
            
            if use_description:
                text = f"skill: {skill_info['name']} \n description: {skill_info['description']}"
            else:
                text = f"skill: {skill_info['name']}"
            
            skill_texts.append(text)
            idf_weights.append(skill_dict.get('idf', 1.0))
    
    if not skill_texts:
        return np.array([]), np.array([])
    
    # Encode all skills at once with GPU support
    skill_embeddings = encoder.encode(
        skill_texts,
        convert_to_numpy=True,
        show_progress_bar=False,
        device=device
    )
    
    return skill_embeddings, np.array(idf_weights)


def pool_skills_with_idf(
    skill_embeddings: np.ndarray,
    idf_weights: np.ndarray
) -> np.ndarray:
    """
    Pool skill embeddings using IDF weights.
    
    Args:
        skill_embeddings: Array of shape [n_skills, embed_dim]
        idf_weights: Array of shape [n_skills]
        
    Returns:
        Pooled embedding of shape [embed_dim]
    """
    if len(skill_embeddings) == 0:
        return None
    
    # Normalize weights
    if np.sum(idf_weights) > 0:
        normalized_weights = idf_weights / np.sum(idf_weights)
    else:
        normalized_weights = np.ones(len(idf_weights)) / len(idf_weights)
    
    # Weighted sum
    pooled = np.sum(skill_embeddings * normalized_weights[:, np.newaxis], axis=0)
    
    return pooled


def pool_jobs_with_log_decay(
    job_vectors: List[np.ndarray],
    alpha: float = 0.5
) -> np.ndarray:
    """
    Pool job vectors using logarithmic position weighting or mean pooling.
    
    - If alpha == 0: Use mean pooling (uniform weights)
    - If alpha > 0: Use logarithmic position weighting
    
    Weight formula: w_i = log(1 + α * i) where i is position (0-indexed)
    Last job gets highest weight when alpha > 0.
    
    Args:
        job_vectors: List of job embeddings
        alpha: Decay parameter (default 0.5). Set to 0 for mean pooling.
        
    Returns:
        Pooled career path embedding
    """
    if not job_vectors:
        return None
    
    n_jobs = len(job_vectors)
    
    # Choose pooling strategy based on alpha
    if alpha == 0:
        # Mean pooling: uniform weights
        weights = np.ones(n_jobs) / n_jobs
    else:
        # Logarithmic position weighting
        weights = np.array([np.log(1 + alpha * i) for i in range(n_jobs)])
        
        # Normalize weights
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            # Fallback to uniform weights if something goes wrong
            weights = np.ones(n_jobs) / n_jobs
    
    # Stack and pool
    job_matrix = np.stack(job_vectors, axis=0)  # [n_jobs, embed_dim]
    pooled = np.sum(job_matrix * weights[:, np.newaxis], axis=0)
    
    return pooled


def process_career_path_batch(
    batch: Dict[str, Any],
    skill_desc_map: Dict[str, Dict[str, str]],
    encoder: SentenceTransformer,
    alpha: float = 0.5,
    use_skill_description: bool = True,
    device: str = None,
    precomputed_skill_embeddings: Dict[str, np.ndarray] = None,
) -> Tuple[List[np.ndarray], List[str]]:
    """
    Process a batch of career paths to generate skill-based representations.
    
    Optimized version: Batch all skill encoding together for efficiency.
    
    Args:
        batch: Batch dict from dataloader
        skill_desc_map: Skill descriptions
        encoder: SentenceTransformer model
        alpha: Logarithmic decay parameter
        use_skill_description: Whether to use skill descriptions
        device: Device to use for encoding (GPU acceleration)
        
    Returns:
        Tuple of (career_path_embeddings, target_texts)
    """
    batch_size = len(batch['job_skills_list'])
    career_path_embeddings = []
    target_texts = []

    # ------------------------------------------------------------------
    # Fast path: use precomputed skill embeddings (one vector per skillUri)
    # ------------------------------------------------------------------
    if precomputed_skill_embeddings is not None:
        career_job_vectors = [[] for _ in range(batch_size)]

        for career_idx in range(batch_size):
            job_skills_list = batch['job_skills_list'][career_idx]

            for skill_info_list in job_skills_list:
                if not skill_info_list:
                    continue

                job_skill_embeds = []
                job_idf_weights = []

                for skill_dict in skill_info_list:
                    skill_uri = skill_dict['skillUri']
                    if skill_uri in precomputed_skill_embeddings:
                        job_skill_embeds.append(precomputed_skill_embeddings[skill_uri])
                        job_idf_weights.append(skill_dict.get('idf', 1.0))

                if job_skill_embeds:
                    job_skill_embeds_arr = np.stack(job_skill_embeds, axis=0)
                    job_idf_weights_arr = np.array(job_idf_weights)
                    job_vector = pool_skills_with_idf(job_skill_embeds_arr, job_idf_weights_arr)
                    if job_vector is not None:
                        career_job_vectors[career_idx].append(job_vector)
    else:
        # ------------------------------------------------------------------
        # Original path: encode all skill texts in the batch on-the-fly
        # ------------------------------------------------------------------
        all_skill_texts = []
        all_skill_idf_weights = []
        skill_to_job_mapping = []  # Track which skills belong to which job in which career path

        for career_idx in range(batch_size):
            job_skills_list = batch['job_skills_list'][career_idx]

            for job_idx, skill_info_list in enumerate(job_skills_list):
                if not skill_info_list:
                    continue

                job_start_idx = len(all_skill_texts)

                for skill_dict in skill_info_list:
                    skill_uri = skill_dict['skillUri']

                    if skill_uri in skill_desc_map:
                        skill_info = skill_desc_map[skill_uri]

                        if use_skill_description:
                            text = f"skill: {skill_info['name']} \n description: {skill_info['description']}"
                        else:
                            text = f"skill: {skill_info['name']}"

                        all_skill_texts.append(text)
                        all_skill_idf_weights.append(skill_dict.get('idf', 1.0))

                job_end_idx = len(all_skill_texts)

                if job_end_idx > job_start_idx:
                    skill_to_job_mapping.append({
                        'career_idx': career_idx,
                        'job_idx': job_idx,
                        'start': job_start_idx,
                        'end': job_end_idx
                    })

        # Encode all skills at once (MUCH faster!) – original implementation
        if all_skill_texts:
            all_skill_embeddings = encoder.encode(
                all_skill_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                device=device,
                batch_size=128  # Larger batch for encoding
            )
            all_skill_idf_weights = np.array(all_skill_idf_weights)
        else:
            all_skill_embeddings = np.array([])

        # Reconstruct career paths from encoded skills
        career_job_vectors = [[] for _ in range(batch_size)]

        for mapping in skill_to_job_mapping:
            career_idx = mapping['career_idx']
            start = mapping['start']
            end = mapping['end']

            # Get skills for this job
            job_skill_embeds = all_skill_embeddings[start:end]
            job_idf_weights = all_skill_idf_weights[start:end]

            # Pool skills with IDF
            job_vector = pool_skills_with_idf(job_skill_embeds, job_idf_weights)
            if job_vector is not None:
                career_job_vectors[career_idx].append(job_vector)
    
    # Pool jobs with logarithmic decay
    embed_dim = encoder.get_sentence_embedding_dimension()
    for career_idx in range(batch_size):
        job_vectors = career_job_vectors[career_idx]
        
        if job_vectors:
            career_embedding = pool_jobs_with_log_decay(job_vectors, alpha)
            career_path_embeddings.append(career_embedding)
        else:
            # Handle cases with no valid skills
            career_path_embeddings.append(np.zeros(embed_dim))
        
        # Create target text
        target_text = f"role: {batch['target_titles'][career_idx]} \n description: {batch['target_descriptions'][career_idx]}"
        target_texts.append(target_text)
    
    return career_path_embeddings, target_texts


def precompute_skill_embeddings(
    job_skill_map: Dict[str, List[Dict]],
    skill_desc_map: Dict[str, Dict[str, str]],
    encoder: SentenceTransformer,
    use_skill_description: bool = True,
    device: str = None,
    batch_size: int = 1024,
) -> Dict[str, np.ndarray]:
    """
    Precompute one embedding per unique skillUri used in job_skill_map.

    This allows us to reuse fixed skill embeddings during training and
    evaluation, avoiding repeated text encoding inside each training batch.
    """
    # Collect all unique skill URIs that actually appear in occupations
    unique_skill_uris = []
    seen = set()
    for skills in job_skill_map.values():
        for skill_dict in skills:
            uri = skill_dict['skillUri']
            if uri not in seen and uri in skill_desc_map:
                seen.add(uri)
                unique_skill_uris.append(uri)

    if not unique_skill_uris:
        return {}

    # Build texts in a deterministic order
    skill_texts = []
    for uri in unique_skill_uris:
        info = skill_desc_map[uri]
        if use_skill_description:
            text = f"skill: {info['name']} \n description: {info['description']}"
        else:
            text = f"skill: {info['name']}"
        skill_texts.append(text)

    # Encode all skills once
    skill_embeddings = encoder.encode(
        skill_texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        device=device,
        batch_size=batch_size,
    )

    # Map back to uri -> embedding
    uri_to_emb = {
        uri: skill_embeddings[i]
        for i, uri in enumerate(unique_skill_uris)
    }
    return uri_to_emb


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def evaluate_ranking(
    model: SentenceTransformer,
    dataloader: DataLoader,
    skill_desc_map: Dict[str, Dict[str, str]],
    all_target_embeddings: np.ndarray,
    all_target_texts: List[str],
    alpha: float,
    use_skill_description: bool,
    device: str,
    precomputed_skill_embeddings: Dict[str, np.ndarray] = None
) -> Dict[str, float]:
    """
    Evaluate model using ranking metrics (MRR, Recall@K).
    
    Args:
        model: SentenceTransformer model
        dataloader: Evaluation dataloader
        skill_desc_map: Skill descriptions
        all_target_embeddings: Pre-computed embeddings of all targets
        all_target_texts: List of all target texts (for matching)
        alpha: Logarithmic decay parameter
        use_skill_description: Whether to use skill descriptions
        device: Device to use
        
    Returns:
        Dict with evaluation metrics
    """
    model.eval()
    
    pred_embeddings = []
    true_target_texts = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            # Process batch to get career path embeddings (with GPU acceleration)
            career_embeds, target_texts = process_career_path_batch(
                batch,
                skill_desc_map,
                model,
                alpha,
                use_skill_description,
                device,
                precomputed_skill_embeddings=precomputed_skill_embeddings,
            )
            
            pred_embeddings.extend(career_embeds)
            true_target_texts.extend(target_texts)
    
    # Convert to numpy
    pred_embeddings = np.array(pred_embeddings)
    
    # Calculate similarities
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(pred_embeddings, all_target_embeddings)
    
    # Find ranks
    sorted_indices = np.argsort(sim_matrix, axis=1)[:, ::-1]
    
    # Match true targets
    target_text_to_idx = {text: idx for idx, text in enumerate(all_target_texts)}
    
    reciprocal_ranks = []
    recall_at_1 = 0
    recall_at_5 = 0
    recall_at_10 = 0
    recall_at_20 = 0
    
    for i, true_text in enumerate(true_target_texts):
        if true_text in target_text_to_idx:
            true_idx = target_text_to_idx[true_text]
            rank_list = list(sorted_indices[i])
            
            if true_idx in rank_list:
                rank = rank_list.index(true_idx) + 1
                reciprocal_ranks.append(1.0 / rank)
                
                if rank <= 1:
                    recall_at_1 += 1
                if rank <= 5:
                    recall_at_5 += 1
                if rank <= 10:
                    recall_at_10 += 1
                if rank <= 20:
                    recall_at_20 += 1
            else:
                reciprocal_ranks.append(0.0)
    
    n_samples = len(true_target_texts)
    
    return {
        'MRR': np.mean(reciprocal_ranks),
        'R@1': recall_at_1 / n_samples,
        'R@5': recall_at_5 / n_samples,
        'R@10': recall_at_10 / n_samples,
        'R@20': recall_at_20 / n_samples
    }


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_model(
    model: SentenceTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    skill_desc_map: Dict[str, Dict[str, str]],
    all_target_embeddings: np.ndarray,
    all_target_texts: List[str],
    args,
    device: str,
    precomputed_skill_embeddings: Dict[str, np.ndarray] = None,
):
    """
    Train the model with MultipleNegativesRankingLoss.
    
    Args:
        model: SentenceTransformer model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        skill_desc_map: Skill descriptions
        all_target_embeddings: Pre-computed target embeddings
        all_target_texts: All target texts
        args: Command-line arguments
        device: Device to use
    """
    logger.info("=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80)
    
    return manual_train_loop(
        model,
        train_loader,
        val_loader,
        skill_desc_map,
        all_target_embeddings,
        all_target_texts,
        args,
        device,
        precomputed_skill_embeddings=precomputed_skill_embeddings,
    )


def manual_train_loop(
    model: SentenceTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    skill_desc_map: Dict[str, Dict[str, str]],
    all_target_embeddings: np.ndarray,
    all_target_texts: List[str],
    args,
    device: str,
    precomputed_skill_embeddings: Dict[str, np.ndarray] = None,
):
    """
    Optimized training loop with mixed precision and gradient accumulation.
    """
    from torch.optim import AdamW
    
    # Setup mixed precision scaler if enabled
    scaler = None
    # Derive device_type string for torch.amp API (e.g. "cuda" / "cpu")
    device_type = device if isinstance(device, str) else getattr(device, "type", "cuda")
    if args.mixed_precision:
        # Use the new torch.amp API instead of deprecated torch.cuda.amp
        from torch.amp import GradScaler
        # GradScaler in this Torch version expects the first argument to be the device string
        scaler = GradScaler(device=device_type)
        logger.info("🔥 Mixed precision training enabled for A100 optimization")
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    best_val_mrr = 0.0
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Training
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        accumulation_counter = 0
        data_time = 0
        gpu_time = 0
        batch_start = time.time()
        
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            # Time data loading
            data_end = time.time()
            data_time += data_end - batch_start
            
            gpu_start = time.time()
            
            if accumulation_counter == 0:
                optimizer.zero_grad()
            
            # First, process the batch to obtain fixed career-path embeddings (no gradients)
            career_embeds, target_texts = process_career_path_batch(
                batch,
                skill_desc_map,
                model,
                args.alpha_decay,
                args.use_skill_description,
                device,
                precomputed_skill_embeddings=precomputed_skill_embeddings,
            )
            
            # Convert career embeddings to tensors (no grad needed - these are fixed features)
            career_embeds_tensor = torch.tensor(
                np.array(career_embeds),
                dtype=torch.float32
            ).to(device)
            
            if args.mixed_precision and scaler is not None:
                # Only the target-encoding & loss computation run under autocast
                with torch.amp.autocast(device_type=device_type, enabled=(device_type == "cuda")):
                    # Encode target texts WITH GRADIENTS (use model forward pass, not encode())
                    # Tokenize
                    target_features = model.tokenize(target_texts)
                    target_features = {key: val.to(device) for key, val in target_features.items()}
                    
                    # Forward pass through model (with gradients!)
                    target_embeds = model(target_features)['sentence_embedding']
                    
                    # MultipleNegativesRankingLoss implementation
                    # Normalize embeddings
                    career_embeds_norm = torch.nn.functional.normalize(career_embeds_tensor, p=2, dim=1)
                    target_embeds_norm = torch.nn.functional.normalize(target_embeds, p=2, dim=1)
                    
                    # Cosine similarity matrix (scaled)
                    sim_matrix = torch.matmul(career_embeds_norm, target_embeds_norm.t()) * 20.0
                    
                    # Labels: diagonal elements are positives
                    labels = torch.arange(len(career_embeds_tensor), device=device)
                    
                    # Cross-entropy loss
                    loss = torch.nn.functional.cross_entropy(sim_matrix, labels)
                
                # Scale loss by accumulation steps and call backward
                loss = loss / args.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                # Encode target texts WITH GRADIENTS (use model forward pass, not encode())
                # Tokenize
                target_features = model.tokenize(target_texts)
                target_features = {key: val.to(device) for key, val in target_features.items()}
                
                # Forward pass through model (with gradients!)
                target_embeds = model(target_features)['sentence_embedding']
                
                # MultipleNegativesRankingLoss implementation
                # Normalize embeddings
                career_embeds_norm = torch.nn.functional.normalize(career_embeds_tensor, p=2, dim=1)
                target_embeds_norm = torch.nn.functional.normalize(target_embeds, p=2, dim=1)
                
                # Cosine similarity matrix (scaled)
                sim_matrix = torch.matmul(career_embeds_norm, target_embeds_norm.t()) * 20.0
                
                # Labels: diagonal elements are positives
                labels = torch.arange(len(career_embeds_tensor), device=device)
                
                # Cross-entropy loss
                loss = torch.nn.functional.cross_entropy(sim_matrix, labels)
                
                # Scale loss by accumulation steps and call backward
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
            
            epoch_loss += loss.item()
            accumulation_counter += 1
            
            # Update parameters every gradient_accumulation_steps
            if accumulation_counter >= args.gradient_accumulation_steps:
                if args.mixed_precision and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                accumulation_counter = 0
            
            # Time GPU computation
            gpu_time += time.time() - gpu_start
            batch_start = time.time()
            num_batches += 1
        
        # Handle remaining accumulated gradients at end of epoch
        if accumulation_counter > 0:
            if args.mixed_precision and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        
        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        logger.info(f"  Train Loss: {avg_train_loss:.4f}")
        
        # Log timing breakdown for profiling
        if len(train_loader) > 0 and args.profile_data_loading:
            total_time = data_time + gpu_time
            if total_time > 0:
                logger.info(f"⌛️  Epoch timing: Data loading: {data_time:.2f}s ({data_time/total_time*100:.1f}%), "
                           f"GPU compute: {gpu_time:.2f}s ({gpu_time/total_time*100:.1f}%)")
                if data_time / total_time > 0.3:  # More than 30% time in data loading
                    logger.warning("⚠️  Data loading bottleneck detected! Consider increasing --num_workers or optimizing skill encoding.")
        
        # Validation
        logger.info("  Running validation...")
        val_metrics = evaluate_ranking(
            model,
            val_loader,
            skill_desc_map,
            all_target_embeddings,
            all_target_texts,
            args.alpha_decay,
            args.use_skill_description,
            device,
            precomputed_skill_embeddings=precomputed_skill_embeddings,
        )
        
        logger.info(f"  Val MRR: {val_metrics['MRR']:.4f}")
        logger.info(f"  Val R@1: {val_metrics['R@1']:.4f}")
        logger.info(f"  Val R@5: {val_metrics['R@5']:.4f}")
        logger.info(f"  Val R@10: {val_metrics['R@10']:.4f}")
        
        # Log to wandb
        if WANDB_AVAILABLE and args.use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': avg_train_loss,
                **{f'val_{k.lower()}': v for k, v in val_metrics.items()}
            })
        
        # Early stopping and model saving
        if val_metrics['MRR'] > best_val_mrr:
            best_val_mrr = val_metrics['MRR']
            patience_counter = 0
            
            # Save best model
            if args.save_model:
                model_save_path = os.path.join(args.output_dir, 'best_model')
                model.save(model_save_path)
                logger.info(f"  ✓ Saved best model to {model_save_path}")
        else:
            patience_counter += 1
        
        if patience_counter >= args.patience:
            logger.info(f"  Early stopping triggered after {epoch + 1} epochs")
            break
    
    logger.info(f"\n✓ Training complete. Best Val MRR: {best_val_mrr:.4f}")
    return best_val_mrr


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Skill-Based Sentence Transformer Finetuning"
    )
    
    # Data paths
    parser.add_argument(
        "--data_type",
        type=str,
        default="karrierewege_100k",
        help="Dataset type (default: karrierewege_100k)"
    )
    parser.add_argument(
        "--job_title_skills_csv",
        type=str,
        default="results/karrierewege_esco_100k_esco_ground_truth/job_title_skills_master.csv",
        help="Path to job title skills mapping CSV"
    )
    parser.add_argument(
        "--skills_csv",
        type=str,
        default="data/esco_datasets/skills_en.csv",
        help="Path to ESCO skills CSV"
    )
    parser.add_argument(
        "--occupations_csv",
        type=str,
        default="data/esco_datasets/occupations_en.csv",
        help="Path to ESCO occupations CSV"
    )
    
    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="ElenaSenger/career-path-representation-mpnet-karrierewege",
        help="Base sentence transformer model"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--alpha_decay",
        type=float,
        default=0.5,
        help="Logarithmic decay parameter for job position weighting (default: 0.5, set to 0 for mean pooling)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Training batch size (optimized for A100)"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=512,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--use_skill_description",
        action='store_true',
        help="Include skill descriptions in encoding"
    )
    
    # GPU Optimization parameters
    parser.add_argument(
        "--mixed_precision",
        action='store_true',
        help="Enable mixed precision training (FP16) for A100 optimization"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients (simulates larger batch size)"
    )
    parser.add_argument(
        "--profile_data_loading",
        action='store_true',
        help="Enable data loading vs GPU compute timing diagnostics"
    )
    parser.add_argument(
        "--precompute_skill_embeddings",
        action='store_true',
        help="Precompute and freeze skill text embeddings (much faster, but skill side no longer updates with model weights)"
    )
    
    # Output and logging
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/cpp_skills",
        help="Output directory for models and logs"
    )
    parser.add_argument(
        "--save_model",
        action='store_true',
        help="Save the trained model"
    )
    parser.add_argument(
        "--use_wandb",
        action='store_true',
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="cpp-skills",
        help="W&B project name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity name"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="skill_based_training",
        help="Run name for logging"
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of DataLoader workers (auto-detects from SLURM_CPUS_PER_TASK if not set)"
    )
    
    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()
    
    # Fix CUDA multiprocessing issue: use 'spawn' instead of 'fork'
    # This must be done before any CUDA operations (encoder loading)
    try:
        multiprocessing.set_start_method('spawn', force=True)
        logger.info("🖥️  CUDA multiprocessing set to 'spawn'")
    except RuntimeError:
        # Already set, which is fine
        pass
    
    logger.info("=" * 80)
    logger.info("Optimized Skill-Based Sentence Transformer Finetuning")
    logger.info("=" * 80)
    logger.info(f"Configuration: {vars(args)}\n")
    
    # Initialize wandb
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=args.run_name
        )
        logger.info(f"🚀 W&B logging enabled for run: {args.run_name}\n")
    
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
    
    # --- Step 1: Load data ---
    logger.info("[1/8] Loading career path data...")
    data = Data(DATA_TYPE=args.data_type, ONLY_TITLES=False)
    train_pairs, val_pairs, test_pairs = data.get_data(stage='embedding_finetuning')
    logger.info(f"  ✓ Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}\n")
    
    # --- Step 2: Load skill mappings ---
    logger.info("[2/8] Loading skill mappings and descriptions...")
    job_skill_map = load_skill_mappings(args.job_title_skills_csv)
    skill_desc_map = load_skill_descriptions(args.skills_csv)
    isco_map = load_occupation_isco_groups(args.occupations_csv)
    logger.info("")
    
    # --- Step 3: Calculate IDF scores ---
    logger.info("[3/8] Calculating IDF scores...")
    job_skill_map = calculate_idf_scores(job_skill_map)
    logger.info("")
    
    # --- Step 4: Create target occupation map ---
    logger.info("[4/8] Creating target occupation mappings...")
    all_pairs = train_pairs + val_pairs + test_pairs
    target_occupation_map = create_target_occupation_map(all_pairs, isco_map)
    logger.info("")
    
    # --- Step 5: Load model ---
    logger.info("[5/8] Loading sentence transformer model...")
    model = SentenceTransformer(args.model_name)
    model.to(device)
    logger.info(f"  ✓ Model loaded: {args.model_name}")
    logger.info(f"  ✓ Embedding dimension: {model.get_sentence_embedding_dimension()}\n")
    # Optionally precompute one embedding per skillUri for faster training
    precomputed_skill_embeddings = None
    if args.precompute_skill_embeddings:
        logger.info("[5.5/8] Precomputing skill text embeddings (one per skillUri)...")
        precomputed_skill_embeddings = precompute_skill_embeddings(
            job_skill_map=job_skill_map,
            skill_desc_map=skill_desc_map,
            encoder=model,
            use_skill_description=args.use_skill_description,
            device=device,
            batch_size=1024,
        )
        logger.info(f"  ✓ Precomputed {len(precomputed_skill_embeddings)} unique skill embeddings\n")
    
    # --- Step 6: Create datasets ---
    logger.info("[6/8] Creating datasets...")
    
    train_dataset = SkillBasedCareerPathDataset(
        data_pairs=train_pairs,
        job_skill_map=job_skill_map,
        target_occupation_map=target_occupation_map,
        sep_token=SEP_TOKEN
    )
    
    val_dataset = SkillBasedCareerPathDataset(
        data_pairs=val_pairs,
        job_skill_map=job_skill_map,
        target_occupation_map=target_occupation_map,
        sep_token=SEP_TOKEN
    )
    
    test_dataset = SkillBasedCareerPathDataset(
        data_pairs=test_pairs,
        job_skill_map=job_skill_map,
        target_occupation_map=target_occupation_map,
        sep_token=SEP_TOKEN
    )
    
    # Create dataloaders with ISCO-aware sampling for train
    train_sampler = ISCOGroupBatchSampler(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False
    )
    
    # Optimized DataLoader configuration for A100
    use_cuda = device.type == 'cuda'
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_skill_batch,
        pin_memory=use_cuda,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_skill_batch,
        pin_memory=use_cuda,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_skill_batch,
        pin_memory=use_cuda,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    logger.info(f"  ✓ Created dataloaders")
    logger.info(f"    Train batches: {len(train_loader)}")
    logger.info(f"    Val batches: {len(val_loader)}")
    logger.info(f"    Test batches: {len(test_loader)}\n")
    
    # --- Step 7: Pre-compute all target embeddings for evaluation ---
    logger.info("[7/8] Pre-computing target embeddings for evaluation...")
    
    # Build unique target occupation texts directly from the occupation map
    # instead of sweeping all DataLoader batches, which is much more efficient.
    all_target_texts = [
        f"role: {info['title']} \n description: {info['description']}"
        for info in target_occupation_map.values()
    ]
    
    logger.info(f"  Encoding {len(all_target_texts)} unique target occupations with optimized batching...")
    all_target_embeddings = model.encode(
        all_target_texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=256,  # Larger batch size for encoding
        device=device
    )
    logger.info(f"  ✓ Target embeddings shape: {all_target_embeddings.shape}\n")
    logger.info(f"  ✓ Optimizations enabled: Mixed Precision={args.mixed_precision}, "
               f"Gradient Accumulation={args.gradient_accumulation_steps}, "
               f"Profiling={args.profile_data_loading}")
    
    # --- Step 8: Train model ---
    logger.info("[8/8] Training model...")
    
    best_val_mrr = train_model(
        model,
        train_loader,
        val_loader,
        skill_desc_map,
        all_target_embeddings,
        all_target_texts,
        args,
        device,
        precomputed_skill_embeddings=precomputed_skill_embeddings,
    )
    
    # --- Step 9: Final evaluation on test set ---
    logger.info("\n[9/9] Final evaluation on test set...")
    
    # Load best model if saved
    if args.save_model:
        model_path = os.path.join(args.output_dir, 'best_model')
        if os.path.exists(model_path):
            logger.info(f"  Loading best model from {model_path}...")
            model = SentenceTransformer(model_path)
            model.to(device)
    
    test_metrics = evaluate_ranking(
        model,
        test_loader,
        skill_desc_map,
        all_target_embeddings,
        all_target_texts,
        args.alpha_decay,
        args.use_skill_description,
        device,
        precomputed_skill_embeddings=precomputed_skill_embeddings,
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("FINAL TEST SET RESULTS")
    logger.info("=" * 80)
    for metric, value in test_metrics.items():
        logger.info(f"  {metric}: {value:.4f}")
    logger.info("=" * 80)
    
    # Log test metrics to wandb
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.log({f'test_{k.lower()}': v for k, v in test_metrics.items()})
    
    # --- Save results to CSV ---
    logger.info("\nSaving results...")
    results_csv = os.path.join(args.output_dir, 'experiment_results.csv')
    
    results_data = {
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'run_name': args.run_name,
        'model_name': args.model_name,
        'alpha_decay': args.alpha_decay,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'num_epochs': args.num_epochs,
        'use_skill_description': args.use_skill_description,
        'best_val_mrr': best_val_mrr,
        'mixed_precision': args.mixed_precision,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        'effective_batch_size': args.batch_size * args.gradient_accumulation_steps,
        **{f'test_{k}': v for k, v in test_metrics.items()}
    }
    
    results_df = pd.DataFrame([results_data])
    
    try:
        if os.path.exists(results_csv):
            results_df.to_csv(results_csv, mode='a', header=False, index=False)
        else:
            results_df.to_csv(results_csv, mode='w', header=True, index=False)
        logger.info(f"  ✓ Results saved to: {results_csv}")
    except Exception as e:
        logger.error(f"  ✗ Error saving results: {e}")
    
    logger.info("\n" + "=" * 80)
    logger.info("Script execution complete")
    logger.info("=" * 80)
    
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

