"""
Enhanced CPP Training v3: last-job skill documents with IDF-capped skills.

This script is a variant of `train_cpp_enhanced_v2.py` that:

- Uses the same **career-path pairs** and overall training pipeline as v2.
- But for the **skill-text modality**, it:
  - Takes the **last job in the career path history** (doc_1).
  - Maps that job title to its ESCO skills.
  - Recomputes IDF scores per skill and applies the **IDF + lexicographic
    top‑k capping** from `finetune_last_job_skills.py` / `skill_pooling.py`.
  - Builds a single **skill document** per sample using the **same text
    formatting** as `finetune_last_job_skills.py`:

        "skill: <name> \\n description: <description>"  (joined with SEP_TOKEN)

- Encodes these last‑job skill documents with the (fine‑tuned) skill encoder
  (typically passed via `--encoder_skill`) and uses the resulting vectors as
  `h_skill_text` features in the multi‑modal CPP model.

Everything else (Optuna, early stopping, metrics, final training, logging)
follows `train_cpp_enhanced_v2.py`.
"""

import argparse
import os
import sys
import time
import random
import multiprocessing
import gc
import hashlib
import json
import copy
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity  # kept for consistency
from tqdm import tqdm, trange

from loguru import logger

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from src.cpp.data_classes import Data
from src.cpp.utils import SEP_TOKEN
from src.cpp.cpp_dataset import CareerPathDataset, collate_career_path_batch
from src.cpp.data_loaders import (
    load_all_vocabs,
    load_job_and_skill_data,
    precompute_target_embeddings,
    precompute_input_embeddings_with_job_ids,
)
from src.cpp.skill_pooling import (
    calculate_idf_scores,
    calculate_idf_scores_by_job_id,
    cap_skills_per_job_lexicographic,
    cap_skills_per_job_stratified,
    load_raw_esco_taxonomy,
    load_skill_mappings,
    load_skill_descriptions,
    load_skills_by_job_id,
)


# Reuse model and training utilities from v2
from src.cpp.train_cpp_enhanced_v2 import (
    MultiModalCPPModel,
    SimpleConcatModel,
    calculate_ranking_metrics_gpu,
    compute_similarity_matrix_gpu,
    compute_and_save_scores,
    train_epoch,
    evaluate,
    objective,
    logger_callback,
    extract_raw_titles_from_doc,
    is_repetitive_pair,
    filter_repetitive_samples_with_job_ids,
)


# Configure logging (similar to v2)
logger.remove()
logger.add(
    "logs/debug.log",
    format="{time} | {level} | {message}",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    enqueue=True,
)
logger.add(
    sys.stdout,
    format="<green>{time}</green> | <level>{message}</level>",
    level="INFO",
)


# ============================================================================
# HELPER FUNCTIONS FOR LAST-JOB SKILL DOCS
# ============================================================================

def _compute_data_hash(text_list: List[str]) -> str:
    """Compute a deterministic hash for a list of strings for cache validation."""
    hasher = hashlib.md5()
    # Include length to catch drops quickly
    hasher.update(str(len(text_list)).encode('utf-8'))
    # Hash content
    for text in text_list:
        hasher.update(text.encode('utf-8'))
    return hasher.hexdigest()


def _skills_to_doc(
    skills: List[Dict[str, Any]],
    skill_desc_map: Dict[str, Dict[str, str]],
    include_descriptions: bool = True,
) -> str:
    """
    Convert a list of skills into the *same* document format used in
    `finetune_last_job_skills.py`:

        - With descriptions:
            "skill: ... \n description: ...<SEP>skill: ..."
        - Without descriptions:
            "skill_1<SEP>skill_2..."
    """
    segments: List[str] = []
    for skill in skills:
        uri = skill.get("skillUri")
        skill_meta = skill_desc_map.get(uri, {})
        # Prefer explicit name from desc-map, then any 'skill' label in dict
        name = skill_meta.get("name") or skill.get("skill") or ""
        if include_descriptions:
            description = skill_meta.get("description") or ""
            segments.append(f"skill: {name} \n description: {description}")
        else:
            segments.append(name)
    return SEP_TOKEN.join(segments)


def build_last_job_skill_embeddings(
    data_pairs: List[Tuple[str, str]],
    job_ids_list: List[List[str]],
    job_skill_map: Dict[str, List[Dict[str, Any]]],
    skill_desc_map: Dict[str, Dict[str, str]],
    encoder_skill: SentenceTransformer,
    include_skill_descriptions: bool,
    device: torch.device,
    cache_dir: str = None,
    encoder_name: str = None,
    force_recompute: bool = False,
) -> np.ndarray:
    """
    For each (history_doc, target_doc) pair, build a single skill document
    for the **last job in the history** and encode it with `encoder_skill`.

    Steps:
      1. Parse `history_doc` and `job_ids_list`.
      2. Take the **last job_id** (preferred) or **last title** (fallback).
      3. Look up its skills in `job_skill_map`.
      4. Convert those skills to text with `_skills_to_doc(...)`.
      5. Encode with the fine-tuned SentenceTransformer.

    Returns:
        np.ndarray of shape [n_samples, skill_embed_dim]
    """
    docs: List[str] = []
    
    # Diagnostic counters for debugging skill coverage
    stats = {
        "total_samples": len(data_pairs),
        "empty_job_ids": 0,
        "none_last_job_id": 0,
        "job_id_found": 0,
        "title_fallback_found": 0,
        "no_skills_found": 0,
        "empty_skill_doc": 0,
    }

    for (history_doc, _), job_ids in zip(data_pairs, job_ids_list):
        skills = []
        
        # 1. Check if job_ids is available
        if not job_ids:
            stats["empty_job_ids"] += 1
            docs.append("")
            continue
        
        # 2. Try job_id lookup (most accurate for decorte-style datasets)
        last_job_id = job_ids[-1]
        if last_job_id is None:
            stats["none_last_job_id"] += 1
        else:
            last_job_id_str = str(last_job_id)
            if last_job_id_str in job_skill_map:
                skills = job_skill_map[last_job_id_str]
                stats["job_id_found"] += 1
        
        # 3. Fallback to title lookup (for legacy or ESCO-based datasets)
        if not skills:
            titles = extract_raw_titles_from_doc(history_doc)
            if titles:
                last_title = titles[-1]
                title_norm = last_title.strip().lower()
                skills = job_skill_map.get(title_norm, [])
                if skills:
                    stats["title_fallback_found"] += 1

        if not skills:
            stats["no_skills_found"] += 1
            docs.append("")  # no skills for this job -> empty doc
            continue

        doc = _skills_to_doc(
            skills=skills,
            skill_desc_map=skill_desc_map,
            include_descriptions=include_skill_descriptions,
        )
        if not doc or not doc.strip():
            stats["empty_skill_doc"] += 1
        docs.append(doc if doc else "")
    
    # Log diagnostic statistics
    logger.info("  📊 Skill Document Generation Statistics:")
    logger.info(f"     Total samples: {stats['total_samples']}")
    logger.info(f"     Empty job_ids lists: {stats['empty_job_ids']}")
    logger.info(f"     None last_job_id: {stats['none_last_job_id']}")
    logger.info(f"     Skills found via job_id: {stats['job_id_found']}")
    logger.info(f"     Skills found via title fallback: {stats['title_fallback_found']}")
    logger.info(f"     No skills found: {stats['no_skills_found']}")
    logger.info(f"     Empty skill documents: {stats['empty_skill_doc']}")
    
    coverage = (stats["job_id_found"] + stats["title_fallback_found"]) / max(stats["total_samples"], 1) * 100
    logger.info(f"     Skill coverage: {coverage:.1f}%")
    
    # Log sample skill document to verify format matches finetuned model
    non_empty_docs = [d for d in docs if d.strip()]
    if non_empty_docs:
        sample_doc = non_empty_docs[0]  # First 500 chars
        logger.info(f"  📝 Sample skill document format:")
        logger.info(f"     {repr(sample_doc)}")
        if not include_skill_descriptions:
            logger.warning("  ⚠️ Skill descriptions DISABLED! If your finetuned model was trained WITH")
            logger.warning("     descriptions, add --use_skill_description to match the expected format:")
            logger.warning("     Expected: 'skill: <name> \\n description: <desc>[SEP]...'")
            logger.warning("     Current:  '<name>[SEP]...'")
    
    if coverage < 10:
        logger.warning("  ⚠️ CRITICAL: Less than 10% skill coverage! Check job_id/skill_map alignment.")
        # Log sample job_ids for debugging
        sample_job_ids = [str(jids[-1]) if jids and jids[-1] is not None else "EMPTY/NONE" 
                        for jids in job_ids_list[:5]]
        sample_map_keys = list(job_skill_map.keys())[:5]
        logger.warning(f"     Sample job_ids from data: {sample_job_ids}")
        logger.warning(f"     Sample keys in job_skill_map: {sample_map_keys}")

    # --- Caching Logic ---
    if cache_dir and encoder_name:
        # Compute hash of the generated documents
        data_hash = _compute_data_hash(docs)
        cache_filename = f"skill_doc_emb_{encoder_name}_{data_hash}.npy"
        cache_path = os.path.join(cache_dir, cache_filename)

        if os.path.exists(cache_path) and not force_recompute:
            try:
                logger.info(f"  > Found cached skill doc embeddings: {cache_filename}")
                embeddings = np.load(cache_path)
                if len(embeddings) == len(docs):
                    logger.info("  ✓ Cache loaded and validated.")
                    return embeddings
                else:
                    logger.warning("  ⚠️ Cache size mismatch. Recomputing.")
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to load cache: {e}")

    embed_dim = encoder_skill.get_sentence_embedding_dimension()
    embeddings = np.zeros((len(docs), embed_dim), dtype=np.float32)

    # Encode only non-empty docs
    non_empty_indices = [i for i, d in enumerate(docs) if d.strip()]
    if non_empty_indices:
        logger.info(f"  > Encoding {len(non_empty_indices)} non-empty skill documents...")
        texts = [docs[i] for i in non_empty_indices]
        skill_embs = encoder_skill.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=256,
            device=str(device) if device.type == "cuda" else None,
        ).astype(np.float32)
        for idx, emb in zip(non_empty_indices, skill_embs):
            embeddings[idx] = emb

    # Save to cache
    if cache_dir and encoder_name:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            np.save(cache_path, embeddings)
            logger.info(f"  ✓ Saved skill doc embeddings to {cache_filename}")
        except Exception as e:
            logger.warning(f"  ⚠️ Failed to save cache: {e}")

    return embeddings


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Enhanced CPP Training v3 with last-job skill docs (IDF+top-k)."
    )

    # Data paths
    parser.add_argument("--data_type", type=str, default="decorte")
    parser.add_argument(
        "--master_skill_file",
        type=str,
        default=(
            "/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/"
            "results/decorte_jobbert_v2_baseline/job_title_skills_master.csv"
        ),
    )
    parser.add_argument(
        "--esco_skills_file",
        type=str,
        default="data/esco_datasets/skills_en.csv",
    )
    parser.add_argument(
        "--raw_esco_dir", 
        type=str, 
        default=None, 
        help="Path to directory containing raw ESCO taxonomy files (occupations_en.csv, etc). If provided, overrides CSV mappings."
    )

    parser.add_argument(
        "--vocab_dir",
        type=str,
        default="data/processed/master_datasets_2/",
    )
    parser.add_argument(
        "--skill_properties_file",
        type=str,
        default="data/processed/master_datasets_2/skill_properties_map.json",
    )

    # Encoder configuration
    parser.add_argument(
        "--encoder_text",
        type=str,
        default="ElenaSenger/career-path-representation-mpnet-decorte",
        help="Encoder for text history and target jobs.",
    )
    parser.add_argument(
        "--encoder_skill",
        type=str,
        default="",
        help=(
            "Encoder for skills. If empty, reuse encoder_text.\n"
            "Pass the fine-tuned model from finetune_last_job_skills.py here "
            "to fully exploit last-job skill docs."
        ),
    )

    # Feature configuration
    parser.add_argument(
        "--use_text_description",
        action="store_true",
        help="Include job descriptions in text history (default: titles only).",
    )
    parser.add_argument(
        "--use_skill_description",
        action="store_true",
        help="Include skill descriptions in skill docs.",
    )
    parser.add_argument(
        "--last_job_only",
        action="store_true",
        help="Restrict to samples where history consists of a single job.",
    )

    # New: explicit top-k for last-job skills (IDF + lexicographic)
    parser.add_argument(
        "--top_k_skills",
        type=int,
        default=10,
        help=(
            "Max number of skills per job in last-job skill docs. "
            "Skills are first sorted by IDF (desc), then description length, "
            "then lexicographically by name."
        ),
    )
    parser.add_argument(
        "--skill_selection_strategy",
        type=str,
        default="top_k",
        choices=["top_k", "stratified"],
        help="Strategy to select skills: 'top_k' (highest score/IDF) or 'stratified' (sample from high/mid/low tiers).",
    )
    parser.add_argument(
        "--scoring_mode",
        type=str,
        default="idf_only",
        choices=["idf_only", "scores_only", "weighted"],
        help="How to score/rank skills for selection (mostly relevant if using predicted skills, but also for IDF logic).",
    )
    parser.add_argument(
        "--skill_scores_json",
        type=str,
        default=None,
        help="Path to fused_predictions.json with skill predictions per job_id. Required for weighted/scores_only mode with free-text datasets.",
    )
    parser.add_argument(
        "--importance_weight",
        type=float,
        default=0.5,
        help="Weight for per-job scores in weighted scoring (0-1). Default: 0.5.",
    )


    # Modality selection (for ablation studies)
    parser.add_argument(
        "--use_text_history",
        action="store_true",
        help="Include job history text features.",
    )
    parser.add_argument(
        "--use_skill_text",
        action="store_true",
        help="Include last-job skill text features.",
    )
    parser.add_argument(
        "--use_structured",
        action="store_true",
        help="Include structured meta-features.",
    )

    # Architecture
    parser.add_argument(
        "--use_advanced",
        action="store_true",
        help="Use multi-modal architecture (MultiModalCPPModel).",
    )

    # Optuna configuration
    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of Optuna trials.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help="Max epochs per trial.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=2,
        help="Early stopping patience for final training.",
    )
    parser.add_argument(
        "--optuna_patience",
        type=int,
        default=3,
        help="Early stopping patience for Optuna trials.",
    )
    parser.add_argument(
        "--val_sample_ratio",
        type=float,
        default=0.1,
        help="Fraction of validation set to use during Optuna.",
    )
    parser.add_argument(
        "--train_sample_ratio",
        type=float,
        default=1.0,
        help="Fraction of training set to use during Optuna (default 1.0 = full set).",
    )
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.001,
        help="Minimum MRR improvement to reset patience.",
    )

    # Training configuration
    parser.add_argument("--batch_size", type=int, default=None,
                       help="Training batch size. If None and --optuna is enabled, it will be searched in {16, 32, 64}.")
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4092,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help=(
            "Number of DataLoader workers (auto-detect if None; "
            "uses SLURM_CPUS_PER_TASK when available)."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Output
    parser.add_argument("--output_dir", type=str, default="results/cpp_v3")
    parser.add_argument(
        "--study_name",
        type=str,
        default="cpp_optuna_study_v3",
    )
    parser.add_argument(
        "--results_csv_path",
        type=str,
        default="results/cpp/experiment_results_v3.csv",
        help="Path to save experiment results CSV file.",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save the final model.",
    )

    # Wandb logging
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable wandb logging.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="cpp-enhanced-v3",
        help="Wandb project name.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Wandb entity name.",
    )

    # Static parameters
    parser.add_argument(
        "--optuna",
        action="store_true",
        help="Run Optuna optimization.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
        help="Learning rate (static mode).",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension (static mode).",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=1,
        help="Number of layers (static mode).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate (static mode).",
    )
    parser.add_argument(
        "--use_modality_weights",
        action="store_true",
        help="Use modality weights in MultiModalCPPModel.",
    )

    # Optimizer configuration
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Optimizer type.",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD (static mode).",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay (static mode).",
    )
    parser.add_argument(
        "--nesterov",
        action="store_true",
        help="Use Nesterov momentum for SGD.",
    )

    # Mixed precision training
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Enable mixed precision training (FP16).",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Steps to accumulate gradients.",
    )

    # Output options
    parser.add_argument(
        "--save_study",
        action="store_true",
        help="Save Optuna study to pickle file.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="cpp_enhanced_v3",
        help="Run name.",
    )
    parser.add_argument(
        "--profile_data_loading",
        action="store_true",
        help="Enable data loading vs GPU compute timing diagnostics.",
    )

    # Embeddings cache
    parser.add_argument(
        "--embeddings_cache_dir",
        type=str,
        default="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/embeddings",
        help="Directory to save/load pre-computed embeddings cache.",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Force recomputation of embeddings even if cache exists.",
    )

    # Score saving (for fusion with skill overlap scores)
    parser.add_argument(
        "--save_scores",
        action="store_true",
        help="Save prediction scores (S_text) for all splits to enable fusion with skill overlap scores.",
    )
    parser.add_argument(
        "--scores_output_dir",
        type=str,
        default=None,
        help="Directory to save prediction scores (defaults to output_dir/scores).",
    )

    # Data configuration (synced from v2)
    parser.add_argument(
        "--no_subspans",
        action="store_true",
        help="Disable using all subspans of length at least 2 in data loading.",
    )
    parser.add_argument(
        "--eval_clean_test",
        action="store_true",
        help="Evaluate on clean test set (no subspans) in addition to regular test set.",
    )
    parser.add_argument(
        "--filter_repetitive",
        action="store_true",
        help="Filter out samples where input history ends with the same occupation as target.",
    )

    # GPU optimization
    parser.add_argument(
        "--pin_embeddings_to_gpu",
        action="store_true",
        help="Pin pre-computed embeddings to GPU memory (faster but uses GPU memory).",
    )

    # Repro / study-alignment toggles (synced from v2)
    parser.add_argument(
        "--normalize_input",
        action="store_true",
        help="L2-normalize input embeddings before the mapping network (matches realistic-career-path-prediction vector_transformation.py).",
    )
    parser.add_argument(
        "--early_stop_metric",
        type=str,
        default="mrr",
        choices=["mrr", "loss"],
        help="Metric used for best-checkpoint selection / early stopping in static training. Use 'loss' to match the study pipeline.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()

    # Hardcode parameters to enforce "last job's skills" logic and satisfy downstream APIs
    args.pooling_strategy = "weighted_idf"  # Always use IDF for skill selection/capping
    args.alpha = 1.0
    args.beta = 1.0
    args.use_skill_path_log_pooling = False
    args.skill_path_alpha_decay = 0.5
    # args.last_job_only is now controlled by command line argument

    # Reproducibility (synced from v2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Validate modality selection
    n_active_modalities = sum(
        [args.use_text_history, args.use_skill_text, args.use_structured]
    )
    if n_active_modalities == 0:
        raise ValueError(
            "At least one modality must be enabled! "
            "Use --use_text_history, --use_skill_text, or --use_structured."
        )

    # Auto-detect num_workers if not specified
    if args.num_workers is None:
        # When using GPU-pinned embeddings, num_workers should be 0 (single process)
        if args.pin_embeddings_to_gpu:
            args.num_workers = 0
            logger.info(f"🖥️  Auto-set num_workers=0 (GPU-pinned embeddings don't benefit from multi-process loading)")
        else:
            slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
            if slurm_cpus:
                args.num_workers = max(1, int(slurm_cpus) - 2)
                logger.info(
                    f"🖥️  Auto-detected num_workers={args.num_workers} "
                    f"from SLURM_CPUS_PER_TASK={slurm_cpus}"
                )
            else:
                args.num_workers = min(16, max(1, multiprocessing.cpu_count() - 1))
                logger.info(
                    f"🖥️  Auto-detected num_workers={args.num_workers} "
                    f"from CPU count"
                )
    else:
        logger.info(f"🖥️  Using specified num_workers={args.num_workers}")
        if args.pin_embeddings_to_gpu and args.num_workers > 0:
            logger.warning("⚠️  Warning: Using num_workers > 0 with GPU-pinned embeddings may not improve performance")

    # Use pin_memory based on device and GPU pinning
    use_pin_memory = (torch.cuda.is_available()) and not args.pin_embeddings_to_gpu
    logger.info(f"🖥️  DataLoader configuration: num_workers={args.num_workers}, "
                f"pin_memory={use_pin_memory}")

    logger.info("=" * 80)
    logger.info("Enhanced Career Path Prediction Training (v3 - last-job skills)")
    logger.info("=" * 80)
    logger.info(
        f"Architecture: {'Multi-modal (Advanced)' if args.use_advanced else 'Simple Concatenation'}"
    )
    logger.info(f"Active Modalities ({n_active_modalities}):")
    logger.info(
        f"  - Text History: {'✓' if args.use_text_history else '✗'}"
        + (
            f" ({'with descriptions' if args.use_text_description else 'titles only'})"
            if args.use_text_history
            else ""
        )
    )
    logger.info(
        f"  - Skill Text (last-job skills): {'✓' if args.use_skill_text else '✗'}"
        + (
            f" ({'with descriptions' if args.use_skill_description else 'names only'})"
            if args.use_skill_text
            else ""
        )
    )
    logger.info(
        f"  - Structured Features: {'✓' if args.use_structured else '✗'}"
    )
    logger.info(f"Configuration: {vars(args)}\n")

    # Initialize wandb
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args,
            name=args.run_name,
            reinit=True,
        )
        logger.info(f"🚀 wandb logging enabled for run: {args.run_name}")

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Add per-run logging to the output directory
    run_log_path = os.path.join(args.output_dir, "training.log")
    logger.add(
        run_log_path,
        format="{time} | {level} | {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="31 days",
        enqueue=True,
    )
    logger.info(f"📜 Logging this run to: {run_log_path}")

    device = torch.device(args.device)

    # --- Step 1: Load encoders ---
    logger.info("[1/8] Loading encoder models...")
    encoder_text = SentenceTransformer(args.encoder_text)

    # Use separate skill encoder if specified (ideally the fine-tuned model)
    if args.encoder_skill:
        logger.info(f"  > Using separate skill encoder: {args.encoder_skill}")
        encoder_skill = SentenceTransformer(args.encoder_skill)
        skill_text_dim = encoder_skill.get_sentence_embedding_dimension()
    else:
        logger.info("  > Using same encoder for skills as for text.")
        encoder_skill = encoder_text
        skill_text_dim = encoder_text.get_sentence_embedding_dimension()

    text_dim = encoder_text.get_sentence_embedding_dimension()
    logger.info(
        f"  ✓ Text encoder dim: {text_dim}, Skill encoder dim: {skill_text_dim}\n"
    )

    # --- Step 2: Load data pairs ---
    logger.info("[2/8] Loading career path data...")
    data = Data(
        DATA_TYPE=args.data_type,
        ONLY_TITLES=not args.use_text_description,
        consider_subspans=not args.no_subspans,
        LOAD_CLEAN_TEST=args.eval_clean_test,
    )
    
    if args.eval_clean_test:
        (train_pairs, train_job_ids), (val_pairs, val_job_ids), (test_pairs, test_job_ids), (test_clean_pairs, test_clean_job_ids) = data.get_data_with_job_ids(
            stage="transformation_finetuning", include_clean_test=True
        )
    else:
        (train_pairs, train_job_ids), (val_pairs, val_job_ids), (test_pairs, test_job_ids) = data.get_data_with_job_ids(stage="transformation_finetuning")
        test_clean_pairs = []
        test_clean_job_ids = []

    if args.last_job_only:
        logger.info("  > Filtering for 'last job only' pairs (single-history jobs)...")
        # Filter pairs and job_ids together to keep them aligned
        train_filtered = [(p, j) for p, j in zip(train_pairs, train_job_ids) if SEP_TOKEN not in p[0]]
        train_pairs = [p for p, _ in train_filtered]
        train_job_ids = [j for _, j in train_filtered]

        val_filtered = [(p, j) for p, j in zip(val_pairs, val_job_ids) if SEP_TOKEN not in p[0]]
        val_pairs = [p for p, _ in val_filtered]
        val_job_ids = [j for _, j in val_filtered]

        test_filtered = [(p, j) for p, j in zip(test_pairs, test_job_ids) if SEP_TOKEN not in p[0]]
        test_pairs = [p for p, _ in test_filtered]
        test_job_ids = [j for _, j in test_filtered]

        if args.eval_clean_test:
            test_clean_filtered = [(p, j) for p, j in zip(test_clean_pairs, test_clean_job_ids) if SEP_TOKEN not in p[0]]
            test_clean_pairs = [p for p, _ in test_clean_filtered]
            test_clean_job_ids = [j for _, j in test_clean_filtered]

    if args.filter_repetitive:
        logger.info("  > Filtering out repetitive samples (where history ends with target)...")
        logger.info(
            f"    Before filtering - Train: {len(train_pairs)}, Val: {len(val_pairs)}, "
            f"Test: {len(test_pairs)}" + (f", Test (clean): {len(test_clean_pairs)}" if args.eval_clean_test else "")
        )
        # Filter pairs and job_ids together using the helper function
        train_pairs, train_job_ids = filter_repetitive_samples_with_job_ids(train_pairs, train_job_ids)
        val_pairs, val_job_ids = filter_repetitive_samples_with_job_ids(val_pairs, val_job_ids)
        test_pairs, test_job_ids = filter_repetitive_samples_with_job_ids(test_pairs, test_job_ids)
        
        if args.eval_clean_test:
            test_clean_pairs, test_clean_job_ids = filter_repetitive_samples_with_job_ids(test_clean_pairs, test_clean_job_ids)

    logger.info(
        f"  ✓ Train: {len(train_pairs)}, Val: {len(val_pairs)}, Test: {len(test_pairs)}"
        + (f", Test (clean): {len(test_clean_pairs)}" if args.eval_clean_test else "") + "\n"
    )

    # --- Data Verification ---
    logger.info("  🔍 Data Verification (Random Training Sample):")
    idx = min(300, len(train_pairs) - 1)
    logger.info(f"     History (first 300 chars): {repr(train_pairs[idx][0][:300])}")
    logger.info(f"     Target (first 300 chars): {repr(train_pairs[idx][1][:300])}")
    logger.info(f"     Job IDs in history: {train_job_ids[idx]}")
    
    # --- Job ID Statistics ---
    logger.info("  📊 Job ID Statistics:")
    empty_job_ids = sum(1 for jids in train_job_ids if not jids)
    none_in_job_ids = sum(1 for jids in train_job_ids if jids and any(j is None for j in jids))
    valid_last_job_ids = sum(1 for jids in train_job_ids if jids and jids[-1] is not None)
    logger.info(f"     Total train samples: {len(train_job_ids)}")
    logger.info(f"     Samples with empty job_ids: {empty_job_ids}")
    logger.info(f"     Samples with None in job_ids: {none_in_job_ids}")
    logger.info(f"     Samples with valid last job_id: {valid_last_job_ids}")
    
    if valid_last_job_ids < len(train_job_ids) * 0.5:
        logger.warning("  ⚠️ WARNING: Less than 50% of samples have valid job_ids!")
        logger.warning("     Check that the master dataset CSV exists and job_id lookup is working.")
    
    # Show sample job_id values
    sample_last_ids = [str(jids[-1]) if jids and jids[-1] is not None else "NONE" 
                      for jids in train_job_ids[:5]]
    logger.info(f"     Sample last job_ids (first 5): {sample_last_ids}")
    logger.info("  " + "-" * 60 + "\n")
    # -------------------------

    # --- Step 2b: Extract train+val job titles for IDF (data_loaders) ---
    logger.info("[2b/8] Extracting train+val job titles for IDF calculation...")
    train_val_jobs = set()
    for history_doc, target_doc in train_pairs + val_pairs:
        train_val_jobs.update(extract_raw_titles_from_doc(history_doc))
        train_val_jobs.update(extract_raw_titles_from_doc(target_doc))
    logger.info(f"  ✓ Extracted {len(train_val_jobs)} unique train+val job titles\n")

    # --- Step 3: Load vocabularies and base skill mappings ---
    logger.info("[3/8] Loading vocabularies and skill mappings...")
    all_vocabs = load_all_vocabs(args.vocab_dir)
    structured_dim = sum(len(vocab) for vocab in all_vocabs.values())

    # --- Step 3b: Determine Loading Mode and Load Skill Data ---
    job_skill_map = {}
    esco_skill_text_map = {}
    skill_properties_map = {}
    uses_predicted_skills = False

    if args.skill_scores_json:
        # 1. Job ID-based predicted skills (Standard for decorte dataset)
        logger.info(f"  > [MODE] Loading predicted skills from {args.skill_scores_json}...")
        
        # We use a combined approach: need descriptions from esco_skills_file and properties from skill_properties_file
        # but the mapping comes from the JSON
        from src.cpp.skill_pooling import load_skills_by_job_id, load_skill_descriptions
        
        # Mapping: job_id -> list of skills
        job_skill_map = load_skills_by_job_id(args.skill_scores_json)
        uses_predicted_skills = True
        
        # --- Verify job_id format alignment ---
        sample_skill_map_keys = list(job_skill_map.keys())[:5]
        sample_data_job_ids = [str(jids[-1]) if jids and jids[-1] is not None else "NONE" 
                              for jids in train_job_ids[:5]]
        logger.info(f"  📊 Job ID Format Verification:")
        logger.info(f"     Sample keys from skill_map: {sample_skill_map_keys}")
        logger.info(f"     Sample last job_ids from data: {sample_data_job_ids}")
        
        # Check if any data job_ids are in the skill_map
        all_data_job_ids = set(str(jids[-1]) for jids in train_job_ids if jids and jids[-1] is not None)
        matched = len(all_data_job_ids & set(job_skill_map.keys()))
        logger.info(f"     Unique job_ids in data: {len(all_data_job_ids)}")
        logger.info(f"     Matched in skill_map: {matched} ({matched/max(len(all_data_job_ids),1)*100:.1f}%)")
        
        if matched == 0:
            logger.error("  ❌ CRITICAL: No job_ids from data match skill_map keys!")
            logger.error("     This means skill embeddings will be ALL ZEROS.")
            logger.error("     Check that job_ids format matches between:")
            logger.error(f"       1. Data class (master CSV lookup)")
            logger.error(f"       2. fused_predictions.json ({args.skill_scores_json})")
        
        # Descriptions
        full_descriptions = load_skill_descriptions(args.esco_skills_file)
        # Convert format to match internal expectation
        esco_skill_text_map = {uri: {"name": meta["name"], "desc": meta["description"]} for uri, meta in full_descriptions.items()}
        
        # Properties
        with open(args.skill_properties_file, 'r') as f:
            skill_properties_map = json.load(f)
            
    elif args.raw_esco_dir:
        # 2. Raw ESCO taxonomy override
        logger.info(f"  > [MODE] Loading raw ESCO taxonomy from {args.raw_esco_dir}...")
        job_skill_map, skill_desc_map_raw = load_raw_esco_taxonomy(args.raw_esco_dir)
        uses_predicted_skills = False
        
        # Sync maps
        esco_skill_text_map = {uri: {"name": meta["name"], "desc": meta["description"]} for uri, meta in skill_desc_map_raw.items()}
        with open(args.skill_properties_file, 'r') as f:
            skill_properties_map = json.load(f)
            
    else:
        # 3. Legacy/Default mode: Load from master skill CSV
        logger.info(f"  > [MODE] Loading standard skill mappings from {args.master_skill_file}...")
        job_skill_map, esco_skill_text_map, skill_properties_map = load_job_and_skill_data(
            master_skill_file=args.master_skill_file,
            esco_skills_file=args.esco_skills_file,
            skill_properties_file=args.skill_properties_file,
            pooling_strategy="mean",
            alpha=args.alpha,
            beta=args.beta,
            train_val_occ=train_val_jobs,
        )
        uses_predicted_skills = False

    # Sync skill_desc_map for downstream capping/formatting logic
    skill_desc_map = {uri: {"name": meta["name"], "description": meta.get("desc") or meta.get("description", "")} 
                     for uri, meta in esco_skill_text_map.items()}

    logger.info(f"  ✓ Structured feature dim: {structured_dim}\n")

    # --- Step 3c: Calculate IDF / Scores ---
    logger.info(f"[3c/8] Calculating Skill Scores (Mode: {args.scoring_mode})...")
    
    if uses_predicted_skills:
        # Free-text / Predicted Skills logic
        if args.scoring_mode == "scores_only":
            # No IDF needed for scores_only
             pass
        elif args.scoring_mode == "weighted":
             job_skill_map = calculate_idf_scores_by_job_id(
                 job_skill_map, use_job_scores=True, importance_weight=args.importance_weight
             )
        else: # idf_only
             job_skill_map = calculate_idf_scores_by_job_id(job_skill_map, use_job_scores=False)
    else:
        # Taxonomy / ESCO logic 
        # Note: If we didn't override, we might be recalculating IDF on the base map, which is fine/desired.
        if args.scoring_mode != "idf_only" and args.scoring_mode != "idf_only": # Check redundancy
            logger.warning(f"  ⚠️ scoring_mode={args.scoring_mode} ignored for ESCO taxonomy skills. Using IDF only.")
        
        # We assume taxonomy skills don't have 'score' or it's 1.0. 
        # If we loaded from CSV, they might have scores but we usually ignore them for V3 unless specified.
        # But let's just apply IDF.
        job_skill_map = calculate_idf_scores(job_skill_map)

    # --- Step 3d: Cap Skills (Top-K or Stratified) ---
    logger.info(f"[3d/8] Capping skills per job (Strategy: {args.skill_selection_strategy}, k={args.top_k_skills})...")
    
    use_weighted = (args.scoring_mode == "weighted")
    
    # Ensure skills have 'skill' label for lexicographic sort if missing (from raw loading)
    for job_id, skills in job_skill_map.items():
        for s in skills:
            if "skill" not in s or not s["skill"]:
                uri = s.get("skillUri")
                if uri and uri in skill_desc_map:
                    s["skill"] = skill_desc_map[uri]["name"]

    if args.skill_selection_strategy == "stratified":
        job_skill_map = cap_skills_per_job_stratified(
            job_skill_map,
            max_skills_per_job=args.top_k_skills,
            use_weighted_idf=use_weighted
        )
    else:
        # Top-K
        if uses_predicted_skills and args.scoring_mode == "scores_only":
             # Special case for predictions without IDF
              from src.cpp.skill_pooling import cap_skills_per_job_by_score
              job_skill_map = cap_skills_per_job_by_score(
                  job_skill_map, 
                  max_skills_per_job=args.top_k_skills,
                  skill_desc_map=skill_desc_map
              )
        else:
             # Standard IDF/Weighted-IDF top-k
             job_skill_map = cap_skills_per_job_lexicographic(
                job_skill_map,
                max_skills_per_job=args.top_k_skills,
                skill_desc_map=skill_desc_map,
                use_weighted_idf=use_weighted
             )

    # --- Step 3e: Coverage diagnostics ---
    logger.info("[3e/8] Checking job-to-skill coverage...")
    
    if uses_predicted_skills:
        # For predicted skills (job_id mode), coverage was already verified in Step 3b
        # The job_skill_map keys are job_ids, not job titles, so title-based check doesn't apply
        logger.info("  > Skipping title-based coverage check (using job_id-based skills)")
        logger.info("  > Job ID coverage was verified in Step 3b above")
    else:
        # For ESCO taxonomy mode, check job title coverage
        train_val_jobs_in_dataset = set()
        for history_doc, target_doc in train_pairs + val_pairs:
            train_val_jobs_in_dataset.update(extract_raw_titles_from_doc(history_doc))
            train_val_jobs_in_dataset.update(extract_raw_titles_from_doc(target_doc))

        jobs_with_skills = set(job_skill_map.keys())
        train_val_missing = train_val_jobs_in_dataset - jobs_with_skills
        train_val_coverage = len(train_val_jobs_in_dataset - train_val_missing) / max(
            len(train_val_jobs_in_dataset), 1
        )

        logger.info(f"  > Train+Val unique jobs: {len(train_val_jobs_in_dataset)}")
        logger.info(f"  > Train+Val coverage: {100 * train_val_coverage:.1f}%")
        
        if train_val_missing:
            logger.warning(f"  ⚠️  {len(train_val_missing)} train+val jobs will receive zero skill embeddings")

    # --- Step 4: Compute embeddings for text history + targets ---
    logger.info("[4/8] Computing embeddings (text history + targets)...")
    
    # Ensure cache directory exists
    os.makedirs(args.embeddings_cache_dir, exist_ok=True)
    encoder_text_name = args.encoder_text.split('/')[-1]
    
    # 4a: Compute target embeddings (with caching)
    logger.info("  [4a] Computing target embeddings...")
    # Deterministic target ordering improves reproducibility and stabilizes caching/indices
    all_target_labels = sorted(set([t for _, t in train_pairs + val_pairs + test_pairs]))
    Y_target_dict, Y_target_all = precompute_target_embeddings(
        encoder_text,
        all_target_labels,
        show_progress=True,
        cache_dir=args.embeddings_cache_dir,
        encoder_name=encoder_text_name,
        force_recompute=args.force_recompute,
    )
    output_dim = Y_target_all.shape[1]
    logger.info(f"  ✓ Target embedding dim: {output_dim}\n")
    
    # 4b: Compute text history embeddings only (skill embeddings computed in Step 5)
    # We use the version with job_ids to ensure alignment is preserved during any internal filtering.
    logger.info("  [4b] Computing text history embeddings for train set...")
    train_pairs, train_job_ids, train_h_text, _ = precompute_input_embeddings_with_job_ids(
        train_pairs, train_job_ids, Y_target_dict, encoder_text, encoder_skill,
        job_skill_map, esco_skill_text_map,
        use_skill_description=args.use_skill_description,
        pooling_strategy=args.pooling_strategy,
        alpha=args.alpha, beta=args.beta,
        use_text_history=args.use_text_history,
        use_skill_text=False,  # Disabled - v3 computes its own skill embeddings
        cache_dir=args.embeddings_cache_dir,
        encoder_skill_name=encoder_text_name,
        force_recompute=args.force_recompute,
        split_name="train",
    )
    
    logger.info("  [4c] Computing text history embeddings for val set...")
    val_pairs, val_job_ids, val_h_text, _ = precompute_input_embeddings_with_job_ids(
        val_pairs, val_job_ids, Y_target_dict, encoder_text, encoder_skill,
        job_skill_map, esco_skill_text_map,
        use_skill_description=args.use_skill_description,
        pooling_strategy=args.pooling_strategy,
        alpha=args.alpha, beta=args.beta,
        use_text_history=args.use_text_history,
        use_skill_text=False,  # Disabled - v3 computes its own skill embeddings
        cache_dir=args.embeddings_cache_dir,
        encoder_skill_name=encoder_text_name,
        force_recompute=False,
        split_name="val",
    )
    
    logger.info("  [4d] Computing text history embeddings for test set...")
    test_pairs, test_job_ids, test_h_text, _ = precompute_input_embeddings_with_job_ids(
        test_pairs, test_job_ids, Y_target_dict, encoder_text, encoder_skill,
        job_skill_map, esco_skill_text_map,
        use_skill_description=args.use_skill_description,
        pooling_strategy=args.pooling_strategy,
        alpha=args.alpha, beta=args.beta,
        use_text_history=args.use_text_history,
        use_skill_text=False,  # Disabled - v3 computes its own skill embeddings
        cache_dir=args.embeddings_cache_dir,
        encoder_skill_name=encoder_text_name,
        force_recompute=False,
        split_name="test",
    )
    logger.info("  ✓ Text history embeddings computed\n")

    # --- Step 5: Build last-job skill embeddings using finetuned-style docs ---
    logger.info("[5/8] Computing last-job skill embeddings (finetuned-style docs)...")
    include_skill_descriptions = args.use_skill_description
    encoder_skill_name = args.encoder_skill.split('/')[-1] if args.encoder_skill else encoder_text_name

    logger.info("  > Computing skill doc embeddings for train set...")
    train_h_skill_last = (
        build_last_job_skill_embeddings(
            train_pairs,
            train_job_ids,
            job_skill_map,
            skill_desc_map,
            encoder_skill,
            include_skill_descriptions,
            device,
            cache_dir=args.embeddings_cache_dir,
            encoder_name=encoder_skill_name,
            force_recompute=args.force_recompute,
        )
        if args.use_skill_text
        else None
    )
    logger.info("  > Computing skill doc embeddings for val set...")
    val_h_skill_last = (
        build_last_job_skill_embeddings(
            val_pairs,
            val_job_ids,
            job_skill_map,
            skill_desc_map,
            encoder_skill,
            include_skill_descriptions,
            device,
            cache_dir=args.embeddings_cache_dir,
            encoder_name=encoder_skill_name,
            force_recompute=False,
        )
        if args.use_skill_text
        else None
    )
    logger.info("  > Computing skill doc embeddings for test set...")
    test_h_skill_last = (
        build_last_job_skill_embeddings(
            test_pairs,
            test_job_ids,
            job_skill_map,
            skill_desc_map,
            encoder_skill,
            include_skill_descriptions,
            device,
            cache_dir=args.embeddings_cache_dir,
            encoder_name=encoder_skill_name,
            force_recompute=False,
        )
        if args.use_skill_text
        else None
    )

    # Clean test set embeddings (if enabled)
    test_clean_h_text = None
    test_clean_h_skill_last = None
    if args.eval_clean_test and test_clean_pairs:
        logger.info("  > Computing text history embeddings for clean test set...")
        test_clean_pairs, test_clean_job_ids, test_clean_h_text, _ = precompute_input_embeddings_with_job_ids(
            test_clean_pairs, test_clean_job_ids, Y_target_dict, encoder_text, encoder_skill,
            job_skill_map, esco_skill_text_map,
            use_skill_description=args.use_skill_description,
            pooling_strategy=args.pooling_strategy,
            alpha=args.alpha, beta=args.beta,
            use_text_history=args.use_text_history,
            use_skill_text=False,  # Disabled - v3 computes its own skill embeddings
            cache_dir=args.embeddings_cache_dir,
            encoder_skill_name=encoder_text_name,
            force_recompute=False,
            split_name="test_clean",
        )
        logger.info("  > Computing skill doc embeddings for clean test set...")
        test_clean_h_skill_last = (
            build_last_job_skill_embeddings(
                test_clean_pairs,
                test_clean_job_ids,
                job_skill_map,
                skill_desc_map,
                encoder_skill,
                include_skill_descriptions,
                device,
                cache_dir=args.embeddings_cache_dir,
                encoder_name=encoder_skill_name,
                force_recompute=False,
            )
            if args.use_skill_text
            else None
        )

    # --- Step 6: Create datasets & loaders (using precomputed features) ---
    logger.info("[6/8] Creating datasets and dataloaders...")

    # Log GPU memory optimization settings (synced from v2)
    if args.pin_embeddings_to_gpu:
        logger.info("📌 GPU Memory Optimization: Pinning embeddings to GPU")
        logger.info("   This will use GPU memory but eliminate CPU->GPU transfers")
    else:
        logger.info("💾 Shared Memory Optimization: Using shared memory for multi-process DataLoader")
        logger.info("   Consider using --pin_embeddings_to_gpu if you have enough GPU memory")

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
        pre_h_skill_text=train_h_skill_last,
        device=device,
        pin_embeddings_to_gpu=args.pin_embeddings_to_gpu,
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
        pre_h_skill_text=val_h_skill_last,
        device=device,
        pin_embeddings_to_gpu=args.pin_embeddings_to_gpu,
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
        pre_h_skill_text=test_h_skill_last,
        device=device,
        pin_embeddings_to_gpu=args.pin_embeddings_to_gpu,
    )

    # Clean test dataset (if enabled)
    test_clean_dataset = None
    if args.eval_clean_test and test_clean_pairs:
        test_clean_dataset = CareerPathDataset(
            data_pairs=test_clean_pairs,
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
            pre_h_text=test_clean_h_text,
            pre_h_skill_text=test_clean_h_skill_last,
            device=device,
            pin_embeddings_to_gpu=args.pin_embeddings_to_gpu,
        )

    # Determine effective batch size for initial loaders
    effective_batch_size = args.batch_size if args.batch_size is not None else 512
    if args.batch_size is None:
        logger.info(f"💡 No --batch_size provided. Using {effective_batch_size} for initialization.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_career_path_batch,
        pin_memory=use_pin_memory,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_career_path_batch,
        pin_memory=use_pin_memory,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_career_path_batch,
        pin_memory=use_pin_memory,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # Clean test loader (if enabled)
    test_clean_loader = None
    if args.eval_clean_test and test_clean_dataset is not None:
        test_clean_loader = DataLoader(
            test_clean_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_career_path_batch,
            pin_memory=use_pin_memory,
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=2 if args.num_workers > 0 else None,
        )

    # Validation subset for Optuna
    val_sample_size = max(1, int(len(val_dataset) * args.val_sample_ratio))
    seed = getattr(args, "seed", 42)
    generator = torch.Generator()
    generator.manual_seed(seed)
    val_sample_indices = torch.randperm(
        len(val_dataset), generator=generator
    )[:val_sample_size].tolist()
    val_sample_dataset = torch.utils.data.Subset(val_dataset, val_sample_indices)
    val_sample_loader = DataLoader(
        val_sample_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_career_path_batch,
        pin_memory=use_pin_memory,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # Create sampled train loader for Optuna trials
    train_sample_loader = train_loader
    if args.optuna and args.train_sample_ratio < 1.0:
        train_sample_size = max(1, int(len(train_dataset) * args.train_sample_ratio))
        logger.info(
            f"  > Subsampling training set for Optuna: {args.train_sample_ratio*100:.0f}% "
            f"({train_sample_size} samples)"
        )
        seed = getattr(args, "seed", 42)
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        train_sample_indices = torch.randperm(
            len(train_dataset), generator=generator
        )[:train_sample_size].tolist()
        
        train_sample_dataset = torch.utils.data.Subset(train_dataset, train_sample_indices)
        
        train_sample_loader = DataLoader(
            train_sample_dataset,
            batch_size=effective_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_career_path_batch,
            pin_memory=use_pin_memory,
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=2 if args.num_workers > 0 else None,
        )

    logger.info(
        f"  ✓ Created dataloaders (val: {len(val_dataset)}, "
        f"val_sample: {len(val_sample_dataset)}, test: {len(test_dataset)}"
        + (f", test_clean: {len(test_clean_dataset)}" if test_clean_dataset else "") + ")\n"
    )

    # --- Step 7: Hyperparameter selection (Optuna or static) ---
    logger.info("[7/8] Hyperparameter selection (Optuna or static)...")

    skip_final_training = False
    if args.optuna:
        logger.info(
            f"  > Running {args.n_trials} Optuna trials with max {args.max_epochs} epochs each"
        )
        logger.info(
            f"  > Validation sample ratio: {args.val_sample_ratio:.2f} "
            f"({len(val_sample_dataset)} samples)"
        )

        scaler = None
        if args.mixed_precision:
            from torch.cuda.amp import GradScaler

            scaler = GradScaler()
            logger.info("🔥 Mixed precision training enabled for Optuna.")

        import optuna

        study = optuna.create_study(
            direction="maximize",
            study_name=args.study_name,
            pruner=optuna.pruners.MedianPruner(),
        )

        start_time = time.time()

        study.optimize(
            lambda trial: objective(
                trial,
                train_sample_loader,
                val_sample_loader,
                Y_target_all,
                args,
                text_dim,
                skill_text_dim,
                structured_dim,
                output_dim,
                scaler,
            ),
            n_trials=args.n_trials,
            show_progress_bar=True,
            callbacks=[logger_callback],
        )

        elapsed = time.time() - start_time
        logger.info(f"\n  ✓ Optimization complete in {elapsed/60:.2f} minutes")
        logger.info(f"  ✓ Best trial: {study.best_trial.number}")
        logger.info(f"  ✓ Best validation MRR: {study.best_value:.4f}")
        final_epochs = study.best_trial.user_attrs["best_epoch"] + 1
        logger.info(f"  ✓ Optimal training epochs: {final_epochs}")
        for key, value in study.best_params.items():
            logger.info(f"      {key}: {value}")

        hidden_dim = study.best_params["hidden_dim"]
        n_layers = study.best_params["n_layers"]
        dropout = study.best_params["dropout"]
        lr = study.best_params["lr"]
        weight_decay = study.best_params.get("weight_decay", 0.0)
        use_modality_weights = study.best_params.get("use_modality_weights", False)

        # Update batch_size if it was searched
        if "batch_size" in study.best_params:
            args.batch_size = study.best_params["batch_size"]
            logger.info(f"  ✓ Optimal batch_size: {args.batch_size}")

        if args.optimizer == "sgd":
            momentum = study.best_params.get("momentum", 0.9)
            nesterov = study.best_params.get("nesterov", False)
        else:
            momentum = None
            nesterov = None

    else:
        logger.info(
            "[7/8] Using static hyperparameters... Training once with best model selection."
        )
        skip_final_training = True

        static_scaler = None
        if args.mixed_precision:
            from torch.cuda.amp import GradScaler

            static_scaler = GradScaler()
            logger.info("🔥 Mixed precision training enabled for static run.")

        hidden_dim = args.hidden_dim
        n_layers = args.n_layers
        dropout = args.dropout
        lr = args.lr
        use_modality_weights = args.use_modality_weights
        weight_decay = args.weight_decay

        if args.optimizer == "sgd":
            momentum = args.momentum
            nesterov = args.nesterov
        else:
            momentum = None
            nesterov = None

        # Build final model directly
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
                use_struct=args.use_structured,
            ).to(device)
        else:
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
                use_struct=args.use_structured,
            ).to(device)

        if args.optimizer == "sgd":
            optimizer = optim.SGD(
                final_model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=nesterov,
            )
        else:
            optimizer = optim.Adam(
                final_model.parameters(), lr=lr, weight_decay=weight_decay
            )

        criterion = nn.CosineEmbeddingLoss()
        
        # Early stop metric selection (synced from v2)
        if args.early_stop_metric == "loss":
            best_val_score = float("inf")
        else:
            best_val_score = 0.0
        epochs_no_improve = 0
        best_epoch = 0
        import copy
        best_model_state = None

        for epoch in trange(args.max_epochs, desc="Epochs", unit="epoch"):
            epoch_start_time = time.time()
            train_loss = train_epoch(
                final_model,
                train_loader,
                optimizer,
                criterion,
                device,
                static_scaler,
                args.mixed_precision,
                args.gradient_accumulation_steps,
                args.profile_data_loading,
                normalize_input=args.normalize_input,
            )
            val_metrics = evaluate(
                final_model, val_loader, Y_target_all, device, criterion
            )
            val_mrr = val_metrics["MRR"]
            val_loss = val_metrics.get("loss", None)
            epoch_time = time.time() - epoch_start_time
            logger.info(
                f"  Static HP Run | Epoch {epoch+1}/{args.max_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {(val_loss if val_loss is not None else float('nan')):.4f} | "
                f"Val MRR: {val_mrr:.4f} | Time: {epoch_time:.1f}s"
            )

            # Early stopping logic (synced from v2)
            if args.early_stop_metric == "loss":
                if val_loss is None:
                    raise ValueError("early_stop_metric='loss' requires val loss; got None.")
                improved = val_loss < (best_val_score - 1e-12)
            else:
                improved = val_mrr > (best_val_score + args.min_delta)

            if improved:
                best_val_score = val_loss if args.early_stop_metric == "loss" else val_mrr
                epochs_no_improve = 0
                best_epoch = epoch + 1
                best_model_state = copy.deepcopy(final_model.state_dict())
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= args.patience:
                logger.info(
                    f"  Early stopping triggered after {epoch+1} epochs "
                    f"(Static training patience: {args.patience}, metric: {args.early_stop_metric})."
                )
                break

        if best_model_state is not None:
            final_model.load_state_dict(best_model_state)
            if args.early_stop_metric == "loss":
                logger.info(f"  ✓ Loaded best model from epoch {best_epoch} (Val Loss: {best_val_score:.4f})")
            else:
                logger.info(f"  ✓ Loaded best model from epoch {best_epoch} (Val MRR: {best_val_score:.4f})")

        final_epochs = best_epoch if best_epoch > 0 else 1
        logger.info(f"  ✓ Optimal training epochs found: {final_epochs}\n")

    if not skip_final_training:
        # --- Step 8: Final training on train+val and test evaluation ---
        logger.info("[8/8] Training final model on train+val with best hyperparameters...")
        logger.info("  > Using full train+val set for final training (no sampling)")

        combined_pairs = train_pairs + val_pairs

        def _concat_optional(a, b):
            if a is None and b is None:
                return None
            if a is None:
                return b
            if b is None:
                return a
            return np.concatenate([a, b], axis=0)

        combined_h_text = _concat_optional(train_h_text, val_h_text)
        combined_h_skill = _concat_optional(train_h_skill_last, val_h_skill_last)

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
            combined_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=collate_career_path_batch,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=2 if args.num_workers > 0 else None,
        )

        # Build final model
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
                use_struct=args.use_structured,
            ).to(device)
        else:
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
                use_struct=args.use_structured,
            ).to(device)

        if args.optimizer == "sgd":
            optimizer = optim.SGD(
                final_model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=nesterov,
            )
        else:
            optimizer = optim.Adam(
                final_model.parameters(), lr=lr, weight_decay=weight_decay
            )

        criterion = nn.CosineEmbeddingLoss()

        final_scaler = None
        if args.mixed_precision:
            from torch.cuda.amp import GradScaler

            final_scaler = GradScaler()
            logger.info("🔥 Mixed precision training enabled for final training.")

        logger.info(f"  > Training final model for {final_epochs} epochs...")
        logger.info(f"  > Model architecture: {final_model}")

        final_training_start = time.time()
        for epoch in tqdm(range(final_epochs), desc="Final training"):
            epoch_start_time = time.time()
            train_epoch(
                final_model,
                combined_loader,
                optimizer,
                criterion,
                device,
                final_scaler,
                args.mixed_precision,
                args.gradient_accumulation_steps,
                args.profile_data_loading,
                normalize_input=args.normalize_input,
            )
            epoch_time = time.time() - epoch_start_time
            if epoch < 2 or epoch % 5 == 0:
                logger.info(
                    f"    Final training epoch {epoch+1}/{final_epochs} "
                    f"completed in {epoch_time:.1f}s"
                )

        final_training_time = time.time() - final_training_start
        logger.info(
            f"  > Final training completed in {final_training_time:.1f}s "
            f"({final_training_time / final_epochs:.1f}s per epoch)"
        )

    # Evaluate on test set
    logger.info("\n  > Evaluating on test set...")
    test_metrics = evaluate(final_model, test_loader, Y_target_all, device)

    logger.info("\n" + "=" * 80)
    logger.info("FINAL TEST SET RESULTS (v3)")
    logger.info("=" * 80)
    for metric, value in test_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    logger.info("=" * 80)

    # Evaluate on clean test set (if enabled)
    test_clean_metrics = {}
    if args.eval_clean_test and test_clean_loader is not None:
        logger.info("\n  > Evaluating on clean test set...")
        test_clean_metrics = evaluate(final_model, test_clean_loader, Y_target_all, device)
        
        logger.info("\n" + "=" * 80)
        logger.info("FINAL CLEAN TEST SET RESULTS (v3)")
        logger.info("=" * 80)
        for metric, value in test_clean_metrics.items():
            logger.info(f"{metric}: {value:.4f}")
        logger.info("=" * 80)

    # Save prediction scores for fusion (if requested)
    if args.save_scores:
        scores_dir = args.scores_output_dir if args.scores_output_dir else os.path.join(args.output_dir, "scores")
        os.makedirs(scores_dir, exist_ok=True)
        
        # Build target labels list in same order as Y_target_all
        # IMPORTANT: Use Y_target_dict keys to ensure order matches Y_target_all (indices)
        target_labels = list(Y_target_dict.keys())
        
        logger.info("\n[9/8] Saving prediction scores for fusion...")
        
        # Need to recreate train and val loaders for score saving
        # (they were deleted earlier for memory optimization)
        # Note: In static mode (skip_final_training=True), combined_h_text/combined_h_skill 
        # are not defined, so we use the original train/val embeddings directly.
        train_pre_h_text = train_h_text
        train_pre_h_skill = train_h_skill_last
        val_pre_h_text = val_h_text
        val_pre_h_skill = val_h_skill_last
        
        temp_train_dataset = CareerPathDataset(
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
            pre_h_text=train_pre_h_text,
            pre_h_skill_text=train_pre_h_skill,
        )
        temp_train_loader = DataLoader(
            temp_train_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=collate_career_path_batch,
            pin_memory=(device.type == "cuda"),
        )
        
        temp_val_dataset = CareerPathDataset(
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
            pre_h_text=val_pre_h_text,
            pre_h_skill_text=val_pre_h_skill,
        )
        temp_val_loader = DataLoader(
            temp_val_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=collate_career_path_batch,
            pin_memory=(device.type == "cuda"),
        )
        
        # Save test scores
        compute_and_save_scores(
            final_model, test_loader, Y_target_all, target_labels, device,
            output_path=os.path.join(scores_dir, "test_scores_text.pkl"),
            data_pairs=test_pairs, job_ids_list=test_job_ids, split_name="test"
        )
        
        # Save train scores
        compute_and_save_scores(
            final_model, temp_train_loader, Y_target_all, target_labels, device,
            output_path=os.path.join(scores_dir, "train_scores_text.pkl"),
            data_pairs=train_pairs, job_ids_list=train_job_ids, split_name="train"
        )
        
        # Save val scores
        compute_and_save_scores(
            final_model, temp_val_loader, Y_target_all, target_labels, device,
            output_path=os.path.join(scores_dir, "val_scores_text.pkl"),
            data_pairs=val_pairs, job_ids_list=val_job_ids, split_name="val"
        )
        
        # Save clean test scores (if enabled)
        if args.eval_clean_test and test_clean_loader is not None:
            compute_and_save_scores(
                final_model, test_clean_loader, Y_target_all, target_labels, device,
                output_path=os.path.join(scores_dir, "test_clean_scores_text.pkl"),
                data_pairs=test_clean_pairs, job_ids_list=test_clean_job_ids, split_name="test_clean"
            )
        
        
        # Cleanup temporary loaders
        del temp_train_dataset, temp_train_loader, temp_val_dataset, temp_val_loader
        gc.collect()

    if WANDB_AVAILABLE and args.use_wandb:
        # Prefix test metrics with 'test_'
        wandb_test_metrics = {f"test_{k.lower()}": v for k, v in test_metrics.items()}
        wandb.log(wandb_test_metrics)
        
        if args.eval_clean_test and test_clean_metrics:
            wandb_test_clean_metrics = {f"test_clean_{k.lower()}": v for k, v in test_clean_metrics.items()}
            wandb.log(wandb_test_clean_metrics)
            
        wandb.finish()

    # Save model and results
    if args.save_model:
        logger.info(f"  > Saving model to {args.output_dir}/final_model_v3.pt")
        checkpoint = {
            "model_state_dict": final_model.state_dict(),
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "dropout": dropout,
            "lr": lr,
            "optimizer": args.optimizer,
            "weight_decay": weight_decay,
            "use_modality_weights": use_modality_weights,
            "test_metrics": test_metrics,
            "test_clean_metrics": test_clean_metrics,
            "args": vars(args),
        }
        if args.optimizer == "sgd":
            checkpoint["momentum"] = momentum
            checkpoint["nesterov"] = nesterov

        save_path = os.path.join(args.output_dir, "final_model_v3.pt")
        torch.save(checkpoint, save_path)
        logger.info(f"\nModel saved to: {save_path}")

    # Save study (Optuna) if requested
    if args.optuna and args.save_study:
        study_path = os.path.join(args.output_dir, "optuna_study_v3.pkl")
        import pickle

        with open(study_path, "wb") as f:
            pickle.dump(study, f)
        logger.info(f"Study saved to: {study_path}")

    # Save summary results to CSV
    if args.optimizer == "sgd":
        optimizer_details = (
            f"SGD(lr={lr:.6f}, momentum={momentum:.4f}, "
            f"weight_decay={weight_decay:.6f}, nesterov={nesterov})"
        )
    else:
        optimizer_details = f"Adam(lr={lr:.6f}, weight_decay={weight_decay:.6f})"

    results_data = {
        "timestamp": pd.to_datetime("now").strftime("%Y-%m-%d %H:%M:%S"),
        "run_name": args.run_name,
        "architecture": "MultiModal" if args.use_advanced else "SimpleConcat",
        "text_history": args.use_text_history,
        "skill_text": args.use_skill_text,
        "structured": args.use_structured,
        "text_encoder": args.encoder_text,
        "skill_encoder": args.encoder_skill if args.encoder_skill else args.encoder_text,
        "pooling_strategy": args.pooling_strategy,
        "last_job_only": args.last_job_only,
        "top_k_skills": args.top_k_skills,
        "final_epochs": final_epochs,
        "lr": lr,
        "hidden_dim": hidden_dim,
        "n_layers": n_layers,
        "dropout": dropout,
        "use_modality_weights": use_modality_weights
        if args.use_advanced
        else "N/A",
        **{f"test_{k}": v for k, v in test_metrics.items()},
        **{f"test_clean_{k}": v for k, v in test_clean_metrics.items()},
        "optimizer": args.optimizer,
        "optimizer_details": optimizer_details,
    }

    results_df = pd.DataFrame([results_data])
    try:
        if os.path.exists(args.results_csv_path):
            results_df.to_csv(
                args.results_csv_path, mode="a", header=False, index=False
            )
        else:
            results_df.to_csv(
                args.results_csv_path, mode="w", header=True, index=False
            )
        logger.info(f"📈 Results appended to: {args.results_csv_path}")
    except Exception as e:
        logger.error(f"Error saving results to CSV: {e}")

    # --- Save results to JSON (for easier parsing) ---
    try:
        # Include configurations from results_data for reproducibility
        metrics_to_save = copy.deepcopy(results_data)
        metrics_to_save.update(
            {
                "test_metrics": test_metrics,
                "test_clean_metrics": test_clean_metrics if args.eval_clean_test else None,
            }
        )

        json_path = os.path.join(args.output_dir, "test_metrics.json")
        with open(json_path, "w") as f:
            json.dump(metrics_to_save, f, indent=4)
        logger.info(f"📊 Test metrics and configurations saved to: {json_path}")
    except Exception as e:
        logger.error(f"Error saving results to JSON: {e}")


if __name__ == "__main__":
    main()


