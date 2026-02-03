"""
category_inference.py - Inference script for Category Predictor

Predicts skill category scores for jobs using a trained CategoryClassifier.
Can use precomputed embeddings or raw text.

Usage:
    python -m skill_mapping.v2.category_inference \
        --jobs_csv data/processed/job_titles.csv \
        --model_dir outputs/category_model \
        --output_path outputs/category_scores.json \
        --embeddings_path /path/to/embeddings/dir_or_file.npy
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from tqdm import tqdm

from .category_trainer import (
    CategoryClassifier, 
    LabelEncoder, 
    EmbeddingDataset, 
    SoftLabelLoss, 
    evaluate, 
    load_esco_data, 
    build_validation_samples, 
    calculate_pos_weights,
    extract_hidden_dims_from_params
)

def load_config(model_dir: Path) -> Dict:
    """Load training configuration."""
    results_path = model_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Config not found at {results_path}")
    with open(results_path, "r") as f:
        return json.load(f)

def load_model(
    model_dir: Path,
    input_dim: int,
    device: torch.device
) -> tuple[CategoryClassifier, LabelEncoder, Dict]:
    """Load trained CategoryClassifier and LabelEncoder."""
    config = load_config(model_dir)
    
    # Load LabelEncoder
    le_path = model_dir / "label_encoder.json"
    if not le_path.exists():
        raise FileNotFoundError(f"LabelEncoder not found at {le_path}")
    label_encoder = LabelEncoder.load(str(le_path))
    
    num_classes = len(label_encoder)
    best_params = config.get("best_params", {})
    
    hidden_dims = extract_hidden_dims_from_params(best_params)
    use_batchnorm = best_params.get("use_batchnorm", True)
        
    # Initialize model
    model = CategoryClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=0.0, # Dropout not used in inference
        use_batchnorm=use_batchnorm
    )
    
    # Load weights
    weights_path = model_dir / "category_classifier.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at {weights_path}")
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, label_encoder, config

def load_embeddings(
    embeddings_path: Path, 
    job_ids: List[str]
) -> np.ndarray:
    """
    Load embeddings from file or directory.
    If directory, expects job_embeddings.npy and optionally metadata.
    """
    if embeddings_path.is_dir():
        emb_file = embeddings_path / "job_embeddings.npy"
        meta_file = embeddings_path / "job_embeddings_meta.json"
        
        if not emb_file.exists():
             raise FileNotFoundError(f"Embeddings file not found in {embeddings_path}")
        
        logger.info(f"Loading embeddings from {emb_file}")
        embeddings = np.load(emb_file)
        
        # Verify alignment if metadata exists
        if meta_file.exists():
            with open(meta_file, "r") as f:
                meta = json.load(f)
            # Basic check
            if meta.get("num_jobs") != len(job_ids):
                logger.warning(
                    f"Embedding count ({meta.get('num_jobs')}) matches "
                    f"metadata but differs from CSV job count ({len(job_ids)}). "
                    "Ensure the embeddings correspond to the input CSV and are in the same order."
                )
    else:
        logger.info(f"Loading embeddings from {embeddings_path}")
        embeddings = np.load(embeddings_path)
        
    if len(embeddings) != len(job_ids):
         logger.warning(
            f"Number of embeddings ({len(embeddings)}) does not match "
            f"number of jobs ({len(job_ids)}). Alignment may be incorrect if order differs."
        )
         
    return embeddings.astype(np.float32)

def encode_texts(
    texts: List[str],
    model_name: str,
    device: str,
    batch_size: int = 64
) -> np.ndarray:
    """Encode texts using SentenceTransformer."""
    logger.info(f"Loading encoder: {model_name}")
    model = SentenceTransformer(model_name, device=device)
    
    logger.info(f"Encoding {len(texts)} texts...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    # L2 normalize
    embeddings = normalize(embeddings, norm="l2", axis=1).astype(np.float32)
    return embeddings

def predict_scores(
    model: CategoryClassifier,
    embeddings: np.ndarray,
    label_encoder: LabelEncoder,
    device: torch.device,
    batch_size: int = 256,
    soft_labels: bool = False
) -> List[Dict[str, Dict[str, float]]]:
    """
    Predict category scores and logits for all embeddings.
    Returns list of dicts: [{"Category A": {"score": 0.9, "logit": 1.2}, ...}, ...]
    """
    model.eval()
    
    # Create dataset/loader for batching
    dataset = TensorDataset(torch.from_numpy(embeddings))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_results = []
    
    logger.info("Predicting category scores and logits...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            emb = batch[0].to(device)
            logits = model(emb)
            
            if soft_labels:
                # If trained with soft labels (KLDiv), output is log-probs or logits
                # We want probabilities summing to 1
                probs = torch.softmax(logits, dim=-1)
            else:
                # If trained with BCE, independent probabilities
                probs = torch.sigmoid(logits)
                
            probs_np = probs.cpu().numpy()
            logits_np = logits.cpu().numpy()
            
            # Map to category names
            for row_probs, row_logits in zip(probs_np, logits_np):
                row_dict = {}
                for i, (score, logit) in enumerate(zip(row_probs, row_logits)):
                    cat_name = label_encoder.idx2str[i]
                    row_dict[cat_name] = {
                        "score": float(score),
                        "logit": float(logit)
                    }
                all_results.append(row_dict)
                
    return all_results

def build_validation_samples_from_esco_id(
    val_df: pd.DataFrame,
    esco_df: pd.DataFrame,
    cat_col: str,
    soft_labels: bool = False,
    esco_id_col: str = "esco_id"
) -> List[Dict]:
    """
    Build validation samples by mapping esco_id (in jobs_csv) -> occupationUri (in ESCO) -> Categories.
    """
    # 1. Map ESCO occupationUri -> Categories (from ESCO dataset)
    # Group ESCO data by occupationUri
    occ_to_cats = {}
    occ_to_cat_counts = {}
    
    for occ_uri, group in esco_df.groupby("occupationUri"):
        cats = group[cat_col].dropna().astype(str).unique().tolist()
        if cats:
            occ_to_cats[occ_uri] = cats
            occ_to_cat_counts[occ_uri] = group[cat_col].value_counts().to_dict()
            
    # 2. Build samples from Job Data
    samples = []
    
    # Check if esco_id column exists
    if esco_id_col not in val_df.columns:
        logger.warning(f"Column '{esco_id_col}' not found in jobs CSV. Cannot map to ground truth.")
        return []
        
    # Group by job_id to handle duplicates (though usually one job -> one esco_id)
    for job_id, group in val_df.groupby("job_id"):
        # Get esco_id (occupationUri)
        first_row = group.iloc[0]
        occ_uri = str(first_row.get(esco_id_col, "")).strip()
        
        if not occ_uri or occ_uri == "nan":
            continue
            
        if occ_uri not in occ_to_cats:
            # logger.debug(f"ESCO ID '{occ_uri}' not found in taxonomy.")
            continue
            
        text = str(first_row.get("processed_text", "")).strip() # Or other text col
        if not text:
             text = str(first_row.get("text", "")).strip()
        
        sample = {
            "job_id": job_id,
            "text": text,
            "categories": occ_to_cats[occ_uri],
        }
        
        if soft_labels:
            sample["category_counts"] = occ_to_cat_counts[occ_uri]
            
        samples.append(sample)
        
    logger.info(f"Built {len(samples)} ground truth samples from ESCO IDs")
    return samples

def main():
    parser = argparse.ArgumentParser(description="Inference for Category Classifier")
    
    # Input Data
    parser.add_argument("--jobs_csv", type=str, required=True, help="Path to CSV with job texts")
    parser.add_argument("--text_col", type=str, default="processed_text", help="Column name for job text")
    parser.add_argument("--id_col", type=str, default=None, help="Column name for job ID (defaults to row index)")
    parser.add_argument("--esco_id_col", type=str, default="esco_id", help="Column name for ESCO occupation URI in jobs CSV (for metrics)")
    
    # Model
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing model artifacts (results.json, .pt, .json)")
    parser.add_argument("--embeddings_path", type=str, default=None, help="Path to precomputed embeddings (.npy or dir)")
    parser.add_argument("--encoder_name", type=str, default=None, help="Encoder name if embeddings not provided (defaults to config)")
    
    # Output
    parser.add_argument("--output_path", type=str, required=True, help="Folder name for outputs (relative to model_dir)")
    
    # Metrics
    parser.add_argument("--compute_metrics", action="store_true", help="Compute evaluation metrics (requires ground truth in CSV)")
    parser.add_argument("--esco_path", type=str, default=None, help="Path to ESCO master CSV (required if computing metrics)")
    
    # Config
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # 1. Load Jobs
    logger.info(f"Loading jobs from {args.jobs_csv}")
    df = pd.read_csv(args.jobs_csv)
    
    if args.text_col not in df.columns:
        # Fallback if text_col missing but 'text' exists?
        if "text" in df.columns:
             logger.warning(f"Column '{args.text_col}' not found, using 'text'")
             args.text_col = "text"
        else:
             raise ValueError(f"Column '{args.text_col}' not found in CSV")
             
    texts = df[args.text_col].astype(str).tolist()
    
    # Implicitly use job_id if present and no explicit ID column
    id_col_to_use = args.id_col
    if not id_col_to_use and "job_id" in df.columns:
        logger.info("Found 'job_id' column, using it as ID.")
        id_col_to_use = "job_id"
        
    if id_col_to_use and id_col_to_use in df.columns:
        job_ids = df[id_col_to_use].astype(str).tolist()
    else:
        job_ids = [str(i) for i in range(len(df))]
        
    # 2. Get Embeddings
    if args.embeddings_path:
        embeddings = load_embeddings(Path(args.embeddings_path), job_ids)
    else:
        # Need to encode
        # Check config for model name if not provided
        config = load_config(Path(args.model_dir))
        encoder_name = args.encoder_name or config.get("config", {}).get("model_path") or "pj-mathematician/JobSkillBGE-large-en-v1.5"
        embeddings = encode_texts(texts, encoder_name, args.device, args.batch_size)
        
    input_dim = embeddings.shape[1]
    
    # 3. Load Classifier
    model, label_encoder, config = load_model(Path(args.model_dir), input_dim, device)
    
    # Check if soft labels were used
    soft_labels = config.get("config", {}).get("soft_labels", False)
    if soft_labels:
        logger.info("Model trained with soft labels (LDL), using softmax for probabilities.")
    else:
        logger.info("Model trained with hard labels, using sigmoid for probabilities.")
        
    # 4. Predict
    category_scores = predict_scores(
        model, 
        embeddings, 
        label_encoder, 
        device, 
        args.batch_size, 
        soft_labels
    )
    
    # Resolve Output Directory
    # If output_path is absolute, use it. If not, treat as subdir of model_dir.
    out_path_arg = Path(args.output_path)
    if out_path_arg.is_absolute():
        output_dir = out_path_arg
    else:
        output_dir = Path(args.model_dir) / out_path_arg
        
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Outputs will be saved to: {output_dir}")
    
    # 5. Compute Metrics (Optional)
    metrics = {}
    if args.compute_metrics:
        if not args.esco_path:
            logger.error("--esco_path is required when --compute_metrics is set")
        else:
            logger.info("Computing evaluation metrics...")
            
            # Load ESCO
            target_level = config.get("config", {}).get("target_level", 1)
            esco_df, cat_col = load_esco_data(args.esco_path, target_level)
            
            # Prepare validation dataframe (rename id col to 'job_id' for build_validation_samples)
            val_df = df.copy()
            
            # If we identified a specific ID column, rename it to 'job_id' unless it's already named that
            if id_col_to_use and id_col_to_use in val_df.columns:
                 if id_col_to_use != "job_id":
                     val_df = val_df.rename(columns={id_col_to_use: "job_id"})
            elif "job_id" not in val_df.columns:
                 # Create job_id from index if not present
                 val_df["job_id"] = [str(i) for i in range(len(val_df))]
            
            # Build Ground Truth Samples (aggregates by job_id)
            # Strategy:
            # 1. If 'skillUri' exists in val_df, use build_validation_samples (direct skill mapping).
            # 2. Else if 'esco_id' exists, map esco_id -> occupationUri -> categories.
            
            if "skillUri" in val_df.columns:
                logger.info("Found 'skillUri' column, building ground truth from direct skill mappings...")
                gt_samples = build_validation_samples(
                    val_df, 
                    esco_df, 
                    cat_col, 
                    soft_labels=soft_labels
                )
            elif args.esco_id_col in val_df.columns:
                logger.info(f"Found '{args.esco_id_col}' column, mapping to ESCO occupation categories...")
                gt_samples = build_validation_samples_from_esco_id(
                    val_df,
                    esco_df,
                    cat_col,
                    soft_labels=soft_labels,
                    esco_id_col=args.esco_id_col
                )
            else:
                logger.warning(f"Neither 'skillUri' nor '{args.esco_id_col}' found. Cannot build ground truth.")
                gt_samples = []
            
            if not gt_samples:
                logger.warning("No valid ground truth samples found (check skillUri column)")
            else:
                # Map job_id -> sample
                gt_map = {str(s["job_id"]): s for s in gt_samples}
                
                # Align embeddings with ground truth
                # We need one embedding per unique job in ground truth
                # Since input 'embeddings' corresponds to input rows (which might be duplicated),
                # we need to pick representative embedding for each unique job.
                
                # Create map: job_id -> embedding_index
                # If duplicates exist, this picks the last one (or first if we iterate reverse)
                # Assuming embeddings for same job_id are identical.
                job_to_idx = {str(jid): i for i, jid in enumerate(job_ids)}
                
                eval_embeddings_list = []
                eval_targets_list = []
                
                found_count = 0
                for job_id, sample in gt_map.items():
                    if job_id in job_to_idx:
                        idx = job_to_idx[job_id]
                        
                        # Get embedding
                        eval_embeddings_list.append(embeddings[idx])
                        
                        # Get target
                        if soft_labels:
                            t = label_encoder.encode_soft(sample["category_counts"])
                        else:
                            t = label_encoder.encode_multi(sample["categories"])
                        eval_targets_list.append(t)
                        found_count += 1
                
                logger.info(f"Matched {found_count} jobs with ground truth out of {len(gt_samples)} available.")
                
                if found_count > 0:
                    eval_embeddings = torch.from_numpy(np.stack(eval_embeddings_list))
                    eval_targets = torch.stack(eval_targets_list)
                    
                    # Create dataset/loader
                    eval_dataset = EmbeddingDataset(eval_embeddings.numpy(), eval_targets)
                    eval_loader = torch.utils.data.DataLoader(
                        eval_dataset, 
                        batch_size=args.batch_size, 
                        shuffle=False
                    )
                    
                    # Loss function
                    if soft_labels:
                        criterion = SoftLabelLoss()
                    else:
                        # For evaluation loss, we need pos_weights? 
                        # evaluate() calculates loss too.
                        # We can calculate weights based on these eval targets or just use 1.0
                        # Usually we use train weights but we don't have them here.
                        # Let's calculate weights from this set to be safe or just use default BCE
                        # Trainer uses calculate_pos_weights(train_targets).
                        # We'll calculate on eval_targets to avoid error, though it's not strictly 'correct' loss for test
                        # But metrics (F1 etc) don't depend on loss.
                        pos_weight = calculate_pos_weights(eval_targets, device)
                        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                    
                    metrics = evaluate(
                        model, 
                        eval_loader, 
                        criterion, 
                        device, 
                        soft_labels=soft_labels
                    )
                    
                    logger.info("Evaluation Metrics:")
                    for k, v in metrics.items():
                        logger.info(f"  {k}: {v:.4f}")
                        
                    # Save metrics
                    metrics_path = output_dir / "metrics.json"
                    with open(metrics_path, "w") as f:
                        json.dump(metrics, f, indent=2)
                    logger.info(f"Saved metrics to {metrics_path}")

    # 6. Save Results
    # Format: { "job_id": [ {"category": "A", "score": 0.9, "logit": 1.2}, ... ] }
    results = {}
    for job_id, cat_results in zip(job_ids, category_scores):
        # Sort by score descending
        sorted_cats = sorted(cat_results.items(), key=lambda x: x[1]["score"], reverse=True)
        results[job_id] = [
            {
                "category": cat, 
                "score": round(res["score"], 6),
                "logit": round(res["logit"], 6)
            }
            for cat, res in sorted_cats
        ]
        
    scores_path = output_dir / "category_scores.json"
    
    with open(scores_path, "w") as f:
        json.dump(results, f, indent=2)
        
    logger.success(f"Saved category scores for {len(results)} jobs to {scores_path}")

if __name__ == "__main__":
    main()

