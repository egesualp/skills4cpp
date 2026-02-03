"""
isco_inference.py - Inference script for ISCO group predictor

Predicts ISCO group probabilities (single-label classification) for jobs/occupations
using a trained ISCOClassifier from `isco_trainer.py`.

Supports:
  - Raw text -> embeddings via SentenceTransformer (encoder inferred from results.json unless overridden)
  - Precomputed embeddings (.npy file or directory with job_embeddings.npy + optional metadata)
  - Optional metrics if ground truth ISCO codes are present in the input CSV

Usage:
    python -m skill_mapping.v2.isco_inference \
        --jobs_csv data/jobs.csv \
        --model_dir outputs/isco_model \
        --output_path outputs \
        --text_col processed_text \
        --compute_metrics \
        --isco_col iscoGroup \
        --isco_level 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .isco_trainer import (
    ISCOClassifier,
    SingleLabelEncoder,
    EmbeddingDataset,
    truncate_isco,
    evaluate,
    extract_hidden_dims_from_params,
    format_text,
)


def load_config(model_dir: Path) -> Dict:
    """Load training configuration (results.json)."""
    results_path = model_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Config not found at {results_path}")
    with open(results_path, "r") as f:
        return json.load(f)


def load_model(
    model_dir: Path,
    input_dim: int,
    device: torch.device,
) -> Tuple[ISCOClassifier, SingleLabelEncoder, Dict]:
    """Load trained ISCOClassifier and SingleLabelEncoder."""
    config = load_config(model_dir)

    le_path = model_dir / "label_encoder.json"
    if not le_path.exists():
        raise FileNotFoundError(f"Label encoder not found at {le_path}")
    label_encoder = SingleLabelEncoder.load(str(le_path))

    num_classes = len(label_encoder)
    best_params = config.get("best_params", {})
    hidden_dims = extract_hidden_dims_from_params(best_params)
    use_batchnorm = bool(best_params.get("use_batchnorm", True))

    model = ISCOClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=0.0,  # dropout disabled for inference
        use_batchnorm=use_batchnorm,
    )

    weights_path = model_dir / "isco_classifier.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at {weights_path}")

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    return model, label_encoder, config


def load_embeddings(embeddings_path: Path, job_ids: List[str]) -> np.ndarray:
    """
    Load embeddings from file or directory.
    If directory, expects job_embeddings.npy and optionally metadata job_embeddings_meta.json.
    """
    if embeddings_path.is_dir():
        emb_file = embeddings_path / "job_embeddings.npy"
        meta_file = embeddings_path / "job_embeddings_meta.json"

        if not emb_file.exists():
            raise FileNotFoundError(f"Embeddings file not found in {embeddings_path}")

        logger.info(f"Loading embeddings from {emb_file}")
        embeddings = np.load(emb_file)

        if meta_file.exists():
            try:
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                if meta.get("num_jobs") != len(job_ids):
                    logger.warning(
                        f"Embeddings metadata num_jobs={meta.get('num_jobs')} differs from CSV jobs={len(job_ids)}. "
                        "Ensure embeddings correspond to the input CSV and are in the same order."
                    )
            except Exception as e:
                logger.warning(f"Failed to parse {meta_file}: {e}")
    else:
        logger.info(f"Loading embeddings from {embeddings_path}")
        embeddings = np.load(embeddings_path)

    if len(embeddings) != len(job_ids):
        logger.warning(
            f"Number of embeddings ({len(embeddings)}) does not match number of jobs ({len(job_ids)}). "
            "If order differs, alignment may be incorrect."
        )

    return embeddings.astype(np.float32)


def encode_texts(
    texts: List[str],
    model_name: str,
    device: str,
    batch_size: int = 64,
    num_workers: int = 0,
) -> np.ndarray:
    """Encode texts using SentenceTransformer and L2-normalize.
    
    Args:
        texts: List of texts to encode.
        model_name: SentenceTransformer model name or path.
        device: Device to use ('cpu' or 'cuda').
        batch_size: Batch size for encoding.
        num_workers: Number of CPU workers for parallel encoding. 
                     If > 0 and device is 'cpu', uses multi-process encoding.
    """
    logger.info(f"Loading encoder: {model_name}")
    model = SentenceTransformer(model_name, device=device)

    logger.info(f"Encoding {len(texts)} texts...")
    
    # Use multi-process pool for CPU parallelization
    if num_workers > 0 and device == "cpu":
        logger.info(f"Using multi-process encoding with {num_workers} CPU workers")
        pool = model.start_multi_process_pool(target_devices=["cpu"] * num_workers)
        try:
            # Use encode() with pool parameter (encode_multi_process is deprecated)
            embeddings = model.encode(
                texts,
                pool=pool,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
        finally:
            model.stop_multi_process_pool(pool)
    else:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )
    
    return normalize(embeddings, norm="l2", axis=1).astype(np.float32)


@torch.no_grad()
def predict_topk(
    model: ISCOClassifier,
    embeddings: np.ndarray,
    label_encoder: SingleLabelEncoder,
    device: torch.device,
    *,
    batch_size: int = 256,
    topk: int = 10,
    num_workers: int = 0,
) -> List[Dict[str, object]]:
    """
    Predict per-row top-k labels and probabilities.

    Returns list of dicts:
      {
        "top1": {"label": "...", "score": 0.9, "logit": 1.2},
        "topk": [{"label": "...", "score": 0.9, "logit": 1.2}, ...]
      }
    """
    model.eval()
    dataset = TensorDataset(torch.from_numpy(embeddings))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    k = int(max(1, topk))
    all_results: List[Dict[str, object]] = []

    for batch in tqdm(loader, desc="Inference"):
        emb = batch[0].to(device)
        logits = model(emb)  # [B, C]
        probs = torch.softmax(logits, dim=-1)

        k_eff = min(k, logits.shape[-1])
        topk_logits, topk_idx = torch.topk(logits, k=k_eff, dim=-1)
        topk_probs = torch.gather(probs, dim=-1, index=topk_idx)

        topk_idx_np = topk_idx.cpu().numpy()
        topk_logits_np = topk_logits.cpu().numpy()
        topk_probs_np = topk_probs.cpu().numpy()

        for row_idx, row_logits, row_probs in zip(topk_idx_np, topk_logits_np, topk_probs_np):
            items = []
            for cls_idx, logit, score in zip(row_idx.tolist(), row_logits.tolist(), row_probs.tolist()):
                label = label_encoder.idx2str[int(cls_idx)]
                items.append({"label": label, "score": round(float(score), 6), "logit": round(float(logit), 6)})

            all_results.append({"top1": items[0], "topk": items})

    return all_results


def _build_eval_subset(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    job_ids: List[str],
    *,
    isco_col: str,
    override_isco_codes: Optional[List[object]] = None,
    isco_level: int,
    label_encoder: SingleLabelEncoder,
) -> Tuple[np.ndarray, torch.Tensor, List[str], Dict[str, int]]:
    """
    Build evaluation subset: only rows with valid, in-vocab ISCO labels.
    Returns (eval_embeddings, eval_targets, eval_job_ids, stats).
    """
    codes = override_isco_codes
    if codes is None:
        if isco_col not in df.columns:
            return (
                np.zeros((0, embeddings.shape[1]), dtype=np.float32),
                torch.empty((0,), dtype=torch.long),
                [],
                {"missing_isco_col": 1},
            )
        codes = df[isco_col].tolist()

    kept_embeddings = []
    kept_targets = []
    kept_job_ids = []

    stats = {
        "rows_total": int(len(df)),
        "rows_missing_isco": 0,
        "rows_invalid_isco": 0,
        "rows_oov_label": 0,
        "rows_kept": 0,
    }

    for i, raw_code in enumerate(codes):
        if raw_code is None or str(raw_code).strip().lower() == "nan":
            stats["rows_missing_isco"] += 1
            continue

        # Normalize and truncate to model level
        truncated = truncate_isco(str(raw_code), isco_level)
        if not truncated:
            stats["rows_invalid_isco"] += 1
            continue

        if truncated not in label_encoder.str2idx:
            stats["rows_oov_label"] += 1
            continue

        kept_embeddings.append(embeddings[i])
        kept_targets.append(label_encoder.encode(truncated))
        kept_job_ids.append(job_ids[i])
        stats["rows_kept"] += 1

    if not kept_embeddings:
        return np.zeros((0, embeddings.shape[1]), dtype=np.float32), torch.empty((0,), dtype=torch.long), [], stats

    eval_embeddings = np.stack(kept_embeddings).astype(np.float32)
    eval_targets = torch.stack(kept_targets)
    return eval_embeddings, eval_targets, kept_job_ids, stats


def _map_gt_isco_from_esco_id(
    df: pd.DataFrame,
    *,
    esco_path: Path,
    esco_id_col: str,
    isco_col_in_esco: str,
) -> List[object]:
    """
    Map each row's ESCO occupation ID (URI) to an ISCO code using an ESCO master CSV.

    - `esco_id_col`: column in jobs CSV containing occupation URI (e.g., occupationUri / conceptUri)
    - `isco_col_in_esco`: column in ESCO CSV containing ISCO group (default 'iscoGroup')
    """
    if esco_id_col not in df.columns:
        raise ValueError(f"Column '{esco_id_col}' not found in jobs CSV; cannot map to ground truth ISCO.")

    logger.info(f"Loading ESCO for ISCO mapping from {esco_path}")
    esco_df = pd.read_csv(esco_path)
    if isco_col_in_esco not in esco_df.columns:
        raise ValueError(
            f"Column '{isco_col_in_esco}' not found in ESCO CSV. Available: {list(esco_df.columns)}"
        )

    # Robust: ESCO exports vary; try common columns for occupation URI.
    uri_col = None
    for c in ["occupationUri", "conceptUri", "esco_id"]:
        if c in esco_df.columns:
            uri_col = c
            break
    if uri_col is None:
        raise ValueError(
            f"Could not find occupation URI column in ESCO CSV. Expected one of "
            f"['occupationUri','conceptUri','esco_id']. Available: {list(esco_df.columns)}"
        )

    # Map occupation URI -> iscoGroup (first non-null)
    esco_df = esco_df[[uri_col, isco_col_in_esco]].copy()
    esco_df[uri_col] = esco_df[uri_col].astype(str)
    # Keep first non-null per URI
    occ_to_isco = (
        esco_df.dropna(subset=[isco_col_in_esco])
        .groupby(uri_col)[isco_col_in_esco]
        .first()
        .to_dict()
    )

    mapped = []
    missing = 0
    for raw_uri in df[esco_id_col].tolist():
        uri = str(raw_uri).strip()
        if not uri or uri.lower() == "nan":
            mapped.append(None)
            continue
        val = occ_to_isco.get(uri)
        if val is None or str(val).strip().lower() == "nan":
            missing += 1
            mapped.append(None)
        else:
            mapped.append(val)

    logger.info(
        f"Ground truth mapping via ESCO IDs: {len(mapped) - missing} rows mapped, {missing} rows unmapped "
        f"(of {len(mapped)} total)."
    )
    return mapped


def main():
    parser = argparse.ArgumentParser(description="Inference for ISCO group classifier")

    # Input data
    parser.add_argument("--jobs_csv", type=str, required=True, help="Path to CSV with job texts (and optionally ISCO codes)")
    parser.add_argument("--text_col", type=str, default="processed_text", help="Column name for job text (or title if formatting)")
    parser.add_argument("--desc_col", type=str, default=None, help="Column name for description (used if include_description is True)")
    parser.add_argument("--prefix", type=str, default=None, help="Prefix for text formatting (e.g. 'role'). Defaults to training config.")
    parser.add_argument("--include_description", action="store_true", help="Include description in text. Defaults to training config preference.")
    parser.add_argument("--id_col", type=str, default=None, help="Column name for job ID (defaults to row index)")
    parser.add_argument("--isco_col", type=str, default="iscoGroup", help="Column name for ISCO codes (for metrics)")
    parser.add_argument("--isco_level", type=int, default=4, choices=[1, 2, 3, 4], help="ISCO digits to evaluate/predict")
    parser.add_argument("--esco_id_col", type=str, default="esco_id", help="Column name for ESCO occupation URI (for metrics mapping)")

    # Model/artifacts
    parser.add_argument("--model_dir", type=str, required=True, help="Directory containing model artifacts")
    parser.add_argument("--embeddings_path", type=str, default=None, help="Path to embeddings (.npy or directory)")
    parser.add_argument("--encoder_name", type=str, default=None, help="Encoder name if embeddings not provided (defaults to config)")

    # Output
    parser.add_argument("--output_path", type=str, required=True, help="Output folder (absolute or relative to model_dir)")

    # Inference config
    parser.add_argument("--topk", type=int, default=10, help="Top-k predictions to save per job")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for embedding/inference")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda/cpu)")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for data loading/embedding (CPU only)")

    # Metrics
    parser.add_argument("--compute_metrics", action="store_true", help="Compute evaluation metrics if ground truth is present")
    parser.add_argument(
        "--esco_path",
        type=str,
        default=None,
        help="Path to ESCO master CSV. If provided, can map --esco_id_col to ISCO codes for metrics.",
    )
    parser.add_argument(
        "--esco_isco_col",
        type=str,
        default="iscoGroup",
        help="Column name of ISCO codes in ESCO CSV (used when mapping via --esco_path).",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # 1) Load config & jobs
    model_dir = Path(args.model_dir)
    config = load_config(model_dir)
    train_config = config.get("config", {})

    # Determine preprocessing parameters
    prefix = args.prefix if args.prefix is not None else train_config.get("prefix")
    include_description = args.include_description or train_config.get("include_description", False)

    logger.info(f"Text processing: prefix={prefix}, include_description={include_description}")

    jobs_csv = Path(args.jobs_csv)
    logger.info(f"Loading jobs from {jobs_csv}")
    df = pd.read_csv(jobs_csv)

    texts = []
    # If formatting is required (prefix or description)
    if prefix or include_description:
        if args.text_col not in df.columns:
            # Fallback to 'text' if not found, similar to legacy behavior
            if "text" in df.columns:
                logger.warning(f"Column '{args.text_col}' not found, using 'text'")
                args.text_col = "text"
            else:
                raise ValueError(f"Column '{args.text_col}' not found in CSV")
        
        # Determine description column if needed
        actual_desc_col = args.desc_col
        if include_description and not actual_desc_col:
            # Try to auto-detect commonly used description columns
            for candidate in ["description", "occupationDescription", "raw_description", "definition"]:
                 if candidate in df.columns:
                     actual_desc_col = candidate
                     break
            if not actual_desc_col:
                 logger.warning("include_description=True but no description column found/provided. Using empty description.")
        
        logger.info(f"Formatting text from title='{args.text_col}' and desc='{actual_desc_col}' with prefix='{prefix}'")
        
        for _, row in df.iterrows():
            title = str(row.get(args.text_col, "")).strip()
            desc = str(row.get(actual_desc_col, "")).strip() if actual_desc_col else ""
            texts.append(format_text(title, desc, prefix, include_description))

    else:
        # Legacy/Simple mode: use text_col as is
        if args.text_col not in df.columns:
            if "text" in df.columns:
                logger.warning(f"Column '{args.text_col}' not found, using 'text'")
                args.text_col = "text"
            else:
                raise ValueError(f"Column '{args.text_col}' not found in CSV")

        texts = df[args.text_col].astype(str).tolist()

    # Determine job IDs
    id_col_to_use = args.id_col
    if not id_col_to_use and "job_id" in df.columns:
        logger.info("Found 'job_id' column, using it as ID.")
        id_col_to_use = "job_id"

    if id_col_to_use and id_col_to_use in df.columns:
        job_ids = df[id_col_to_use].astype(str).tolist()
    else:
        job_ids = [str(i) for i in range(len(df))]
    if len(set(job_ids)) != len(job_ids):
        logger.warning(
            "Duplicate job IDs detected. Output JSON is keyed by job_id, so later rows will overwrite earlier ones. "
            "If you need per-row output, consider using a unique id column or omit --id_col to fall back to row indices."
        )

    # 2) Embeddings
    model_dir = Path(args.model_dir)
    if args.embeddings_path:
        embeddings = load_embeddings(Path(args.embeddings_path), job_ids)
    else:
        config = load_config(model_dir)
        encoder_name = (
            args.encoder_name
            or config.get("config", {}).get("model_path")
            or "pj-mathematician/JobSkillBGE-large-en-v1.5"
        )
        embeddings = encode_texts(texts, encoder_name, args.device, batch_size=args.batch_size, num_workers=args.num_workers)

    input_dim = int(embeddings.shape[1])

    # 3) Load classifier + label encoder
    model, label_encoder, config = load_model(model_dir, input_dim, device)

    # 4) Predict top-k
    pred_rows = predict_topk(
        model=model,
        embeddings=embeddings,
        label_encoder=label_encoder,
        device=device,
        batch_size=args.batch_size,
        topk=args.topk,
        num_workers=args.num_workers,
    )

    # Resolve output directory
    out_path_arg = Path(args.output_path)
    if out_path_arg.is_absolute():
        output_dir = out_path_arg
    else:
        output_dir = model_dir / out_path_arg
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Outputs will be saved to: {output_dir}")

    # Save predictions
    results = {job_id: row for job_id, row in zip(job_ids, pred_rows)}
    preds_path = output_dir / "isco_predictions.json"
    with open(preds_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.success(f"Saved ISCO predictions for {len(results)} jobs to {preds_path}")

    # 5) Optional metrics
    if args.compute_metrics:
        logger.info("Computing evaluation metrics from ground truth ISCO codes (if available)...")

        override_codes = None
        if args.isco_col in df.columns:
            logger.info(f"Found '{args.isco_col}' in jobs CSV; using it as ground truth.")
        elif args.esco_path and args.esco_id_col in df.columns:
            logger.info(
                f"'{args.isco_col}' not found; mapping '{args.esco_id_col}' -> ISCO using ESCO CSV ({args.esco_path})."
            )
            override_codes = _map_gt_isco_from_esco_id(
                df,
                esco_path=Path(args.esco_path),
                esco_id_col=args.esco_id_col,
                isco_col_in_esco=args.esco_isco_col,
            )
        else:
            logger.warning(
                f"No ground truth found: neither '{args.isco_col}' is present in jobs CSV nor "
                f"('--esco_path' + '{args.esco_id_col}') is available for mapping."
            )

        eval_embeddings, eval_targets, eval_job_ids, stats = _build_eval_subset(
            df,
            embeddings,
            job_ids,
            isco_col=args.isco_col,
            override_isco_codes=override_codes,
            isco_level=args.isco_level,
            label_encoder=label_encoder,
        )

        logger.info(f"Metric subset stats: {stats}")

        if len(eval_job_ids) == 0:
            logger.warning(
                "No valid evaluation rows found. Ensure your CSV has a ground truth ISCO column "
                f"('{args.isco_col}') and its codes match the model's ISCO level={args.isco_level}."
            )
        else:
            eval_dataset = EmbeddingDataset(eval_embeddings, eval_targets)
            eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
            criterion = torch.nn.CrossEntropyLoss()
            metrics = evaluate(model, eval_loader, criterion, device)

            metrics_path = output_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump({"metrics": metrics, "subset_stats": stats}, f, indent=2)
            logger.success(f"Saved metrics for {len(eval_job_ids)} eval rows to {metrics_path}")


if __name__ == "__main__":
    main()


