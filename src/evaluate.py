import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from box import Box
from loguru import logger

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}")

from .config import load_config
from .utils import load_esco_titles, load_pairs
from .indexing import build_or_load_faiss_index, search_faiss_index
from .metrics import map_esco_id_to_row, METRICS
from .model import BiEncoder


def seed_all(seed: int):
    """Seed all random number generators."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pretty_print_metrics(metrics: dict):
    """Prints a compact table of metrics."""
    df = pd.DataFrame([metrics])
    df = df.round(4)
    logger.info("\n" + df.to_string(index=False))


def main(cfg: Box):
    """Main evaluation script."""
    t_start = time.monotonic()
    seed_all(cfg.project.seed)
    
    # Create a unique model identifier for caching from the model path/name.
    model_id = Path(cfg.model.hf_id).name
    if cfg.eval.get("ckpt_path"):
        model_id += "_" + Path(cfg.eval.ckpt_path).stem

    cache_dir = Path(cfg.eval.cache_dir) / model_id

    if cfg.eval.get("clean_cache"):
        logger.info(f"Cleaning cache directory: {cache_dir}")
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

    cache_dir.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load model
    logger.info("Loading model...")
    model_cfg = cfg.model
    # For raw sentence-transformer models, there's no projection head.
    if not cfg.eval.get("ckpt_path"):
        logger.info("No checkpoint path provided, using raw sentence-transformer. Setting proj_dim to None.")
        model_cfg.proj_dim = None

    model = BiEncoder(model_cfg, device).to(device)

    if cfg.eval.get("ckpt_path"):
        logger.info(f"Loading model checkpoint from {cfg.eval.ckpt_path}")
        ckpt = torch.load(cfg.eval.ckpt_path, map_location=device)
        model.load_state_dict(ckpt)
    model.eval()

    # 2. Load and encode ESCO titles
    logger.info("Loading ESCO titles...")
    esco_ids, esco_titles = load_esco_titles(cfg.data.esco_path)

    emb_path = cache_dir / "esco_emb.npy"
    ids_path = cache_dir / "esco_ids.json"

    if cfg.eval.save_embeddings and emb_path.exists() and ids_path.exists():
        logger.info("Loading cached ESCO embeddings...")
        esco_emb = np.load(emb_path)
        with open(ids_path) as f:
            cached_ids = json.load(f)
        if cached_ids == esco_ids:
            assert esco_emb.dtype == np.float32
        else:
            # Re-encode if IDs don't match
            logger.info("ESCO IDs mismatch, re-encoding...")
            esco_emb = model.encode_esco(
                esco_titles,
                batch_size=cfg.eval.batch_size,
                normalize=True,
                show_progress_bar=True,
            )
            np.save(emb_path, esco_emb)
            with open(ids_path, "w") as f:
                json.dump(esco_ids, f)
    else:
        logger.info("Encoding ESCO titles...")
        esco_emb = model.encode_esco(
            esco_titles,
            batch_size=cfg.eval.batch_size,
            normalize=True,
            show_progress_bar=True,
        )
        if cfg.eval.save_embeddings:
            np.save(emb_path, esco_emb)
            with open(ids_path, "w") as f:
                json.dump(esco_ids, f)

    assert esco_emb.dtype == np.float32

    # 3. Build or load FAISS index
    if cfg.eval.use_faiss:
        logger.info("Building or loading FAISS index...")
        index = build_or_load_faiss_index(
            esco_emb, cache_dir / "index.faiss", use_cache=cfg.eval.save_embeddings
        )

    # 4. Load pairs and encode job titles
    logger.info("Loading evaluation pairs...")
    pairs = load_pairs(cfg.data.pairs_path)
    job_texts = [p["job_title"] for p in pairs]
    gold_ids = [p["esco_id"] for p in pairs]

    logger.info("Encoding job titles...")
    t_encode_start = time.monotonic()
    job_emb = model.encode_job(
        job_texts,
        batch_size=cfg.eval.batch_size,
        normalize=True,
        show_progress_bar=True,
    )
    t_encode_end = time.monotonic()
    assert job_emb.dtype == np.float32

    # 5. Retrieve top-k
    logger.info("Retrieving top-k candidates...")
    if cfg.eval.use_faiss:
        _, top_k_indices = search_faiss_index(index, job_emb, cfg.eval.topk)
    else:
        scores = job_emb @ esco_emb.T
        top_k_indices = np.argpartition(scores, -cfg.eval.topk, axis=1)[:, -cfg.eval.topk:]
        # Sort the top-k indices by score
        rows = np.arange(len(job_emb))[:, np.newaxis]
        top_k_scores = scores[rows, top_k_indices]
        sorted_idx = np.argsort(top_k_scores, axis=1)[:, ::-1]
        top_k_indices = top_k_indices[rows, sorted_idx]

    # 6. Compute metrics
    logger.info("Computing metrics...")
    gold_rows, coverage = map_esco_id_to_row(gold_ids, esco_ids)

    if coverage < 0.95:
        logger.warning(
            f"Low coverage ({coverage:.2f}). "
            "This might indicate an ESCO version mismatch between your pairs and titles."
        )

    metrics = {}
    for metric_fn in METRICS:
        metrics.update(metric_fn(top_k_indices, gold_rows))

    N_eval = len(job_texts)
    encode_ms_per_query = (t_encode_end - t_encode_start) * 1000 / N_eval

    metrics.update({
        "coverage": coverage,
        "N_eval": N_eval,
        "encode_ms_per_query": encode_ms_per_query,
    })

    # 7. Print and save results
    logger.info("Evaluation results:")
    pretty_print_metrics(metrics)

    with open(cache_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if cfg.eval.save_predictions:
        logger.info("Saving predictions...")
        predictions = []
        for i in range(len(job_texts)):
            preds = [esco_ids[j] for j in top_k_indices[i]]
            predictions.append(
                {
                    "job_title": job_texts[i],
                    "gold_esco_id": gold_ids[i],
                    "predicted_esco_ids": preds,
                }
            )
        with open(cache_dir / "predictions.jsonl", "w") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")

    logger.info(f"Total evaluation time: {time.monotonic() - t_start:.2f}s")
    logger.info(f"Results saved to {cache_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a BiEncoder model.")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config file.")
    args = parser.parse_args()
    config = load_config(args.cfg)
    main(config)

