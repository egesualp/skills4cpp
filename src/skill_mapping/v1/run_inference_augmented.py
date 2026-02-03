"""
run_inference_augmented.py

Goal
----
Create job→skill mappings *only* from augmented text fields:
  - Occupation/job text comes from occupations CSV column: `skill_brief`
  - Skill text comes from skills CSV column: `job_brief`

We then retrieve the most similar skills for each occupation using FAISS.

Optional HYBRID mode
--------------------
If `--cat_model_ckpt` is provided, we also:
  - Predict job categories with the trained CategoryPredictor
  - Re-rank retrieved skills using: final_score = P(category(skill)) * sim(job, skill)

Note: category mapping is derived from a hierarchy CSV (default is the existing master file),
keyed by `skillUri`. This file is used only for taxonomy metadata (categories), not for text.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# HuggingFace helper for checkpoint subfolder loading (see src/seq_transformer/extract_features.py)
from huggingface_hub import snapshot_download

from skill_mapping.v1.data import HIER_COL_MAP
from skill_mapping.v1.indexing import load_category_model, predict_categories
from skill_mapping.v1.utils import normalize_embeddings, set_seed


def load_sentence_transformer(
    model_name: str,
    device: str,
    checkpoint_subfolder: Optional[str] = None,
) -> SentenceTransformer:
    """
    Load a SentenceTransformer using the same approach as:
    `src/seq_transformer/extract_features.py` (145-171).
    """
    logger.info(f"Loading model: {model_name} (subfolder={checkpoint_subfolder})")
    try:
        if "bge" in model_name.lower() or checkpoint_subfolder is not None:
            if checkpoint_subfolder is None:
                # Keep a sane default for the common bge repo layout.
                # Users can override via --*_encoder_subfolder.
                checkpoint_subfolder = ""

            if checkpoint_subfolder:
                snapshot_path = snapshot_download(
                    repo_id=model_name,
                    allow_patterns=[f"{checkpoint_subfolder}/*"],
                )
                model_path = os.path.join(snapshot_path, checkpoint_subfolder)
                logger.info(f"Loading model from local path: {model_path}")
                model = SentenceTransformer(model_path)
            else:
                # No subfolder specified: fall back to standard load
                model = SentenceTransformer(model_name, device=device)
        else:
            model = SentenceTransformer(model_name, device=device)
            if "BERT" in model_name:
                # Keep behavior consistent with extract_features.py
                model = SentenceTransformer(modules=[model[0], model[1]], device=device)
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise
    return model


def _require_cols(df: pd.DataFrame, cols: List[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}. Available: {list(df.columns)}")


def _normalize_job_title(s: str) -> str:
    return str(s).strip().lower()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Augmented job→skill inference (baseline or hybrid).")

    # --- Data paths ---
    p.add_argument(
        "--occupations_path",
        type=str,
        required=True,
        help="Augmented occupations CSV (e.g., augmented_decorte_occupations_with_desc.csv).",
    )
    p.add_argument(
        "--skills_path",
        type=str,
        required=True,
        help="Augmented skills CSV (e.g., augmented_esco_skills.csv).",
    )

    # --- Text columns (swapped on purpose per project note) ---
    p.add_argument("--job_text_col", type=str, default="skill_brief", help="Occupation text column to encode.")
    p.add_argument("--skill_text_col", type=str, default="job_brief", help="Skill text column to encode.")

    # --- Identifiers / labels ---
    p.add_argument("--job_title_col", type=str, default="raw_title", help="Occupation title column for output CSV.")
    p.add_argument(
        "--job_id_col",
        type=str,
        default="Unnamed: 0",
        help="Occupation row id column. If not present, dataframe index is used.",
    )
    p.add_argument("--occupation_uri_col", type=str, default="esco_id", help="Occupation ESCO URI column (optional).")

    p.add_argument("--skill_uri_col", type=str, default="conceptUri", help="Skill URI column in skills CSV.")
    p.add_argument("--skill_label_col", type=str, default="preferredLabel", help="Skill label column in skills CSV.")

    # --- Hybrid / category predictor ---
    p.add_argument("--cat_model_ckpt", type=str, default=None, help="Path to trained CategoryPredictor .pt")
    p.add_argument("--hier_level", type=int, default=None, help="Hierarchy level 0-3 (required in hybrid mode).")
    p.add_argument(
        "--hierarchy_path",
        type=str,
        default="data/processed/master_datasets_2/master_complete_hierarchy_w_occ.csv",
        help="CSV that provides skill hierarchy columns (used only for skill categories in hybrid mode).",
    )
    p.add_argument("--cat_threshold", type=float, default=0.1, help="Minimum category prob to use in reranking.")

    # --- Encoders ---
    p.add_argument("--skill_encoder_ckpt", type=str, default="BAAI/bge-large-en-v1.5")
    p.add_argument("--skill_encoder_subfolder", type=str, default=None)

    p.add_argument("--base_encoder_ckpt", type=str, default="all-MiniLM-L6-v2")
    p.add_argument("--base_encoder_subfolder", type=str, default=None)
    p.add_argument("--hidden_dim", type=int, default=None)

    # --- Inference config ---
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)

    # --- Output ---
    p.add_argument("--out_dir", type=str, default="results/")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--normalize_job_titles", action="store_true")

    return p.parse_args()


def build_faiss_index_from_texts(
    encoder: SentenceTransformer,
    texts: List[str],
    batch_size: int,
    device: str,
) -> Tuple[faiss.Index, np.ndarray]:
    emb = encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=device,
    ).cpu().numpy()
    emb = normalize_embeddings(emb).astype(np.float32)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)
    return index, emb


def retrieve_topk(
    encoder: SentenceTransformer,
    job_texts: List[str],
    faiss_index: faiss.Index,
    top_k: int,
    batch_size: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      D: [N, top_k] scores
      I: [N, top_k] indices into the skill corpus
    """
    all_D: List[np.ndarray] = []
    all_I: List[np.ndarray] = []

    for i in tqdm(range(0, len(job_texts), batch_size), desc="Encoding+Retrieval"):
        batch = job_texts[i : i + batch_size]
        if not batch:
            continue
        q = encoder.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            device=device,
        ).cpu().numpy()
        q = normalize_embeddings(q).astype(np.float32)
        D, I = faiss_index.search(q, top_k)
        all_D.append(D)
        all_I.append(I)

    return np.vstack(all_D), np.vstack(all_I)


def load_skill_uri_to_category(
    hierarchy_path: str,
    hier_level: int,
) -> Tuple[Dict[str, str], List[str]]:
    cat_col = HIER_COL_MAP[hier_level]
    usecols = ["skillUri", cat_col]
    logger.info(f"Loading skill categories from {hierarchy_path} (cols={usecols})")
    df = pd.read_csv(hierarchy_path, usecols=usecols, low_memory=False)
    df = df.dropna(subset=["skillUri", cat_col])
    df["skillUri"] = df["skillUri"].astype(str)
    df[cat_col] = df[cat_col].astype(str)

    uri_to_cat = pd.Series(df[cat_col].values, index=df["skillUri"]).to_dict()
    all_cats = sorted(df[cat_col].unique().tolist())
    logger.info(f"Loaded {len(uri_to_cat)} skillUri→category mappings and {len(all_cats)} unique categories.")
    return uri_to_cat, all_cats


def rerank_hybrid(
    retrieved_idx: np.ndarray,
    retrieved_sim: np.ndarray,
    skill_uris: List[str],
    skill_uri_to_cat: Dict[str, str],
    cat_probs: Dict[str, float],
    cat_threshold: float,
) -> List[Tuple[int, float, float, Optional[str], float]]:
    """
    Returns list of tuples:
      (skill_idx, sim, final_score, category_label, cat_prob_used)
    """
    merged: List[Tuple[int, float, float, Optional[str], float]] = []
    for j in range(retrieved_idx.shape[0]):
        s_idx = int(retrieved_idx[j])
        sim = float(retrieved_sim[j])
        s_uri = skill_uris[s_idx]
        cat = skill_uri_to_cat.get(s_uri)
        if cat is None:
            continue
        p = float(cat_probs.get(cat, 0.0))
        if p < cat_threshold:
            p = 0.0
        merged.append((s_idx, sim, p * sim, cat, p))

    merged.sort(key=lambda x: x[2], reverse=True)
    return merged


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    run_hybrid_mode = args.cat_model_ckpt is not None
    if run_hybrid_mode and args.hier_level is None:
        raise ValueError("--hier_level is required when --cat_model_ckpt is set (hybrid mode).")

    process = "HYBRID" if run_hybrid_mode else "BASELINE"
    run_name = args.run_name or f"augmented_{process}_top{args.top_k}"
    logger.add(Path(args.out_dir) / f"{run_name}.log")
    logger.info(f"Starting: {run_name}")
    logger.info(f"Args: {vars(args)}")

    # --- Load data ---
    occ_df = pd.read_csv(args.occupations_path)
    skills_df = pd.read_csv(args.skills_path, low_memory=False)

    _require_cols(occ_df, [args.job_text_col, args.job_title_col], "occupations CSV")
    _require_cols(skills_df, [args.skill_text_col, args.skill_uri_col, args.skill_label_col], "skills CSV")

    # Prepare job ids/titles/texts
    if args.job_id_col in occ_df.columns:
        job_row_ids = occ_df[args.job_id_col].astype(int).tolist()
    else:
        job_row_ids = occ_df.index.astype(int).tolist()

    job_titles = occ_df[args.job_title_col].astype(str).tolist()
    if args.normalize_job_titles:
        job_titles = [_normalize_job_title(t) for t in job_titles]

    job_texts = occ_df[args.job_text_col].fillna("").astype(str).tolist()
    job_texts = [("" if t.strip().lower() == "nan" else t) for t in job_texts]
    if args.occupation_uri_col in occ_df.columns:
        occ_uris = occ_df[args.occupation_uri_col].astype(str).fillna("").tolist()
    else:
        occ_uris = [""] * len(occ_df)

    # Prepare skills corpus
    skills_df = skills_df.dropna(subset=[args.skill_uri_col, args.skill_label_col])
    skill_uris = skills_df[args.skill_uri_col].astype(str).tolist()
    skill_labels = skills_df[args.skill_label_col].astype(str).tolist()
    skill_texts = skills_df[args.skill_text_col].fillna("").astype(str).tolist()
    skill_texts = [("" if t.strip().lower() == "nan" else t) for t in skill_texts]

    # Drop empty texts (cannot embed meaningful vectors)
    keep = [i for i, t in enumerate(skill_texts) if str(t).strip() != ""]
    skill_uris = [skill_uris[i] for i in keep]
    skill_labels = [skill_labels[i] for i in keep]
    skill_texts = [skill_texts[i] for i in keep]
    logger.info(f"Skill corpus size after filtering empty texts: {len(skill_texts)}")

    # --- Load encoders & build FAISS ---
    skill_encoder = load_sentence_transformer(
        args.skill_encoder_ckpt, device=args.device, checkpoint_subfolder=args.skill_encoder_subfolder
    )
    faiss_index, _ = build_faiss_index_from_texts(
        encoder=skill_encoder,
        texts=skill_texts,
        batch_size=args.batch_size,
        device=args.device,
    )

    # --- Optional: hybrid components ---
    category_model = None
    cat_idx2label: Dict[int, str] = {}
    skill_uri_to_cat: Dict[str, str] = {}

    if run_hybrid_mode:
        skill_uri_to_cat, all_cat_labels = load_skill_uri_to_category(args.hierarchy_path, args.hier_level)
        cat_idx2label = {i: c for i, c in enumerate(all_cat_labels)}

        base_encoder = load_sentence_transformer(
            args.base_encoder_ckpt, device=args.device, checkpoint_subfolder=args.base_encoder_subfolder
        )
        if args.hidden_dim is None:
            args.hidden_dim = base_encoder.get_sentence_embedding_dimension()
            logger.info(f"Inferred hidden_dim: {args.hidden_dim}")

        category_model = load_category_model(
            ckpt_path=Path(args.cat_model_ckpt),
            base_encoder=base_encoder,
            hidden_dim=args.hidden_dim,
            categories_idx2str=cat_idx2label,
            device=torch.device(args.device),
        )

    # --- Retrieve ---
    top_k = min(args.top_k, faiss_index.ntotal)
    D, I = retrieve_topk(
        encoder=skill_encoder,
        job_texts=job_texts,
        faiss_index=faiss_index,
        top_k=top_k,
        batch_size=args.batch_size,
        device=args.device,
    )

    # --- Optional: predict categories in batches (hybrid) ---
    cat_probs_batch: List[Dict[str, float]] = []
    if run_hybrid_mode:
        assert category_model is not None
        logger.info("Predicting categories for occupations...")
        for i in tqdm(range(0, len(job_texts), args.batch_size), desc="Category prediction"):
            batch = job_texts[i : i + args.batch_size]
            if not batch:
                continue
            cat_probs_batch.extend(predict_categories(category_model, batch, cat_idx2label))
        if len(cat_probs_batch) != len(job_texts):
            raise RuntimeError(
                f"Category prediction batch size mismatch: {len(cat_probs_batch)} != {len(job_texts)}"
            )

    # --- Assemble outputs ---
    all_predictions: Dict[str, List[Dict[str, object]]] = {}
    master_mapping: List[Dict[str, object]] = []
    flat_rows: List[Dict[str, object]] = []

    for n in range(len(job_texts)):
        job_id = f"job_{job_row_ids[n]}"

        if run_hybrid_mode:
            reranked = rerank_hybrid(
                retrieved_idx=I[n],
                retrieved_sim=D[n],
                skill_uris=skill_uris,
                skill_uri_to_cat=skill_uri_to_cat,
                cat_probs=cat_probs_batch[n],
                cat_threshold=args.cat_threshold,
            )
            if reranked:
                ranked_items = reranked[:top_k]
                pred_items = []
                for s_idx, sim, final_score, cat_label, cat_prob_used in ranked_items:
                    pred_items.append(
                        {
                            "skill": skill_labels[s_idx],
                            "skillUri": skill_uris[s_idx],
                            "score": float(final_score),
                            "sim": float(sim),
                            "category": cat_label,
                            "category_prob": float(cat_prob_used),
                        }
                    )
            else:
                # Fallback: if a skill has no category mapping, do baseline ranking.
                pred_items = []
                for j in range(top_k):
                    s_idx = int(I[n, j])
                    pred_items.append(
                        {
                            "skill": skill_labels[s_idx],
                            "skillUri": skill_uris[s_idx],
                            "score": float(D[n, j]),
                            "sim": float(D[n, j]),
                        }
                    )
        else:
            pred_items = []
            for j in range(top_k):
                s_idx = int(I[n, j])
                pred_items.append(
                    {
                        "skill": skill_labels[s_idx],
                        "skillUri": skill_uris[s_idx],
                        "score": float(D[n, j]),
                        "sim": float(D[n, j]),
                    }
                )

        all_predictions[job_id] = pred_items

        master_mapping.append(
            {
                "job_id": job_id,
                "job_title": job_titles[n],
                "occupationUri": occ_uris[n],
                "skills": [{"skill": it["skill"], "score": it["score"], "skillUri": it["skillUri"]} for it in pred_items],
            }
        )

        for it in pred_items:
            flat_rows.append(
                {
                    "original_row_index": int(job_row_ids[n]),
                    "job_title": job_titles[n],
                    "skill": it["skill"],
                    "score": it["score"],
                    "skillUri": it["skillUri"],
                }
            )

    # --- Save ---
    out_dir = Path(args.out_dir)
    out_preds = out_dir / f"{run_name}_predictions.json"
    out_master_json = out_dir / f"{run_name}_job_title_skills_master.json"
    out_master_csv = out_dir / f"{run_name}_job_title_skills_master.csv"

    with open(out_preds, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, indent=2)
    with open(out_master_json, "w", encoding="utf-8") as f:
        json.dump(master_mapping, f, indent=2)

    pd.DataFrame(flat_rows).to_csv(out_master_csv, index=False)

    logger.success(f"Wrote: {out_preds}")
    logger.success(f"Wrote: {out_master_json}")
    logger.success(f"Wrote: {out_master_csv}")


if __name__ == "__main__":
    main(parse_args())


