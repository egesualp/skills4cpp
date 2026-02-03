import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import faiss
from box import Box
from loguru import logger

# --- Optimized Imports & Components ---
from config import load_config
from utils import load_esco_titles, load_pairs
from metrics import map_esco_id_to_row, METRICS, load_skills_per_occupation, compute_skill_coverage
from model import BiEncoder

logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}")

def seed_all(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main(cfg: Box):
    t_start = time.monotonic()
    seed_all(cfg.project.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results_dir = Path("experiments/results") / cfg.eval.run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Define model_id for metadata
    model_id = Path(cfg.model.hf_id).name
    if cfg.eval.get("ckpt_path"):
        model_id += "_" + Path(cfg.eval.ckpt_path).stem

    # 1. Load model and optimize for H100
    logger.info("Loading model and optimizing for H100...")
    model = BiEncoder(cfg.model, device).to(device)
    
    if cfg.eval.get("ckpt_path"):
        ckpt = torch.load(cfg.eval.ckpt_path, map_location=device)
        model.load_state_dict(ckpt)
    
    model.eval()
    # Use BF16 for H100 - massive speedup for 7B models
    model = model.to(torch.bfloat16) 
    
    # 2. Encode ESCO Titles (The Corpus)
    esco_ids, esco_titles = load_esco_titles(cfg.data.esco_path)
    
    logger.info(f"Encoding {len(esco_titles)} ESCO titles...")
    # Use a larger batch size for H100
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            esco_emb = model.encode_esco(
                esco_titles, 
                batch_size=cfg.eval.get("batch_size", 128), 
                normalize=True, 
                show_progress_bar=True
            )
    esco_emb = esco_emb.astype('float32') # FAISS prefers float32

    # 3. Encode Job Titles (The Queries)
    logger.info(f"Loading pairs from {cfg.data.pairs_path}")
    
    # --- DEBUG LOGGING ---
    try:
        _paths = cfg.data.pairs_path
        # Handle single path or list of paths (BoxList behaves like list)
        if isinstance(_paths, (str, Path)):
            _paths = [_paths]
        
        df_debug = pd.concat([pd.read_csv(p) for p in _paths], ignore_index=True)
        logger.info(f"DEBUG: Raw input has {len(df_debug)} rows (aggregated).")
        if 'raw_title' in df_debug.columns:
            logger.info(f"DEBUG: Unique raw_titles: {df_debug['raw_title'].nunique()}")
        if 'raw_description' in df_debug.columns:
            logger.info(f"DEBUG: Unique title+desc combos: {df_debug[['raw_title', 'raw_description']].drop_duplicates().shape[0]}")
    except Exception as e:
        logger.warning(f"DEBUG: Could not inspect raw file: {e}")
    # ---------------------

    pairs = load_pairs(
        cfg.data.pairs_path,
        lowercase_raw=(cfg.data.get("lowercase") in ["raw", "both"]),
        lowercase_esco=(cfg.data.get("lowercase") in ["esco", "both"]),
        ground_truth_col=cfg.data.get("ground_truth_col", "esco_id"),
        group_by_col=cfg.data.get("group_by_col", "raw_title"),
    )
    logger.info(f"DEBUG: load_pairs returned {len(pairs)} processed pairs.")
    # Support use_description
    job_texts = []
    use_description = cfg.data.get("use_description", False)
    if use_description:
        logger.info("Appending job descriptions to titles...")

    for p in pairs:
        text = p["job_title"]
        if use_description:
            desc = p.get("raw_description", "")
            if desc:
                text = f"{text}</s>{desc}"
        job_texts.append(text)
    
    logger.info(f"Encoding {len(job_texts)} job titles...")
    t_encode_start = time.monotonic()
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            job_emb = model.encode_job(
                job_texts, 
                batch_size=cfg.eval.get("batch_size", 128), 
                normalize=True, 
                show_progress_bar=True
            )
    job_emb = job_emb.astype('float32')
    logger.info(f"Encoding took {time.monotonic() - t_encode_start:.2f}s")

    # 4. BLAZING FAST RETRIEVAL (The Bottleneck Fix)
    #logger.info("Starting GPU-accelerated retrieval...")
    #t_retrieve_start = time.monotonic()
    #
    #d = esco_emb.shape[1]
    #top_k = cfg.eval.topk # e.g., 100
    #
    ## Use GPU FAISS
    #res = faiss.StandardGpuResources()
    ## FlatL2 is actually Inner Product if vectors are normalized
    #flat_config = faiss.GpuIndexFlatConfig()
    #flat_config.device = 0 
    #
    #gpu_index = faiss.GpuIndexFlatIP(res, d, flat_config)
    #gpu_index.add(esco_emb)
    #
    ## Search in one go (FAISS handles the batching internally on GPU)
    ## We retrieve slightly more than top_k to account for ID collapsing
    #search_k = min(len(esco_ids), top_k * 2) 
    #distances, I = gpu_index.search(job_emb, search_k)
    # FAISS REPLACEMENT (Pure PyTorch on GPU)
    logger.info("Starting PyTorch-accelerated retrieval (H100)...")
    t_retrieve_start = time.monotonic()
    top_k = cfg.eval.topk
    search_k = min(len(esco_ids), top_k * 2)

    # Move embeddings to GPU
    job_emb_torch = torch.from_numpy(job_emb).to('cuda').to(torch.bfloat16)
    esco_emb_torch = torch.from_numpy(esco_emb).to('cuda').to(torch.bfloat16)

    # Batched search to avoid OOM
    I_list = []
    batch_size = 2000 
    for i in range(0, len(job_emb_torch), batch_size):
        batch = job_emb_torch[i : i + batch_size]
        # Matrix multiply on H100 (Blazing fast)
        scores = torch.matmul(batch, esco_emb_torch.T)
        # Get top-k
        _, indices = torch.topk(scores, k=search_k, dim=1)
        I_list.append(indices.cpu().numpy())

    I = np.vstack(I_list)
    t_retrieve_end = time.monotonic()
    logger.info(f"PyTorch retrieval took {t_retrieve_end - t_retrieve_start:.2f}s")
    
    # 5. VECTORIZED ID COLLAPSING (The "Forever" Loop Fix)
    logger.info("Collapsing duplicate ESCO IDs via Vectorized Mapping...")
    
    # Create unique ID map
    unique_ids_list = sorted(list(set(esco_ids)))
    id_to_unique_idx = {uid: i for i, uid in enumerate(unique_ids_list)}
    index_to_unique = np.array([id_to_unique_idx[eid] for eid in esco_ids], dtype=np.int32)
    
    # Map retrieved indices to unique indices
    I_mapped = index_to_unique[I]
    
    # We use a fast row-wise unique operation (only for the top results)
    final_I = []
    for row in I_mapped:
        # This is the only loop, but it's much faster because it only sees 'search_k' elements
        _, idx = np.unique(row, return_index=True)
        final_I.append(row[np.sort(idx)][:top_k])
    
    I = np.array(final_I)
    t_retrieve_end = time.monotonic()
    logger.info(f"Retrieval and collapsing took {t_retrieve_end - t_retrieve_start:.2f}s")

    # 6. Compute Metrics
    logger.info("Computing metrics...")
    gold_id_lists = [p[cfg.data.get("ground_truth_col", "esco_id")] for p in pairs]
    gold_rows, coverage = map_esco_id_to_row(gold_id_lists, unique_ids_list)

    metrics = {}
    for metric_fn in METRICS:
        metrics.update(metric_fn(I, gold_rows))
    
    # Skill Coverage
    if cfg.eval.get("compute_skill_coverage", False):
        skills_path = cfg.data.get("skills_path", "data/skills_per_occupations.csv")
        if Path(skills_path).exists():
            logger.info("Computing skill coverage...")
            skills_by_occupation = load_skills_per_occupation(skills_path)
            skill_coverage_metrics = compute_skill_coverage(
                I, gold_rows, unique_ids_list, skills_by_occupation, ks=(1, 5, 10)
            )
            metrics.update(skill_coverage_metrics)
        else:
            logger.warning(f"compute_skill_coverage=True but {skills_path} not found.")

    # Add run details
    run_details = {
        "model_id": model_id,
        "data_path": str(cfg.data.pairs_path),
        "proj_dim": cfg.model.get("proj_dim"),
        "topk": cfg.eval.topk,
        "use_faiss": False, 
        "normalize_embeddings": True,
        "optimization": "H100_torch",
        "use_description": use_description
    }
    metrics["run_details"] = run_details

    # --- SAVE ARTIFACTS ---
    if cfg.eval.get("save_embeddings", False):
        logger.info(f"Saving embeddings to {results_dir / 'embeddings.pt'}")
        # Use pickle_protocol=4 to handle objects > 4GB
        torch.save({
            "job_emb": job_emb,
            "esco_emb": esco_emb,
            "esco_ids": esco_ids,
            "esco_titles": esco_titles,
            # "job_texts": job_texts # Omitted to save space, redundant if we have ids
        }, results_dir / "embeddings.pt", pickle_protocol=4)

    if cfg.eval.get("save_predictions", False):
        logger.info(f"Saving predictions to {results_dir / 'predictions.csv'} and {results_dir / 'predictions.jsonl'}")
        preds_csv = []
        preds_jsonl = []
        id_col = cfg.data.get("group_by_col", "job_id")
        ground_truth_col = cfg.data.get("ground_truth_col", "esco_id")
        
        for i, idx_row in enumerate(I):
            p = pairs[i]
            # Get Job ID if available, else index
            jid = p.get(id_col) if id_col in p else p.get("job_id", i)
            
            # idx_row contains indices into unique_ids_list
            pred_ids = [unique_ids_list[idx] for idx in idx_row]
            
            # Get gold values
            gold_val = p.get(ground_truth_col, [])
            if not isinstance(gold_val, (list, tuple, set)):
                gold_val = [gold_val]
            gold_val = list(gold_val)
            
            # CSV format (pipe-separated)
            preds_csv.append({
                "job_id": jid,
                "job_title": p.get("job_title", ""),
                "predicted_ids": "|".join(pred_ids)
            })
            
            # JSONL format (for fused_scorer compatibility)
            preds_jsonl.append({
                "job_id": jid,
                "job_title": p.get("job_title", ""),
                f"gold_{ground_truth_col}": gold_val,
                "predicted_esco_ids": pred_ids,
            })
        
        # Save CSV
        pd.DataFrame(preds_csv).to_csv(results_dir / "predictions.csv", index=False)
        
        # Save JSONL
        with open(results_dir / "predictions.jsonl", "w") as f:
            for pred in preds_jsonl:
                f.write(json.dumps(pred) + "\n")
    # ----------------------

    # 7. Finalize Results
    metrics.update({"coverage": coverage, "total_time": time.monotonic() - t_start})
    logger.info("Evaluation results:")
    print(pd.DataFrame([metrics]).round(4).to_string(index=False))

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Done. Total time: {time.monotonic() - t_start:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()
    main(load_config(args.cfg))