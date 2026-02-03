"""
similarity_scorer_opt.py - High-Speed Semantic Retrieval for Job-to-Skill Mapping (Optimized)

Maps Job Titles to ESCO Skills using cosine similarity via PyTorch on GPU.
Optimized for large-scale retrieval (e.g., 400k jobs x 3000 skills).

Usage:
    python -m skill_mapping.v2.similarity_scorer_opt \
        --jobs_csv data/processed/job_titles.csv \
        --index_dir outputs/skill_index \
        --model_path pj-mathematician/JobSkillBGE-large-en-v1.5 \
        --output_path outputs/similarity_scores.json \
        --top_k 3000 \
        --batch_size 1024
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Set

import faiss
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from huggingface_hub import snapshot_download

# Reuse metrics from existing codebase
from .metrics_utils import compute_precision_at_k, compute_recall_at_k, compute_map, compute_mrr


def load_index_and_vectors(index_dir: str | Path) -> tuple[np.ndarray, List[Dict[str, str]]]:
    """
    Load FAISS index, extract vectors, and load skill metadata.
    
    Returns:
        skill_vectors: Float32 array of shape (n_skills, dim)
        skills: List of skill metadata dicts
    """
    index_dir = Path(index_dir)
    
    # Load FAISS index
    index_path = index_dir / "skill.index"
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")
    
    logger.info(f"Loading FAISS index from {index_path}")
    index = faiss.read_index(str(index_path))
    
    # Extract vectors for GPU matmul
    # reconstruct_n(start, count) works for IndexFlatIP/L2
    logger.info(f"Extracting {index.ntotal} vectors from index...")
    skill_vectors = index.reconstruct_n(0, index.ntotal)
    
    # Load metadata
    metadata_path = index_dir / "skill_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    skills = metadata["skills"]
    logger.info(f"Loaded metadata for {len(skills)} skills")
    
    return skill_vectors, skills


def load_skill_relations(csv_path: str | Path) -> Dict[str, Set[str]]:
    """
    Load occupation-skill relations.
    Returns: dict {occupation_uri: {skill_uri, ...}}
    """
    logger.info(f"Loading skill relations from {csv_path}")
    df = pd.read_csv(csv_path)
    # Ensure columns exist
    if "occupationUri" not in df.columns or "skillUri" not in df.columns:
        logger.warning(f"Relation CSV missing columns. Available: {df.columns}")
        return {}
    
    relations = {}
    # GroupBy is faster than iterating
    for occ, group in df.groupby("occupationUri"):
        relations[occ] = set(group["skillUri"].tolist())
    
    logger.info(f"Loaded relations for {len(relations)} occupations")
    return relations


def load_jobs_multilabel(
    csv_path: str | Path,
    text_column: str = "processed_text",
    id_column: str | None = None,
    esco_id_column: str = "esco_id",
    use_raw_features: bool = False,
    title_column: str = "raw_title",
    desc_column: str = "raw_description",
    separator: str = ". ",
) -> tuple[List[str], List[str], List[Set[str]]]:
    """
    Load job texts from CSV and group multiple ESCO IDs for the same job.
    
    Returns:
        job_ids: Unique job identifiers (one per unique query)
        job_texts: Unique job descriptions (one per unique query)
        esco_id_sets: List of sets, where each set contains *all* valid ESCO IDs for that job
    """
    logger.info(f"Loading jobs from {csv_path} (Multi-label support)")
    df = pd.read_csv(csv_path)
    
    # 1. Construct Full Text
    if use_raw_features:
        logger.info(f"Using raw features. Title: {title_column}, Desc: {desc_column}")
        if title_column not in df.columns:
            raise ValueError(f"Column '{title_column}' not found.")
        
        df[title_column] = df[title_column].fillna("")
        
        if desc_column and desc_column in df.columns:
             df[desc_column] = df[desc_column].fillna("")
             texts = (df[title_column] + separator + df[desc_column])
        else:
             texts = df[title_column]
             
        df['_temp_text'] = texts.astype(str).str.strip()
    else:
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found.")
        df['_temp_text'] = df[text_column].astype(str)
    
    # 2. Handle ID Column
    if id_column and id_column in df.columns:
        df['_temp_id'] = df[id_column].astype(str)
    else:
        # Fallback: Treat the text itself as the unique ID if no ID provided
        # (Or just use row index, but that breaks aggregation. Let's assume unique text = unique job)
        logger.warning("No ID column provided. Using text content as grouping key.")
        df['_temp_id'] = df['_temp_text']

    # 3. Aggregate: Group by Job ID (or Text) and collect all ESCO IDs
    # Filter out rows with empty ESCO IDs first if necessary, or handle empty strings later
    if esco_id_column in df.columns:
        # Collect set of ESCO IDs per job
        grouped = df.groupby(['_temp_id', '_temp_text'])[esco_id_column].apply(lambda x: set(x.dropna().astype(str))).reset_index()
    else:
        # No ground truth column
        grouped = df[['_temp_id', '_temp_text']].drop_duplicates()
        grouped[esco_id_column] = [set() for _ in range(len(grouped))]
    
    job_ids = grouped['_temp_id'].tolist()
    job_texts = grouped['_temp_text'].tolist()
    esco_id_sets = grouped[esco_id_column].tolist()
    
    logger.info(f"Loaded {len(job_ids)} unique jobs (collapsed from {len(df)} rows)")
    return job_ids, job_texts, esco_id_sets

def encode_queries(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 64,
) -> np.ndarray:
    """
    Encode texts and L2-normalize for cosine similarity.
    """
    logger.info(f"Encoding {len(texts)} queries...")
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    
    # L2-normalize using sklearn
    embeddings = normalize(embeddings, norm="l2", axis=1).astype(np.float32)
    
    logger.info(f"Encoded queries with shape {embeddings.shape}")
    return embeddings


def search_index_optimized(
    queries: np.ndarray,
    skill_vectors: np.ndarray,
    top_k: int = 100,
    batch_size: int = 1024,
    device: str = "cuda"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform batched matrix multiplication on GPU to find top-k nearest neighbors.
    Much faster than CPU FAISS for large k and large query sets.
    """
    if not torch.cuda.is_available() and device == "cuda":
        logger.warning("CUDA not available. Falling back to CPU (will be slow).")
        device = "cpu"

    logger.info(f"Starting optimized retrieval on {device}...")
    logger.info(f"Queries: {queries.shape}, Skills: {skill_vectors.shape}, Top-K: {top_k}")
    
    t_start = time.monotonic()
    
    # Move skills to device once
    logger.info("Moving skill vectors to device...")
    skills_tensor = torch.from_numpy(skill_vectors).to(device)
    # Optionally use float16/bfloat16 if memory is tight, but float32 is safer for precision
    # skills_tensor = skills_tensor.to(torch.bfloat16) 
    
    # Store results
    all_scores = []
    all_indices = []
    
    # Process queries in batches
    num_queries = len(queries)
    
    for i in range(0, num_queries, batch_size):
        end_idx = min(i + batch_size, num_queries)
        batch_queries = queries[i:end_idx]
        
        # Move batch to device
        batch_tensor = torch.from_numpy(batch_queries).to(device)
        
        # Matrix Multiplication (Cosine Similarity if normalized)
        # (B, D) @ (M, D).T -> (B, M)
        scores = torch.matmul(batch_tensor, skills_tensor.T)
        
        # Top-K
        # If top_k is very large, this can be memory intensive, but usually fine for 3k
        batch_scores, batch_indices = torch.topk(scores, k=top_k, dim=1)
        
        # Move back to CPU
        all_scores.append(batch_scores.cpu().numpy())
        all_indices.append(batch_indices.cpu().numpy())
        
        if (i // batch_size) % 10 == 0:
            logger.info(f"Processed {end_idx}/{num_queries} queries...")
            
    # Concatenate all batches
    final_scores = np.vstack(all_scores)
    final_indices = np.vstack(all_indices)
    
    t_end = time.monotonic()
    logger.info(f"Retrieval finished in {t_end - t_start:.2f}s")
    
    return final_scores, final_indices.astype(np.int64)


def format_results(
    job_ids: List[str],
    scores: np.ndarray,
    indices: np.ndarray,
    skills: List[Dict[str, str]],
    uri_key: str = "conceptUri",
) -> Dict[str, List[Dict]]:
    """
    Format search results as a dictionary.
    """
    results = {}
    
    for i, job_id in enumerate(job_ids):
        job_results = []
        for rank, (score, idx) in enumerate(zip(scores[i], indices[i]), start=1):
            if idx < 0:
                continue
            skill_info = skills[idx]
            job_results.append({
                "skill_uri": skill_info[uri_key],
                "score": round(float(score), 6),
                "rank": rank,
            })
        results[job_id] = job_results
    
    return results


def save_query_embeddings(
    embeddings: np.ndarray,
    job_ids: List[str],
    output_dir: str | Path,
    model_path: str,
) -> None:
    """
    Save query embeddings and job ID mapping for reuse.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings
    embeddings_path = output_dir / "job_embeddings.npy"
    np.save(embeddings_path, embeddings)
    logger.info(f"Saved job embeddings to {embeddings_path} (shape: {embeddings.shape})")
    
    # Save metadata: job_id -> index mapping
    metadata_path = output_dir / "job_embeddings_meta.json"
    metadata = {
        "model_path": model_path,
        "num_jobs": len(job_ids),
        "embedding_dim": embeddings.shape[1],
        "job_id_to_index": {job_id: i for i, job_id in enumerate(job_ids)},
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved job embeddings metadata to {metadata_path}")


def load_query_embeddings(
    load_dir: str | Path,
    expected_job_ids: List[str],
    model_path: str,
) -> np.ndarray:
    """
    Load precomputed query embeddings from disk and align with expected jobs.
    """
    load_dir = Path(load_dir)
    embeddings_path = load_dir / "job_embeddings.npy"
    metadata_path = load_dir / "job_embeddings_meta.json"
    
    if not embeddings_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Embeddings not found in {load_dir}")
        
    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
        
    saved_model = metadata.get("model_path")
    if saved_model != model_path:
        logger.warning(f"Model mismatch: Loaded embeddings were created with '{saved_model}', but current run uses '{model_path}'")
    
    saved_job_map = metadata["job_id_to_index"]
    
    logger.info(f"Loading embeddings from {embeddings_path}")
    all_embeddings = np.load(embeddings_path)
    
    dim = all_embeddings.shape[1]
    aligned_embeddings = np.zeros((len(expected_job_ids), dim), dtype=np.float32)
    
    missing_ids = []
    for i, job_id in enumerate(expected_job_ids):
        if job_id in saved_job_map:
            idx = saved_job_map[job_id]
            aligned_embeddings[i] = all_embeddings[idx]
        else:
            missing_ids.append(job_id)
            
    if missing_ids:
        msg = f"Found {len(missing_ids)} jobs missing from precomputed embeddings"
        if len(missing_ids) > 5:
            msg += f" (e.g., {missing_ids[:5]}...)"
        raise ValueError(msg)
        
    logger.info(f"Successfully loaded and aligned {len(aligned_embeddings)} embeddings")
    return aligned_embeddings


def main():
    parser = argparse.ArgumentParser(
        description="Map Job Titles to ESCO Skills using GPU optimization"
    )
    parser.add_argument(
        "--jobs_csv",
        type=str,
        required=True,
        help="Path to CSV with job texts",
    )
    parser.add_argument(
        "--index_dir",
        type=str,
        required=True,
        help="Directory containing skill.index and skill_metadata.json",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="pj-mathematician/JobSkillBGE-large-en-v1.5",
        help="SentenceTransformer model path",
    )
    parser.add_argument(
        "--checkpoint_subfolder",
        type=str,
        default=None,
        help='Checkpoint of the model.'
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save output JSON",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="processed_text",
        help="Column name for job text",
    )
    parser.add_argument(
        "--id_column",
        type=str,
        default=None,
        help="Column name for job ID (defaults to row index)",
    )
    parser.add_argument(
        "--use_raw_features",
        action="store_true",
        help="Use raw_title and raw_description instead of a single text column",
    )
    parser.add_argument(
        "--title_column",
        type=str,
        default="raw_title",
        help="Column name for job title (if using raw features)",
    )
    parser.add_argument(
        "--desc_column",
        type=str,
        default=None,
        help="Column name for job description (optional, if using raw features)",
    )
    parser.add_argument(
        "--separator",
        type=str,
        default=". ",
        help="Separator between title and description",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Number of top candidates to retrieve per job",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for retrieval (and default for encoding if suitable)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for encoding and retrieval (cuda/cpu)",
    )
    parser.add_argument(
        "--save_embeddings",
        type=str,
        default=None,
        help="Optional: Directory to save query embeddings for reuse",
    )
    parser.add_argument(
        "--load_embeddings",
        type=str,
        default=None,
        help="Optional: Directory to load precomputed embeddings from",
    )
    parser.add_argument(
        "--skill_relations_csv",
        type=str,
        default="./data/esco_datasets/occupationSkillRelations_en.csv",
        help="Path to occupationSkillRelations.csv for evaluation",
    )
    parser.add_argument(
        "--test_metrics",
        action="store_true",
        help="Calculate test metrics after matching."
    )
    parser.add_argument(
        "--save_scores",
        action="store_true",
        help="Saves scores."
    )
    args = parser.parse_args()
    
    # 1. Load Skills (Vectors)
    skill_vectors, skills = load_index_and_vectors(args.index_dir)
    
    # 2. Load Jobs (Multi-label support)
    job_ids, job_texts, esco_id_sets = load_jobs_multilabel(
        args.jobs_csv,
        text_column=args.text_column,
        id_column=args.id_column,
        use_raw_features=args.use_raw_features,
        title_column=args.title_column,
        desc_column=args.desc_column,
        separator=args.separator,
    )
    
    # 3. Get Query Embeddings
    queries = None
    if args.load_embeddings:
        queries = load_query_embeddings(args.load_embeddings, job_ids, args.model_path)

    if queries is None:
        logger.info(f"Loading model: {args.model_path}")
        # Model loading logic (reused from original)
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
                        allow_patterns=[f"{checkpoint}/*"]
                    )
                model_path = os.path.join(snapshot_path, checkpoint)
                model = SentenceTransformer(model_path, device=args.device)
            else:
                model = SentenceTransformer(args.model_path, device=args.device)
                if 'BERT' in args.model_path:
                    model = SentenceTransformer(modules=[model[0], model[1]], device=args.device)
        except Exception as e:
            logger.error(f"Failed to load model {args.model_path}: {e}")
            raise
        
        # Use smaller batch size for encoding if needed, though args.batch_size is high for retrieval
        encoding_batch_size = 64 if args.batch_size > 128 else args.batch_size
        queries = encode_queries(model, job_texts, batch_size=encoding_batch_size)
    
    # Save embeddings if requested
    if args.save_embeddings and not args.load_embeddings:
        save_query_embeddings(
            queries,
            job_ids,
            args.save_embeddings,
            model_path=args.model_path,
        )
    
    # 4. Optimized Retrieval
    scores, indices = search_index_optimized(
        queries, 
        skill_vectors, 
        top_k=args.top_k, 
        batch_size=args.batch_size,
        device=args.device
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 5. Evaluation (Multi-label)
    if args.skill_relations_csv and args.test_metrics:
        try:
            skill_relations = load_skill_relations(args.skill_relations_csv)
            skill_uri_to_idx = {s.get("conceptUri"): i for i, s in enumerate(skills)}
            
            gold_sets = []
            found_in_gold = 0
            
            # Iterate over SETS of ESCO IDs per job
            for occ_uris in esco_id_sets:
                job_gold_indices = set()
                
                # Union of skills from ALL mapped occupations
                for occ_uri in occ_uris:
                    if occ_uri and occ_uri in skill_relations:
                        skill_uris = skill_relations[occ_uri]
                        # Add indices to our combined set
                        job_gold_indices.update(
                            skill_uri_to_idx[uri] for uri in skill_uris if uri in skill_uri_to_idx
                        )

                gold_sets.append(job_gold_indices)
                if job_gold_indices:
                    found_in_gold += 1
            
            metrics = {}
            metrics["gold_coverage"] = found_in_gold / len(job_ids) if job_ids else 0.0
            metrics["map_full"] = compute_map(indices, gold_sets, k=None)
            metrics["mrr_full"] = compute_mrr(indices, gold_sets, k=None)
            
            for k in [1, 5, 10, 20, 50, 100]:
                if k > args.top_k:
                    continue
                metrics[f"precision@{k}"] = compute_precision_at_k(indices, gold_sets, k)
                metrics[f"recall@{k}"] = compute_recall_at_k(indices, gold_sets, k)
                metrics[f"map@{k}"] = compute_map(indices, gold_sets, k)
                metrics[f"mrr@{k}"] = compute_mrr(indices, gold_sets, k)
            
            logger.info("Evaluation results:")
            logger.info("=" * 60)
            logger.info(f"Gold Coverage: {metrics['gold_coverage']:.4f} ({found_in_gold}/{len(job_ids)} jobs)")
            logger.info("-" * 60)
            logger.info(f"MAP (full):    {metrics['map_full']:.4f}")
            logger.info(f"MRR (full):    {metrics['mrr_full']:.4f}")
            logger.info("-" * 60)
            
            k_values = [k for k in [1, 5, 10, 20, 50, 100] if k <= args.top_k]
            if k_values:
                logger.info(f"{'K':<6} {'Precision':<12} {'Recall':<12} {'MAP':<12} {'MRR':<12}")
                logger.info("-" * 60)
                for k in k_values:
                    p = metrics.get(f"precision@{k}", 0.0)
                    r = metrics.get(f"recall@{k}", 0.0)
                    m = metrics.get(f"map@{k}", 0.0)
                    mrr = metrics.get(f"mrr@{k}", 0.0)
                    logger.info(f"{k:<6} {p:<12.4f} {r:<12.4f} {m:<12.4f} {mrr:<12.4f}")
            logger.info("=" * 60)
            
            metrics_path = Path(args.output_path).parent / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved metrics to {metrics_path}")

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")

    # 6. Save Results (Streaming to avoid OOM)
    if args.save_scores:
        logger.info(f"Saving results to {output_path} (streaming)...")
        # We manually construct the JSON to stream it: { "job_id": [...], ... }
        # This keeps memory usage low by processing one job at a time.
        
        with open(output_path, "w") as f:
            f.write("{\n")
            
            num_jobs = len(job_ids)
            for i, job_id in enumerate(job_ids):
                # Construct result for this job only
                job_results = []
                for rank, (score, idx) in enumerate(zip(scores[i], indices[i]), start=1):
                    if idx < 0:
                        continue
                    skill_info = skills[idx]
                    # Create a minimal dict
                    job_results.append({
                        "skill_uri": skill_info.get("conceptUri"),
                        "score": round(float(score), 6),
                        "rank": rank,
                    })
                
                # key: value
                # Use json.dumps for safe string escaping of key and value
                line = f"{json.dumps(job_id)}: {json.dumps(job_results)}"
                
                # Add comma if not the last item
                if i < num_jobs - 1:
                    line += ",\n"
                else:
                    line += "\n"
                
                f.write(line)
                
                if (i + 1) % 10000 == 0:
                    logger.info(f"Saved {i + 1}/{num_jobs} jobs...")

            f.write("}")
            
        logger.success(f"Saved results for {num_jobs} jobs to {output_path}")
    else:
        logger.info("Not saving results.")

if __name__ == "__main__":
    main()
