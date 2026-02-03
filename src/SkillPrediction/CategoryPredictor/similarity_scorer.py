"""
similarity_scorer.py - High-Speed Semantic Retrieval for Job-to-Skill Mapping

Maps Job Titles to ESCO Skills using cosine similarity via FAISS.

Usage:
    python -m skill_mapping.v2.similarity_scorer \
        --jobs_csv data/processed/job_titles.csv \
        --index_dir outputs/skill_index \
        --model_path pj-mathematician/JobSkillBGE-large-en-v1.5 \
        --output_path outputs/similarity_scores.json \
        --top_k 100 \
        --save_embeddings outputs/job_embeddings  # Optional: save query vectors
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Set

import faiss
import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from huggingface_hub import snapshot_download

from .metrics_utils import compute_precision_at_k, compute_recall_at_k, compute_map, compute_mrr


def load_index(index_dir: str | Path) -> tuple[faiss.Index, List[Dict[str, str]]]:
    """
    Load FAISS index and skill metadata from disk.
    
    Returns:
        index: FAISS IndexFlatIP
        skills: List of skill metadata dicts (index i -> skill info)
    """
    index_dir = Path(index_dir)
    
    # Load FAISS index
    index_path = index_dir / "skill.index"
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")
    index = faiss.read_index(str(index_path))
    logger.info(f"Loaded FAISS index with {index.ntotal} vectors")
    
    # Load metadata
    metadata_path = index_dir / "skill_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    skills = metadata["skills"]
    logger.info(f"Loaded metadata for {len(skills)} skills")
    
    return index, skills


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


def load_jobs(
    csv_path: str | Path,
    text_column: str = "processed_text",
    id_column: str | None = None,
    esco_id_column: str = "esco_id",
    use_raw_features: bool = False,
    title_column: str = "raw_title",
    desc_column: str = "raw_description",
    separator: str = ". ",
) -> tuple[List[str], List[str], List[str]]:
    """
    Load job texts from CSV.
    
    Returns:
        job_ids: List of job identifiers
        job_texts: List of job descriptions to encode
        esco_ids: List of ESCO occupation URIs for ground truth
    """
    logger.info(f"Loading jobs from {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=['raw_title', 'raw_description', 'esco_id', 'job_id'], keep='first')
    
    if use_raw_features:
        logger.info(f"Using raw features. Title: {title_column}, Desc: {desc_column}")
        # Validate columns
        if title_column not in df.columns:
            raise ValueError(f"Column '{title_column}' not found.")
        
        # Fill NA with empty string
        df[title_column] = df[title_column].fillna("")
        
        texts = []
        if desc_column and desc_column in df.columns:
             df[desc_column] = df[desc_column].fillna("")
             # Combine: Title + separator + Description
             texts = (df[title_column] + separator + df[desc_column]).tolist()
        else:
             if desc_column:
                 logger.warning(f"Description column '{desc_column}' not found. Using only title.")
             else:
                 logger.info("No description column specified. Using only title.")
             texts = df[title_column].tolist()
             
        job_texts = [t.strip() for t in texts]
    else:
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found. Available: {list(df.columns)}")
        job_texts = df[text_column].astype(str).tolist()
    
    # Use provided ID column, or fallback to index
    if id_column in df.columns:
        job_ids = df[id_column].astype(str).tolist()
    else:
        job_ids = [str(i) for i in range(len(df))]
    
    esco_ids = []
    if esco_id_column in df.columns:
        esco_ids = df[esco_id_column].astype(str).tolist()
    else:
        # Pad with empty if missing, so we don't break, but evaluation will be skipped
        esco_ids = [""] * len(df)
    
    logger.info(f"Loaded {len(job_texts)} jobs")
    return job_ids, job_texts, esco_ids


def encode_queries(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 64,
) -> np.ndarray:
    """
    Encode texts and L2-normalize for cosine similarity.
    
    Returns:
        Normalized float32 embeddings of shape (n_texts, dim)
    """
    logger.info(f"Encoding {len(texts)} queries...")
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # We normalize with sklearn
    )
    
    # L2-normalize using sklearn (Inner Product = Cosine Similarity)
    embeddings = normalize(embeddings, norm="l2", axis=1).astype(np.float32)
    
    logger.info(f"Encoded queries with shape {embeddings.shape}")
    return embeddings


def search_index(
    index: faiss.Index,
    queries: np.ndarray,
    top_k: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Search FAISS index for top-k nearest neighbors.
    
    Args:
        index: FAISS IndexFlatIP
        queries: L2-normalized float32 embeddings (n_queries, dim)
        top_k: Number of nearest neighbors to retrieve
    
    Returns:
        scores: Cosine similarity scores (n_queries, top_k), range [-1, 1]
        indices: FAISS indices (n_queries, top_k)
    """
    if queries.dtype != np.float32:
        raise ValueError("Query vectors must be float32")
    
    # Verify normalization
    norms = np.linalg.norm(queries, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("Query vectors must be L2-normalized")
    
    scores, indices = index.search(queries, top_k)
    
    # Clip scores to [-1, 1] (numerical precision)
    scores = np.clip(scores, -1.0, 1.0).astype(np.float32)
    
    return scores, indices.astype(np.int64)


def format_results(
    job_ids: List[str],
    scores: np.ndarray,
    indices: np.ndarray,
    skills: List[Dict[str, str]],
    uri_key: str = "conceptUri",
) -> Dict[str, List[Dict]]:
    """
    Format search results as a dictionary.
    
    Returns:
        { "job_id": [ {"skill_uri": "...", "score": 0.985, "rank": 1}, ... ] }
    """
    results = {}
    
    for i, job_id in enumerate(job_ids):
        job_results = []
        for rank, (score, idx) in enumerate(zip(scores[i], indices[i]), start=1):
            if idx < 0:  # FAISS returns -1 for missing results
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
    
    Outputs:
        - job_embeddings.npy: Embeddings array (n_jobs, dim)
        - job_embeddings_meta.json: Mapping of job_id to embedding index
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
        
    # Warn if model mismatch
    saved_model = metadata.get("model_path")
    if saved_model != model_path:
        logger.warning(f"Model mismatch: Loaded embeddings were created with '{saved_model}', but current run uses '{model_path}'")
    
    saved_job_map = metadata["job_id_to_index"]
    
    # Load embeddings
    logger.info(f"Loading embeddings from {embeddings_path}")
    all_embeddings = np.load(embeddings_path)
    
    # Align embeddings: Create array matching expected_job_ids order
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
        description="Map Job Titles to ESCO Skills using semantic similarity"
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
        default=64,
        help="Batch size for encoding",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for encoding (cuda/cpu)",
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
    
    # Load FAISS index and metadata
    index, skills = load_index(args.index_dir)
    
    # Load job texts
    job_ids, job_texts, esco_ids = load_jobs(
        args.jobs_csv,
        text_column=args.text_column,
        id_column=args.id_column,
        use_raw_features=args.use_raw_features,
        title_column=args.title_column,
        desc_column=args.desc_column,
        separator=args.separator,
    )
    
    queries = None
    if args.load_embeddings:
        queries = load_query_embeddings(args.load_embeddings, job_ids, args.model_path)

    # Load model and encode queries ONLY if not loaded
    if queries is None:
        logger.info(f"Loading model: {args.model_path}")
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
                model = SentenceTransformer(model_path, device=args.device)
            else:
                model = SentenceTransformer(args.model_path, device=args.device)
                if 'BERT' in args.model_path:
                    model = SentenceTransformer(modules=[model[0], model[1]], device=args.device)
        except Exception as e:
            logger.error(f"Failed to load model {args.model_path}: {e}")
            raise
        queries = encode_queries(model, job_texts, batch_size=args.batch_size)
    
    # Optionally save query embeddings
    if args.save_embeddings and not args.load_embeddings:
        save_query_embeddings(
            queries,
            job_ids,
            args.save_embeddings,
            model_path=args.model_path,
        )
    
    # Search
    logger.info(f"Searching for top-{args.top_k} skills per job...")
    scores, indices = search_index(index, queries, top_k=args.top_k)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Evaluation
    if args.skill_relations_csv and args.test_metrics:
        try:
            skill_relations = load_skill_relations(args.skill_relations_csv)
            
            # Map skill URI to index in 'skills' list
            skill_uri_to_idx = {s.get("conceptUri"): i for i, s in enumerate(skills)}
            
            # Construct gold sets (indices)
            gold_sets = []
            found_in_gold = 0
            for occ_uri in esco_ids:
                if occ_uri and occ_uri in skill_relations:
                    skill_uris = skill_relations[occ_uri]
                    # Convert to indices
                    gold_indices = {skill_uri_to_idx[uri] for uri in skill_uris if uri in skill_uri_to_idx}
                    gold_sets.append(gold_indices)
                    found_in_gold += 1
                else:
                    gold_sets.append(set())
            
            # Compute metrics
            metrics = {}
            
            # Coverage
            metrics["gold_coverage"] = found_in_gold / len(esco_ids) if esco_ids else 0.0

            # Full Corpus Metrics
            metrics["map_full"] = compute_map(indices, gold_sets, k=None)
            metrics["mrr_full"] = compute_mrr(indices, gold_sets, k=None)
            
            # Top-K Metrics
            for k in [1, 5, 10, 20, 50, 100]:
                if k > args.top_k:
                    continue
                metrics[f"precision@{k}"] = compute_precision_at_k(indices, gold_sets, k)
                metrics[f"recall@{k}"] = compute_recall_at_k(indices, gold_sets, k)
                metrics[f"map@{k}"] = compute_map(indices, gold_sets, k)
                metrics[f"mrr@{k}"] = compute_mrr(indices, gold_sets, k)
            
            # Print
            logger.info("Evaluation results:")
            # Format metrics for display
            logger.info("=" * 60)
            logger.info(f"Gold Coverage: {metrics['gold_coverage']:.4f} ({found_in_gold}/{len(esco_ids)} jobs)")
            logger.info("-" * 60)
            
            # Full corpus metrics
            logger.info(f"MAP (full):    {metrics['map_full']:.4f}")
            logger.info(f"MRR (full):    {metrics['mrr_full']:.4f}")
            logger.info("-" * 60)
            
            # Top-K metrics in table format
            k_values = [k for k in [1, 5, 10, 20, 50, 100] if k <= args.top_k]
            if k_values:
                logger.info("Top-K Metrics:")
                logger.info(f"{'K':<6} {'Precision':<12} {'Recall':<12} {'MAP':<12} {'MRR':<12}")
                logger.info("-" * 60)
                for k in k_values:
                    p = metrics.get(f"precision@{k}", 0.0)
                    r = metrics.get(f"recall@{k}", 0.0)
                    m = metrics.get(f"map@{k}", 0.0)
                    mrr = metrics.get(f"mrr@{k}", 0.0)
                    logger.info(f"{k:<6} {p:<12.4f} {r:<12.4f} {m:<12.4f} {mrr:<12.4f}")
            logger.info("=" * 60)
            
            # Save metrics
            metrics_path = Path(args.output_path).parent / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Saved metrics to {metrics_path}")

        except Exception as e:
            logger.error(f"Error during evaluation: {e}")

    # Format and save results
    if args.save_scores:
        results = format_results(job_ids, scores, indices, skills)
    
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.success(f"Saved results for {len(results)} jobs to {output_path}")
    else:
        logger.info("Not saving results.")


if __name__ == "__main__":
    main()

