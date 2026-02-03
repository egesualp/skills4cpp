"""
skill_indexer.py - ESCO Skill Vector Store Builder

Creates a static FAISS index and metadata for the entire ESCO skill library.

Usage:
    python -m skill_mapping.v2.skill_indexer \
        --skills_csv data/processed/augmentation/augmented_esco_skills.csv \
        --model_name pj-mathematician/JobSkillBGE-large-en-v1.5 \
        --output_dir outputs/skill_index \
        --text_column job_brief
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download



def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalizes embeddings for cosine similarity via inner product."""
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norm + 1e-12)


def load_skills(
    csv_path: str | Path,
    text_column: str = "job_brief",
    uri_column: str = "conceptUri",
    label_column: str = "preferredLabel",
    use_raw_features: bool = False,
    desc_column: str = "description",
    separator: str = ". ",
) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Load skills from CSV and return texts + metadata.
    
    Returns:
        texts: List of skill descriptions to encode
        metadata: List of dicts with conceptUri and preferredLabel
    """
    logger.info(f"Loading skills from {csv_path}")
    df = pd.read_csv(csv_path)
    df_o = pd.read_csv(r'/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/occupationSkillRelations_en.csv')
    df = df.merge(df_o[['skillUri']].drop_duplicates(), left_on='conceptUri', right_on='skillUri', how='inner')
    
    if use_raw_features:
        logger.info(f"Using raw features: {label_column} + '{separator}' + {desc_column}")
        # Validate columns
        required_cols = [uri_column, label_column]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")
        
        # Prepare texts
        df[label_column] = df[label_column].fillna("")
        
        texts = []
        if desc_column in df.columns:
            df[desc_column] = df[desc_column].fillna("")
            texts = (df[label_column] + separator + df[desc_column]).tolist()
        else:
            logger.warning(f"Description column '{desc_column}' not found. Using only label.")
            texts = df[label_column].astype(str).tolist()
            
        # Clean up texts
        texts = [t.strip() for t in texts]

        # Filter empty texts if any
        # (Though usually label exists, so text won't be empty)
        
    else:
        # Validate required columns
        required_cols = [text_column, uri_column, label_column]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")
    
        # Drop rows with missing text
        df = df.dropna(subset=[text_column]).reset_index(drop=True)
        texts = df[text_column].astype(str).tolist()

    # Drop duplicates by URI (common for both modes)
    df = df.drop_duplicates(subset=[uri_column]).reset_index(drop=True)
    
    # We need to ensure texts list aligns with df after drop_duplicates
    # Since we computed texts BEFORE drop_duplicates/dropna in the raw case (partially), 
    # let's re-unify the logic to be safe and consistent.
    
    # RE-IMPLEMENTATION for consistency:
    # 1. Drop duplicates first
    # (Re-loading df to apply drop_duplicates first)
    # Actually, simpler to just start over with clean logic order
    pass 

# ... (Self-correction: rewriting the whole function body for clarity in ReplacementContent)

def load_skills(
    csv_path: str | Path,
    text_column: str = "job_brief",
    uri_column: str = "conceptUri",
    label_column: str = "preferredLabel",
    use_raw_features: bool = False,
    desc_column: str = "description",
    separator: str = ". ",
) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Load skills from CSV and return texts + metadata.
    
    Returns:
        texts: List of skill descriptions to encode
        metadata: List of dicts with conceptUri and preferredLabel
    """
    logger.info(f"Loading skills from {csv_path}")
    df = pd.read_csv(csv_path)
    df_o = pd.read_csv(r'/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/occupationSkillRelations_en.csv')
    df = df.merge(df_o[['skillUri']].drop_duplicates(), left_on='conceptUri', right_on='skillUri', how='inner')
    
    # Drop duplicates by URI first to have a clean set of skills
    df = df.drop_duplicates(subset=[uri_column]).reset_index(drop=True)

    texts = []
    if use_raw_features:
        logger.info(f"Using raw features. Label: {label_column}")
        
        if label_column not in df.columns:
             raise ValueError(f"Label column '{label_column}' not found.")
        
        # Fill NA
        df[label_column] = df[label_column].fillna("")
        
        if desc_column and desc_column in df.columns:
            logger.info(f"Adding description from: {desc_column} (sep: '{separator}')")
            df[desc_column] = df[desc_column].fillna("")
            texts = (df[label_column] + separator + df[desc_column]).tolist()
        else:
            if desc_column:
                logger.warning(f"Description column '{desc_column}' not found. Using only label.")
            else:
                logger.info("No description column specified. Using only label.")
            texts = df[label_column].astype(str).tolist()
            
        texts = [t.strip() for t in texts]
        
    else:
        # Validate required columns
        required_cols = [text_column, uri_column, label_column]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")
        
        # Drop rows with missing text
        df = df.dropna(subset=[text_column]).reset_index(drop=True)
        texts = df[text_column].astype(str).tolist()
    
    metadata = df[[uri_column, label_column]].to_dict("records")
    
    logger.info(f"Loaded {len(texts)} unique skills")
    return texts, metadata


def build_index(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 64,
) -> Tuple[faiss.Index, np.ndarray]:
    """
    Encode texts and build FAISS index.
    
    Returns:
        index: FAISS IndexFlatIP for cosine similarity
        embeddings: Raw embeddings (normalized)
    """
    logger.info(f"Encoding {len(texts)} texts...")
    
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,  # We normalize ourselves
    )
    
    # Normalize for cosine similarity
    embeddings = normalize_embeddings(embeddings).astype(np.float32)
    
    # Build FAISS index (Inner Product = Cosine Sim for normalized vectors)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    logger.info(f"Built FAISS index with {index.ntotal} vectors (dim={dim})")
    return index, embeddings


def save_index(
    index: faiss.Index,
    embeddings: np.ndarray,
    metadata: List[Dict[str, str]],
    output_dir: str | Path,
    model_name: str,
    text_column: str,
) -> None:
    """Save FAISS index, embeddings, and metadata to disk."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save FAISS index
    index_path = output_dir / "skill.index"
    faiss.write_index(index, str(index_path))
    logger.info(f"Saved FAISS index to {index_path}")
    
    # Save embeddings for reuse (clustering, visualization, etc.)
    embeddings_path = output_dir / "skill_embeddings.npy"
    np.save(embeddings_path, embeddings)
    logger.info(f"Saved embeddings to {embeddings_path} (shape: {embeddings.shape})")
    
    # Save metadata: list mapping index ID -> skill info
    metadata_path = output_dir / "skill_metadata.json"
    full_metadata = {
        "model_name": model_name,
        "text_column": text_column,
        "num_skills": len(metadata),
        "embedding_dim": embeddings.shape[1],
        "skills": metadata,  # List where index i corresponds to FAISS ID i
    }
    with open(metadata_path, "w") as f:
        json.dump(full_metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS index for ESCO skills"
    )
    parser.add_argument(
        "--skills_csv",
        type=str,
        required=True,
        help="Path to augmented ESCO skills CSV",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="pj-mathematician/JobSkillBGE-large-en-v1.5",
        help="HuggingFace model name for encoding",
    )
    parser.add_argument(
        "--checkpoint_subfolder",
        type=str,
        default=None,
        help='Checkpoint of the model.'
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save index and metadata",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="job_brief",
        help="Column name for skill text to encode",
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
        "--use_raw_features",
        action="store_true",
        help="Use label + description instead of job_brief column",
    )
    parser.add_argument(
        "--desc_column",
        type=str,
        default=None,
        help="Column name for skill description (optional, if using raw features)",
    )
    parser.add_argument(
        "--separator",
        type=str,
        default=". ",
        help="Separator between label and description",
    )
    args = parser.parse_args()
    
    # Load model
    logger.info(f"Loading model: {args.model_name}")

    if args.checkpoint_subfolder:
        checkpoint = args.checkpoint_subfolder
    elif args.model_name == "pj-mathematician/JobSkillBGE-large-en-v1.5-v2":
        checkpoint = "checkpoint-2240"
    elif args.model_name == "pj-mathematician/JobSkillBGE-large-en-v1.5":
        checkpoint = "checkpoint-4480"
    else:
        checkpoint = None

    try:
        if checkpoint:
            snapshot_path = snapshot_download(
                    repo_id=args.model_name,
                    allow_patterns=[f"{checkpoint}/*"]  # This downloads only the checkpoint files
                )
            model_path = os.path.join(snapshot_path, checkpoint)
            model = SentenceTransformer(model_path, device=args.device)
        else:
            model = SentenceTransformer(args.model_name, device=args.device)
            if 'BERT' in args.model_name:
                model = SentenceTransformer(modules=[model[0], model[1]], device=args.device)
    except Exception as e:
        logger.error(f"Failed to load model {args.model_name}: {e}")
        raise
    
    # Load skills
    texts, metadata = load_skills(
        args.skills_csv,
        text_column=args.text_column,
        use_raw_features=args.use_raw_features,
        desc_column=args.desc_column,
        separator=args.separator,
    )
    
    # Build index
    index, embeddings = build_index(model, texts, batch_size=args.batch_size)
    
    # Determine text_column label for metadata
    if args.use_raw_features:
        if args.desc_column:
            text_column_meta = f"preferredLabel + {args.desc_column}"
        else:
            text_column_meta = "preferredLabel (Title Only)"
    else:
        text_column_meta = args.text_column

    # Save
    save_index(
        index,
        embeddings,
        metadata,
        args.output_dir,
        model_name=args.model_name,
        text_column=text_column_meta,
    )
    
    logger.success(f"Done! Index saved to {args.output_dir}")


if __name__ == "__main__":
    main()

