# indexing.py (Modified for your new plan)

import torch
import numpy as np
from typing import Dict, List, Tuple, Callable
from pathlib import Path
from sentence_transformers import SentenceTransformer
from loguru import logger
import faiss
import pandas as pd

# Import our models and data helpers
from skill_mapping.v1.model import CategoryPredictor, TextEncoderWrapper # No SkillSimilarityModel needed
from skill_mapping.v1.data import build_skill_lookups, HIER_COL_MAP
from skill_mapping.v1.utils import normalize_embeddings

# -----------------------------
# 1. MODEL LOADING
# -----------------------------

def load_category_model(
    ckpt_path: Path,
    base_encoder: SentenceTransformer, # Pass in the base encoder
    hidden_dim: int,
    categories_idx2str: Dict[int, str],
    device: str = "cpu",
) -> CategoryPredictor:
    """
    Loads the trained CategoryPredictor (Step 1).
    This model's encoder is FROZEN and just used for feature extraction.
    """
    logger.info(f"Loading CategoryPredictor from {ckpt_path}...")
    
    # Create the wrapper for the *frozen* base encoder
    encoder_wrapper = TextEncoderWrapper(
        lambda texts: base_encoder.encode(texts, convert_to_tensor=True, device=device)
    )

    model = CategoryPredictor(
        encoder=encoder_wrapper,
        hidden_dim=hidden_dim,
        num_categories=len(categories_idx2str),
    ).to(device)

    # Load the saved state_dict (this is just the classifier head)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    logger.success("CategoryPredictor loaded.")
    return model

# -----------------------------
# 2. FAISS INDEXING
# -----------------------------

def build_faiss_index(
    skill_encoder: SentenceTransformer, # Takes the SOTA model directly
    all_skill_texts: List[str]
) -> Tuple[faiss.Index, Dict[int, str]]:
    """
    Encodes all skills using the SOTA skill encoder and builds a FAISS index.
    """
    logger.info(f"Building FAISS index for {len(all_skill_texts)} skills...")
    
    # 1. Encode all skill texts using the provided SOTA model
    skill_emb = skill_encoder.encode(
        all_skill_texts, 
        convert_to_tensor=True, 
        show_progress_bar=True,
        batch_size=32 # Adjust as needed
    ).cpu().numpy()
    
    # 2. L2 Normalize
    skill_emb = normalize_embeddings(skill_emb).astype(np.float32)
    hidden_dim = skill_emb.shape[1]

    # 3. Build the FAISS index
    index = faiss.IndexFlatIP(hidden_dim)
    index.add(skill_emb)
    
    # 4. Create the mapping from FAISS index ID -> skill label
    # We assume all_skill_texts is in the same order as all_skill_labels
    id_map = {i: label for i, label in enumerate(all_skill_texts)}
    
    logger.success(f"FAISS index built with {index.ntotal} skills.")
    return index, id_map


# -----------------------------
# 3. HYBRID INFERENCE
# -----------------------------

@torch.no_grad()
def predict_categories(
    model: CategoryPredictor,
    text_batch: List[str],
    idx2cat: Dict[int, str],
) -> List[Dict[str, float]]:
    """
    Returns per input text a dict of (category_label -> score).
    """
    # This function remains the same
    probs = model.predict_proba(text_batch)
    probs = probs.cpu().numpy()
    out_batch = []
    for row in probs:
        cats = {idx2cat[i]: float(p) for i, p in enumerate(row)}
        out_batch.append(cats)
    return out_batch


@torch.no_grad()
def retrieve_skills(
    job_texts: List[str],
    skill_encoder: SentenceTransformer, # Takes the SOTA model directly
    faiss_index: faiss.Index,
    id_map: Dict[int, str],
    top_k: int = 50
) -> List[List[Tuple[str, float]]]:
    """
    Retrieves the Top-K most similar skills for a batch of job texts.
    """
    # 1. Embed the job texts using the SOTA model
    job_emb = skill_encoder.encode(
        job_texts, 
        convert_to_tensor=True, 
        show_progress_bar=False
    ).cpu().numpy()
    job_emb = normalize_embeddings(job_emb).astype(np.float32)
    
    # 2. Search the FAISS index
    D, I = faiss_index.search(job_emb, top_k)
    
    results = []
    for i in range(len(job_texts)):
        this_job_skills = []
        for j in range(top_k):
            faiss_id = I[i, j]
            score = D[i, j]
            label = id_map[faiss_id] # Assumes id_map uses text labels
            this_job_skills.append((label, float(score)))
        results.append(this_job_skills)
        
    return results


def merge_hybrid_scores(
    category_prob_map: Dict[str, float],
    retrieved_skills: List[Tuple[str, float]],
    skill_to_cat_map: Dict[str, str],
    category_threshold: float = 0.1
) -> List[Tuple[str, float]]:
    """
    Implements the "Smarter Hybrid" (Model 3).
    Final Score = P(Category) * Sim(Skill)
    """
    # This function remains the same
    merged_scores = {}
    for skill_label, skill_sim in retrieved_skills:
        skill_category = skill_to_cat_map.get(skill_label)
        if skill_category is None:
            continue
        
        cat_prob = category_prob_map.get(skill_category, 0.0)
        
        if cat_prob < category_threshold:
            cat_prob = 0.0
        
        final_score = cat_prob * skill_sim
        merged_scores[skill_label] = final_score

    ranked = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked

# -----------------------------
# 4. MAIN INFERENCE PIPELINE
# -----------------------------

def load_hybrid_pipeline(
    esco_df: pd.DataFrame,
    cat_model_ckpt: str,
    base_encoder_ckpt: str,  # e.g., "all-MiniLM-L6-v2"
    skill_encoder_ckpt: str, # e.g., "pjmathematician/talentclef-model"
    hidden_dim: int,
    hier_level: int,
    text_fields: str,
    is_structured: bool,
    device: str = "cpu"
) -> Callable:
    """
    Loads all components needed for the full hybrid pipeline.
    """
    # 1. Build the lookup maps from the ESCO data
    cat_col = HIER_COL_MAP[hier_level]
    all_cat_labels = esco_df[cat_col].dropna().astype(str).unique().tolist()
    cat_idx2label = {i: label for i, label in enumerate(all_cat_labels)}
    
    # Use the skill LABEL as the key
    skill_label_col = "skillLabel"
    unique_skills_df = esco_df.drop_duplicates(subset=[skill_label_col])
    
    all_skill_labels = []
    all_skill_texts = []
    skill_to_cat_map = {}
    
    for _, row in unique_skills_df.iterrows():
        label = row[skill_label_col]
        cat = row[cat_col]
        if pd.isna(label) or pd.isna(cat):
            continue
        label, cat = str(label), str(cat)
        
        all_skill_labels.append(label)
        all_skill_texts.append(build_skill_text(row, text_fields, is_structured))
        skill_to_cat_map[label] = cat

    # 2. Load the *BASE* encoder for the Category Model
    base_encoder = SentenceTransformer(base_encoder_ckpt).to(device)

    # 3. Load the *Category Predictor* (Step 1 Model)
    cat_model = load_category_model(
        ckpt_path=Path(cat_model_ckpt),
        base_encoder=base_encoder,
        hidden_dim=hidden_dim,
        categories_idx2str=cat_idx2label,
        device=device
    )
    
    # 4. Load the SOTA *Skill Encoder* (Step 2 Model)
    skill_encoder = SentenceTransformer(skill_encoder_ckpt).to(device)

    # 5. Build the FAISS index using the SOTA skill_encoder
    faiss_index, faiss_id_map = build_faiss_index(
        skill_encoder=skill_encoder,
        all_skill_texts=all_skill_labels # Use labels as the text to index
    )
    # Note: We must ensure all_skill_labels == all_skill_texts if we use
    # labels as the key. Let's adjust build_faiss_index to take labels.

    # --- Let's correct that ---
    # We should index the TEXTS and map the IDs back to LABELS
    
    # 5. Build FAISS index
    index_to_label_map = {i: label for i, label in enumerate(all_skill_labels)}
    
    logger.info("Re-building FAISS index with proper text/label mapping...")
    skill_emb = skill_encoder.encode(
        all_skill_texts, 
        convert_to_tensor=True, 
        show_progress_bar=True
    ).cpu().numpy()
    skill_emb = normalize_embeddings(skill_emb).astype(np.float32)
    faiss_index = faiss.IndexFlatIP(skill_emb.shape[1])
    faiss_index.add(skill_emb)
    logger.success(f"FAISS index built with {faiss_index.ntotal} skills.")
    
    # 6. Create a single inference function
    def run_inference(
        job_texts: List[str], 
        top_k: int = 50, 
        category_threshold: float = 0.1
    ) -> List[List[Tuple[str, float]]]:
        
        # Step 1: Get category probabilities
        cat_probs_batch = predict_categories(cat_model, job_texts, cat_idx2label)
        
        # Step 2: Get global skill similarities
        job_emb = skill_encoder.encode(
            job_texts, 
            convert_to_tensor=True
        ).cpu().numpy()
        job_emb = normalize_embeddings(job_emb).astype(np.float32)
        
        D, I = faiss_index.search(job_emb, top_k)
        
        # Step 3: Merge them
        final_results = []
        for i in range(len(job_texts)): # For each job
            
            # Reconstruct retrieved skills
            retrieved_skills = []
            for j in range(top_k):
                faiss_id = I[i, j]
                score = D[i, j]
                label = index_to_label_map[faiss_id]
                retrieved_skills.append((label, float(score)))

            # Run hybrid merge
            ranked_skills = merge_hybrid_scores(
                category_prob_map=cat_probs_batch[i],
                retrieved_skills=retrieved_skills,
                skill_to_cat_map=skill_to_cat_map,
                category_threshold=category_threshold
            )
            final_results.append(ranked_skills)
            
        return final_results

    return run_inference