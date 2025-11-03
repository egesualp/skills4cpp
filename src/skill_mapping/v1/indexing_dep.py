# indexing.py

import torch
import numpy as np
from typing import Dict, List, Tuple
from pathlib import Path
from sentence_transformers import SentenceTransformer
from loguru import logger

from model import CategoryPredictor, SkillPredictor, TextEncoderWrapper, SimilarityModel
from data import LabelEncoder
from utils import normalize_embeddings


############################################
# 1. Stage 1 inference: predict categories
############################################

def load_category_model(
    ckpt_path: Path,
    encoder_ckpt: str,
    hidden_dim: int,
    categories_idx2str: Dict[int, str],
    device: str = "cpu",
) -> CategoryPredictor:
    """
    ckpt_path: path to the saved .pt state_dict for the CategoryPredictor
    encoder_ckpt: which SentenceTransformer to load for text encoding
    categories_idx2str: idx -> category label (we need len() for head size)
    """
    encoder = SentenceTransformer(encoder_ckpt)
    encoder_wrapper = TextEncoderWrapper(
        lambda texts: torch.tensor(
            encoder.encode(texts, convert_to_numpy=True),
            dtype=torch.float32,
        ).to(device)
    )

    model = CategoryPredictor(
        encoder=encoder_wrapper,
        hidden_dim=hidden_dim,
        num_categories=len(categories_idx2str),
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_categories(
    model: CategoryPredictor,
    text_batch: List[str],
    idx2cat: Dict[int, str],
    threshold: float = 0.2,
) -> List[List[Tuple[str, float]]]:
    """
    Returns per input text a list of (category_label, score)
    filtered by threshold.
    """
    probs = model.predict_proba(text_batch)  # [B, C]
    probs = probs.cpu().numpy()

    out = []
    for row in probs:
        cats = []
        for idx, p in enumerate(row):
            if p >= threshold:
                cats.append((idx2cat[idx], float(p)))
        # sort high to low
        cats.sort(key=lambda x: x[1], reverse=True)
        out.append(cats)
    return out


############################################
# 2. Stage 2 inference: predict skills
############################################

def load_skill_model(
    ckpt_path: Path,
    encoder_ckpt: str,
    hidden_dim: int,
    skills_idx2str: Dict[int, str],
    device: str = "cpu",
) -> SkillPredictor:
    """
    ckpt_path: path to saved .pt for SkillPredictor for ONE category.
    skills_idx2str: idx -> skillUri (or skillLabel), used for head size.
    """
    encoder = SentenceTransformer(encoder_ckpt)
    encoder_wrapper = TextEncoderWrapper(
        lambda texts: torch.tensor(
            encoder.encode(texts, convert_to_numpy=True),
            dtype=torch.float32,
        ).to(device)
    )

    model = SkillPredictor(
        encoder=encoder_wrapper,
        hidden_dim=hidden_dim,
        num_skills=len(skills_idx2str),
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_skills_for_category(
    model: SkillPredictor,
    text_batch: List[str],
    idx2skill: Dict[int, str],
) -> List[List[Tuple[str, float]]]:
    """
    For each job text:
      - returns list of (skill_id, prob) pairs sorted desc
    This is run per category (pillar/group).
    """
    probs = model.predict_proba(text_batch)  # [B, K]
    probs = probs.cpu().numpy()

    out = []
    for row in probs:
        skills = []
        for idx, p in enumerate(row):
            skills.append((idx2skill[idx], float(p)))
        skills.sort(key=lambda x: x[1], reverse=True)
        out.append(skills)
    return out


############################################
# 3. Merge stage: category weighting
############################################

def merge_category_and_skill_scores(
    category_preds: List[Tuple[str, float]],
    # e.g. [("skills", 0.9), ("knowledge", 0.4)]
    per_category_skill_preds: Dict[str, List[Tuple[str, float]]],
    # e.g. {
    #   "skills":    [("sql", 0.8), ("python", 0.7)],
    #   "knowledge": [("accounting_principles", 0.6)]
    # }
) -> List[Tuple[str, float]]:
    """
    Final step of the pipeline:
    final_score(skill) = P(category) * P(skill | category)

    We assume you already ran predict_skills_for_category() for
    each predicted category and stored results in per_category_skill_preds.
    """

    merged_scores = {}

    for cat_label, cat_score in category_preds:
        skills_for_cat = per_category_skill_preds.get(cat_label, [])
        for skill_id, skill_score in skills_for_cat:
            weighted = cat_score * skill_score
            if skill_id not in merged_scores or weighted > merged_scores[skill_id]:
                merged_scores[skill_id] = weighted

    # sort by final weighted score
    ranked = sorted(merged_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked  # list of (skill_id, final_score)


############################################
# 4. End-to-end helper
############################################

def infer_job_skills(
    job_text: str,
    category_model: CategoryPredictor,
    cat_idx2label: Dict[int, str],
    skill_models: Dict[str, SkillPredictor],
    skill_idx_maps: Dict[str, Dict[int, str]],
    category_threshold: float = 0.2,
    top_skills_per_cat: int = 20,
) -> List[Tuple[str, float]]:
    """
    Full pipeline for a SINGLE job text:
    1. Predict relevant categories.
    2. For each predicted category, run its skill model.
    3. Merge weighted by category confidence.

    Inputs:
      - category_model: trained Stage 1 model
      - cat_idx2label: {idx: "skills"} etc.
      - skill_models: {"skills": model_for_skills, "knowledge": model_for_knowledge, ...}
      - skill_idx_maps: {"skills": {idx: skillUri,...}, ...}

    Returns:
      list of (skill_id, final_score) sorted desc
    """
    # 1. Category prediction (Stage 1)
    cat_preds_batch = predict_categories(
        category_model,
        [job_text],
        idx2cat=cat_idx2label,
        threshold=category_threshold,
    )
    # cat_preds_batch is a list of length 1
    cat_preds = cat_preds_batch[0]  # [("skills",0.91), ("knowledge",0.44), ...]

    # 2. For each category, predict skills using Stage 2 models
    cat_to_skill_scores = {}
    for (cat_label, _cat_prob) in cat_preds:
        model_for_cat = skill_models.get(cat_label)
        idx2skill_for_cat = skill_idx_maps.get(cat_label)
        if model_for_cat is None or idx2skill_for_cat is None:
            continue

        skill_preds_batch = predict_skills_for_category(
            model_for_cat,
            [job_text],
            idx2skill_for_cat,
        )
        # also length 1
        skill_preds = skill_preds_batch[0][:top_skills_per_cat]  # top K per cat
        cat_to_skill_scores[cat_label] = skill_preds

    # 3. Merge with category weights
    final_ranked = merge_category_and_skill_scores(
        category_preds=cat_preds,
        per_category_skill_preds=cat_to_skill_scores,
    )

    return final_ranked


############################################
# 5. Retrieval / similarity mode using FAISS
############################################

def build_skill_embedding_index(
    skill_texts: List[str],
    encoder_ckpt: str,
) -> Tuple[np.ndarray, dict]:
    """
    Build embeddings for all skills and return:
    - normalized embeddings matrix [N, D] (float32)
    - id map: {row_idx: skill_text or skill_id}
    You can then send this to FAISS.

    This is for the 'similarity' baseline where we compare job text embedding
    vs skill text embedding directly.
    """
    enc = SentenceTransformer(encoder_ckpt)
    emb = enc.encode(skill_texts, convert_to_numpy=True)  # (N, D) float32
    emb = normalize_embeddings(emb)  # L2 normalize
    id_map = {i: skill_texts[i] for i in range(len(skill_texts))}
    return emb.astype(np.float32), id_map


def embed_job_texts(job_texts: List[str], encoder_ckpt: str) -> np.ndarray:
    """
    Encode job texts with same encoder.
    """
    enc = SentenceTransformer(encoder_ckpt)
    emb = enc.encode(job_texts, convert_to_numpy=True)
    emb = normalize_embeddings(emb)
    return emb.astype(np.float32)


def retrieve_topk_skills(
    index,
    job_emb: np.ndarray,
    id_map: dict,
    topk: int = 10,
) -> List[List[Tuple[str, float]]]:
    """
    Uses your FAISS helpers to query nearest skill texts.
    Returns per query: [(skill_text, score), ...]
    """
    from index import search_faiss_index  # reuse your FAISS search
    D, I = search_faiss_index(index, job_emb, topk=topk)

    results = []
    for row_scores, row_ids in zip(D, I):
        this_query = []
        for score, idx in zip(row_scores, row_ids):
            this_query.append((id_map[int(idx)], float(score)))
        results.append(this_query)
    return results
