"""
Shared utilities for skill-based pooling and ESCO data loading.

These functions implement:
- Loading job->skill mappings and ESCO skill/occupation information
- IDF weighting per skill
- Logarithmic pooling over jobs in a career path
- Batch processing from SkillBasedCareerPathDataset-style batches

They are used by training scripts such as train_cpp_skills_v3.py and can
also be reused by other modules that need the same pooling behaviour.
"""

from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import re

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


# ============================================================================
# CSV / METADATA LOADERS
# ============================================================================


def load_skill_mappings(job_title_skills_csv: str) -> Dict[str, List[Dict]]:
    """
    Load job title to skills mapping from CSV.

    Returns a dict mapping normalized job_title to a list of
    skill dicts: {'skill', 'skillUri', 'score'}.
    """
    print(f"Loading skill mappings from {job_title_skills_csv}...")
    df = pd.read_csv(job_title_skills_csv)

    job_skill_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for _, row in df.iterrows():
        job_title = str(row["job_title"]).strip().lower()
        skill_dict = {
            "skill": row["skill"],
            "skillUri": row["skillUri"],
            "score": float(row["score"]) if "score" in row else 1.0,
        }
        job_skill_map[job_title].append(skill_dict)

    print(f"  ✓ Loaded skills for {len(job_skill_map)} job titles")
    return dict(job_skill_map)


def load_skill_descriptions(skills_csv: str) -> Dict[str, Dict[str, str]]:
    """
    Load skill descriptions from ESCO skills CSV.

    Returns a dict mapping skillUri to {'name', 'description'}.
    """
    print(f"Loading skill descriptions from {skills_csv}...")
    df = pd.read_csv(skills_csv)

    skill_desc_map: Dict[str, Dict[str, str]] = {}
    for _, row in df.iterrows():
        skill_uri = row["conceptUri"]
        skill_desc_map[skill_uri] = {
            "name": row["preferredLabel"],
            "description": row.get("description", ""),
        }

    print(f"  ✓ Loaded descriptions for {len(skill_desc_map)} skills")
    return skill_desc_map


def load_occupation_isco_groups(occupations_csv: str) -> Dict[str, str]:
    """
    Load ISCO groups for ESCO occupations.

    Returns a dict mapping normalized occupation title to ISCO group.
    """
    print(f"Loading ISCO groups from {occupations_csv}...")
    df = pd.read_csv(occupations_csv)

    isco_map: Dict[str, str] = {}
    for _, row in df.iterrows():
        title = str(row["preferredLabel"]).strip().lower()
        isco_group = str(row["iscoGroup"])
        isco_map[title] = isco_group

    print(f"  ✓ Loaded ISCO groups for {len(isco_map)} occupations")
    return isco_map


def calculate_idf_scores(
    job_skill_map: Dict[str, List[Dict]],
) -> Dict[str, List[Dict]]:
    """
    Calculate IDF scores for each skill based on occupation frequency.

    IDF = log(total_occupations / occupation_count)
    Adds an 'idf' field to each skill dict.
    """
    print("Calculating IDF scores for skills...")

    # Count in how many occupations each skill appears
    skill_occupation_count: Dict[str, int] = defaultdict(int)
    for _, skills in job_skill_map.items():
        seen_skills = set()
        for skill_dict in skills:
            skill_uri = skill_dict["skillUri"]
            if skill_uri not in seen_skills:
                skill_occupation_count[skill_uri] += 1
                seen_skills.add(skill_uri)

    total_occupations = len(job_skill_map)
    skill_idf: Dict[str, float] = {}
    for skill_uri, count in skill_occupation_count.items():
        if count > 0:
            skill_idf[skill_uri] = float(np.log(total_occupations / count))
        else:
            skill_idf[skill_uri] = 0.0

    # Attach IDF to each skill entry
    updated_map: Dict[str, List[Dict]] = {}
    for job_title, skills in job_skill_map.items():
        updated_skills: List[Dict[str, Any]] = []
        for skill_dict in skills:
            skill_uri = skill_dict["skillUri"]
            new_dict = dict(skill_dict)
            new_dict["idf"] = skill_idf.get(skill_uri, 0.0)
            updated_skills.append(new_dict)
        updated_map[job_title] = updated_skills

    print(f"  ✓ Calculated IDF for {len(skill_idf)} unique skills")
    return updated_map


def cap_skills_per_job_lexicographic(
    job_skill_map: Dict[str, List[Dict[str, Any]]],
    max_skills_per_job: int,
    skill_desc_map: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Deterministically cap the number of skills per job.

    Within each job we:
      1) Sort skills by IDF descending (higher IDF = more job-specific).
      2) Break ties lexicographically by skill label (field 'skill'),
         using the label that we actually encode (name/description).
      3) Keep the first `max_skills_per_job` skills.

    This gives us:
      - Preference for job-specific / rare skills (via IDF),
      - A stable, reproducible selection within IDF ties (via label).

    If max_skills_per_job <= 0, the map is returned unchanged.
    """
    if max_skills_per_job is None or max_skills_per_job <= 0:
        return job_skill_map

    capped_map: Dict[str, List[Dict[str, Any]]] = {}
    total_before = 0
    total_after = 0

    for job_title, skills in job_skill_map.items():
        total_before += len(skills)
        if len(skills) <= max_skills_per_job:
            capped_skills = skills
        else:
            # Sort by:
            #   1) -idf (most job-specific first)
            #   2) -len(description) (longer descriptions first, if available)
            #   3) skill label lexicographically (deterministic tie-break)
            def sort_key(s: Dict[str, Any]):
                idf_val = -float(s.get("idf", 0.0))
                # Look up description length if skill_desc_map is provided
                desc_len = 0
                if skill_desc_map is not None:
                    uri = s.get("skillUri")
                    if uri in skill_desc_map:
                        desc_text = skill_desc_map[uri].get("description", "") or ""
                        desc_len = -len(desc_text)
                label = str(s.get("skill", "")).lower()
                return (idf_val, desc_len, label)

            capped_skills = sorted(skills, key=sort_key)[:max_skills_per_job]
        total_after += len(capped_skills)
        capped_map[job_title] = capped_skills

    if total_before > 0:
        kept_ratio = total_after / total_before
    else:
        kept_ratio = 1.0

    print(
        f"  ✓ Applied IDF+lexicographic cap of {max_skills_per_job} skills/job "
        f"(avg kept ratio over all jobs: {kept_ratio:.2f})"
    )
    return capped_map


def create_target_occupation_map(
    data_pairs: List[Tuple[str, str]],
    isco_map: Dict[str, str],
) -> Dict[str, Dict[str, str]]:
    """
    Create mapping from full target_doc text to occupation information.

    Each entry:
      target_doc -> {'title', 'description', 'isco_group'}
    """
    print("Creating target occupation map...")

    target_map: Dict[str, Dict[str, str]] = {}
    missing_isco = 0

    for _, target_doc in data_pairs:
        if target_doc in target_map:
            continue

        # target_doc format:
        #   "esco role: <title> \n description: <description>"
        title_match = re.search(r"esco role: (.*?)\n", target_doc)
        desc_match = re.search(r"description: (.*)$", target_doc, re.DOTALL)

        if title_match and desc_match:
            title = title_match.group(1).strip()
            description = desc_match.group(1).strip()

            title_normalized = title.lower()
            isco_group = isco_map.get(title_normalized, "unknown")
            if isco_group == "unknown":
                missing_isco += 1

            target_map[target_doc] = {
                "title": title,
                "description": description,
                "isco_group": isco_group,
            }

    print(f"  ✓ Created map for {len(target_map)} target occupations")
    if missing_isco > 0:
        print(f"  ⚠️  {missing_isco} occupations missing ISCO group")

    return target_map


# ============================================================================
# POOLING UTILITIES
# ============================================================================


def pool_skills_with_idf(
    skill_embeddings: np.ndarray,
    idf_weights: np.ndarray,
) -> np.ndarray:
    """
    Pool skill embeddings using IDF weights (normalized weighted sum).
    """
    if len(skill_embeddings) == 0:
        return None

    if np.sum(idf_weights) > 0:
        normalized_weights = idf_weights / np.sum(idf_weights)
    else:
        normalized_weights = np.ones(len(idf_weights)) / len(idf_weights)

    pooled = np.sum(skill_embeddings * normalized_weights[:, np.newaxis], axis=0)
    return pooled


def pool_jobs_with_log_decay(
    job_vectors: List[np.ndarray],
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Pool job vectors using logarithmic position weighting or mean pooling.

    - If alpha == 0: mean pooling (uniform weights)
    - If alpha > 0: w_i = log(1 + alpha * i), last job gets highest weight
    """
    if not job_vectors:
        return None

    n_jobs = len(job_vectors)

    if alpha == 0:
        weights = np.ones(n_jobs) / n_jobs
    else:
        weights = np.array([np.log(1 + alpha * i) for i in range(n_jobs)])
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(n_jobs) / n_jobs

    job_matrix = np.stack(job_vectors, axis=0)  # [n_jobs, dim]
    pooled = np.sum(job_matrix * weights[:, np.newaxis], axis=0)
    return pooled


# ============================================================================
# BATCH PROCESSING (USED BY TRAINING SCRIPTS)
# ============================================================================


def process_career_path_batch(
    batch: Dict[str, Any],
    skill_desc_map: Dict[str, Dict[str, str]],
    encoder: SentenceTransformer,
    alpha: float = 0.5,
    use_skill_description: bool = True,
    device: torch.device = None,
    precomputed_skill_embeddings: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Any], List[str]]:
    """
    Convert a batch from SkillBasedCareerPathDataset into:
      - career_path_embeddings: pooled skill-based vectors per career
      - target_texts: ESCO target occupation texts

    This preserves:
      - IDF-weighted pooling across skills in a job
      - Logarithmic position pooling across jobs in the career
    """
    batch_size = len(batch["job_skills_list"])
    career_path_embeddings: List[Any] = []
    target_texts: List[str] = []

    # Fast path: use precomputed skill embeddings (one vector per skillUri)
    if precomputed_skill_embeddings is not None:
        career_job_vectors: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]

        example_tensor = next(iter(precomputed_skill_embeddings.values()))
        embed_device = example_tensor.device

        for career_idx in range(batch_size):
            job_skills_list = batch["job_skills_list"][career_idx]

            for skill_info_list in job_skills_list:
                if not skill_info_list:
                    continue

                job_skill_embeds: List[torch.Tensor] = []
                job_idf_weights: List[float] = []

                for skill_dict in skill_info_list:
                    skill_uri = skill_dict["skillUri"]
                    if skill_uri in precomputed_skill_embeddings:
                        job_skill_embeds.append(precomputed_skill_embeddings[skill_uri])
                        job_idf_weights.append(float(skill_dict.get("idf", 1.0)))

                if job_skill_embeds:
                    job_skill_embeds_tensor = torch.stack(job_skill_embeds, dim=0).to(
                        embed_device
                    )
                    job_idf_weights_tensor = torch.tensor(
                        job_idf_weights,
                        dtype=torch.float32,
                        device=embed_device,
                    )

                    if torch.sum(job_idf_weights_tensor) > 0:
                        normalized_weights = (
                            job_idf_weights_tensor / torch.sum(job_idf_weights_tensor)
                        )
                    else:
                        normalized_weights = (
                            torch.ones_like(job_idf_weights_tensor)
                            / len(job_idf_weights_tensor)
                        )

                    pooled_job = torch.sum(
                        job_skill_embeds_tensor * normalized_weights.unsqueeze(-1),
                        dim=0,
                    )
                    career_job_vectors[career_idx].append(pooled_job)
    else:
        # Encode skill texts on the fly (slower but keeps skill side trainable)
        all_skill_texts: List[str] = []
        all_skill_idf_weights: List[float] = []
        skill_to_job_mapping: List[Dict[str, int]] = []

        for career_idx in range(batch_size):
            job_skills_list = batch["job_skills_list"][career_idx]

            for job_idx, skill_info_list in enumerate(job_skills_list):
                if not skill_info_list:
                    continue

                job_start_idx = len(all_skill_texts)

                for skill_dict in skill_info_list:
                    skill_uri = skill_dict["skillUri"]
                    if skill_uri in skill_desc_map:
                        skill_info = skill_desc_map[skill_uri]
                        if use_skill_description:
                            text = (
                                f"skill: {skill_info['name']} \n "
                                f"description: {skill_info['description']}"
                            )
                        else:
                            text = f"skill: {skill_info['name']}"

                        all_skill_texts.append(text)
                        all_skill_idf_weights.append(skill_dict.get("idf", 1.0))

                job_end_idx = len(all_skill_texts)
                if job_end_idx > job_start_idx:
                    skill_to_job_mapping.append(
                        {
                            "career_idx": career_idx,
                            "job_idx": job_idx,
                            "start": job_start_idx,
                            "end": job_end_idx,
                        }
                    )

        if all_skill_texts:
            all_skill_embeddings = encoder.encode(
                all_skill_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                device=device,
                batch_size=128,
            )
            all_skill_idf_weights_arr = np.array(all_skill_idf_weights)
        else:
            all_skill_embeddings = np.array([])
            all_skill_idf_weights_arr = np.array([])

        career_job_vectors_np: List[List[np.ndarray]] = [[] for _ in range(batch_size)]

        for mapping in skill_to_job_mapping:
            career_idx = mapping["career_idx"]
            start = mapping["start"]
            end = mapping["end"]

            job_skill_embeds = all_skill_embeddings[start:end]
            job_idf_weights = all_skill_idf_weights_arr[start:end]

            job_vector = pool_skills_with_idf(job_skill_embeds, job_idf_weights)
            if job_vector is not None:
                career_job_vectors_np[career_idx].append(job_vector)

        career_job_vectors = career_job_vectors_np  # type: ignore[assignment]

    # Pool jobs with logarithmic decay
    embed_dim = encoder.get_sentence_embedding_dimension()
    for career_idx in range(batch_size):
        job_vectors = career_job_vectors[career_idx]

        if job_vectors:
            if precomputed_skill_embeddings is not None:
                n_jobs = len(job_vectors)
                device_for_jobs = job_vectors[0].device
                if alpha == 0:
                    weights = torch.ones(n_jobs, device=device_for_jobs) / n_jobs
                else:
                    positions = torch.arange(
                        n_jobs, dtype=torch.float32, device=device_for_jobs
                    )
                    weights = torch.log1p(alpha * positions)
                    if torch.sum(weights) > 0:
                        weights = weights / torch.sum(weights)
                    else:
                        weights = torch.ones_like(weights) / n_jobs

                job_matrix = torch.stack(job_vectors, dim=0)
                pooled_career = torch.sum(
                    job_matrix * weights.unsqueeze(-1), dim=0
                )
                career_path_embeddings.append(pooled_career)
            else:
                career_embedding = pool_jobs_with_log_decay(job_vectors, alpha)
                career_path_embeddings.append(career_embedding)
        else:
            if precomputed_skill_embeddings is not None:
                zero_vec = torch.zeros(embed_dim, device=embed_device)
                career_path_embeddings.append(zero_vec)
            else:
                career_path_embeddings.append(np.zeros(embed_dim))

        target_text = (
            f"role: {batch['target_titles'][career_idx]} \n "
            f"description: {batch['target_descriptions'][career_idx]}"
        )
        target_texts.append(target_text)

    return career_path_embeddings, target_texts


# ============================================================================
# SKILL EMBEDDING PRECOMPUTATION
# ============================================================================


def precompute_skill_embeddings(
    job_skill_map: Dict[str, List[Dict]],
    skill_desc_map: Dict[str, Dict[str, str]],
    encoder: SentenceTransformer,
    use_skill_description: bool = True,
    device: torch.device = None,
    batch_size: int = 1024,
) -> Dict[str, Any]:
    """
    Precompute one embedding per unique skillUri used in job_skill_map.

    This is useful when you want the skill-side to be fixed text embeddings,
    and only train the mapping from pooled skills to ESCO targets.
    """
    unique_skill_uris: List[str] = []
    seen = set()
    for skills in job_skill_map.values():
        for skill_dict in skills:
            uri = skill_dict["skillUri"]
            if uri not in seen and uri in skill_desc_map:
                seen.add(uri)
                unique_skill_uris.append(uri)

    if not unique_skill_uris:
        return {}

    skill_texts: List[str] = []
    for uri in unique_skill_uris:
        info = skill_desc_map[uri]
        if use_skill_description:
            text = f"skill: {info['name']} \n description: {info['description']}"
        else:
            text = f"skill: {info['name']}"
        skill_texts.append(text)

    skill_embeddings = encoder.encode(
        skill_texts,
        convert_to_numpy=False,
        show_progress_bar=True,
        device=device,
        batch_size=batch_size,
    )

    uri_to_emb: Dict[str, Any] = {}
    for i, uri in enumerate(unique_skill_uris):
        uri_to_emb[uri] = skill_embeddings[i]

    return uri_to_emb


def process_career_path_batch_train(
    batch: Dict[str, Any],
    skill_desc_map: Dict[str, Dict[str, str]],
    encoder: SentenceTransformer,
    alpha: float = 0.5,
    use_skill_description: bool = True,
    device: torch.device = None,
    precomputed_skill_embeddings: Optional[Dict[str, Any]] = None,
) -> Tuple[List[torch.Tensor], List[str]]:
    """
    Training-time variant of career-path processing.

    Differences from process_career_path_batch:
      - When precomputed_skill_embeddings is None, skill texts are encoded
        via a full model forward pass that preserves gradients (no NumPy).
      - All returned career_path_embeddings are torch.Tensors so gradients
        can flow back to the encoder from the skill side.
    """
    batch_size = len(batch["job_skills_list"])
    career_path_embeddings: List[torch.Tensor] = []
    target_texts: List[str] = []

    # Always store job-level vectors as torch tensors
    career_job_vectors: List[List[torch.Tensor]] = [[] for _ in range(batch_size)]

    # ------------------------------------------------------------------
    # 1) Fast path: use precomputed skill embeddings (already torch)
    # ------------------------------------------------------------------
    if precomputed_skill_embeddings is not None:
        example_tensor = next(iter(precomputed_skill_embeddings.values()))
        embed_device = example_tensor.device

        for career_idx in range(batch_size):
            job_skills_list = batch["job_skills_list"][career_idx]

            for skill_info_list in job_skills_list:
                if not skill_info_list:
                    continue

                job_skill_embeds: List[torch.Tensor] = []
                job_idf_weights: List[float] = []

                for skill_dict in skill_info_list:
                    skill_uri = skill_dict["skillUri"]
                    if skill_uri in precomputed_skill_embeddings:
                        job_skill_embeds.append(precomputed_skill_embeddings[skill_uri])
                        job_idf_weights.append(float(skill_dict.get("idf", 1.0)))

                if job_skill_embeds:
                    job_skill_embeds_tensor = torch.stack(job_skill_embeds, dim=0).to(
                        embed_device
                    )
                    job_idf_weights_tensor = torch.tensor(
                        job_idf_weights,
                        dtype=torch.float32,
                        device=embed_device,
                    )

                    if torch.sum(job_idf_weights_tensor) > 0:
                        normalized_weights = (
                            job_idf_weights_tensor / torch.sum(job_idf_weights_tensor)
                        )
                    else:
                        normalized_weights = (
                            torch.ones_like(job_idf_weights_tensor)
                            / len(job_idf_weights_tensor)
                        )

                    pooled_job = torch.sum(
                        job_skill_embeds_tensor * normalized_weights.unsqueeze(-1),
                        dim=0,
                    )
                    career_job_vectors[career_idx].append(pooled_job)

    # ------------------------------------------------------------------
    # 2) Grad-enabled path: encode skill texts on-the-fly with model(...)
    # ------------------------------------------------------------------
    else:
        all_skill_texts: List[str] = []
        all_skill_idf_weights: List[float] = []
        skill_to_job_mapping: List[Dict[str, int]] = []

        for career_idx in range(batch_size):
            job_skills_list = batch["job_skills_list"][career_idx]

            for job_idx, skill_info_list in enumerate(job_skills_list):
                if not skill_info_list:
                    continue

                job_start_idx = len(all_skill_texts)

                for skill_dict in skill_info_list:
                    skill_uri = skill_dict["skillUri"]
                    if skill_uri in skill_desc_map:
                        skill_info = skill_desc_map[skill_uri]
                        if use_skill_description:
                            text = (
                                f"skill: {skill_info['name']} \n "
                                f"description: {skill_info['description']}"
                            )
                        else:
                            text = f"skill: {skill_info['name']}"

                        all_skill_texts.append(text)
                        all_skill_idf_weights.append(skill_dict.get("idf", 1.0))

                job_end_idx = len(all_skill_texts)
                if job_end_idx > job_start_idx:
                    skill_to_job_mapping.append(
                        {
                            "career_idx": career_idx,
                            "job_idx": job_idx,
                            "start": job_start_idx,
                            "end": job_end_idx,
                        }
                    )

        if all_skill_texts:
            # Encode all skills in one model forward pass with gradients
            features = encoder.tokenize(all_skill_texts)
            if device is not None:
                features = {k: v.to(device) for k, v in features.items()}
            outputs = encoder(features)
            all_skill_embeddings = outputs["sentence_embedding"]  # [N, dim]
            all_skill_idf_weights_tensor = torch.tensor(
                all_skill_idf_weights,
                dtype=torch.float32,
                device=all_skill_embeddings.device,
            )
        else:
            all_skill_embeddings = None
            all_skill_idf_weights_tensor = None

        if all_skill_embeddings is not None:
            for mapping in skill_to_job_mapping:
                career_idx = mapping["career_idx"]
                start = mapping["start"]
                end = mapping["end"]

                job_skill_embeds = all_skill_embeddings[start:end]  # [n_skills, dim]
                job_idf_weights = all_skill_idf_weights_tensor[start:end]  # [n_skills]

                if torch.sum(job_idf_weights) > 0:
                    normalized_weights = job_idf_weights / torch.sum(job_idf_weights)
                else:
                    normalized_weights = (
                        torch.ones_like(job_idf_weights) / len(job_idf_weights)
                    )

                pooled_job = torch.sum(
                    job_skill_embeds * normalized_weights.unsqueeze(-1), dim=0
                )
                career_job_vectors[career_idx].append(pooled_job)

    # ------------------------------------------------------------------
    # 3) Pool jobs with logarithmic decay (always in torch)
    # ------------------------------------------------------------------
    embed_dim = encoder.get_sentence_embedding_dimension()
    for career_idx in range(batch_size):
        job_vectors = career_job_vectors[career_idx]

        if job_vectors:
            n_jobs = len(job_vectors)
            device_for_jobs = job_vectors[0].device

            if alpha == 0:
                weights = torch.ones(n_jobs, device=device_for_jobs) / n_jobs
            else:
                positions = torch.arange(
                    n_jobs, dtype=torch.float32, device=device_for_jobs
                )
                weights = torch.log1p(alpha * positions)
                if torch.sum(weights) > 0:
                    weights = weights / torch.sum(weights)
                else:
                    weights = torch.ones_like(weights) / n_jobs

            job_matrix = torch.stack(job_vectors, dim=0)  # [n_jobs, dim]
            pooled_career = torch.sum(job_matrix * weights.unsqueeze(-1), dim=0)
            career_path_embeddings.append(pooled_career)
        else:
            # No valid skills → zero vector on requested device (or CPU fallback)
            target_device = device if device is not None else torch.device("cpu")
            zero_vec = torch.zeros(embed_dim, device=target_device)
            career_path_embeddings.append(zero_vec)

        target_text = (
            f"role: {batch['target_titles'][career_idx]} \n "
            f"description: {batch['target_descriptions'][career_idx]}"
        )
        target_texts.append(target_text)

    return career_path_embeddings, target_texts

