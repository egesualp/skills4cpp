"""
Shared utilities for skill-based pooling and ESCO data loading.

These functions implement:
- Loading job->skill mappings and ESCO skill/occupation information
- IDF weighting per skill
- Skill importance scores from fused scorer predictions
- Logarithmic pooling over jobs in a career path
- Batch processing from SkillBasedCareerPathDataset-style batches

They are used by training scripts such as train_cpp_skills_v3.py and can
also be reused by other modules that need the same pooling behaviour.
"""

from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
import re
import random
import math
import json

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


def load_fused_skill_scores(
    fused_predictions_json: str,
    decorte_map_csv: str,
    aggregation: str = "max",
) -> Dict[str, Dict[str, float]]:
    """
    Load skill importance scores from fused scorer predictions.
    
    Aggregates skill scores per ESCO occupation title across all jobs
    in the decorte dataset that share the same occupation.
    
    Args:
        fused_predictions_json: Path to fused_predictions.json
            Format: {job_id: [[skill_uri, score], ...], ...}
        decorte_map_csv: Path to decorte_master CSV with job_id -> esco_title mapping
        aggregation: How to aggregate scores across jobs with same occupation
            - "max": Use maximum score for each skill
            - "mean": Use mean score for each skill
    
    Returns:
        Dict mapping normalized occupation title to dict of {skill_uri: score}
    """
    print(f"Loading fused skill scores from {fused_predictions_json}...")
    
    # Load fused predictions
    with open(fused_predictions_json, 'r') as f:
        fused_preds = json.load(f)
    
    # Load decorte mapping
    decorte_df = pd.read_csv(decorte_map_csv)
    job_to_occ = {}
    for _, row in decorte_df.iterrows():
        job_id = str(row["job_id"])
        esco_title = str(row["esco_title"]).strip().lower()
        job_to_occ[job_id] = esco_title
    
    # Aggregate skill scores per occupation
    occ_skill_scores: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    
    for job_id, skill_list in fused_preds.items():
        job_id_str = str(job_id)
        if job_id_str not in job_to_occ:
            continue
        
        occ_title = job_to_occ[job_id_str]
        for skill_uri, score in skill_list:
            occ_skill_scores[occ_title][skill_uri].append(float(score))
    
    # Aggregate to final scores
    result: Dict[str, Dict[str, float]] = {}
    for occ_title, skill_dict in occ_skill_scores.items():
        result[occ_title] = {}
        for skill_uri, scores in skill_dict.items():
            if aggregation == "max":
                result[occ_title][skill_uri] = max(scores)
            elif aggregation == "mean":
                result[occ_title][skill_uri] = sum(scores) / len(scores)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
    
    total_skills = sum(len(skills) for skills in result.values())
    print(f"  ✓ Loaded fused scores for {len(result)} occupations ({total_skills} skill entries)")
    return result


def load_global_skill_scores(
    fused_predictions_json: str,
    aggregation: str = "max",
) -> Dict[str, float]:
    """
    Load global skill importance scores from fused scorer predictions.
    
    Aggregates skill scores across ALL jobs to get a single importance
    score per skill URI.
    
    Args:
        fused_predictions_json: Path to fused_predictions.json
        aggregation: How to aggregate scores across all jobs
            - "max": Use maximum score for each skill
            - "mean": Use mean score for each skill
    
    Returns:
        Dict mapping skill_uri to global importance score
    """
    print(f"Loading global skill scores from {fused_predictions_json}...")
    
    with open(fused_predictions_json, 'r') as f:
        fused_preds = json.load(f)
    
    skill_scores: Dict[str, List[float]] = defaultdict(list)
    
    for job_id, skill_list in fused_preds.items():
        for skill_uri, score in skill_list:
            skill_scores[skill_uri].append(float(score))
    
    result: Dict[str, float] = {}
    for skill_uri, scores in skill_scores.items():
        if aggregation == "max":
            result[skill_uri] = max(scores)
        elif aggregation == "mean":
            result[skill_uri] = sum(scores) / len(scores)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    
    print(f"  ✓ Loaded global scores for {len(result)} unique skills")
    return result


def load_skills_by_job_id(
    fused_predictions_json: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load skills per job_id directly from fused_predictions.json.
    
    This function returns skills indexed by job_id (not by job title),
    preserving the prediction scores for each skill.
    
    Args:
        fused_predictions_json: Path to fused_predictions.json
            Format: {job_id: [[skill_uri, score], ...], ...}
    
    Returns:
        Dict mapping job_id (as string) to list of skill dicts:
        [{'skillUri': str, 'score': float}, ...]
        Skills are sorted by score descending.
    """
    print(f"Loading skills by job_id from {fused_predictions_json}...")
    
    # Special handling for karrierewege_cp dataset (kw_cp) which uses JSONL format
    # Other data sources use regular JSON format
    is_karrierewege_cp = 'kw_cp' in fused_predictions_json and fused_predictions_json.endswith('.jsonl')
    
    if is_karrierewege_cp:
        print(f"  > Detected karrierewege_cp JSONL format")
        fused_preds = {}
        with open(fused_predictions_json, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    # Handle format: {"job_id": "...", "predictions": [...]}
                    job_id = str(record.get('job_id', record.get('id', '')))
                    predictions = record.get('predictions', record.get('skills', []))
                    if job_id:
                        fused_preds[job_id] = predictions
                except json.JSONDecodeError as e:
                    print(f"  ⚠️ Warning: Could not parse line {line_num}: {e}")
                    continue
    else:
        # Regular JSON format for other data sources
        with open(fused_predictions_json, 'r') as f:
            fused_preds = json.load(f)
    
    job_skill_map: Dict[str, List[Dict[str, Any]]] = {}
    total_skills = 0
    
    for job_id, skill_list in fused_preds.items():
        job_id_str = str(job_id)
        skills = []
        for skill_item in skill_list:
            # Handle both formats:
            # 1. List/tuple format: [skill_uri, score]
            # 2. Dict format: {"skill_uri": "...", "score": ...}
            if isinstance(skill_item, (list, tuple)):
                skill_uri = skill_item[0]
                score = skill_item[1] if len(skill_item) > 1 else 1.0
            elif isinstance(skill_item, dict):
                skill_uri = skill_item.get('skill_uri') or skill_item.get('skillUri')
                score = skill_item.get('score', 1.0)
            else:
                # Fallback for unexpected types (or if unpacking logic was previously used)
                continue

            if skill_uri:
                skills.append({
                    "skillUri": skill_uri,
                    "score": float(score),
                })
        # Sort by score descending (highest first)
        skills.sort(key=lambda s: -s["score"])
        job_skill_map[job_id_str] = skills
        total_skills += len(skills)
    
    print(f"  ✓ Loaded skills for {len(job_skill_map)} job_ids ({total_skills} total skill entries)")
    return job_skill_map


def load_raw_esco_taxonomy(esco_dir: str) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict[str, str]]]:
    """
    Load and join raw ESCO taxonomy files from a directory.
    
    Required files in esco_dir:
    - occupations_en.csv: Maps occupationUri to preferredLabel
    - skills_en.csv: Maps skillUri to preferredLabel and description
    - occupationSkillRelations_en.csv: Maps occupationUri to skillUri
    
    Returns:
        Tuple containing:
        1. job_skill_map: Dict mapping normalized job_title to list of skill dicts
        2. skill_desc_map: Dict mapping skillUri to {'name', 'description'}
    """
    import os
    print(f"Loading raw ESCO taxonomy from {esco_dir}...")
    
    occ_file = os.path.join(esco_dir, "occupations_en.csv")
    skills_file = os.path.join(esco_dir, "skills_en.csv")
    rel_file = os.path.join(esco_dir, "occupationSkillRelations_en.csv")
    
    for f in [occ_file, skills_file, rel_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Required ESCO file not found: {f}")
            
    # 1. Load Skills and Metadata
    print("  - Loading skill descriptions...")
    skills_df = pd.read_csv(skills_file)
    skill_desc_map: Dict[str, Dict[str, str]] = {}
    for _, row in skills_df.iterrows():
        uri = row["conceptUri"]
        skill_desc_map[uri] = {
            "name": row["preferredLabel"],
            "description": row.get("description", "") if pd.notna(row.get("description")) else ""
        }
    
    # 2. Load Occupations to get Titles
    print("  - Loading occupation titles...")
    occ_df = pd.read_csv(occ_file)
    uri_to_title = {}
    for _, row in occ_df.iterrows():
        uri = row["conceptUri"]
        title = str(row["preferredLabel"]).strip().lower()
        uri_to_title[uri] = title
        
    # 3. Load Relations and Join
    print("  - Joining occupations and skills...")
    rel_df = pd.read_csv(rel_file)
    job_skill_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    for _, row in rel_df.iterrows():
        occ_uri = row["occupationUri"]
        skill_uri = row["skillUri"]
        
        if occ_uri in uri_to_title:
            job_title = uri_to_title[occ_uri]
            skill_name = skill_desc_map[skill_uri]["name"] if skill_uri in skill_desc_map else "Unknown Skill"
            
            job_skill_map[job_title].append({
                "skill": skill_name,
                "skillUri": skill_uri,
                "score": 1.0  # Taxonomy relations are unweighted
            })
            
    print(f"  ✓ Processed {len(job_skill_map)} occupations and {len(skill_desc_map)} skills")
    return dict(job_skill_map), skill_desc_map


def calculate_idf_scores_by_job_id(
    job_skill_map: Dict[str, List[Dict]],
    use_job_scores: bool = False,
    importance_weight: float = 0.5,
) -> Dict[str, List[Dict]]:
    """
    Calculate IDF scores for skills indexed by job_id.
    
    Optionally combines IDF with the per-job prediction scores already present
    in each skill dict (from load_skills_by_job_id).

    IDF = log(total_jobs / job_count_with_skill)
    
    If use_job_scores is True:
        weighted_idf = (1 - importance_weight) * normalized_idf + importance_weight * job_score
        where job_score is the 'score' field already in each skill dict (per-job, not global)
    
    Adds 'idf' and optionally 'weighted_idf' fields to each skill dict.
    
    Args:
        job_skill_map: Dict mapping job_id to list of skill dicts (from load_skills_by_job_id)
            Each skill dict should have 'skillUri' and optionally 'score' (prediction score)
        use_job_scores: If True, combine IDF with per-job prediction scores
        importance_weight: Weight for job scores in weighted_idf (0-1)
            - 0.0: Use only IDF
            - 1.0: Use only job scores
            - 0.5: Equal weight (default)
    """
    print("Calculating IDF scores for skills (by job_id)...")

    # Count in how many jobs each skill appears
    skill_job_count: Dict[str, int] = defaultdict(int)
    for _, skills in job_skill_map.items():
        seen_skills = set()
        for skill_dict in skills:
            skill_uri = skill_dict["skillUri"]
            if skill_uri not in seen_skills:
                skill_job_count[skill_uri] += 1
                seen_skills.add(skill_uri)

    total_jobs = len(job_skill_map)
    skill_idf: Dict[str, float] = {}
    for skill_uri, count in skill_job_count.items():
        if count > 0:
            skill_idf[skill_uri] = float(np.log(total_jobs / count))
        else:
            skill_idf[skill_uri] = 0.0

    # Normalize IDF scores to 0-1 range for combining with job scores
    idf_values = list(skill_idf.values())
    if idf_values:
        idf_min = min(idf_values)
        idf_max = max(idf_values)
        idf_range = idf_max - idf_min if idf_max > idf_min else 1.0
    else:
        idf_min, idf_range = 0.0, 1.0
    
    skill_idf_normalized: Dict[str, float] = {}
    for skill_uri, idf_val in skill_idf.items():
        skill_idf_normalized[skill_uri] = (idf_val - idf_min) / idf_range

    # Attach IDF (and optionally weighted_idf) to each skill entry
    updated_map: Dict[str, List[Dict]] = {}
    skills_with_scores = 0
    
    for job_id, skills in job_skill_map.items():
        updated_skills: List[Dict[str, Any]] = []
        for skill_dict in skills:
            skill_uri = skill_dict["skillUri"]
            new_dict = dict(skill_dict)
            new_dict["idf"] = skill_idf.get(skill_uri, 0.0)
            
            if use_job_scores:
                idf_norm = skill_idf_normalized.get(skill_uri, 0.0)
                # Use the per-job score that's already in the skill dict
                job_score = float(skill_dict.get("score", 0.0))
                
                if job_score > 0:
                    skills_with_scores += 1
                
                # Combine normalized IDF with per-job prediction score
                weighted = (1 - importance_weight) * idf_norm + importance_weight * job_score
                new_dict["weighted_idf"] = weighted
            
            updated_skills.append(new_dict)
        updated_map[job_id] = updated_skills

    print(f"  ✓ Calculated IDF for {len(skill_idf)} unique skills across {total_jobs} jobs")
    if use_job_scores:
        print(f"  ✓ Combined with per-job prediction scores (weight={importance_weight})")
        print(f"  ✓ Skills with scores: {skills_with_scores}")
    
    return updated_map


def cap_skills_per_job_by_score(
    job_skill_map: Dict[str, List[Dict[str, Any]]],
    max_skills_per_job: int,
    skill_desc_map: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Cap skills per job using only the prediction score (scores_only mode).
    
    Skills are sorted by their 'score' field (prediction score from fused scorer)
    and the top-k are kept.
    
    Args:
        job_skill_map: Dict mapping job_id to list of skill dicts
        max_skills_per_job: Maximum number of skills to keep per job
        skill_desc_map: Optional dict for description length tie-breaking
    """
    if max_skills_per_job is None or max_skills_per_job <= 0:
        return job_skill_map
    
    capped_map: Dict[str, List[Dict[str, Any]]] = {}
    total_before = 0
    total_after = 0

    for job_id, skills in job_skill_map.items():
        total_before += len(skills)
        if len(skills) <= max_skills_per_job:
            capped_skills = skills
        else:
            # Sort by:
            #   1) -score (highest prediction score first)
            #   2) -len(description) (longer descriptions first, if available)
            #   3) skillUri lexicographically (deterministic tie-break)
            def sort_key(s: Dict[str, Any]):
                score_val = -float(s.get("score", 0.0))
                desc_len = 0
                if skill_desc_map is not None:
                    uri = s.get("skillUri")
                    if uri in skill_desc_map:
                        desc_text = skill_desc_map[uri].get("description", "") or ""
                        desc_len = len(desc_text)
                uri = str(s.get("skillUri", ""))
                return (score_val, -desc_len, uri)

            capped_skills = sorted(skills, key=sort_key)[:max_skills_per_job]
        total_after += len(capped_skills)
        capped_map[job_id] = capped_skills

    if total_before > 0:
        kept_ratio = total_after / total_before
    else:
        kept_ratio = 1.0

    print(
        f"  ✓ Applied score-only cap of {max_skills_per_job} skills/job "
        f"(avg kept ratio: {kept_ratio:.2f})"
    )
    return capped_map


def calculate_idf_scores(
    job_skill_map: Dict[str, List[Dict]],
    skill_importance_scores: Optional[Dict[str, float]] = None,
    importance_weight: float = 0.5,
) -> Dict[str, List[Dict]]:
    """
    Calculate IDF scores for each skill based on occupation frequency.
    Optionally combines with external skill importance scores.

    IDF = log(total_occupations / occupation_count)
    
    If skill_importance_scores is provided:
        weighted_idf = (1 - importance_weight) * normalized_idf + importance_weight * importance_score
    
    Adds 'idf' and optionally 'weighted_idf' fields to each skill dict.
    
    Args:
        job_skill_map: Dict mapping job_title to list of skill dicts
        skill_importance_scores: Optional dict mapping skill_uri to importance score (0-1)
        importance_weight: Weight for importance scores in weighted_idf (0-1)
            - 0.0: Use only IDF
            - 1.0: Use only importance scores
            - 0.5: Equal weight (default)
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

    # Normalize IDF scores to 0-1 range for combining with importance scores
    idf_values = list(skill_idf.values())
    if idf_values:
        idf_min = min(idf_values)
        idf_max = max(idf_values)
        idf_range = idf_max - idf_min if idf_max > idf_min else 1.0
    else:
        idf_min, idf_range = 0.0, 1.0
    
    skill_idf_normalized: Dict[str, float] = {}
    for skill_uri, idf_val in skill_idf.items():
        skill_idf_normalized[skill_uri] = (idf_val - idf_min) / idf_range

    # Attach IDF (and optionally weighted_idf) to each skill entry
    updated_map: Dict[str, List[Dict]] = {}
    skills_with_importance = 0
    
    for job_title, skills in job_skill_map.items():
        updated_skills: List[Dict[str, Any]] = []
        for skill_dict in skills:
            skill_uri = skill_dict["skillUri"]
            new_dict = dict(skill_dict)
            new_dict["idf"] = skill_idf.get(skill_uri, 0.0)
            
            if skill_importance_scores is not None:
                idf_norm = skill_idf_normalized.get(skill_uri, 0.0)
                importance = skill_importance_scores.get(skill_uri, 0.0)
                
                if importance > 0:
                    skills_with_importance += 1
                
                # Combine normalized IDF with importance score
                weighted = (1 - importance_weight) * idf_norm + importance_weight * importance
                new_dict["weighted_idf"] = weighted
                new_dict["importance_score"] = importance
            
            updated_skills.append(new_dict)
        updated_map[job_title] = updated_skills

    print(f"  ✓ Calculated IDF for {len(skill_idf)} unique skills")
    if skill_importance_scores is not None:
        print(f"  ✓ Combined with importance scores (weight={importance_weight})")
        print(f"  ✓ Skills with importance scores: {skills_with_importance}")
    
    return updated_map


def cap_skills_per_job_lexicographic(
    job_skill_map: Dict[str, List[Dict[str, Any]]],
    max_skills_per_job: int,
    skill_desc_map: Optional[Dict[str, Dict[str, str]]] = None,
    use_weighted_idf: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Deterministically cap the number of skills per job.

    Within each job we:
      1) Sort skills by IDF (or weighted_idf if available and use_weighted_idf=True) descending.
      2) Break ties lexicographically by skill label (field 'skill'),
         using the label that we actually encode (name/description).
      3) Keep the first `max_skills_per_job` skills.

    This gives us:
      - Preference for job-specific / rare skills (via IDF),
      - A stable, reproducible selection within IDF ties (via label).

    If max_skills_per_job <= 0, the map is returned unchanged.
    
    Args:
        job_skill_map: Dict mapping job_title to list of skill dicts
        max_skills_per_job: Maximum number of skills to keep per job
        skill_desc_map: Optional dict for description length tie-breaking
        use_weighted_idf: If True, use 'weighted_idf' field instead of 'idf'
    """
    if max_skills_per_job is None or max_skills_per_job <= 0:
        return job_skill_map

    score_field = "weighted_idf" if use_weighted_idf else "idf"
    
    capped_map: Dict[str, List[Dict[str, Any]]] = {}
    total_before = 0
    total_after = 0

    for job_title, skills in job_skill_map.items():
        total_before += len(skills)
        if len(skills) <= max_skills_per_job:
            capped_skills = skills
        else:
            # Sort by:
            #   1) -score (most important first)
            #   2) -len(description) (longer descriptions first, if available)
            #   3) skill label lexicographically (deterministic tie-break)
            def sort_key(s: Dict[str, Any]):
                # Use weighted_idf if available and requested, else fall back to idf
                if use_weighted_idf and "weighted_idf" in s:
                    score_val = -float(s.get("weighted_idf", 0.0))
                else:
                    score_val = -float(s.get("idf", 0.0))
                # Look up description length if skill_desc_map is provided
                desc_len = 0
                if skill_desc_map is not None:
                    uri = s.get("skillUri")
                    if uri in skill_desc_map:
                        desc_text = skill_desc_map[uri].get("description", "") or ""
                        desc_len = len(desc_text)
                label = str(s.get("skill", "")).lower()
                return (score_val, -desc_len, label)

            capped_skills = sorted(skills, key=sort_key)[:max_skills_per_job]
        total_after += len(capped_skills)
        capped_map[job_title] = capped_skills

    if total_before > 0:
        kept_ratio = total_after / total_before
    else:
        kept_ratio = 1.0

    score_name = "weighted_idf" if use_weighted_idf else "IDF"
    print(
        f"  ✓ Applied {score_name}+lexicographic cap of {max_skills_per_job} skills/job "
        f"(avg kept ratio over all jobs: {kept_ratio:.2f})"
    )
    return capped_map


def cap_skills_per_job_stratified(
    job_skill_map: Dict[str, List[Dict[str, Any]]],
    max_skills_per_job: int,
    use_weighted_idf: bool = False,
    score_source: str = "auto",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Cap skills using stratified sampling based on a chosen score source.

    Strategy for max_skills_per_job=10:
      - High score: 4 skills
      - Mid score: 3 skills
      - Low score: 3 skills
    
    If max_skills_per_job is different, we scale the proportions accordingly.
    
    The sampling within tiers is random to encourage robustness.
    Skills are sorted by score first to define the tiers.
    
    Score source selection:
      - If score_source == "auto" (default), we preserve the legacy behavior:
          1) weighted_idf (if use_weighted_idf=True and field exists)
          2) idf (if field exists)
          3) score (prediction score, fallback for scores_only mode)
      - If score_source is one of {"idf", "weighted_idf", "score"}, we use that
        field explicitly (with minimal fallbacks if missing).
    
    Args:
        job_skill_map: Dict mapping job_id/job_title to list of skill dicts
        max_skills_per_job: Maximum number of skills to keep per job
        use_weighted_idf: If True, use 'weighted_idf' field instead of 'idf'
            for both bucketing and final sorting
        score_source: Which field to use for stratification/sorting:
            - "auto" (default): legacy priority order (see above)
            - "idf": use only 'idf'
            - "weighted_idf": use only 'weighted_idf' (fallback to 'idf'/'score' if missing)
            - "score": use only 'score' (prediction score)
    """
    if max_skills_per_job is None or max_skills_per_job <= 0:
        return job_skill_map

    # Define proportions (approx 40%, 30%, 30%)
    p_high = 0.4
    p_mid = 0.3
    p_low = 0.3
    
    # Calculate target counts
    k_high = int(round(max_skills_per_job * p_high))
    k_mid = int(round(max_skills_per_job * p_mid))
    k_low = max_skills_per_job - k_high - k_mid
    
    capped_map: Dict[str, List[Dict[str, Any]]] = {}
    total_before = 0
    total_after = 0
    
    # Use a fixed seed for reproducibility across runs, 
    # but varying per call if we wanted dynamic sampling. 
    # For dataset construction, usually fixed seed is better.
    rng = random.Random(42)
    
    def get_score(s: Dict[str, Any]) -> float:
        """Get the score to use for sorting/bucketing.
        
        If score_source == "auto": weighted_idf > idf > score (prediction score)
        Else: use the requested field explicitly (with minimal fallbacks).
        """
        if score_source not in ("auto", "idf", "weighted_idf", "score"):
            raise ValueError(
                f"Invalid score_source={score_source!r}. "
                "Expected one of: 'auto', 'idf', 'weighted_idf', 'score'."
            )

        # Explicit modes
        if score_source == "idf":
            # If missing, fall back to score to avoid returning all zeros.
            if "idf" in s:
                return float(s.get("idf", 0.0))
            return float(s.get("score", 0.0))

        if score_source == "weighted_idf":
            if "weighted_idf" in s:
                return float(s.get("weighted_idf", 0.0))
            if "idf" in s:
                return float(s.get("idf", 0.0))
            return float(s.get("score", 0.0))

        if score_source == "score":
            return float(s.get("score", 0.0))

        # Legacy / auto mode
        if use_weighted_idf and "weighted_idf" in s:
            return float(s.get("weighted_idf", 0.0))
        if "idf" in s:
            return float(s.get("idf", 0.0))
        return float(s.get("score", 0.0))

    for job_title, skills in job_skill_map.items():
        total_before += len(skills)
        n_skills = len(skills)
        
        if n_skills <= max_skills_per_job:
            capped_skills = skills
        else:
            # Sort skills by score descending
            sorted_skills = sorted(skills, key=lambda s: -get_score(s))
            
            # Split into 3 buckets
            # We use n_skills to split into roughly equal thirds
            chunk_size = math.ceil(n_skills / 3.0)
            
            high_bucket = sorted_skills[:chunk_size]
            mid_bucket = sorted_skills[chunk_size : 2 * chunk_size]
            low_bucket = sorted_skills[2 * chunk_size :]
            
            selected = []
            
            # Helper to sample without replacement
            def sample_bucket(bucket, k):
                if len(bucket) <= k:
                    return bucket
                return rng.sample(bucket, k)

            # Sample from each bucket
            sel_high = sample_bucket(high_bucket, k_high)
            sel_mid = sample_bucket(mid_bucket, k_mid)
            sel_low = sample_bucket(low_bucket, k_low)
            
            selected.extend(sel_high)
            selected.extend(sel_mid)
            selected.extend(sel_low)
            
            # If we are short (due to small buckets), fill from remaining
            if len(selected) < max_skills_per_job:
                current_uris = {s["skillUri"] for s in selected}
                remaining = [s for s in sorted_skills if s["skillUri"] not in current_uris]
                needed = max_skills_per_job - len(selected)
                if remaining:
                    fill = rng.sample(remaining, min(len(remaining), needed))
                    selected.extend(fill)
            
            # Sort selected by score again for consistency in output format
            capped_skills = sorted(selected, key=lambda s: -get_score(s))
            
        total_after += len(capped_skills)
        capped_map[job_title] = capped_skills

    if total_before > 0:
        kept_ratio = total_after / total_before
    else:
        kept_ratio = 1.0

    # Determine score name for logging based on what fields are present
    if score_source != "auto":
        score_name = score_source
    elif use_weighted_idf:
        score_name = "weighted_idf"
    else:
        # Check first job's skills to see what fields are available
        sample_skills = next(iter(job_skill_map.values()), [])
        if sample_skills and "idf" in sample_skills[0]:
            score_name = "IDF"
        else:
            score_name = "prediction_score"
    
    print(
        f"  ✓ Applied Stratified {score_name} sampling (High:{k_high}, Mid:{k_mid}, Low:{k_low}) "
        f"for cap {max_skills_per_job} (avg kept ratio: {kept_ratio:.2f})"
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
                            text = skill_info['name']

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
            text = info['name']
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
                            text = skill_info['name']

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
