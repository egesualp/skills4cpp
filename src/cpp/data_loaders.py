"""
Utility functions for loading skill data, vocabularies, and helper maps.

These functions handle:
- Loading vocabularies and skill mappings
- Pre-computing and caching raw embeddings (by job title / skill URI)
- Pooling embeddings at runtime (fast once raw embeddings are cached)
"""

import json
import os
import pickle
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional, Set
from loguru import logger
import hashlib


# ============================================================================
# EMBEDDING CACHE UTILITIES
# ============================================================================

def _get_cache_path(cache_dir: str, cache_type: str, encoder_name: str, 
                    use_description: bool = False, extra_suffix: str = "") -> str:
    """Generate a cache file path for embeddings.
    
    Args:
        cache_dir: Directory to store cache files
        cache_type: Type of embeddings ('skills', 'targets', 'jobs')
        encoder_name: Name of the encoder model
        use_description: Whether descriptions are included
        extra_suffix: Additional suffix for the filename
        
    Returns:
        Full path to the cache file
    """
    encoder_short = encoder_name.split('/')[-1]
    desc_suffix = "_with_desc" if use_description else ""
    filename = f"{cache_type}_{encoder_short}{desc_suffix}{extra_suffix}.pkl"
    return os.path.join(cache_dir, filename)


def load_embedding_cache(cache_path: str) -> Optional[Dict[str, np.ndarray]]:
    """Load embeddings from cache file.
    
    Args:
        cache_path: Path to the cache file
        
    Returns:
        Dictionary mapping keys to embeddings, or None if cache doesn't exist
    """
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        cache_size_mb = os.path.getsize(cache_path) / (1024**2)
        logger.info(f"  ✓ Loaded {len(cache)} embeddings from cache ({cache_size_mb:.1f} MB)")
        return cache
    except Exception as e:
        logger.warning(f"  ⚠️ Failed to load cache: {e}")
        return None


def save_embedding_cache(cache_path: str, embeddings: Dict[str, np.ndarray]) -> bool:
    """Save embeddings to cache file.
    
    Args:
        cache_path: Path to save the cache
        embeddings: Dictionary mapping keys to embeddings
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(embeddings, f, protocol=pickle.HIGHEST_PROTOCOL)
        cache_size_mb = os.path.getsize(cache_path) / (1024**2)
        logger.info(f"  ✓ Saved {len(embeddings)} embeddings to cache ({cache_size_mb:.1f} MB)")
        return True
    except Exception as e:
        logger.error(f"  ❌ Failed to save cache: {e}")
        return False


def load_precomputed_skill_embeddings(skill_embeddings_dir: str) -> Dict[str, np.ndarray]:
    """Load precomputed skill embeddings from a directory.
    
    The directory must contain:
    - skill_embeddings.npy: Numpy array of shape [num_skills, embedding_dim]
    - skill_metadata.json: JSON with model_name, num_skills, embedding_dim, and skills list
    
    Each skill in the metadata must have 'conceptUri' (the skill URI) and 'preferredLabel'.
    
    Args:
        skill_embeddings_dir: Path to directory containing skill_embeddings.npy and skill_metadata.json
        
    Returns:
        Dictionary mapping skill URIs (conceptUri) to their embeddings
        
    Raises:
        FileNotFoundError: If required files are missing
        ValueError: If metadata doesn't match embeddings shape
    """
    embeddings_path = os.path.join(skill_embeddings_dir, "skill_embeddings.npy")
    metadata_path = os.path.join(skill_embeddings_dir, "skill_metadata.json")
    
    # Check files exist
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"skill_embeddings.npy not found in {skill_embeddings_dir}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"skill_metadata.json not found in {skill_embeddings_dir}")
    
    # Load metadata
    logger.info(f"  > Loading skill metadata from {metadata_path}...")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    model_name = metadata.get("model_name", "unknown")
    num_skills = metadata["num_skills"]
    embedding_dim = metadata["embedding_dim"]
    skills_list = metadata["skills"]
    
    logger.info(f"    Model: {model_name}")
    logger.info(f"    Skills: {num_skills}, Embedding dim: {embedding_dim}")
    
    # Load embeddings
    logger.info(f"  > Loading skill embeddings from {embeddings_path}...")
    embeddings = np.load(embeddings_path)
    
    # Validate shape
    if embeddings.shape[0] != num_skills:
        raise ValueError(f"Embeddings shape {embeddings.shape} doesn't match metadata num_skills={num_skills}")
    if embeddings.shape[1] != embedding_dim:
        raise ValueError(f"Embeddings dim {embeddings.shape[1]} doesn't match metadata embedding_dim={embedding_dim}")
    
    # Build URI -> embedding map
    skill_embedding_map = {}
    for idx, skill_info in enumerate(skills_list):
        uri = skill_info["conceptUri"]
        skill_embedding_map[uri] = embeddings[idx].astype(np.float32)
    
    logger.info(f"  ✓ Loaded {len(skill_embedding_map)} precomputed skill embeddings")
    
    return skill_embedding_map


def get_or_compute_skill_embeddings(
    unique_skill_uris: Set[str],
    encoder_skill,
    esco_skill_text_map: Dict[str, Dict],
    use_skill_description: bool,
    cache_dir: Optional[str] = None,
    encoder_name: Optional[str] = None,
    force_recompute: bool = False,
    use_skill_prefix: bool = False,
) -> Dict[str, np.ndarray]:
    """Get or compute embeddings for skills, with optional caching.
    
    This function:
    1. Checks if a cache exists with all needed skills
    2. If yes, returns cached embeddings (filtered to requested skills)
    3. If no, computes embeddings and optionally saves to cache
    
    Args:
        unique_skill_uris: Set of skill URIs to encode
        encoder_skill: Encoder model for skills
        esco_skill_text_map: Map from skill URIs to text
        use_skill_description: Whether to include skill descriptions
        cache_dir: Directory to cache embeddings (None = no caching)
        encoder_name: Name of the encoder (for cache filename)
        force_recompute: Force recomputation even if cache exists
        use_skill_prefix: Use "skill: ..." prefix (for skill-specific encoders)
        
    Returns:
        Dictionary mapping skill URIs to embeddings
    """
    cache_path = None
    cached_embeddings = None
    
    # Try to load from cache
    if cache_dir and encoder_name and not force_recompute:
        prefix_suffix = "_skill_prefix" if use_skill_prefix else ""
        cache_path = _get_cache_path(cache_dir, "skills", encoder_name, 
                                     use_skill_description, prefix_suffix)
        cached_embeddings = load_embedding_cache(cache_path)
    
    # Check if cache covers all needed skills
    if cached_embeddings is not None:
        missing_skills = unique_skill_uris - set(cached_embeddings.keys())
        if not missing_skills:
            # All skills are cached, filter and return
            logger.info(f"  ✓ All {len(unique_skill_uris)} skills found in cache")
            return {uri: cached_embeddings[uri] for uri in unique_skill_uris 
                    if uri in cached_embeddings}
        else:
            logger.info(f"  > Cache has {len(cached_embeddings)} skills, need {len(missing_skills)} more")
            # Will compute missing skills and merge
    else:
        missing_skills = unique_skill_uris
        cached_embeddings = {}
    
    # Compute embeddings for missing skills
    if missing_skills:
        logger.info(f"  > Encoding {len(missing_skills)} skills...")
        skill_texts = []
        skill_uris_ordered = []
        
        for uri in missing_skills:
            if uri in esco_skill_text_map:
                st = esco_skill_text_map[uri]
                if use_skill_prefix:
                    text = (f"skill: {st['name']}\ndescription: {st['desc']}"
                            if use_skill_description else f"skill: {st['name']}")
                else:
                    text = (f"role: {st['name']}\ndescription: {st['desc']}"
                            if use_skill_description else st['name'])
                skill_texts.append(text)
                skill_uris_ordered.append(uri)
        
        if skill_texts:
            new_embeddings = encoder_skill.encode(
                skill_texts, convert_to_numpy=True, 
                show_progress_bar=True, batch_size=512
            )
            for uri, emb in zip(skill_uris_ordered, new_embeddings):
                cached_embeddings[uri] = emb.astype(np.float32)
    
    # Save updated cache
    if cache_path and missing_skills:
        save_embedding_cache(cache_path, cached_embeddings)
    
    # Return only the requested skills
    return {uri: cached_embeddings[uri] for uri in unique_skill_uris 
            if uri in cached_embeddings}


def get_or_compute_target_embeddings(
    target_labels: List[str],
    encoder,
    cache_dir: Optional[str] = None,
    encoder_name: Optional[str] = None,
    force_recompute: bool = False,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Get or compute embeddings for target job labels, with optional caching.
    
    Args:
        target_labels: List of unique target job strings
        encoder: SentenceTransformer encoder model
        cache_dir: Directory to cache embeddings (None = no caching)
        encoder_name: Name of the encoder (for cache filename)
        force_recompute: Force recomputation even if cache exists
        
    Returns:
        Tuple of:
        - Dictionary mapping target strings to their embeddings
        - Numpy array of all target embeddings (for similarity calculations)
    """
    cache_path = None
    cached_embeddings = None
    
    unique_labels = list(set(target_labels))
    
    # Try to load from cache
    if cache_dir and encoder_name and not force_recompute:
        cache_path = _get_cache_path(cache_dir, "targets", encoder_name)
        cached_embeddings = load_embedding_cache(cache_path)
    
    # Check if cache covers all needed targets
    if cached_embeddings is not None:
        missing_labels = set(unique_labels) - set(cached_embeddings.keys())
        if not missing_labels:
            logger.info(f"  ✓ All {len(unique_labels)} targets found in cache")
            Y_target_dict = {label: cached_embeddings[label] for label in unique_labels}
            Y_target_all = np.array([Y_target_dict[label] for label in unique_labels])
            return Y_target_dict, Y_target_all
        else:
            logger.info(f"  > Cache has {len(cached_embeddings)} targets, need {len(missing_labels)} more")
    else:
        missing_labels = set(unique_labels)
        cached_embeddings = {}
    
    # Compute embeddings for missing targets
    if missing_labels:
        logger.info(f"  > Encoding {len(missing_labels)} targets...")
        missing_list = list(missing_labels)
        new_embeddings = encoder.encode(
            missing_list, show_progress_bar=True, 
            convert_to_numpy=True, batch_size=512
        )
        for label, emb in zip(missing_list, new_embeddings):
            cached_embeddings[label] = emb.astype(np.float32)
    
    # Save updated cache
    if cache_path and missing_labels:
        save_embedding_cache(cache_path, cached_embeddings)
    
    # Build return values
    Y_target_dict = {label: cached_embeddings[label] for label in unique_labels}
    Y_target_all = np.array([Y_target_dict[label] for label in unique_labels])
    
    return Y_target_dict, Y_target_all


# ============================================================================
# VOCABULARY AND DATA LOADING
# ============================================================================


def load_all_vocabs(vocab_dir: str) -> dict:
    """
    Loads all _vocab.json files from the vocab directory.
    
    Args:
        vocab_dir: Path to directory containing vocabulary JSON files
        
    Returns:
        Dictionary mapping feature names to their vocabularies
    """
    print(f"Loading all vocabularies from {vocab_dir}...")
    all_vocabs = {}
    try:
        for filename in os.listdir(vocab_dir):
            if filename.endswith("_vocab.json"):
                feature_name = filename.replace("_vocab.json", "")
                with open(os.path.join(vocab_dir, filename), 'r') as f:
                    all_vocabs[feature_name] = json.load(f)
                print(f"  > Loaded vocab '{feature_name}' with {len(all_vocabs[feature_name])} entries.")
    except FileNotFoundError:
        print(f"Error: Vocab directory not found: {vocab_dir}")
        raise
    return all_vocabs


def load_job_and_skill_data(
    master_skill_file: str,
    esco_skills_file: str,
    skill_properties_file: str,
    pooling_strategy: str = "mean",
    alpha: float = 1.0,
    beta: float = 1.0,
    train_val_occ: Optional[Set[str]] = None 
):
    """
    Loads all necessary skill data.
    
    IMPORTANT: Job titles in the returned job_skill_map are normalized (lowercase + stripped).
    When looking up skills, ensure job titles are normalized using .strip().lower()
    
    Args:
        master_skill_file: Path to CSV linking jobs to skillUris and scores
        esco_skills_file: Path to ESCO CSV with skill descriptions
        skill_properties_file: Path to JSON with skill meta-features
        pooling_strategy: Pooling strategy (determines if IDF is calculated)
        alpha: Exponent for confidence score (for weighted_idf)
        beta: Exponent for IDF score (for weighted_idf)
        train_val_occ: Optional set of job titles from train+val splits for IDF calculation
                      (to avoid test set leakage). If None, IDF calculated on all jobs.
                      Note: These should also be normalized (lowercase + stripped).
    
    Returns:
        Tuple of:
        - job_skill_map: { job_title (normalized) -> [{'skillUri': ..., 'score': ..., 'idf': ...}] }
        - esco_skill_text_map: { skillUri -> {'name': ..., 'desc': ...} }
        - skill_properties_map: { skillUri -> { 'skillType': [...], 'reuseLevel': [...] } }
    """
    
    # --- 1. Load Master Skill File (for links) ---
    print(f"Loading master skill map from: {master_skill_file}")
    try:
        df_full = pd.read_csv(master_skill_file)
    except FileNotFoundError as e:
        print(f"Error: Skill file not found. {e}")
        raise
    
    required_cols = ['job_title', 'skillUri', 'score']
    if not all(col in df_full.columns for col in required_cols):
        raise ValueError(f"CSV must contain {required_cols}")
    
    print(f"  > Loaded {len(df_full)} job-skill mappings for {df_full['job_title'].nunique()} unique jobs")

    # --- 2. Calculate IDF (if requested) ---
    if pooling_strategy == "weighted_idf":
        print("Calculating IDF scores from master_skill_file...")
        
        # Filter to only train+val jobs for IDF calculation (avoid test leakage)
        if train_val_occ is not None:
            df_for_idf = df_full[df_full['job_title'].isin(train_val_occ)].copy()
            print(f"  > Filtering to {len(train_val_occ)} unique train+val jobs for IDF calculation")
            print(f"  > IDF will be calculated from {len(df_for_idf)} job-skill pairs")
        else:
            df_for_idf = df_full.copy()
            print("  > Warning: Computing IDF on all jobs (including potential test jobs)")
        
        # Check if we have data after filtering
        if len(df_for_idf) == 0:
            raise ValueError("No job-skill mappings found after filtering to train+val jobs. "
                           "Check that job titles in train_val_occ match those in master_skill_file.")
        
        # N_occ = Total number of unique job titles in train+val
        N_occ = df_for_idf['job_title'].nunique()
        # n_i = Number of unique train+val job titles this skill appears with
        skill_n_occ = df_for_idf.groupby('skillUri')['job_title'].nunique()
        
        if len(skill_n_occ) == 0:
            raise ValueError("No skills found after filtering. Check your data.")
        
        # idf_i = log((N_occ + 1) / (n_i + 1))
        idf_map = np.log((N_occ + 1) / (skill_n_occ + 1))
        
        # Apply IDF scores to the FULL dataframe (including test jobs)
        df_full = df_full.copy()
        df_full['idf'] = df_full['skillUri'].map(idf_map)
        
        # Fill NaN with MAX IDF for skills that don't appear in train+val (e.g., test-only skills)
        # Assumption: Unseen skills are rare and thus important.
        max_idf = idf_map.max()
        n_missing = df_full['idf'].isna().sum()
        if n_missing > 0:
            print(f"  > {n_missing} job-skill pairs have skills not in train+val (setting IDF=max_idf={max_idf:.4f})")
        df_full['idf'] = df_full['idf'].fillna(max_idf)
        
        print(f"  > N_occ (total train+val jobs) = {N_occ}")
        print(f"  > Unique skills in train+val = {len(skill_n_occ)}")
        print(f"  > IDF range: [{idf_map.min():.4f}, {idf_map.max():.4f}]")
        print(f"  > Most common skill: '{skill_n_occ.idxmax()}' (appears in {skill_n_occ.max()} jobs, IDF={idf_map.min():.4f})")
        print(f"  > Rarest skill: '{skill_n_occ.idxmin()}' (appears in {skill_n_occ.min()} jobs, IDF={idf_map.max():.4f})")
        
    # --- 3. Build the final job_skill_map (from FULL dataframe, including test jobs) ---
    job_skill_map = {}
    print("Grouping skills by job title...")
    
    # Define the columns we need to build the map
    cols_to_group = ['skillUri', 'score']
    if pooling_strategy == "weighted_idf":
        cols_to_group.append('idf')
        
    for job_title, group in tqdm(df_full.groupby('job_title'), desc="Building job->skill map"):
        # Normalize job title to ensure consistent matching (lowercase + stripped)
        job_skill_map[job_title.strip().lower()] = group[cols_to_group].to_dict('records')
            
    print(f"Created job-to-skill map with {len(job_skill_map)} unique job titles (train+val+test).")

    # --- 4. Load ESCO Skill Text (Name/Desc) ---
    print(f"Loading ESCO skill text from: {esco_skills_file}")
    try:
        esco_df = pd.read_csv(esco_skills_file, usecols=['conceptUri', 'preferredLabel', 'description'])
        esco_df.columns = ['skillUri', 'skill', 'skill_description']
        # Create a fast lookup map: {skillUri -> {'name': ..., 'desc': ...}}
        esco_skill_text_map = {}
        for _, row in esco_df.iterrows():
            esco_skill_text_map[row['skillUri']] = {
                'name': row['skill'],
                'desc': row['skill_description'] if pd.notna(row['skill_description']) else ""
            }
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading {esco_skills_file}. Make sure it has 'skillUri', 'skill', 'skill_description'.")
        raise
        
    # --- 5. Load Skill Meta-Feature Map ---
    print(f"Loading skill meta-features from: {skill_properties_file}")
    try:
        with open(skill_properties_file, 'r') as f:
            skill_properties_map = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {skill_properties_file} not found. Did you run create_vocabularies.py?")
        raise
        
    return job_skill_map, esco_skill_text_map, skill_properties_map


def load_job_skill_data_by_id(
    skill_scores_file: str,
    esco_skills_file: str,
    skill_properties_file: str,
    pooling_strategy: str = "mean",
    alpha: float = 1.0,
    beta: float = 1.0,
    train_val_job_ids: Optional[Set[str]] = None,
    esco_taxonomy_file: Optional[str] = None,
    min_max_normalize: bool = False,
):
    """
    Loads skill data using job_id as the key (instead of job titles).
    
    This function uses a JSON file with pre-computed skill scores where keys are job_ids.
    The expected format is: {"scores": {"job_id": [{"skill_uri": ..., "score": ...}, ...]}}
    
    Args:
        skill_scores_file: Path to JSON file with skill scores (e.g., best_fused_scores.json)
        esco_skills_file: Path to ESCO CSV with skill descriptions
        skill_properties_file: Path to JSON with skill meta-features
        pooling_strategy: Pooling strategy (determines if IDF is calculated)
        alpha: Exponent for confidence score (for weighted_idf)
        beta: Exponent for IDF score (for weighted_idf)
        train_val_job_ids: Optional set of job_ids from train+val splits for IDF calculation
                          (to avoid test set leakage). If None, IDF calculated on all jobs.
        esco_taxonomy_file: Optional path to ESCO taxonomy CSV (occupationSkillRelations) for static IDF.
                           If provided, IDF is calculated from this file instead of the dataset.
        min_max_normalize: Optional bool to apply Min-Max normalization to skill scores per job.
                          Default False (backward compatibility). Scales scores to [0, 1].
    
    Returns:
        Tuple of:
        - job_skill_map: { job_id (str) -> [{'skillUri': ..., 'score': ..., 'idf': ...}] }
        - esco_skill_text_map: { skillUri -> {'name': ..., 'desc': ...} }
        - skill_properties_map: { skillUri -> { 'skillType': [...], 'reuseLevel': [...] } }
    """
    
    # --- 1. Load Skill Scores File ---
    print(f"Loading skill scores from: {skill_scores_file}")
    try:
        # Special handling for karrierewege_cp dataset (kw_cp) which uses JSONL format
        # Other data sources use regular JSON format
        is_karrierewege_cp = 'kw_cp' in skill_scores_file and skill_scores_file.endswith('.jsonl')
        
        if is_karrierewege_cp:
            # JSONL format for karrierewege_cp: each line is a separate JSON object
            # Expected format: {"job_id": "...", "predictions": [...]}
            print(f"  > Detected karrierewege_cp JSONL format")
            scores_dict = {}
            with open(skill_scores_file, 'r') as f:
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
                            scores_dict[job_id] = predictions
                    except json.JSONDecodeError as e:
                        print(f"  ⚠️ Warning: Could not parse line {line_num}: {e}")
                        continue
            print(f"  > Loaded JSONL file with {len(scores_dict)} job entries")
        else:
            # Regular JSON format for other data sources
            with open(skill_scores_file, 'r') as f:
                data = json.load(f)
            
            # Extract scores dictionary
            if 'scores' in data:
                scores_dict = data['scores']
            else:
                # Assume the file is already the scores dictionary
                scores_dict = data
    except FileNotFoundError as e:
        print(f"Error: Skill scores file not found. {e}")
        raise
    
    print(f"  > Loaded skill scores for {len(scores_dict)} unique job_ids")
    
    # --- 1b. [NEW] Min-Max Normalization ---
    if min_max_normalize:
        print("  > 🧪 Applying per-job Min-Max normalization to skill scores...")
        normalized_count = 0
        for job_id, skills in scores_dict.items():
            if not skills:
                continue
            
            # Extract scores safely
            current_scores = []
            for s in skills:
                # Handle list/tuple or dict format
                if isinstance(s, (list, tuple)):
                    val = s[1] if len(s) > 1 else 1.0
                else:
                    val = s.get('score', 1.0)
                current_scores.append(float(val))
            
            if not current_scores:
                continue
                
            min_s = min(current_scores)
            max_s = max(current_scores)
            range_s = max_s - min_s
            
            # Update scores in place
            for idx, s in enumerate(skills):
                orig_val = current_scores[idx]
                
                if range_s > 1e-9:
                    new_val = (orig_val - min_s) / range_s
                else:
                    # All scores are equal -> set to 1.0 (max confidence)
                    new_val = 1.0
                
                # Write back (handle both formats)
                if isinstance(s, (list, tuple)):
                    # Convert tuple to list to modify
                    s_list = list(s)
                    if len(s_list) > 1:
                        s_list[1] = new_val
                    else:
                        s_list.append(new_val)
                    scores_dict[job_id][idx] = s_list
                else:
                    s['score'] = new_val
            
            normalized_count += 1
            
        print(f"  ✓ Normalized scores for {normalized_count} jobs")
    
    # --- 2. Build job_skill_map and calculate IDF if needed ---
    job_skill_map = {}
    
    # First pass: collect all job-skill mappings
    all_job_skill_pairs = []  # List of (job_id, skill_uri, score)
    for job_id, skill_list in scores_dict.items():
        for skill_info in skill_list:
            # Handle both formats:
            # 1. List/tuple format: [skill_uri, score] (e.g., fused_predictions.json)
            # 2. Dict format: {"skill_uri": "...", "score": ...} (e.g., best_fused_scores.json)
            if isinstance(skill_info, (list, tuple)):
                skill_uri = skill_info[0]
                score = skill_info[1] if len(skill_info) > 1 else 1.0
            else:
                skill_uri = skill_info.get('skill_uri') or skill_info.get('skillUri')
                score = skill_info.get('score', 1.0)
            all_job_skill_pairs.append((str(job_id), skill_uri, score))
    
    print(f"  > Total job-skill mappings: {len(all_job_skill_pairs)}")
    
    # --- 3. Calculate IDF (if requested) ---
    idf_map = {}
    max_idf = 0.0
    if pooling_strategy == "weighted_idf":
        if esco_taxonomy_file and os.path.exists(esco_taxonomy_file):
            print(f"Calculating IDF scores from ESCO taxonomy: {esco_taxonomy_file}")
            try:
                df_esco = pd.read_csv(esco_taxonomy_file)
                # Ensure we have the right columns
                if 'occupationUri' in df_esco.columns and 'skillUri' in df_esco.columns:
                    N_occ = df_esco['occupationUri'].nunique()
                    skill_n_occ = df_esco.groupby('skillUri')['occupationUri'].nunique()
                    
                    # idf_i = log((N_occ + 1) / (n_i + 1))
                    idf_series = np.log((N_occ + 1) / (skill_n_occ + 1))
                    idf_map = idf_series.to_dict()
                    max_idf = idf_series.max()
                    
                    print(f"  > N_occ (total ESCO occupations) = {N_occ}")
                    print(f"  > Unique skills in ESCO = {len(skill_n_occ)}")
                    print(f"  > IDF range: [{idf_series.min():.4f}, {idf_series.max():.4f}]")
                    print(f"  > Static IDF calculation complete.")
                else:
                    print("  ⚠️ ESCO file missing required columns. Falling back to dataset IDF.")
                    # Fallback code will execute below if we reset/continue, but cleaner to just let it fall through 
                    # or handle it. Here I'll raise or fallback. Let's raise to be safe.
                    raise ValueError(f"ESCO file {esco_taxonomy_file} missing 'occupationUri' or 'skillUri'")
            except Exception as e:
                print(f"  ❌ Error reading ESCO taxonomy file: {e}")
                raise
        
        else:
            print("Calculating IDF scores from dataset...")
            
            # Filter to only train+val jobs for IDF calculation
            if train_val_job_ids is not None:
                train_val_pairs = [(jid, suri, sc) for jid, suri, sc in all_job_skill_pairs 
                                  if jid in train_val_job_ids]
                print(f"  > Filtering to {len(train_val_job_ids)} train+val job_ids for IDF calculation")
                print(f"  > IDF will be calculated from {len(train_val_pairs)} job-skill pairs")
            else:
                train_val_pairs = all_job_skill_pairs
                print("  > Warning: Computing IDF on all jobs (including potential test jobs)")
            
            if len(train_val_pairs) == 0:
                raise ValueError("No job-skill mappings found after filtering to train+val job_ids.")
            
            # Create dataframe for IDF calculation
            df_for_idf = pd.DataFrame(train_val_pairs, columns=['job_id', 'skillUri', 'score'])
            
            # N_occ = Total number of unique job_ids in train+val
            N_occ = df_for_idf['job_id'].nunique()
            # n_i = Number of unique train+val job_ids this skill appears with
            skill_n_occ = df_for_idf.groupby('skillUri')['job_id'].nunique()
            
            # idf_i = log((N_occ + 1) / (n_i + 1))
            idf_series = np.log((N_occ + 1) / (skill_n_occ + 1))
            idf_map = idf_series.to_dict()
            max_idf = idf_series.max()
            
            print(f"  > N_occ (total train+val jobs) = {N_occ}")
            print(f"  > Unique skills in train+val = {len(skill_n_occ)}")
            print(f"  > IDF range: [{idf_series.min():.4f}, {idf_series.max():.4f}]")
    
    # --- 4. Build the final job_skill_map ---
    print("Building job->skill map...")
    for job_id, skill_list in tqdm(scores_dict.items(), desc="Building job->skill map"):
        job_id_str = str(job_id)
        skill_infos = []
        for skill_info in skill_list:
            # Handle both formats:
            # 1. List/tuple format: [skill_uri, score] (e.g., fused_predictions.json)
            # 2. Dict format: {"skill_uri": "...", "score": ...} (e.g., best_fused_scores.json)
            if isinstance(skill_info, (list, tuple)):
                skill_uri = skill_info[0]
                score = skill_info[1] if len(skill_info) > 1 else 1.0
            else:
                skill_uri = skill_info.get('skill_uri') or skill_info.get('skillUri')
                score = skill_info.get('score', 1.0)
            
            info = {'skillUri': skill_uri, 'score': score}
            if pooling_strategy == "weighted_idf":
                # Use max_idf for unseen skills (rare assumption) instead of 0.0
                info['idf'] = idf_map.get(skill_uri, max_idf)
            skill_infos.append(info)
        
        job_skill_map[job_id_str] = skill_infos
    
    print(f"Created job-to-skill map with {len(job_skill_map)} unique job_ids.")

    # --- 5. Load ESCO Skill Text (Name/Desc) ---
    print(f"Loading ESCO skill text from: {esco_skills_file}")
    try:
        esco_df = pd.read_csv(esco_skills_file, usecols=['conceptUri', 'preferredLabel', 'description'])
        esco_df.columns = ['skillUri', 'skill', 'skill_description']
        esco_skill_text_map = {}
        for _, row in esco_df.iterrows():
            esco_skill_text_map[row['skillUri']] = {
                'name': row['skill'],
                'desc': row['skill_description'] if pd.notna(row['skill_description']) else ""
            }
    except (FileNotFoundError, KeyError) as e:
        print(f"Error loading {esco_skills_file}. Make sure it has required columns.")
        raise
        
    # --- 6. Load Skill Meta-Feature Map ---
    # --- 6. Load Skill Meta-Feature Map ---
    skill_properties_map = {}
    if skill_properties_file:
        print(f"Loading skill meta-features from: {skill_properties_file}")
        try:
            with open(skill_properties_file, 'r') as f:
                skill_properties_map = json.load(f)
        except FileNotFoundError as e:
            print(f"Error: {skill_properties_file} not found.")
            raise
    else:
        print("Skipping skill meta-features (not provided).")
        
    return job_skill_map, esco_skill_text_map, skill_properties_map


def precompute_target_embeddings(
    encoder, 
    labels: list, 
    show_progress: bool = True,
    cache_dir: Optional[str] = None,
    encoder_name: Optional[str] = None,
    force_recompute: bool = False,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Pre-compute embeddings for all target job labels, with optional caching.

    Args:
        encoder: SentenceTransformer encoder model
        labels: List of unique target job strings
        show_progress: Whether to show progress bar
        cache_dir: Directory to cache embeddings (None = no caching)
        encoder_name: Name of the encoder (for cache filename)
        force_recompute: Force recomputation even if cache exists

    Returns:
        Tuple of:
        - Dictionary mapping target strings to their embeddings
        - Numpy array of all target embeddings
    """
    logger.info(f"  > Pre-computing embeddings for {len(set(labels))} unique target labels...")
    
    if cache_dir and encoder_name:
        return get_or_compute_target_embeddings(
            labels, encoder, cache_dir, encoder_name, force_recompute
        )
    else:
        # No caching - compute directly
        unique_labels = list(set(labels))
        target_embeddings = encoder.encode(
            unique_labels, show_progress_bar=show_progress, 
            convert_to_numpy=True, batch_size=512
        )
        Y_target_dict = {label: emb.astype(np.float32) 
                         for label, emb in zip(unique_labels, target_embeddings)}
        Y_target_all = np.array(list(Y_target_dict.values()))
        logger.info(f"  ✓ Created target embedding dictionary with {len(Y_target_dict)} entries")
        return Y_target_dict, Y_target_all


def _extract_job_titles_from_history(history_doc: str) -> List[str]:
    """Extract raw job titles from a history document (without normalizing).
    
    Handles both formatted documents (e.g., "role: cook\\n description: ...")
    and plain titles separated by SEP_TOKEN.
    """
    # First try to extract from formatted "role: <title>\\n description: ..."
    titles = re.findall(r"role: (.*?)\n", history_doc)
    
    # If no matches found, assume it's plain title(s) or SEP-separated titles
    if not titles:
        from cpp import utils
        titles = [t.strip() for t in history_doc.split(utils.SEP_TOKEN) if t.strip()]
    
    # Ensure whitespace is stripped consistently
    return [t.strip() for t in titles if t.strip()]


def _extract_skill_infos(history_doc: str, job_skill_map: Dict[str, List[Dict]]) -> List[Dict]:
    """Extract skill information from a history document.
    
    Uses the same title parsing logic as `_extract_job_titles_from_history` but
    immediately maps titles to their associated skills and flattens them.
    
    Note: Job titles are normalized (lowercase + stripped) to match the mapping file format.
    """
    titles = _extract_job_titles_from_history(history_doc)
    
    infos = []
    for t in titles:
        # Normalize title to match mapping file format (lowercase + stripped)
        title_clean = t.strip().lower()
        if title_clean in job_skill_map:
            infos.extend(job_skill_map[title_clean])
    return infos


def _extract_skill_infos_by_job_ids(job_ids: List[str], job_skill_map: Dict[str, List[Dict]]) -> List[Dict]:
    """Extract skill information using job_ids directly.
    
    This function looks up skills using pre-mapped job_ids instead of parsing
    titles from the history document. This is more accurate for free-text datasets
    where the same job title can have different descriptions and thus different skills.
    
    Args:
        job_ids: List of job_id strings for jobs in the career history
        job_skill_map: Map from job_id -> list of skill info dicts
        
    Returns:
        Flattened list of skill info dictionaries for all jobs
    """
    infos = []
    for job_id in job_ids:
        if job_id in job_skill_map:
            infos.extend(job_skill_map[job_id])
    return infos


def _pool_jobs_with_log_decay(job_vectors: List[np.ndarray], alpha: float = 0.5) -> np.ndarray:
    """Pool job vectors using logarithmic position weighting or mean pooling.
    
    This mirrors the behaviour in ``train_cpp_skills_v2.pool_jobs_with_log_decay``:
    
    - If alpha == 0: Use mean pooling (uniform weights)
    - If alpha > 0: Use logarithmic position weighting
    
    Weight formula: w_i = log(1 + α * i) where i is position (0-indexed).
    Last job gets highest weight when alpha > 0.
    """
    if not job_vectors:
        return None
    
    n_jobs = len(job_vectors)
    
    if alpha == 0:
        # Mean pooling: uniform weights
        weights = np.ones(n_jobs, dtype=np.float32) / n_jobs
    else:
        # Logarithmic position weighting
        positions = np.arange(n_jobs, dtype=np.float32)
        weights = np.log(1.0 + alpha * positions)
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            # Fallback to uniform weights if something goes wrong
            weights = np.ones(n_jobs, dtype=np.float32) / n_jobs
    
    job_matrix = np.stack(job_vectors, axis=0)  # [n_jobs, dim]
    pooled = np.sum(job_matrix * weights[:, np.newaxis], axis=0)
    return pooled.astype(np.float32)


def _pooled_skill_vec(infos: List[Dict], encoder_skill, esco_skill_text_map: Dict[str, Dict],
                      use_skill_description: bool, pooling_strategy: str, alpha: float, beta: float,
                      embed_dim: int, debug: bool = False, sample_limit: int = 3) -> np.ndarray:
    """Compute pooled skill text embedding vector.
    
    DEPRECATED: This function is kept for backward compatibility but is inefficient.
    Use _pooled_skill_vec_optimized with pre-computed embeddings instead.
    
    Args:
        debug: If True, log sample skill texts being encoded
        sample_limit: Number of sample skills to log when debug=True
    """
    strings, weights = [], []
    
    for idx, info in enumerate(infos):
        uri = info['skillUri']
        if uri in esco_skill_text_map:
            st = esco_skill_text_map[uri]
            text = f"role: {st['name']}\ndescription: {st['desc']}" if use_skill_description else st['name']
            
            # Debug logging: show first N skills being encoded
            if debug and idx < sample_limit:
                logger.debug(f"\n{'='*60}")
                logger.debug(f"🔍 Skill {idx+1}/{len(infos)} - Text Being Encoded:")
                logger.debug(f"{'='*60}")
                logger.debug(f"Skill URI: {uri}")
                logger.debug(f"Use description: {use_skill_description}")
                logger.debug(f"Pooling strategy: {pooling_strategy}")
                logger.debug(f"\n--- Formatted Text (sent to encoder) ---")
                logger.debug(text)
                logger.debug(f"--- End of Text ---")
                logger.debug(f"\nText length: {len(text)} characters")
                logger.debug(f"Contains newline: {chr(10) in text}")
                logger.debug(f"Contains 'role:': {'role:' in text}")
                logger.debug(f"Contains 'description:': {'description:' in text}")
                
                # Show weight information
                if pooling_strategy == "mean":
                    weight = 1.0
                elif pooling_strategy == "weighted_mean":
                    weight = info['score']
                else:
                    c = info['score']
                    idf = info.get('idf', 0)
                    weight = (c ** alpha) * (idf ** beta)
                logger.debug(f"\nWeight for this skill: {weight:.4f}")
                if pooling_strategy not in ["mean"]:
                    logger.debug(f"  - Confidence score: {info['score']:.4f}")
                    if pooling_strategy == "weighted_idf":
                        logger.debug(f"  - IDF: {info.get('idf', 0):.4f}")
                        logger.debug(f"  - Alpha (confidence exponent): {alpha}")
                        logger.debug(f"  - Beta (IDF exponent): {beta}")
                logger.debug(f"{'='*60}\n")
            
            strings.append(text)
            if pooling_strategy == "mean":
                weights.append(1.0)
            elif pooling_strategy == "weighted_mean":
                weights.append(info['score'])
            else:
                c = info['score']
                idf = info.get('idf', 0)
                weights.append((c ** alpha) * (idf ** beta))
    
    if not strings:
        if debug:
            logger.debug("⚠️  No skills found for this job - returning zero vector")
        return np.zeros(embed_dim, dtype=np.float32)
    
    if debug:
        logger.debug(f"📊 Encoding {len(strings)} skill texts into embeddings...")
        logger.debug(f"   Total skills in batch: {len(strings)}")
        logger.debug(f"   Weight sum: {sum(weights):.4f}")
        logger.debug(f"   Weight range: [{min(weights):.4f}, {max(weights):.4f}]")
    
    embs = encoder_skill.encode(strings, convert_to_numpy=True, batch_size=512)
    w = np.array(weights, dtype=np.float32)
    vec = embs.mean(axis=0) if pooling_strategy == "mean" or w.sum() == 0 else np.average(embs, axis=0, weights=w)
    return vec.astype(np.float32)


def extract_unique_skills_from_dataset(data_pairs: List[Tuple[str, str]], 
                                       job_skill_map: Dict[str, List[Dict]]) -> set:
    """Extract all unique skill URIs used in the dataset.
    
    Args:
        data_pairs: List of (history_doc, target_doc) tuples
        job_skill_map: Map from job titles to skill info
        
    Returns:
        Set of unique skill URIs found in the dataset
    """
    unique_skills = set()
    for history_doc, _ in data_pairs:
        infos = _extract_skill_infos(history_doc, job_skill_map)
        for info in infos:
            unique_skills.add(info['skillUri'])
    return unique_skills


def extract_unique_skills_from_job_ids(job_ids_list: List[List[str]], 
                                       job_skill_map: Dict[str, List[Dict]]) -> set:
    """Extract all unique skill URIs using job_ids.
    
    Args:
        job_ids_list: List of job_id lists (one per data sample)
        job_skill_map: Map from job_id to skill info
        
    Returns:
        Set of unique skill URIs found across all job_ids
    """
    unique_skills = set()
    for job_ids in job_ids_list:
        infos = _extract_skill_infos_by_job_ids(job_ids, job_skill_map)
        for info in infos:
            unique_skills.add(info['skillUri'])
    return unique_skills


def precompute_skill_embeddings(unique_skill_uris: set, 
                                encoder_skill, 
                                esco_skill_text_map: Dict[str, Dict],
                                use_skill_description: bool,
                                use_skill_prefix: bool = False) -> Dict[str, np.ndarray]:
    """Pre-compute embeddings for all unique skills in the dataset.
    
    DEPRECATED: Use `get_or_compute_skill_embeddings` instead for caching support.
    
    This function encodes all unique skills once, which is much more efficient than
    encoding the same skill multiple times across different samples.
    
    Args:
        unique_skill_uris: Set of unique skill URIs to encode
        encoder_skill: Encoder for skills
        esco_skill_text_map: Map from skill URIs to text
        use_skill_description: Whether to include skill descriptions
        use_skill_prefix: Use "skill: ..." prefix for skill-specific encoders
        
    Returns:
        Dictionary mapping skill URIs to their embeddings
    """
    import warnings
    warnings.warn(
        "precompute_skill_embeddings is deprecated, use get_or_compute_skill_embeddings instead",
        DeprecationWarning, stacklevel=2
    )
    
    # Delegate to the new caching function (without caching)
    return get_or_compute_skill_embeddings(
        unique_skill_uris,
        encoder_skill,
        esco_skill_text_map,
        use_skill_description,
        cache_dir=None,  # No caching
        encoder_name=None,
        force_recompute=True,
        use_skill_prefix=use_skill_prefix,
    )


def _pooled_skill_vec_optimized(infos: List[Dict], 
                                skill_embedding_map: Dict[str, np.ndarray],
                                pooling_strategy: str, 
                                alpha: float, 
                                beta: float, 
                                embed_dim: int,
                                debug: bool = False,
                                sample_limit: int = 3) -> np.ndarray:
    """Compute pooled skill vector using pre-computed embeddings.
    
    This is the optimized version that uses pre-computed skill embeddings
    instead of encoding skills on-the-fly.
    
    Args:
        infos: List of skill info dictionaries
        skill_embedding_map: Pre-computed embeddings for skills
        pooling_strategy: Pooling strategy (mean, weighted_mean, weighted_idf)
        alpha: Exponent for confidence score
        beta: Exponent for IDF score
        embed_dim: Embedding dimension
        debug: If True, log detailed information
        sample_limit: Number of sample skills to log when debug=True
        
    Returns:
        Pooled skill embedding vector
    """
    embeddings = []
    weights = []
    
    for idx, info in enumerate(infos):
        uri = info['skillUri']
        if uri in skill_embedding_map:
            embeddings.append(skill_embedding_map[uri])
            
            # Debug logging: show first N skills
            if debug and idx < sample_limit:
                logger.debug(f"\n{'='*60}")
                logger.debug(f"🔍 Skill {idx+1}/{len(infos)} - Using Pre-computed Embedding:")
                logger.debug(f"{'='*60}")
                logger.debug(f"Skill URI: {uri}")
                logger.debug(f"Pooling strategy: {pooling_strategy}")
                
                # Show weight information
                if pooling_strategy == "mean":
                    weight = 1.0
                elif pooling_strategy == "weighted_mean":
                    weight = info['score']
                else:
                    c = info['score']
                    idf = info.get('idf', 0)
                    weight = (c ** alpha) * (idf ** beta)
                logger.debug(f"\nWeight for this skill: {weight:.4f}")
                if pooling_strategy not in ["mean"]:
                    logger.debug(f"  - Confidence score: {info['score']:.4f}")
                    if pooling_strategy == "weighted_idf":
                        logger.debug(f"  - IDF: {info.get('idf', 0):.4f}")
                        logger.debug(f"  - Alpha (confidence exponent): {alpha}")
                        logger.debug(f"  - Beta (IDF exponent): {beta}")
                logger.debug(f"{'='*60}\n")
            
            if pooling_strategy == "mean":
                weights.append(1.0)
            elif pooling_strategy == "weighted_mean":
                weights.append(info['score'])
            else:  # weighted_idf
                c = info['score']
                idf = info.get('idf', 0)
                weights.append((c ** alpha) * (idf ** beta))
    
    if not embeddings:
        if debug:
            logger.debug("⚠️  No skills found for this job - returning zero vector")
        return np.zeros(embed_dim, dtype=np.float32)
    
    if debug:
        logger.debug(f"📊 Pooling {len(embeddings)} pre-computed embeddings...")
        logger.debug(f"   Total skills: {len(embeddings)}")
        logger.debug(f"   Weight sum: {sum(weights):.4f}")
        logger.debug(f"   Weight range: [{min(weights):.4f}, {max(weights):.4f}]")
    
    embs = np.array(embeddings)
    w = np.array(weights, dtype=np.float32)
    
    vec = embs.mean(axis=0) if pooling_strategy == "mean" or w.sum() == 0 else np.average(embs, axis=0, weights=w)
    return vec.astype(np.float32)


def _compute_data_hash(text_list: List[str]) -> str:
    """Compute a deterministic hash for a list of strings for cache validation."""
    hasher = hashlib.md5()
    # Include length to catch drops quickly
    hasher.update(str(len(text_list)).encode('utf-8'))
    # Hash content
    for text in text_list:
        hasher.update(text.encode('utf-8'))
    return hasher.hexdigest()


def precompute_input_embeddings(
    data_pairs: List[Tuple[str, str]],
    Y_target_dict: Dict[str, np.ndarray],
    encoder_text,
    encoder_skill,
    job_skill_map: Dict[str, List[Dict]],
    esco_skill_text_map: Dict[str, Dict],
    use_skill_description: bool = False,
    pooling_strategy: str = "mean",
    alpha: float = 1.0,
    beta: float = 1.0,
    use_text_history: bool = True,
    use_skill_text: bool = True,
    debug: Optional[bool] = None,
    use_skill_path_log_pooling: bool = False,
    skill_path_alpha_decay: float = 0.5,
    cache_dir: Optional[str] = None,
    encoder_skill_name: Optional[str] = None,
    force_recompute: bool = False,
    split_name: str = "data"  # <--- New argument for logging
) -> Tuple[List[Tuple[str, str]], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute pooled input embeddings for text history and skill text.
    
    This function:
    1. Loads raw skill embeddings from cache (or computes and caches them)
    2. Performs pooling (IDF-weighted, etc.) at runtime (fast)
    3. Returns per-sample pooled vectors ready for the model
    
    Note: Text history embeddings are always computed fresh since they depend
    on the specific history document format. Skill embeddings are cached by URI.

    Args:
        data_pairs: List of (history_doc, target_doc) tuples
        Y_target_dict: Dictionary mapping target strings to embeddings (for filtering)
        encoder_text: Encoder for text history
        encoder_skill: Encoder for skills
        job_skill_map: Map from job titles to skill info
        esco_skill_text_map: Map from skill URIs to text
        use_skill_description: Whether to include skill descriptions
        pooling_strategy: Pooling strategy for skills
        alpha: Exponent for confidence score
        beta: Exponent for IDF score
        use_text_history: Whether to compute text history embeddings
        use_skill_text: Whether to compute skill text embeddings
        debug: If True, log sample skill texts being encoded
        use_skill_path_log_pooling: Use skills_v2-style per-job pooling
        skill_path_alpha_decay: Log decay for job position weighting
        cache_dir: Directory to cache raw skill embeddings (None = no caching)
        encoder_skill_name: Name of skill encoder (for cache filename)
        force_recompute: Force recomputation of cached embeddings

    Returns:
        Tuple of (filtered_pairs, h_text_embeddings, h_skill_embeddings)
    """
    # Filter pairs to match target dictionary (same as dataset does internally)
    filtered_pairs = [(h, t) for (h, t) in data_pairs if t in Y_target_dict]

    embed_dim = encoder_text.get_sentence_embedding_dimension()
    h_text = None
    h_skill = None

    if use_text_history:
        logger.info(f"  > Processing text history for {split_name} ({len(filtered_pairs)} samples)...")
        histories = [h for (h, _) in filtered_pairs]
        
        # --- Caching Logic Start ---
        loaded_from_cache = False
        if cache_dir:
            # 1. Compute Hash of the text content (The "Key")
            data_hash = _compute_data_hash(histories)
            
            # 2. Construct filename
            enc_name = encoder_text.model_name_or_path.split('/')[-1] if hasattr(encoder_text, 'model_name_or_path') else "text_encoder"
            cache_filename = f"history_emb_{enc_name}_{data_hash}.npy"
            cache_path = os.path.join(cache_dir, cache_filename)
            
            # 3. Try Load
            if os.path.exists(cache_path) and not force_recompute:
                try:
                    logger.info(f"  > Found cached history embeddings: {cache_filename}")
                    h_text = np.load(cache_path)
                    if len(h_text) == len(histories):
                        logger.info("  ✓ Cache loaded and validated.")
                        loaded_from_cache = True
                    else:
                        logger.warning("  ⚠️ Cache size mismatch (hash collision?). Recomputing.")
                except Exception as e:
                    logger.warning(f"  ⚠️ Failed to load cache: {e}")

        if not loaded_from_cache:
            logger.info("  > Computing text history embeddings (this may take a while)...")
            h_text = encoder_text.encode(
                histories, convert_to_numpy=True, 
                show_progress_bar=True, batch_size=512
            ).astype(np.float32)
            
            # 4. Save to cache
            if cache_dir:
                try:
                    np.save(cache_path, h_text)
                    logger.info(f"  ✓ Saved history embeddings to {cache_filename}")
                except Exception as e:
                    logger.warning(f"  ⚠️ Failed to save cache: {e}")
        # --- Caching Logic End ---

    if use_skill_text:
        logger.info("  > Computing skill text embeddings...")
        
        # Step 1: Extract unique skills from dataset
        logger.info("    Step 1: Extracting unique skills from dataset...")
        unique_skills = extract_unique_skills_from_dataset(filtered_pairs, job_skill_map)
        logger.info(f"    Found {len(unique_skills)} unique skills")
        
        # Step 2: Get or compute raw skill embeddings (with caching)
        logger.info("    Step 2: Loading/computing raw skill embeddings...")
        use_skill_prefix = use_skill_path_log_pooling or (encoder_skill is not encoder_text)
        skill_embedding_map = get_or_compute_skill_embeddings(
            unique_skills,
            encoder_skill,
            esco_skill_text_map,
            use_skill_description,
            cache_dir=cache_dir,
            encoder_name=encoder_skill_name,
            force_recompute=force_recompute,
            use_skill_prefix=use_skill_prefix,
        )
        
        if debug:
            logger.info("\n" + "🔍" * 30)
            logger.info("DEBUG: Showing skill aggregation for first sample")
            logger.info("🔍" * 30 + "\n")
        
        # Step 3: Pool skills per sample (fast - just numpy operations)
        logger.info("    Step 3: Pooling skill vectors per sample...")
        skill_vecs = []
        for idx, (h, _) in enumerate(tqdm(filtered_pairs, desc="    Pooling")):
            if use_skill_path_log_pooling:
                # skills_v2-style: pool skills per job, then log-weight jobs
                job_titles = _extract_job_titles_from_history(h)
                job_vectors = []
                for title in job_titles:
                    title_clean = title.strip().lower()
                    infos = job_skill_map.get(title_clean, [])
                    if not infos:
                        continue
                    job_vec = _pooled_skill_vec_optimized(
                        infos, skill_embedding_map, pooling_strategy,
                        alpha, beta, embed_dim, debug=False,
                    )
                    job_vectors.append(job_vec)
                
                if job_vectors:
                    skill_vec = _pool_jobs_with_log_decay(job_vectors, alpha=skill_path_alpha_decay)
                else:
                    skill_vec = np.zeros(embed_dim, dtype=np.float32)
            else:
                # Default: aggregate all skills across the full history
                infos = _extract_skill_infos(h, job_skill_map)
                is_first_sample = (idx == 0)
                skill_vec = _pooled_skill_vec_optimized(
                    infos, skill_embedding_map, pooling_strategy,
                    alpha, beta, embed_dim, debug=(debug and is_first_sample),
                )
            skill_vecs.append(skill_vec)
        
        h_skill = np.stack(skill_vecs, axis=0)
        logger.info(f"  ✓ Skill embeddings pooled: shape {h_skill.shape}")

    return filtered_pairs, h_text, h_skill


def precompute_input_embeddings_with_job_ids(
    data_pairs: List[Tuple[str, str]],
    job_ids_list: List[List[str]],
    Y_target_dict: Dict[str, np.ndarray],
    encoder_text,
    encoder_skill,
    job_skill_map: Dict[str, List[Dict]],
    esco_skill_text_map: Dict[str, Dict],
    use_skill_description: bool = False,
    pooling_strategy: str = "mean",
    alpha: float = 1.0,
    beta: float = 1.0,
    use_text_history: bool = True,
    use_skill_text: bool = True,
    debug: Optional[bool] = None,
    use_skill_path_log_pooling: bool = False,
    skill_path_alpha_decay: float = 0.5,
    cache_dir: Optional[str] = None,
    encoder_skill_name: Optional[str] = None,
    force_recompute: bool = False,
    split_name: str = "data",
    precomputed_skill_embedding_map: Optional[Dict[str, np.ndarray]] = None,
) -> Tuple[List[Tuple[str, str]], List[List[str]], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute pooled input embeddings using job_ids for skill lookup.
    
    This is similar to precompute_input_embeddings but uses pre-mapped job_ids
    instead of extracting job titles from the history document. This is more
    accurate for free-text datasets where the same title can have different skills.
    
    Args:
        data_pairs: List of (history_doc, target_doc) tuples
        job_ids_list: List of job_id lists (one per data sample)
        Y_target_dict: Dictionary mapping target strings to embeddings (for filtering)
        encoder_text: Encoder for text history
        encoder_skill: Encoder for skills (can be None if precomputed_skill_embedding_map is provided)
        job_skill_map: Map from job_id to skill info (NOT job title!)
        esco_skill_text_map: Map from skill URIs to text (can be None if precomputed_skill_embedding_map is provided)
        use_skill_description: Whether to include skill descriptions
        pooling_strategy: Pooling strategy for skills
        alpha: Exponent for confidence score
        beta: Exponent for IDF score
        use_text_history: Whether to compute text history embeddings
        use_skill_text: Whether to compute skill text embeddings
        debug: If True, log sample skill texts being encoded
        use_skill_path_log_pooling: Use per-job pooling then log-weight jobs
        skill_path_alpha_decay: Log decay for job position weighting
        cache_dir: Directory to cache raw skill embeddings (None = no caching)
        encoder_skill_name: Name of skill encoder (for cache filename)
        force_recompute: Force recomputation of cached embeddings
        split_name: Name for logging purposes
        precomputed_skill_embedding_map: Optional pre-loaded skill embeddings (URI -> embedding).
                                         If provided, skips skill encoding entirely.

    Returns:
        Tuple of (filtered_pairs, filtered_job_ids, h_text_embeddings, h_skill_embeddings)
    """
    # Filter pairs and job_ids to match target dictionary
    filtered_pairs = []
    filtered_job_ids = []
    for (h, t), job_ids in zip(data_pairs, job_ids_list):
        if t in Y_target_dict:
            filtered_pairs.append((h, t))
            filtered_job_ids.append(job_ids)

    embed_dim = encoder_text.get_sentence_embedding_dimension()
    h_text = None
    h_skill = None

    if use_text_history:
        logger.info(f"  > Processing text history for {split_name} ({len(filtered_pairs)} samples)...")
        histories = [h for (h, _) in filtered_pairs]
        
        # --- Caching Logic Start ---
        loaded_from_cache = False
        if cache_dir:
            data_hash = _compute_data_hash(histories)
            enc_name = encoder_text.model_name_or_path.split('/')[-1] if hasattr(encoder_text, 'model_name_or_path') else "text_encoder"
            cache_filename = f"history_emb_{enc_name}_{data_hash}.npy"
            cache_path = os.path.join(cache_dir, cache_filename)
            
            if os.path.exists(cache_path) and not force_recompute:
                try:
                    logger.info(f"  > Found cached history embeddings: {cache_filename}")
                    h_text = np.load(cache_path)
                    if len(h_text) == len(histories):
                        logger.info("  ✓ Cache loaded and validated.")
                        loaded_from_cache = True
                    else:
                        logger.warning("  ⚠️ Cache size mismatch (hash collision?). Recomputing.")
                except Exception as e:
                    logger.warning(f"  ⚠️ Failed to load cache: {e}")

        if not loaded_from_cache:
            logger.info("  > Computing text history embeddings (this may take a while)...")
            h_text = encoder_text.encode(
                histories, convert_to_numpy=True, 
                show_progress_bar=True, batch_size=512
            ).astype(np.float32)
            
            if cache_dir:
                try:
                    np.save(cache_path, h_text)
                    logger.info(f"  ✓ Saved history embeddings to {cache_filename}")
                except Exception as e:
                    logger.warning(f"  ⚠️ Failed to save cache: {e}")

    if use_skill_text:
        logger.info("  > Computing skill text embeddings using job_ids...")
        
        # Step 1: Extract unique skills using job_ids
        logger.info("    Step 1: Extracting unique skills from job_ids...")
        unique_skills = extract_unique_skills_from_job_ids(filtered_job_ids, job_skill_map)
        logger.info(f"    Found {len(unique_skills)} unique skills")
        
        # Step 2: Get skill embeddings (precomputed or compute/cache)
        if precomputed_skill_embedding_map is not None:
            # Use precomputed embeddings - just filter to the skills we need
            logger.info("    Step 2: Using precomputed skill embeddings...")
            skill_embedding_map = precomputed_skill_embedding_map
            
            # Check coverage
            missing_skills = unique_skills - set(skill_embedding_map.keys())
            if missing_skills:
                logger.warning(f"    ⚠️ {len(missing_skills)} skills not found in precomputed embeddings (will use zero vectors)")
                if len(missing_skills) <= 10:
                    logger.warning(f"    Missing: {missing_skills}")
            
            # Get embedding dimension from precomputed embeddings
            if skill_embedding_map:
                first_emb = next(iter(skill_embedding_map.values()))
                skill_embed_dim = first_emb.shape[0]
                logger.info(f"    ✓ Precomputed skill embedding dim: {skill_embed_dim}")
            else:
                skill_embed_dim = embed_dim  # Fallback
        else:
            # Compute or load from cache (legacy behavior)
            logger.info("    Step 2: Loading/computing raw skill embeddings...")
            use_skill_prefix = use_skill_path_log_pooling or (encoder_skill is not encoder_text)
            skill_embedding_map = get_or_compute_skill_embeddings(
                unique_skills,
                encoder_skill,
                esco_skill_text_map,
                use_skill_description,
                cache_dir=cache_dir,
                encoder_name=encoder_skill_name,
                force_recompute=force_recompute,
                use_skill_prefix=use_skill_prefix,
            )
            skill_embed_dim = embed_dim
        
        if debug:
            logger.info("\n" + "🔍" * 30)
            logger.info("DEBUG: Showing skill aggregation for first sample (using job_ids)")
            logger.info("🔍" * 30 + "\n")
        
        # Step 3: Pool skills per sample using job_ids
        logger.info("    Step 3: Pooling skill vectors per sample (using job_ids)...")
        skill_vecs = []
        for idx, job_ids in enumerate(tqdm(filtered_job_ids, desc="    Pooling")):
            if use_skill_path_log_pooling:
                # Per-job pooling, then log-weight jobs
                job_vectors = []
                for job_id in job_ids:
                    infos = job_skill_map.get(job_id, [])
                    if not infos:
                        continue
                    job_vec = _pooled_skill_vec_optimized(
                        infos, skill_embedding_map, pooling_strategy,
                        alpha, beta, skill_embed_dim, debug=False,
                    )
                    job_vectors.append(job_vec)
                
                if job_vectors:
                    skill_vec = _pool_jobs_with_log_decay(job_vectors, alpha=skill_path_alpha_decay)
                else:
                    skill_vec = np.zeros(skill_embed_dim, dtype=np.float32)
            else:
                # Default: aggregate all skills across all jobs in history
                infos = _extract_skill_infos_by_job_ids(job_ids, job_skill_map)
                is_first_sample = (idx == 0)
                skill_vec = _pooled_skill_vec_optimized(
                    infos, skill_embedding_map, pooling_strategy,
                    alpha, beta, skill_embed_dim, debug=(debug and is_first_sample),
                )
            skill_vecs.append(skill_vec)
        
        h_skill = np.stack(skill_vecs, axis=0)
        logger.info(f"  ✓ Skill embeddings pooled: shape {h_skill.shape}")

    return filtered_pairs, filtered_job_ids, h_text, h_skill
