"""
Utility functions for loading skill data, vocabularies, and helper maps.

These functions are shared between the pre-computation script (generate_embeddings.py)
and the on-the-fly dataset (cpp_dataset.py).
"""

import json
import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional, Set
from loguru import logger


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
        
        # Fill NaN with 0 for skills that don't appear in train+val (e.g., test-only skills)
        n_missing = df_full['idf'].isna().sum()
        if n_missing > 0:
            print(f"  > {n_missing} job-skill pairs have skills not in train+val (setting IDF=0)")
        df_full['idf'] = df_full['idf'].fillna(0)
        
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


def precompute_target_embeddings(encoder, labels: list, show_progress: bool = True) -> dict:
    """
    Pre-compute embeddings for all target job labels.

    Args:
        encoder: SentenceTransformer encoder model
        labels: List of unique target job strings
        show_progress: Whether to show progress bar

    Returns:
        Dictionary mapping target strings to their embeddings
    """
    print(f"Pre-computing embeddings for {len(labels)} target labels...")
    target_embeddings = encoder.encode(labels, show_progress_bar=show_progress, convert_to_numpy=True, batch_size=512)
    Y_target_dict = dict(zip(labels, target_embeddings))
    print(f"  > Created target embedding dictionary with {len(Y_target_dict)} entries.")
    return Y_target_dict


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
            text = f"role: {st['name']} \n description: {st['desc']}" if use_skill_description else st['name']
            
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


def precompute_skill_embeddings(unique_skill_uris: set, 
                                encoder_skill, 
                                esco_skill_text_map: Dict[str, Dict],
                                use_skill_description: bool,
                                use_skill_prefix: bool = False) -> Dict[str, np.ndarray]:
    """Pre-compute embeddings for all unique skills in the dataset.
    
    This function encodes all unique skills once, which is much more efficient than
    encoding the same skill multiple times across different samples.
    
    Args:
        unique_skill_uris: Set of unique skill URIs to encode
        encoder_skill: Encoder for skills
        esco_skill_text_map: Map from skill URIs to text
        use_skill_description: Whether to include skill descriptions
        
    Returns:
        Dictionary mapping skill URIs to their embeddings
    """
    skill_texts = []
    skill_uris_ordered = []
    
    for uri in unique_skill_uris:
        if uri in esco_skill_text_map:
            st = esco_skill_text_map[uri]
            if use_skill_prefix:
                # Match the skill text template used in train_cpp_skills_v2.encode_skills
                text = (
                    f"skill: {st['name']} \n description: {st['desc']}"
                    if use_skill_description
                    else f"skill: {st['name']}"
                )
            else:
                text = (
                    f"role: {st['name']} \n description: {st['desc']}"
                    if use_skill_description
                    else st['name']
                )
            skill_texts.append(text)
            skill_uris_ordered.append(uri)
    
    # Encode all skills at once (batch encoding is efficient)
    print(f"  > Encoding {len(skill_texts)} unique skills (batch encoding)...")
    skill_embeddings = encoder_skill.encode(skill_texts, convert_to_numpy=True, show_progress_bar=True, batch_size=512)
    
    # Create lookup dictionary
    skill_embedding_map = dict(zip(skill_uris_ordered, skill_embeddings))
    return skill_embedding_map


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
) -> Tuple[List[Tuple[str, str]], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Pre-compute input embeddings for text history and skill text.
    
    OPTIMIZED VERSION: This function now pre-encodes all unique skills once
    instead of encoding the same skill multiple times for efficiency.

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

    Returns:
        Tuple of (filtered_pairs, h_text_embeddings, h_skill_embeddings)
    """
    # Filter pairs to match target dictionary (same as dataset does internally)
    filtered_pairs = [(h, t) for (h, t) in data_pairs if t in Y_target_dict]

    embed_dim = encoder_text.get_sentence_embedding_dimension()
    h_text = None
    h_skill = None

    if use_text_history:
        print("  > Pre-computing text history embeddings...")
        histories = [h for (h, _) in filtered_pairs]
        h_text = encoder_text.encode(histories, convert_to_numpy=True, show_progress_bar=True, batch_size=512).astype(np.float32)

    if use_skill_text:
        print("  > Pre-computing skill text embeddings (OPTIMIZED)...")
        
        # *** NEW: Extract unique skills first ***
        print("  > Step 1: Extracting unique skills from dataset...")
        unique_skills = extract_unique_skills_from_dataset(filtered_pairs, job_skill_map)
        print(f"  > Found {len(unique_skills)} unique skills in dataset")
        
        # Calculate total skills if we encoded naively
        total_skill_instances = sum(
            len(_extract_skill_infos(h, job_skill_map)) 
            for h, _ in filtered_pairs
        )
        print(f"  > Total skill instances across all samples: {total_skill_instances}")
        if total_skill_instances > 0:
            speedup_ratio = total_skill_instances / max(len(unique_skills), 1)
            print(f"  > Efficiency gain: ~{speedup_ratio:.1f}x (encoding {len(unique_skills)} instead of {total_skill_instances})")
        
        # *** Pre-encode all unique skills once ***
        print("  > Step 2: Pre-encoding all unique skills...")
        # Use the "skill: ..." template in two cases:
        #   1) When we explicitly enable skills_v2-style path pooling, to match
        #      the finetuning setup used in train_cpp_skills_v2.py
        #   2) When a dedicated skill encoder is provided (encoder_skill != encoder_text),
        #      to make the input format clearly skill-centric for that encoder.
        use_skill_prefix = use_skill_path_log_pooling or (encoder_skill is not encoder_text)
        skill_embedding_map = precompute_skill_embeddings(
            unique_skills,
            encoder_skill,
            esco_skill_text_map,
            use_skill_description,
            use_skill_prefix=use_skill_prefix,
        )
        
        if debug:
            logger.info("\n" + "🔍" * 30)
            logger.info("DEBUG: Showing skill aggregation for first sample")
            logger.info("🔍" * 30 + "\n")
        
        # *** Now process samples using pre-computed embeddings ***
        print("  > Step 3: Aggregating skill vectors per sample...")
        skill_vecs = []
        for idx, (h, _) in enumerate(tqdm(filtered_pairs, desc="  > Aggregating")):
            # Two modes:
            #  - Legacy (default): aggregate all skills across the full history
            #  - skills_v2-style: pool skills per job with IDF, then log-weight jobs
            if use_skill_path_log_pooling:
                # Extract ordered job titles for this history
                job_titles = _extract_job_titles_from_history(h)
                job_vectors = []
                for title in job_titles:
                    title_clean = title.strip().lower()
                    if title_clean in job_skill_map:
                        infos = job_skill_map[title_clean]
                    else:
                        infos = []
                    if not infos:
                        continue
                    job_vec = _pooled_skill_vec_optimized(
                        infos,
                        skill_embedding_map,
                        pooling_strategy,
                        alpha,
                        beta,
                        embed_dim,
                        debug=False,
                    )
                    job_vectors.append(job_vec)
                
                if job_vectors:
                    skill_vec = _pool_jobs_with_log_decay(job_vectors, alpha=skill_path_alpha_decay)
                else:
                    skill_vec = np.zeros(embed_dim, dtype=np.float32)
            else:
                infos = _extract_skill_infos(h, job_skill_map)
                # Only debug the first sample to avoid flooding logs
                is_first_sample = (idx == 0)
                skill_vec = _pooled_skill_vec_optimized(
                    infos,
                    skill_embedding_map,
                    pooling_strategy,
                    alpha,
                    beta,
                    embed_dim,
                    debug=(debug and is_first_sample),
                )
            skill_vecs.append(skill_vec)
        h_skill = np.stack(skill_vecs, axis=0)
        print(f"  ✓ Skill embeddings computed: shape {h_skill.shape}")

    return filtered_pairs, h_text, h_skill



