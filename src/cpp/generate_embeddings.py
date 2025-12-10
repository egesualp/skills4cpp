"""
Embedding generation and caching utilities for CPP training.

This module handles pre-computation and caching of embeddings to avoid
redundant computation across multiple experiments.
"""

import os
import pickle
from typing import Dict, List, Tuple, Optional
import numpy as np
from loguru import logger


def generate_cache_filename(args) -> str:
    """Generate a unique cache filename based on configuration parameters.
    
    This ensures different configurations get different cache files,
    enabling proper reproducibility and avoiding cache conflicts.
    """
    # Extract encoder name (use basename for brevity)
    encoder_name = args.encoder_text.split('/')[-1]
    encoder_skill_name = args.encoder_skill.split('/')[-1] if args.encoder_skill else "same"
    
    # Build filename components
    # Some callers (older scripts) may not define the newer attributes; use getattr
    # with safe defaults to keep this function backwards compatible.
    use_skill_path_log_pooling = getattr(args, "use_skill_path_log_pooling", False)
    skill_path_alpha_decay = getattr(args, "skill_path_alpha_decay", 0.5)

    components = [
        args.data_type,
        encoder_name,
        f"enc_skill_{encoder_skill_name}" if encoder_skill_name != "same" else None,
        f"text_hist" if args.use_text_history else None,
        f"text_desc" if args.use_text_description else None,
        f"skill_text" if args.use_skill_text else None,
        f"skill_desc" if args.use_skill_description else None,
        f"structured" if args.use_structured else None,
        f"pool_{args.pooling_strategy}",
        f"alpha{args.alpha}" if args.alpha != 1.0 else None,
        f"beta{args.beta}" if args.beta != 1.0 else None,
        f"last_job_only" if args.last_job_only else None,
        # skills_v2-style career-path skill pooling options
        "skillpathlog" if use_skill_path_log_pooling else None,
        (
            f"pathalpha{skill_path_alpha_decay}"
            if use_skill_path_log_pooling and skill_path_alpha_decay != 0.5
            else None
        ),
    ]
    
    # Filter out None values and join
    filename_parts = [c for c in components if c is not None]
    filename = "_".join(filename_parts) + ".pkl"
    
    return filename


def load_embeddings_from_cache(cache_filepath: str) -> Optional[Dict]:
    """Load pre-computed embeddings from cache file.
    
    Args:
        cache_filepath: Path to the cache file
        
    Returns:
        Dictionary containing cached embeddings and metadata, or None if loading fails
    """
    logger.info("=" * 80)
    logger.info("📦 LOADING PRE-COMPUTED EMBEDDINGS FROM CACHE")
    logger.info("=" * 80)
    logger.info(f"Cache file: {cache_filepath}")
    
    try:
        with open(cache_filepath, 'rb') as f:
            cache = pickle.load(f)
        
        # Calculate cache size
        cache_size_gb = os.path.getsize(cache_filepath) / (1024**3)
        
        logger.info(f"  ✓ Loaded embeddings from cache ({cache_size_gb:.2f} GB)")
        logger.info(f"  ✓ Target embedding dim: {cache['output_dim']}")
        logger.info(f"  ✓ Train samples: {len(cache['train_pairs'])}")
        logger.info(f"  ✓ Val samples: {len(cache['val_pairs'])}")
        logger.info(f"  ✓ Test samples: {len(cache['test_pairs'])}")
        logger.info(f"  ✅ Skipped ~2 hours of computation time!")
        logger.info("=" * 80 + "\n")
        
        return cache
        
    except Exception as e:
        logger.error(f"  ❌ Failed to load cache: {e}")
        logger.warning(f"  ⚠️  Will recompute embeddings from scratch...")
        return None


def compute_and_cache_embeddings(
    cache_filepath: str,
    train_pairs: List[Tuple[str, str]],
    val_pairs: List[Tuple[str, str]],
    test_pairs: List[Tuple[str, str]],
    encoder_text,
    encoder_skill,
    job_skill_map: Dict,
    esco_skill_text_map: Dict,
    args,
    precompute_target_embeddings_func,
    precompute_input_embeddings_func
) -> Dict:
    """Compute embeddings and save to cache.
    
    Args:
        cache_filepath: Path where cache will be saved
        train_pairs, val_pairs, test_pairs: Data pairs
        encoder_text: Text encoder model
        encoder_skill: Skill encoder model
        job_skill_map: Job to skill mapping
        esco_skill_text_map: ESCO skill text mapping
        args: Training arguments
        precompute_target_embeddings_func: Function to compute target embeddings
        precompute_input_embeddings_func: Function to compute input embeddings
        
    Returns:
        Dictionary containing all computed embeddings
    """
    logger.info("=" * 80)
    logger.info("🔄 COMPUTING EMBEDDINGS (will be cached for future runs)")
    logger.info("=" * 80)
    if args.force_recompute:
        logger.info("  > Force recompute flag is set")
    logger.info(f"Cache will be saved to: {cache_filepath}\n")
    
    # --- Step 4: Pre-compute target embeddings ---
    logger.info("[4/7] Pre-computing target embeddings...")
    actual_labels = list(set([pair[1] for pair in train_pairs + val_pairs + test_pairs]))
    Y_target_dict = precompute_target_embeddings_func(encoder_text, actual_labels, show_progress=True)
    Y_target_all = np.array(list(Y_target_dict.values()))
    output_dim = Y_target_all.shape[1]
    logger.info(f"  ✓ Target embedding dim: {output_dim}\n")
    
    # --- Step 4b: Pre-compute input embeddings ---
    logger.info("[4b/7] Pre-computing input embeddings...")

    logger.info(f"  > Pre-computing input embeddings for train pairs...")
    train_pairs, train_h_text, train_h_skill = precompute_input_embeddings_func(
        train_pairs,
        Y_target_dict,
        encoder_text,
        encoder_skill,
        job_skill_map,
        esco_skill_text_map,
        use_skill_description=args.use_skill_description,
        pooling_strategy=args.pooling_strategy,
        alpha=args.alpha,
        beta=args.beta,
        use_text_history=args.use_text_history,
        use_skill_text=args.use_skill_text,
        # skills_v2-style career path pooling (safe for callers that define these)
        use_skill_path_log_pooling=getattr(args, "use_skill_path_log_pooling", False),
        skill_path_alpha_decay=getattr(args, "skill_path_alpha_decay", 0.5),
    )
    
    logger.info(f"  > Pre-computing input embeddings for val pairs...")
    val_pairs, val_h_text, val_h_skill = precompute_input_embeddings_func(
        val_pairs,
        Y_target_dict,
        encoder_text,
        encoder_skill,
        job_skill_map,
        esco_skill_text_map,
        use_skill_description=args.use_skill_description,
        pooling_strategy=args.pooling_strategy,
        alpha=args.alpha,
        beta=args.beta,
        use_text_history=args.use_text_history,
        use_skill_text=args.use_skill_text,
        use_skill_path_log_pooling=getattr(args, "use_skill_path_log_pooling", False),
        skill_path_alpha_decay=getattr(args, "skill_path_alpha_decay", 0.5),
    )
    
    logger.info(f"  > Pre-computing input embeddings for test pairs...")
    test_pairs, test_h_text, test_h_skill = precompute_input_embeddings_func(
        test_pairs,
        Y_target_dict,
        encoder_text,
        encoder_skill,
        job_skill_map,
        esco_skill_text_map,
        use_skill_description=args.use_skill_description,
        pooling_strategy=args.pooling_strategy,
        alpha=args.alpha,
        beta=args.beta,
        use_text_history=args.use_text_history,
        use_skill_text=args.use_skill_text,
        use_skill_path_log_pooling=getattr(args, "use_skill_path_log_pooling", False),
        skill_path_alpha_decay=getattr(args, "skill_path_alpha_decay", 0.5),
    )
    logger.info(f"  ✓ Input embeddings pre-computed\n")
    
    # --- Save to cache ---
    logger.info(f"💾 Saving embeddings to cache...")
    try:
        cache_data = {
            'Y_target_dict': Y_target_dict,
            'Y_target_all': Y_target_all,
            'output_dim': output_dim,
            'train_pairs': train_pairs,
            'train_h_text': train_h_text,
            'train_h_skill': train_h_skill,
            'val_pairs': val_pairs,
            'val_h_text': val_h_text,
            'val_h_skill': val_h_skill,
            'test_pairs': test_pairs,
            'test_h_text': test_h_text,
            'test_h_skill': test_h_skill,
            # Also save metadata for verification
            'metadata': {
                'data_type': args.data_type,
                'encoder_text': args.encoder_text,
                'encoder_skill': args.encoder_skill,
                'use_text_history': args.use_text_history,
                'use_text_description': args.use_text_description,
                'use_skill_text': args.use_skill_text,
                'use_skill_description': args.use_skill_description,
                'use_structured': args.use_structured,
                'pooling_strategy': args.pooling_strategy,
                'alpha': args.alpha,
                'beta': args.beta,
                'last_job_only': args.last_job_only,
            }
        }
        
        with open(cache_filepath, 'wb') as f:
            pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        cache_size_gb = os.path.getsize(cache_filepath) / (1024**3)
        logger.info(f"  ✓ Cache saved successfully ({cache_size_gb:.2f} GB)")
        logger.info(f"  ✓ Future runs with same config will load instantly!")
        logger.info("=" * 80 + "\n")
        
        return cache_data
        
    except Exception as e:
        logger.error(f"  ❌ Failed to save cache: {e}")
        logger.warning(f"  ⚠️  Continuing without cache (will need to recompute next time)")
        
        # Return data even if caching fails
        return {
            'Y_target_dict': Y_target_dict,
            'Y_target_all': Y_target_all,
            'output_dim': output_dim,
            'train_pairs': train_pairs,
            'train_h_text': train_h_text,
            'train_h_skill': train_h_skill,
            'val_pairs': val_pairs,
            'val_h_text': val_h_text,
            'val_h_skill': val_h_skill,
            'test_pairs': test_pairs,
            'test_h_text': test_h_text,
            'test_h_skill': test_h_skill,
        }


def get_or_compute_embeddings(
    train_pairs: List[Tuple[str, str]],
    val_pairs: List[Tuple[str, str]],
    test_pairs: List[Tuple[str, str]],
    encoder_text,
    encoder_skill,
    job_skill_map: Dict,
    esco_skill_text_map: Dict,
    args,
    precompute_target_embeddings_func,
    precompute_input_embeddings_func
) -> Tuple:
    """Main function to get embeddings from cache or compute them.
    
    This is the primary interface for the training script.
    
    Returns:
        Tuple of (Y_target_dict, Y_target_all, output_dim, 
                 train_pairs, train_h_text, train_h_skill,
                 val_pairs, val_h_text, val_h_skill,
                 test_pairs, test_h_text, test_h_skill)
    """
    # Setup cache directory and filepath
    os.makedirs(args.embeddings_cache_dir, exist_ok=True)
    cache_filename = generate_cache_filename(args)
    cache_filepath = os.path.join(args.embeddings_cache_dir, cache_filename)
    
    # Try to load from cache
    cache = None
    if os.path.exists(cache_filepath) and not args.force_recompute:
        cache = load_embeddings_from_cache(cache_filepath)
    
    # Compute if cache doesn't exist or loading failed
    if cache is None:
        cache = compute_and_cache_embeddings(
            cache_filepath, train_pairs, val_pairs, test_pairs,
            encoder_text, encoder_skill, job_skill_map, esco_skill_text_map,
            args, precompute_target_embeddings_func, precompute_input_embeddings_func
        )
    
    # Return in the expected format
    return (
        cache['Y_target_dict'],
        cache['Y_target_all'],
        cache['output_dim'],
        cache['train_pairs'],
        cache['train_h_text'],
        cache['train_h_skill'],
        cache['val_pairs'],
        cache['val_h_text'],
        cache['val_h_skill'],
        cache['test_pairs'],
        cache['test_h_text'],
        cache['test_h_skill']
    )

