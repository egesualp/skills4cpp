"""
Script to analyze the variance and spread characteristics of text embeddings vs. pooled skill embeddings.
"""

import argparse
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
from scipy.stats import skew, kurtosis

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.cpp.data_classes import Data
from src.cpp.data_loaders import (
    load_job_skill_data_by_id, 
    load_all_vocabs, 
    precompute_target_embeddings,
    load_precomputed_skill_embeddings,
    load_precomputed_skill_embeddings,
    precompute_input_embeddings_with_job_ids
)
try:
    from src.cpp.train_cpp_enhanced_v3 import build_last_job_skill_embeddings
    from src.cpp.skill_pooling import (
        cap_skills_per_job_lexicographic,
        cap_skills_per_job_by_score,
        load_skill_descriptions,
        calculate_idf_scores_by_job_id,
        load_skill_mappings # Ensure this is available if needed, or rely on load_job_skill_data_by_id
    )
except ImportError:
    logger.warning("V3 modules not found. V3 analysis will fail if requested.")

class EmbeddingVarianceAnalyzer:
    def __init__(self, embeddings_dict):
        """
        Initialize analyzer with a dictionary of embeddings.
        embeddings_dict: {name: embedding_matrix (N, D)}
        """
        self.embeddings = embeddings_dict
        self.results = {}
        
    def analyze_pca_variance(self, n_components=50):
        """Compare explained variance concentration."""
        logger.info("Computing PCA explained variance...")
        result = {}
        
        for name, emb in self.embeddings.items():
            # Standardize logic: handle potentially large matrices
            # Using subset if too large? PCA is usually fine for <10k samples
            pca = PCA(n_components=min(n_components, emb.shape[1], emb.shape[0]))
            pca.fit(emb)
            cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
            
            result[name] = {
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'cumulative_variance': cumulative_variance,
                'top_10': cumulative_variance[9] if len(cumulative_variance) > 9 else cumulative_variance[-1],
                'top_20': cumulative_variance[19] if len(cumulative_variance) > 19 else cumulative_variance[-1],
                'top_50': cumulative_variance[49] if len(cumulative_variance) > 49 else cumulative_variance[-1]
            }
        
        self.results['pca'] = result
        return result
    
    def analyze_pairwise_similarity(self, n_sample=1000, seed=42):
        """Compare within-space pairwise similarity distributions."""
        logger.info(f"Computing pairwise similarities (sampled n={n_sample})...")
        rng = np.random.default_rng(seed)
        result = {}
        
        for name, emb in self.embeddings.items():
            n = emb.shape[0]
            if n > n_sample:
                indices = rng.choice(n, n_sample, replace=False)
                sample_emb = emb[indices]
            else:
                sample_emb = emb
                
            # Compute cosine similarity
            # Normalize first to ensure numerical stability
            norm = np.linalg.norm(sample_emb, axis=1, keepdims=True)
            norm[norm == 0] = 1e-10
            normalized_emb = sample_emb / norm
            
            sim_matrix = normalized_emb @ normalized_emb.T
            
            # Extract upper triangle
            upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
            
            result[name] = {
                'mean': np.mean(upper_tri),
                'std': np.std(upper_tri),
                'min': np.min(upper_tri),
                'max': np.max(upper_tri),
                'skew': skew(upper_tri),
                'values': upper_tri  # Store for histogram
            }
            
        self.results['pairwise'] = result
        return result
    
    def compute_cka(self):
        """Compute CKA between text and skill spaces."""
        logger.info("Computing CKA similarity...")
        
        def cka_linear(X, Y):
            # Center columns
            X = X - X.mean(axis=0)
            Y = Y - Y.mean(axis=0)
            
            XXT = X @ X.T
            YYT = Y @ Y.T
            
            hsic_xy = np.sum(XXT * YYT)
            hsic_xx = np.sum(XXT * XXT)
            hsic_yy = np.sum(YYT * YYT)
            
            if hsic_xx == 0 or hsic_yy == 0:
                return 0.0
                
            return hsic_xy / (np.sqrt(hsic_xx) * np.sqrt(hsic_yy))
        
        # We compute CKA between 'text' and all other keys
        if 'text' not in self.embeddings:
            logger.warning("No 'text' embedding found for CKA baseline.")
            return {}
            
        text_emb = self.embeddings['text']
        result = {}
        
        # Use subset for CKA if dataset is large (it uses N^2 memory)
        n = text_emb.shape[0]
        if n > 2000:
            logger.info("Subsampling for CKA computation (N=2000)...")
            indices = np.random.choice(n, 2000, replace=False)
            text_sub = text_emb[indices]
        else:
            indices = np.arange(n)
            text_sub = text_emb
            
        for name, emb in self.embeddings.items():
            if name == 'text': 
                continue
                
            emb_sub = emb[indices]
            score = cka_linear(text_sub, emb_sub)
            result[name] = score
            
        self.results['cka'] = result
        return result
    
    def generate_visualization(self, save_path):
        """Create multi-panel figure with thesis-standard font sizes."""

        logger.info(f"Generating visualization to {save_path}...")

        n_embs = len(self.embeddings)

        # 1. Global styling
        plt.style.use('seaborn-v0_8-whitegrid')
        # Increased height to 14 to accommodate larger titles and labels without overlap
        fig = plt.figure(figsize=(20, 14))

        LABEL_FONT = 16
        TITLE_FONT = 20
        TICK_FONT = 14

        # 2. Similarity Distributions (Top Panel)
        # Using colspan to make this a wide, hero-style distribution plot
        ax_hist = plt.subplot2grid((2, n_embs), (0, 0), colspan=n_embs)

        for name, stats in self.results['pairwise'].items():
            sns.kdeplot(
                stats['values'], 
                label=f"{name} ($\mu$={stats['mean']:.2f})", 
                ax=ax_hist, 
                fill=True, 
                alpha=0.3,
                linewidth=2
            )

        ax_hist.set_title("Pairwise Cosine Similarity Distribution", fontsize=TITLE_FONT, fontweight='bold', pad=20)
        ax_hist.set_xlabel("Cosine Similarity", fontsize=LABEL_FONT, labelpad=10)
        ax_hist.set_ylabel("Density", fontsize=LABEL_FONT, labelpad=10)
        ax_hist.set_xlim(-0.2, 1.0)
        ax_hist.tick_params(labelsize=TICK_FONT)
        ax_hist.legend(fontsize=14, loc='upper left')

        # 3. PCA Scatter Plots (Bottom Row)
        for i, (name, emb) in enumerate(self.embeddings.items()):
            ax = plt.subplot2grid((2, n_embs), (1, i))

            # Compute 2D PCA
            pca = PCA(n_components=2)
            emb_2d = pca.fit_transform(emb)

            # Plot with slightly larger points for visibility
            ax.scatter(emb_2d[:, 0], emb_2d[:, 1], alpha=0.5, s=15, color='#2c3e50', edgecolors='none')

            # Get Variance info
            cum_var = self.results['pca'][name]['cumulative_variance']
            top_2_var = cum_var[1]
            top_10_var = cum_var[9] if len(cum_var) > 9 else cum_var[-1]

            # Multi-line title for clarity
            ax.set_title(f"{name}\nVar: PC1+2={top_2_var:.1%}", fontsize=16, fontweight='semibold', pad=15)
            ax.set_xlabel("PC1", fontsize=LABEL_FONT)
            ax.set_ylabel("PC2", fontsize=LABEL_FONT)
            ax.tick_params(labelsize=TICK_FONT)

        # Standardizing layout
        plt.tight_layout(pad=3.0) # Increased padding to prevent label overlap
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_report(self):
        """Print summary statistics."""
        print("\n" + "="*80)
        print("EMBEDDING VARIANCE ANALYSIS REPORT")
        print("="*80)
        
        print("\n1. PCA Explained Variance (Cumulative)")
        print(f"{'Embedding':<20} {'Top 10':<10} {'Top 20':<10} {'Top 50':<10}")
        print("-" * 55)
        for name, stats in self.results['pca'].items():
            print(f"{name:<20} {stats['top_10']:.1%}      {stats['top_20']:.1%}      {stats['top_50']:.1%}")
            
        print("\n2. Pairwise Similarity Statistics")
        print(f"{'Embedding':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'Max':<10}")
        print("-" * 65)
        for name, stats in self.results['pairwise'].items():
            print(f"{name:<20} {stats['mean']:.3f}      {stats['std']:.3f}      {stats['min']:.3f}      {stats['max']:.3f}")
            
        if 'cka' in self.results and self.results['cka']:
            print("\n3. CKA Similarity (vs Text)")
            print("-" * 30)
            for name, score in self.results['cka'].items():
                print(f"{name:<20} {score:.4f}")
        
        print("="*80 + "\n")

def pool_skill_embeddings(skill_uris, job_skill_map, skill_embedding_map, strategy="mean", idf_map=None, alpha=1.0, beta=1.0):
    """
    Pool skill embeddings for a list of skill URIs.
    """
    # This logic needs to mirror `CareerPathDataset._generate_skill_text_embedding` 
    # but strictly operating on pre-computed embeddings
    
    vectors = []
    weights = []
    
    # We expect skill_uris to be a list of dicts with {'skillUri', 'score', ...}
    for info in skill_uris:
        uri = info['skillUri']
        score = info.get('score', 1.0)
        
        if uri in skill_embedding_map:
            vectors.append(skill_embedding_map[uri])
            
            if strategy == "mean":
                weights.append(1.0)
            elif strategy == "weighted_idf":
                # Check for idf in info or look it up
                idf = info.get('idf', 0.0)
                if idf_map and uri in idf_map:
                    idf = idf_map[uri]
                    
                w = (score ** alpha) * (idf ** beta)
                weights.append(w)
            elif strategy == "log_pooling":
                # For log pooling, we need the position. Usually applied at JOB level not SKILL level
                # But if we treat skills as a sequence (which they aren't usually in a single job), it's weird.
                # Usually log pooling is for `_pool_jobs_with_log_decay`.
                # If we are pooling skills within a SINGLE job, log pooling assumes order matters (rank).
                # If the input list is ordered by importance, then yes.
                weights.append(1.0) # Placeholder, will apply position weights after
            else:
                weights.append(1.0)
    
    if not vectors:
        return np.zeros(next(iter(skill_embedding_map.values())).shape, dtype=np.float32)
        
    vectors = np.array(vectors)
    weights = np.array(weights)
    
    if strategy == "log_pooling":
        # Logarithmic decay based on rank (index)
        # w_i = 1 / log(i + 2) ?? Or the formula from code: log(1 + alpha * i)
        # Using formula from `data_loaders.py`: w_i = log(1 + alpha * i)
        # Wait, `_pool_jobs_with_log_decay` uses increasing weights for later items (history).
        # For SKILLS in a job, if they are ranked by importance (high to low), we probably want high weights for first items.
        # But commonly we pool skills with simple weighted average.
        # The prompt asked for "Log Pooling" - let's assume standard rank decay: 1/log(rank+1) or similar?
        # OR does it refer to pooling multiple JOBS in history? 
        # "Multimodal: Job Titles + Skills (Names, Log Pooling)" in config implies History Pooling.
        # BUT here we are comparing Embeddings. The Text Embedding is of the WHOLE HISTORY.
        # So we should pool the Skills of the WHOLE HISTORY.
        pass

    if np.sum(weights) == 0:
        return np.mean(vectors, axis=0)
        
    return np.average(vectors, axis=0, weights=weights)


def get_history_skill_embeddings(history_str, job_skill_map, skill_embedding_map, strategy="mean", alpha=1.0, beta=1.0):
    """
    Extract skills from history string and pool them.
    Treats history as a sequence of jobs.
    """
    # 1. Extract jobs from history
    from src.cpp.utils import SEP_TOKEN
    
    # Simple extraction by splitting SEP_TOKEN (assuming pure text history or similar format)
    # The `_extract_job_titles_from_history` handles "role: ... \n description: ..."
    # We will replicate basic logic here or import if possible.
    # For robust extraction, we should use the same logic as Dataset.
    
    # Actually, simpler: in `variance_analysis.py` we can load data with job_ids directly using `Data` class.
    # `train_pairs` comes with `train_job_ids`. We should use `train_job_ids`.
    pass 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_type", type=str, default="decorte")
    parser.add_argument("--skill_embeddings_dir", type=str, 
                       default="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index")
    parser.add_argument("--skill_scores_file", type=str, 
                       default="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_h2_pjmath/fused_predictions.json")
    parser.add_argument("--embeddings_cache_dir", type=str, 
                       default="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/embeddings",
                       help="Directory to save/load pre-computed embeddings cache")
    parser.add_argument("--encoder_text", type=str, default="ElenaSenger/career-path-representation-mpnet-decorte")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples")
    parser.add_argument("--alpha", type=float, default=1.0, help="Confidence weight alpha")
    parser.add_argument("--beta", type=float, default=1.0, help="IDF weight beta")
    
    # V3 Arguments
    parser.add_argument("--mode", type=str, default="v2", choices=["v2", "v3", "both"], help="Analysis mode")
    parser.add_argument("--encoder_skill", type=str, default="", help="Skill encoder for V3")
    parser.add_argument("--top_k_skills", type=int, default=10, help="Top K skills for V3")
    parser.add_argument("--skill_selection_strategy", type=str, default="top_k", choices=["top_k", "stratified"])
    parser.add_argument("--scoring_mode", type=str, default="idf_only", choices=["idf_only", "scores_only", "weighted"])
    parser.add_argument("--importance_weight", type=float, default=0.5)
    
    args = parser.parse_args()
    
    # 1. Load Data (Test Split)
    logger.info("Loading Data...")
    data = Data(DATA_TYPE=args.data_type, ONLY_TITLES=False, consider_subspans=True, LOAD_CLEAN_TEST=False)
    # We only care about Test Set
    _, _, (test_pairs, test_job_ids) = data.get_data_with_job_ids(stage='transformation_finetuning', include_clean_test=False)
    
    if args.limit > 0:
        test_pairs = test_pairs[:args.limit]
        test_job_ids = test_job_ids[:args.limit]
        
    logger.info(f"Loaded {len(test_pairs)} test samples.")
    
    # 2. Load Maps
    logger.info("Loading Skill Maps...")
    # Dummy values for files not strictly needed if we just matching job_ids
    # We need skill_scores_file to know which skills map to which job_id
    job_skill_map, esco_skill_text_map, _ = load_job_skill_data_by_id(
        skill_scores_file=args.skill_scores_file,
        esco_skills_file="data/esco_datasets/skills_en.csv", # Default path, hopefully exists or relative
        skill_properties_file="", # Not needed
        pooling_strategy="weighted_idf", # Pre-calc IDF
        alpha=args.alpha,
        beta=args.beta,
        # We need IDF for train+val, so we might need to load them or just use test for this analysis?
        # Ideally we load all to get correct IDF.
        # For simplicity, calculating IDF on the fly for loaded data or assume it's in the map.
        # `load_job_skill_data_by_id` calculates IDF if we don't pass train_val_job_ids.
        # It will calculate on ALL loaded jobs if we don't pass filter. 
        # We will let it calculate on all for this analysis.
    )
    
    # 3. Load Embeddings
    logger.info("Loading Embeddings...")
    
    # Text Encoder
    logger.info(f"Loading Text Encoder: {args.encoder_text}")
    text_encoder = SentenceTransformer(args.encoder_text)
    
    # Skill Embeddings
    logger.info(f"Loading Skill Embeddings from {args.skill_embeddings_dir}")
    skill_emb_map = load_precomputed_skill_embeddings(args.skill_embeddings_dir)
    skill_dim = next(iter(skill_emb_map.values())).shape[0]
    
    # 4. Generate Embeddings
    embeddings_dict = {}
    
    # A. Text History Embeddings (Using Precompute + Cache)
    logger.info("Generating/Loading Text History Embeddings...")
    
    # Needs Y_target_dict for filtering (to match cache hash)
    # We pass empty list for targets to only load the cache map, not compute new ones if possible
    # But precompute_target_embeddings requires list of targets.
    # To match train script, we ideally should have same targets.
    # Let's collect targets from test_pairs as a baseline.
    test_targets = sorted(list(set([p[1] for p in test_pairs])))
    Y_target_dict, _ = precompute_target_embeddings(
        text_encoder, 
        test_targets, 
        show_progress=True,
        cache_dir=args.embeddings_cache_dir,
        encoder_name=args.encoder_text.split('/')[-1]
    )
    
    # Compute inputs (will hit cache if available)
    # We set use_skill_text=False to only care about text history here
    filtered_pairs, filtered_job_ids, h_text, _ = precompute_input_embeddings_with_job_ids(
        data_pairs=test_pairs,
        job_ids_list=test_job_ids,
        Y_target_dict=Y_target_dict,
        encoder_text=text_encoder,
        encoder_skill=None, # Not needed for text only
        job_skill_map=job_skill_map,
        esco_skill_text_map=esco_skill_text_map,
        use_text_history=True,
        use_skill_text=False, # We compute skills manually below
        cache_dir=args.embeddings_cache_dir,
        split_name="test_analysis"
    )
    
    # Note: filtered_pairs might be slightly smaller if some targets missing from Y_target_dict (unlikely if we used test_targets)
    # Update pairs/ids to match the text embeddings
    test_pairs = filtered_pairs
    test_job_ids = filtered_job_ids
    embeddings_dict['text'] = h_text
    
    # B. Pooled Skill Embeddings
    strategies = ['mean', 'weighted_idf', 'log_pooling']
    
    # Pre-calculate log weights for max history length
    # log_pooling in `train_cpp` usually refers to pooling JOBS in history
    # We align with `_pool_jobs_with_log_decay` in data_loaders.py
    
    for strategy in strategies:
        if args.mode == 'v3': break # Skip V2 strategies if only V3 requested
        logger.info(f"Generating Skill Embeddings ({strategy})...")
        pooled_embs = []
        
        for job_ids, pair in tqdm(zip(test_job_ids, test_pairs), total=len(test_job_ids)):
            # "job_ids" is a list of job IDs in the history
            # For each job, we get its skills, pool them (usually mean or weighted), 
            # then we pool the sequence of job-vectors.
            
            job_vectors = []
            
            for i, job_id in enumerate(job_ids):
                job_id_str = str(job_id)
                if job_id_str in job_skill_map:
                    skills = job_skill_map[job_id_str]
                    # Pool skills for this job
                    # Default skill pooling is usually weighted_idf or mean
                    # The 'strategy' arg primarily controls how JOBS are pooled or global skill pooling?
                    # In `train_cpp`, `pooling_strategy` controls SKILL pooling.
                    # And `use_skill_path_log_pooling` controls JOB pooling.
                    
                    # For this analysis:
                    # 'mean': Mean of all skills in history (flattened) OR Mean of Job vectors (where job is mean of skills)
                    # 'weighted_idf': Weighted Mean of skills 
                    # 'log_pooling': Weighted Mean of skills (per job) + Log Decay of Jobs
                    
                    # Implementation:
                    # 1. Get embedding for each skill
                    # 2. Pool skills to get job vector (using weighted_idf or mean)
                    # 3. Pool job vectors (using mean or log decay)
                    
                    skill_vecs = []
                    skill_weights = []
                    
                    for s in skills:
                        if s['skillUri'] in skill_emb_map:
                            skill_vecs.append(skill_emb_map[s['skillUri']])
                            
                            # Weighting for WITHIN-JOB pooling
                            if strategy in ['weighted_idf', 'log_pooling']:
                                w = (s.get('score', 1.0) ** args.alpha) * (s.get('idf', 0.0) ** args.beta)
                                skill_weights.append(w)
                            else:
                                skill_weights.append(1.0)
                                
                    if skill_vecs:
                        skill_vecs = np.array(skill_vecs)
                        skill_weights = np.array(skill_weights)
                        if np.sum(skill_weights) > 0:
                            job_vec = np.average(skill_vecs, axis=0, weights=skill_weights)
                        else:
                            job_vec = np.mean(skill_vecs, axis=0)
                        job_vectors.append(job_vec)
                        
            # Now pool job vectors
            if not job_vectors:
                pooled_embs.append(np.zeros(skill_dim, dtype=np.float32))
                continue
                
            job_vectors = np.array(job_vectors)
            
            if strategy == 'log_pooling':
                # Log decay pooling for job sequence
                # w_i = log(1 + 0.5 * i)
                n_jobs = len(job_vectors)
                positions = np.arange(n_jobs, dtype=np.float32)
                # alpha decay 0.5 is default in code
                job_weights = np.log(1.0 + 0.5 * positions)
                
                # Normalize
                if np.sum(job_weights) > 0:
                     final_vec = np.average(job_vectors, axis=0, weights=job_weights)
                else:
                     final_vec = np.mean(job_vectors, axis=0)
                     
            else:
                # Mean pooling of jobs for 'mean' and 'weighted_idf' strategies
                # (Unless 'weighted_idf' implies something else? Usually strictly refers to skill weighting)
                final_vec = np.mean(job_vectors, axis=0)
                
            pooled_embs.append(final_vec)
            
        embeddings_dict[f'skill_{strategy}'] = np.array(pooled_embs)
    
    # C. V3 Embeddings (Concatenation)
    if args.mode in ['v3', 'both']:
        logger.info("Generating V3 Skill Embeddings (Concatenation)...")
        
        # Load Skill Encoder
        if args.encoder_skill:
             logger.info(f"Loading Skill Encoder: {args.encoder_skill}")
             v3_skill_encoder = SentenceTransformer(args.encoder_skill)
             v3_encoder_name = args.encoder_skill.split('/')[-1]
        else:
             logger.info("Using Text Encoder as Skill Encoder for V3")
             v3_skill_encoder = text_encoder
             v3_encoder_name = args.encoder_text.split('/')[-1]
        
        # Load Skill Data for V3 (Weighted IDF)
        # We need to reload because V3 params might be different or strictly require weighted_idf base
        logger.info("Loading Data for V3...")
        job_skill_map_v3, _, _ = load_job_skill_data_by_id(
            skill_scores_file=args.skill_scores_file,
            esco_skills_file="data/esco_datasets/skills_en.csv",
            skill_properties_file=None,
            pooling_strategy="weighted_idf",
            alpha=1.0, beta=1.0,
            esco_taxonomy_file=None # Default
        )
        
        # Load descriptions
        skill_desc_map = load_skill_descriptions("data/esco_datasets/skills_en.csv")
        
        # Apply V3 Selection/Capping Logic
        logger.info(f"Applying V3 Selection: Top-{args.top_k_skills}, {args.skill_selection_strategy}, {args.scoring_mode}")
        
        if args.scoring_mode == "weighted":
             job_skill_map_v3 = calculate_idf_scores_by_job_id(
                 job_skill_map_v3, use_job_scores=True, importance_weight=args.importance_weight
             )
             
        if args.skill_selection_strategy == "top_k":
             if args.scoring_mode == "scores_only":
                 job_skill_map_v3 = cap_skills_per_job_by_score(
                     job_skill_map_v3, max_skills_per_job=args.top_k_skills, skill_desc_map=skill_desc_map
                 )
             else:
                 job_skill_map_v3 = cap_skills_per_job_lexicographic(
                     job_skill_map_v3, max_skills_per_job=args.top_k_skills, 
                     skill_desc_map=skill_desc_map, use_weighted_idf=(args.scoring_mode == "weighted")
                 )
                 
        # Build Embeddings
        # We assume use_skill_description=True/False based on... let's match V2 args or defaults?
        # User didn't specify arg for this, assuming True if descriptions available?
        # V3 config usually enables descriptions. Let's default to True for "names + desc" logic unless told otherwise.
        # But wait, `build_last_job_skill_embeddings` takes `include_skill_descriptions`.
        # Taking a safe bet: True usually.
        
        v3_embs = build_last_job_skill_embeddings(
            data_pairs=test_pairs,
            job_ids_list=test_job_ids,
            job_skill_map=job_skill_map_v3,
            skill_desc_map=skill_desc_map,
            encoder_skill=v3_skill_encoder,
            include_skill_descriptions=True, # Defaulting to True for V3
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            cache_dir=args.embeddings_cache_dir,
            encoder_name=v3_encoder_name,
            force_recompute=False
        )
        
        config_name = f"v3_top{args.top_k_skills}_{args.scoring_mode}"
        embeddings_dict[config_name] = v3_embs

    # 5. Run Analysis
    analyzer = EmbeddingVarianceAnalyzer(embeddings_dict)
    analyzer.analyze_pca_variance()
    analyzer.analyze_pairwise_similarity()
    analyzer.compute_cka()
    analyzer.generate_report()
    analyzer.generate_visualization(os.path.join(os.path.dirname(__file__), "embedding_variance_pooled.png"))

if __name__ == "__main__":
    main()
