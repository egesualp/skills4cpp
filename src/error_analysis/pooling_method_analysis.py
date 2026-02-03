#!/usr/bin/env python3
"""
Pooling Method Analysis for CPP (Career Path Prediction).

This script analyzes the effect of skill pooling methods on CPP performance by examining
how skill characteristics (frequency, quality, specificity) relate to prediction accuracy.

===================================================================================
METHODOLOGY
===================================================================================

1. SKILL FREQUENCY & IDF COMPUTATION
   - Frequency = Number of ESCO occupations that use a skill
   - IDF = log((N_occupations + 1) / (n_skill + 1))
   - Genericness = 1 - normalized(IDF), scaled to [0, 1]
   
   Higher genericness = more common/generic skill (e.g., "communicate verbally")
   Lower genericness = rarer/specific skill (e.g., "operate plasma cutting torch")

2. POOLING STRATEGY SIMULATION
   When aggregating skills per career path, different weighting schemes are compared:
   
   - MEAN: w_i = 1/N (equal contribution)
   - WEIGHTED_MEAN: w_i = score_i / Σ(scores)
   - WEIGHTED_IDF: w_i = score_i^α × idf_i^β / Σ(...)
   
   "Effective number of skills" = exp(entropy) measures weight concentration.

3. PER-PATH STATISTICS
   For each career path, we compute:
   - Skill frequency statistics (min, median, max, avg, std, q25, q75)
   - IDF statistics (min, median, max, avg, std)
   - Skill score statistics (min, median, max, avg, std)
   - Genericness statistics
   - Pooling weight entropy and concentration metrics

===================================================================================
OUTPUT FILES
===================================================================================

Data Files:
  - analysis_dataframe.csv: Per-sample metrics (confidence, correctness, ranks)
  - path_statistics.csv: Detailed per-career-path skill statistics
  - statistics_report.txt: Human-readable statistical summary

Visualizations:
  - pooling_analysis.png/pdf: Main 1×2 plot (genericness vs confidence, box plots)
  - pooling_strategy_analysis.png: Pooling strategy comparisons (2×2)
  - frequency_analysis.png: Frequency-based analysis (2×2)
  - skill_confidence_analysis.png: Score distribution and skill count analysis

===================================================================================
KEY METRICS EXPLAINED
===================================================================================

Acc@1: Percentage of samples where top-1 prediction is correct
MRR: Mean Reciprocal Rank = mean(1/rank)
Median Rank: 50th percentile of true target ranks

Genericness: Normalized skill commonality (0=rare, 1=common)
Effective Skills: exp(entropy of weights) - measures weight concentration

===================================================================================
USAGE
===================================================================================

    python pooling_method_analysis.py \\
        --text_scores <path_to_text_scores.pkl> \\
        --skill_scores <path_to_skill_scores.pkl> \\
        --hybrid_scores <path_to_hybrid_scores.pkl> \\
        --master_skill_file <path_to_fused_predictions.json> \\
        --output_dir <output_directory> \\
        [--esco_taxonomy_file <path_to_esco_taxonomy.csv>] \\
        [--alpha <float>] [--beta <float>]

See README_POOLING_ANALYSIS.md for detailed documentation.

Author: Thesis Project
Date: 2026-01
"""


import argparse
import json
import pickle
import os
import sys
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configure matplotlib for better aesthetics
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Constants
ESCO_TAXONOMY_FILE = "/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/occupationSkillRelations_en.csv"
ESCO_SKILLS_FILE = "/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv"


def load_prediction_data(pkl_path: str) -> Dict:
    """Load prediction scores from pickle file."""
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_skill_scores(json_path: str) -> Dict[str, List]:
    """Load skill scores (fused_predictions.json format)."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def load_master_job_data(csv_path: str) -> pd.DataFrame:
    """Load master job data with job_id mapping."""
    return pd.read_csv(csv_path)


def compute_skill_idf_from_taxonomy(taxonomy_path: str) -> Dict[str, float]:
    """
    Compute IDF scores for skills from ESCO taxonomy.
    
    IDF = log((N_occupations + 1) / (n_skill_occurrences + 1))
    Higher IDF = rarer skill = more specific
    Lower IDF = common skill = more generic
    
    Returns:
        skill_idf: {skill_uri: idf_score}
    """
    df = pd.read_csv(taxonomy_path)
    
    # Count unique occupations
    n_occupations = df['occupationUri'].nunique()
    
    # Count occupations per skill
    skill_occ_count = df.groupby('skillUri')['occupationUri'].nunique()
    
    # Compute IDF
    idf_series = np.log((n_occupations + 1) / (skill_occ_count + 1))
    
    print(f"  > IDF statistics: min={idf_series.min():.4f}, max={idf_series.max():.4f}, mean={idf_series.mean():.4f}")
    
    return idf_series.to_dict()


def compute_skill_genericness(skill_idf: Dict[str, float]) -> Dict[str, float]:
    """
    Compute genericness score (inverse of IDF, normalized to [0, 1]).
    Higher genericness = more common/generic skill.
    
    Returns:
        skill_genericness: {skill_uri: genericness_score}
    """
    if not skill_idf:
        return {}
    
    idf_values = np.array(list(skill_idf.values()))
    idf_min, idf_max = idf_values.min(), idf_values.max()
    
    # Normalize IDF to [0, 1] and invert
    # genericness = 1 - normalized_idf
    skill_genericness = {}
    for skill_uri, idf in skill_idf.items():
        normalized_idf = (idf - idf_min) / (idf_max - idf_min + 1e-8)
        skill_genericness[skill_uri] = 1.0 - normalized_idf
    
    return skill_genericness


def compute_skill_frequency_from_taxonomy(taxonomy_path: str) -> Dict[str, int]:
    """
    Compute frequency (number of occupations) per skill from ESCO taxonomy.
    Higher frequency = more common skill.
    
    Returns:
        skill_freq: {skill_uri: n_occupations}
    """
    df = pd.read_csv(taxonomy_path)
    skill_occ_count = df.groupby('skillUri')['occupationUri'].nunique()
    return skill_occ_count.to_dict()


# =============================================================================
# DECORTE DATASET IDF COMPUTATION
# =============================================================================

def compute_skill_idf_from_dataset(
    skill_scores_map: Dict[str, List],
    source_name: str = "dataset",
    master_df: Optional[pd.DataFrame] = None
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    Compute IDF scores for skills from the dataset (Decorte job postings).
    
    This mirrors the IDF calculation in train_cpp_enhanced_v2.py:
    - N_occ = Total number of unique job_ids in the dataset
    - n_i = Number of unique job_ids this skill appears with
    - IDF = log((N_occ + 1) / (n_i + 1))
    
    Args:
        skill_scores_map: {job_id: [(skill_uri, score), ...]} from fused_predictions.json
        source_name: Name for logging purposes
        
    Returns:
        Tuple of:
        - skill_idf: {skill_uri: idf_score}
        - skill_freq: {skill_uri: n_job_ids}
    """
    print(f"\n  Computing IDF from {source_name}...")
    if master_df is not None:
        train_val_job_ids = master_df.query('split != "test"').job_id.unique()
    
    # Collect all (job_id, skill_uri) pairs
    job_skill_pairs = []
    for job_id, skill_list in skill_scores_map.items():
        for skill_info in skill_list:
            if isinstance(skill_info, (list, tuple)):
                skill_uri = skill_info[0]
            else:
                skill_uri = skill_info.get('skill_uri') or skill_info.get('skillUri')
            job_skill_pairs.append((job_id, skill_uri))
    
    # Create DataFrame
    df = pd.DataFrame(job_skill_pairs, columns=['job_id', 'skillUri'])
    df.job_id = df.job_id.astype(int)
    all_skill_n_occ = df.groupby('skillUri')['job_id'].nunique()
    if master_df is not None:
        df_train_val = df.query('job_id in @train_val_job_ids')
        N_occ = df_train_val['job_id'].nunique()
        skill_n_occ = df_train_val.groupby('skillUri')['job_id'].nunique()
    else:
        N_occ = df['job_id'].nunique()
        skill_n_occ = df.groupby('skillUri')['job_id'].nunique()
    
    # IDF = log((N_occ + 1) / (n_i + 1))
    idf_series = np.log((N_occ + 1) / (skill_n_occ + 1))
    max_idf = idf_series.max()
    
    # Create dictionaries
    skill_idf = idf_series.to_dict()
    skill_freq = skill_n_occ.to_dict()

    if master_df is not None:
        # Find skills that only appear in test split
        all_skills = set(df['skillUri'].unique())
        train_val_skills = set(skill_idf.keys())
        test_only_skills = all_skills - train_val_skills
        
        # Assign max IDF to test-only skills
        for skill_uri in test_only_skills:
            skill_idf[skill_uri] = max_idf
    
    print(f"    > N_jobs (total unique jobs) = {N_occ}")
    print(f"    > Unique skills = {len(skill_n_occ)}")
    print(f"    > IDF range: [{idf_series.min():.4f}, {idf_series.max():.4f}], mean={idf_series.mean():.4f}")
    
    # Show most common and rarest skills
    most_common_skill = skill_n_occ.idxmax()
    rarest_skill = skill_n_occ.idxmin()
    print(f"    > Most common skill appears in {skill_n_occ.max()} jobs (IDF={idf_series.min():.4f})")
    print(f"    > Rarest skill appears in {skill_n_occ.min()} jobs (IDF={idf_series.max():.4f})")
    
    return skill_idf, skill_freq

def compute_skill_genericness_from_idf(
    skill_idf: Dict[str, float],
    source_prefix: str = ""
) -> Dict[str, float]:
    """
    Compute genericness score (inverse of IDF, normalized to [0, 1]).
    Higher genericness = more common/generic skill.
    
    This is a unified version that can accept any IDF dictionary.
    
    Args:
        skill_idf: {skill_uri: idf_score}
        source_prefix: Prefix for the source (e.g., "esco_" or "decorte_")
        
    Returns:
        skill_genericness: {skill_uri: genericness_score}
    """
    if not skill_idf:
        return {}
    
    idf_values = np.array(list(skill_idf.values()))
    idf_min, idf_max = idf_values.min(), idf_values.max()
    
    # Normalize IDF to [0, 1] and invert
    skill_genericness = {}
    for skill_uri, idf in skill_idf.items():
        normalized_idf = (idf - idf_min) / (idf_max - idf_min + 1e-8)
        skill_genericness[skill_uri] = 1.0 - normalized_idf
    
    return skill_genericness


def compare_idf_sources(
    esco_idf: Dict[str, float],
    decorte_idf: Dict[str, float]
) -> Dict[str, any]:
    """
    Compare IDF scores between ESCO taxonomy and Decorte dataset.
    
    Returns statistics about the overlap and correlation between sources.
    """
    # Find common skills
    common_skills = set(esco_idf.keys()) & set(decorte_idf.keys())
    esco_only = set(esco_idf.keys()) - set(decorte_idf.keys())
    decorte_only = set(decorte_idf.keys()) - set(esco_idf.keys())
    
    print(f"\n  IDF Source Comparison:")
    print(f"    > Common skills: {len(common_skills)}")
    print(f"    > ESCO-only skills: {len(esco_only)}")
    print(f"    > Decorte-only skills: {len(decorte_only)}")
    
    # Compute correlation for common skills
    if len(common_skills) > 10:
        esco_vals = [esco_idf[s] for s in common_skills]
        decorte_vals = [decorte_idf[s] for s in common_skills]
        
        r, p = stats.pearsonr(esco_vals, decorte_vals)
        rho, p_rho = stats.spearmanr(esco_vals, decorte_vals)
        
        print(f"    > Pearson correlation: r={r:.4f} (p={p:.4e})")
        print(f"    > Spearman correlation: ρ={rho:.4f} (p={p_rho:.4e})")
        
        return {
            'n_common': len(common_skills),
            'n_esco_only': len(esco_only),
            'n_decorte_only': len(decorte_only),
            'pearson_r': r,
            'pearson_p': p,
            'spearman_rho': rho,
            'spearman_p': p_rho
        }
    
    return {
        'n_common': len(common_skills),
        'n_esco_only': len(esco_only),
        'n_decorte_only': len(decorte_only)
    }




def align_predictions(d_text: Dict, d_skill: Dict, d_hybrid: Dict) -> Tuple:
    """
    Align predictions across three models using text predictions as master.
    Returns aligned scores, targets, and labels.
    """
    master_labels = d_text['target_labels']
    
    def reorder_scores(data, master_labels):
        """Reorder score matrix columns to match master label order."""
        label_map = {lbl: i for i, lbl in enumerate(data['target_labels'])}
        new_order = [label_map[lbl] for lbl in master_labels]
        return data['scores'][:, new_order]
    
    # Reorder skill and hybrid scores to match text label order
    scores_skill = reorder_scores(d_skill, master_labels)
    scores_hybrid = reorder_scores(d_hybrid, master_labels)
    
    # Align rows by job_ids
    def make_key(ids):
        return [tuple(p) if isinstance(p, list) else (p,) for p in ids]
    
    keys_text = make_key(d_text['job_ids'])
    keys_skill = make_key(d_skill['job_ids'])
    keys_hybrid = make_key(d_hybrid['job_ids'])
    
    skill_map = {k: i for i, k in enumerate(keys_skill)}
    hybrid_map = {k: i for i, k in enumerate(keys_hybrid)}
    
    # Find common samples
    common_indices = [
        (i, skill_map[k], hybrid_map[k])
        for i, k in enumerate(keys_text)
        if k in skill_map and k in hybrid_map
    ]
    
    if not common_indices:
        raise ValueError("No common samples found across models!")
    
    idx_text, idx_skill, idx_hybrid = zip(*common_indices)
    
    aligned_scores = {
        'text': d_text['scores'][list(idx_text)],
        'skill': scores_skill[list(idx_skill)],
        'hybrid': scores_hybrid[list(idx_hybrid)]
    }
    
    aligned_targets = np.array(d_text['true_target_indices'])[list(idx_text)]
    aligned_job_ids = [d_text['job_ids'][i] for i in idx_text]
    
    print(f"  > Aligned {len(common_indices)} samples across all three models")
    
    return aligned_scores, aligned_targets, master_labels, aligned_job_ids


def compute_prediction_confidence(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Compute prediction confidence as the score for the true target.
    """
    n_samples = scores.shape[0]
    confidences = np.array([scores[i, targets[i]] for i in range(n_samples)])
    return confidences


def compute_is_correct(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Determine if the top-1 prediction is correct.
    """
    predictions = np.argmax(scores, axis=1)
    return predictions == targets


def compute_prediction_rank(scores: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """
    Compute the rank of the true target (1-indexed).
    """
    n_samples = scores.shape[0]
    ranks = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        true_score = scores[i, targets[i]]
        rank = np.sum(scores[i] > true_score) + 1
        ranks[i] = rank
    
    return ranks


def compute_detailed_path_statistics(
    job_ids_list: List[List[str]],
    skill_scores_map: Dict[str, List],
    skill_idf: Dict[str, float],
    skill_freq: Dict[str, int],
    skill_genericness: Dict[str, float],
    decorte_idf: Optional[Dict[str, float]] = None,
    decorte_freq: Optional[Dict[str, int]] = None,
    decorte_genericness: Optional[Dict[str, float]] = None,
    alpha: float = 1.0,
    beta: float = 1.0
) -> pd.DataFrame:
    """
    Compute detailed per-career-path statistics.
    
    For each path, computes:
    - Skill frequency statistics (min, median, max, avg) from ESCO taxonomy
    - Skill frequency statistics from Decorte dataset (if provided)
    - IDF statistics from both sources
    - Skill score statistics
    - Pooling strategy weight simulations
    
    Args:
        job_ids_list: List of job_id lists for each sample
        skill_scores_map: {job_id: [(skill_uri, score), ...]}
        skill_idf: {skill_uri: idf_score} from ESCO taxonomy
        skill_freq: {skill_uri: n_occupations} from ESCO taxonomy
        skill_genericness: {skill_uri: genericness} from ESCO
        decorte_idf: Optional {skill_uri: idf_score} from Decorte dataset
        decorte_freq: Optional {skill_uri: n_jobs} from Decorte dataset
        decorte_genericness: Optional {skill_uri: genericness} from Decorte
        alpha: Exponent for confidence score (weighted_idf)
        beta: Exponent for IDF score (weighted_idf)
    
    Returns:
        DataFrame with detailed per-path statistics
    """
    # Pre-calculate fallback values for alignment with training script
    max_esco_idf = max(skill_idf.values()) if skill_idf else 0.0
    
    path_stats = []
    
    for path_idx, job_ids in enumerate(job_ids_list):
        all_freqs = []
        all_idfs = []
        all_scores = []
        all_genericness = []
        all_skill_uris = set()
        
        # Decorte-specific collections
        all_decorte_freqs = []
        all_decorte_idfs = []
        all_decorte_genericness = []
        
        # Collect all skills and their properties from this path
        for job_id in job_ids:
            job_id_str = str(job_id)
            if job_id_str not in skill_scores_map:
                continue
            
            for skill_info in skill_scores_map[job_id_str]:
                if isinstance(skill_info, (list, tuple)):
                    skill_uri = skill_info[0]
                    score = skill_info[1] if len(skill_info) > 1 else 1.0
                else:
                    skill_uri = skill_info.get('skill_uri') or skill_info.get('skillUri')
                    score = skill_info.get('score', 1.0)
                
                all_skill_uris.add(skill_uri)
                all_scores.append(score)
                
                # ESCO-based statistics
                if skill_uri in skill_freq:
                    all_freqs.append(skill_freq[skill_uri])
                
                # Align with training script: Use max_idf for missing skills
                if skill_uri in skill_idf:
                    all_idfs.append(skill_idf[skill_uri])
                else:
                    all_idfs.append(max_esco_idf)

                if skill_uri in skill_genericness:
                    all_genericness.append(skill_genericness[skill_uri])
                else:
                    # Missing skills are rare -> high IDF -> low genericness (0)
                    all_genericness.append(0.0)
                
                # Decorte-based statistics (if provided)
                if decorte_freq and skill_uri in decorte_freq:
                    all_decorte_freqs.append(decorte_freq[skill_uri])
                if decorte_idf and skill_uri in decorte_idf:
                    all_decorte_idfs.append(decorte_idf[skill_uri])
                if decorte_genericness and skill_uri in decorte_genericness:
                    all_decorte_genericness.append(decorte_genericness[skill_uri])
        
        # Compute statistics
        stats_dict = {'path_idx': path_idx}
        
        if all_freqs:
            stats_dict.update({
                'min_freq': np.min(all_freqs),
                'median_freq': np.median(all_freqs),
                'max_freq': np.max(all_freqs),
                'avg_freq': np.mean(all_freqs),
                'std_freq': np.std(all_freqs),
                'q25_freq': np.percentile(all_freqs, 25),
                'q75_freq': np.percentile(all_freqs, 75),
            })
        else:
            for k in ['min_freq', 'median_freq', 'max_freq', 'avg_freq', 'std_freq', 'q25_freq', 'q75_freq']:
                stats_dict[k] = np.nan
        
        if all_idfs:
            stats_dict.update({
                'min_idf': np.min(all_idfs),
                'median_idf': np.median(all_idfs),
                'max_idf': np.max(all_idfs),
                'avg_idf': np.mean(all_idfs),
                'std_idf': np.std(all_idfs),
            })
        else:
            for k in ['min_idf', 'median_idf', 'max_idf', 'avg_idf', 'std_idf']:
                stats_dict[k] = np.nan
        
        if all_scores:
            stats_dict.update({
                'min_score': np.min(all_scores),
                'median_score': np.median(all_scores),
                'max_score': np.max(all_scores),
                'avg_score': np.mean(all_scores),
                'std_score': np.std(all_scores),
            })
        else:
            for k in ['min_score', 'median_score', 'max_score', 'avg_score', 'std_score']:
                stats_dict[k] = np.nan
        
        if all_genericness:
            stats_dict.update({
                'min_genericness': np.min(all_genericness),
                'median_genericness': np.median(all_genericness),
                'max_genericness': np.max(all_genericness),
                'avg_genericness': np.mean(all_genericness),
                'std_genericness': np.std(all_genericness),
            })
        else:
            for k in ['min_genericness', 'median_genericness', 'max_genericness', 'avg_genericness', 'std_genericness']:
                stats_dict[k] = np.nan
        
        # === DECORTE DATASET STATISTICS ===
        # Frequency from Decorte
        if all_decorte_freqs:
            stats_dict.update({
                'decorte_min_freq': np.min(all_decorte_freqs),
                'decorte_median_freq': np.median(all_decorte_freqs),
                'decorte_max_freq': np.max(all_decorte_freqs),
                'decorte_avg_freq': np.mean(all_decorte_freqs),
                'decorte_std_freq': np.std(all_decorte_freqs),
            })
        else:
            for k in ['decorte_min_freq', 'decorte_median_freq', 'decorte_max_freq', 'decorte_avg_freq', 'decorte_std_freq']:
                stats_dict[k] = np.nan
        
        # IDF from Decorte
        if all_decorte_idfs:
            stats_dict.update({
                'decorte_min_idf': np.min(all_decorte_idfs),
                'decorte_median_idf': np.median(all_decorte_idfs),
                'decorte_max_idf': np.max(all_decorte_idfs),
                'decorte_avg_idf': np.mean(all_decorte_idfs),
                'decorte_std_idf': np.std(all_decorte_idfs),
            })
        else:
            for k in ['decorte_min_idf', 'decorte_median_idf', 'decorte_max_idf', 'decorte_avg_idf', 'decorte_std_idf']:
                stats_dict[k] = np.nan
        
        # Genericness from Decorte
        if all_decorte_genericness:
            stats_dict.update({
                'decorte_min_genericness': np.min(all_decorte_genericness),
                'decorte_median_genericness': np.median(all_decorte_genericness),
                'decorte_max_genericness': np.max(all_decorte_genericness),
                'decorte_avg_genericness': np.mean(all_decorte_genericness),
                'decorte_std_genericness': np.std(all_decorte_genericness),
            })
        else:
            for k in ['decorte_min_genericness', 'decorte_median_genericness', 'decorte_max_genericness', 'decorte_avg_genericness', 'decorte_std_genericness']:
                stats_dict[k] = np.nan
        
        # Count statistics
        stats_dict['num_unique_skills'] = len(all_skill_uris)
        stats_dict['total_skills'] = len(all_scores)
        stats_dict['num_jobs'] = len(job_ids)
        
        # Simulate pooling weights
        if all_scores and all_idfs and len(all_scores) == len(all_idfs):
            scores_arr = np.array(all_scores)
            idfs_arr = np.array(all_idfs)
            
            # Mean pooling: equal weights
            mean_weights = np.ones(len(scores_arr)) / len(scores_arr)
            
            # Weighted mean: weights = score
            weighted_mean_weights = scores_arr / (scores_arr.sum() + 1e-8)
            
            # Weighted IDF: weights = score^alpha * idf^beta
            weighted_idf_weights = (scores_arr ** alpha) * (idfs_arr ** beta)
            weighted_idf_weights = weighted_idf_weights / (weighted_idf_weights.sum() + 1e-8)
            
            stats_dict['mean_weight_entropy'] = -np.sum(mean_weights * np.log(mean_weights + 1e-8))
            stats_dict['weighted_mean_weight_entropy'] = -np.sum(weighted_mean_weights * np.log(weighted_mean_weights + 1e-8))
            stats_dict['weighted_idf_weight_entropy'] = -np.sum(weighted_idf_weights * np.log(weighted_idf_weights + 1e-8))
            
            # Weight concentration (max weight / sum)
            stats_dict['mean_max_weight'] = mean_weights.max()
            stats_dict['weighted_mean_max_weight'] = weighted_mean_weights.max()
            stats_dict['weighted_idf_max_weight'] = weighted_idf_weights.max()
            
            # Effective number of skills (exp(entropy))
            stats_dict['mean_effective_skills'] = np.exp(stats_dict['mean_weight_entropy'])
            stats_dict['weighted_mean_effective_skills'] = np.exp(stats_dict['weighted_mean_weight_entropy'])
            stats_dict['weighted_idf_effective_skills'] = np.exp(stats_dict['weighted_idf_weight_entropy'])
        else:
            for prefix in ['mean', 'weighted_mean', 'weighted_idf']:
                stats_dict[f'{prefix}_weight_entropy'] = np.nan
                stats_dict[f'{prefix}_max_weight'] = np.nan
                stats_dict[f'{prefix}_effective_skills'] = np.nan
        
        path_stats.append(stats_dict)
    
    return pd.DataFrame(path_stats)


def compute_sample_skill_stats(
    job_ids_list: List[List[str]],
    skill_scores_map: Dict[str, List],
    skill_idf: Dict[str, float],
    skill_genericness: Dict[str, float]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute skill-level statistics for each sample.
    
    Returns:
        mean_skill_score: Average skill prediction score per sample
        mean_genericness: Average skill genericness (frequency) per sample
        mean_idf: Average skill IDF per sample
    """
    n_samples = len(job_ids_list)
    mean_skill_scores = np.zeros(n_samples)
    mean_genericness = np.zeros(n_samples)
    mean_genericness = np.zeros(n_samples)
    mean_idf = np.zeros(n_samples)
    
    # Pre-calculate fallback values
    max_idf_val = max(skill_idf.values()) if skill_idf else 0.0
    
    for i, job_ids in enumerate(job_ids_list):
        all_skill_scores = []
        all_genericness = []
        all_idf = []
        
        for job_id in job_ids:
            job_id_str = str(job_id)
            if job_id_str not in skill_scores_map:
                continue
            
            for skill_info in skill_scores_map[job_id_str]:
                if isinstance(skill_info, (list, tuple)):
                    skill_uri = skill_info[0]
                    score = skill_info[1] if len(skill_info) > 1 else 1.0
                else:
                    skill_uri = skill_info.get('skill_uri') or skill_info.get('skillUri')
                    score = skill_info.get('score', 1.0)
                
                all_skill_scores.append(score)
                
                if skill_uri in skill_genericness:
                    all_genericness.append(skill_genericness[skill_uri])
                else:
                    all_genericness.append(0.0)
                
                if skill_uri in skill_idf:
                    all_idf.append(skill_idf[skill_uri])
                else:
                    all_idf.append(max_idf_val)
        
        if all_skill_scores:
            mean_skill_scores[i] = np.mean(all_skill_scores)
        if all_genericness:
            mean_genericness[i] = np.mean(all_genericness)
        if all_idf:
            mean_idf[i] = np.mean(all_idf)
    
    return mean_skill_scores, mean_genericness, mean_idf


def create_analysis_dataframe(
    aligned_scores: Dict[str, np.ndarray],
    aligned_targets: np.ndarray,
    aligned_job_ids: List[List[str]],
    skill_scores_map: Dict[str, List],
    skill_idf: Dict[str, float],
    skill_freq: Dict[str, int],
    skill_genericness: Dict[str, float],
    decorte_idf: Optional[Dict[str, float]] = None,
    decorte_freq: Optional[Dict[str, int]] = None,
    decorte_genericness: Optional[Dict[str, float]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create comprehensive DataFrames with all analysis metrics.
    
    Args:
        aligned_scores: Dictionary of model scores
        aligned_targets: True target indices
        aligned_job_ids: Job IDs per sample
        skill_scores_map: Skill scores per job_id
        skill_idf: ESCO-based IDF scores
        skill_freq: ESCO-based frequency
        skill_genericness: ESCO-based genericness
        decorte_idf: Optional Decorte-based IDF scores
        decorte_freq: Optional Decorte-based frequency
        decorte_genericness: Optional Decorte-based genericness
    
    Returns:
        df: Main analysis DataFrame with prediction metrics
        df_path_stats: Detailed per-path statistics DataFrame
    """
    n_samples = len(aligned_targets)
    
    print("Computing metrics...")
    
    # Compute per-model metrics
    data = {
        'sample_idx': np.arange(n_samples),
        'target_idx': aligned_targets,
    }
    
    for model_name in ['text', 'skill', 'hybrid']:
        scores = aligned_scores[model_name]
        data[f'confidence_{model_name}'] = compute_prediction_confidence(scores, aligned_targets)
        data[f'is_correct_{model_name}'] = compute_is_correct(scores, aligned_targets)
        data[f'rank_{model_name}'] = compute_prediction_rank(scores, aligned_targets)
    
    # Compute skill statistics
    mean_skill_scores, mean_genericness, mean_idf = compute_sample_skill_stats(
        aligned_job_ids, skill_scores_map, skill_idf, skill_genericness
    )
    
    data['mean_skill_score'] = mean_skill_scores
    data['mean_genericness'] = mean_genericness
    data['mean_idf'] = mean_idf
    
    # Compute Decorte-based statistics if available
    if decorte_genericness:
        _, mean_decorte_genericness, _ = compute_sample_skill_stats(
            aligned_job_ids, skill_scores_map, {}, decorte_genericness
        )
        data['mean_decorte_genericness'] = mean_decorte_genericness
    
    # Compute the number of skills per sample
    skill_counts = []
    for job_ids in aligned_job_ids:
        count = 0
        for job_id in job_ids:
            job_id_str = str(job_id)
            if job_id_str in skill_scores_map:
                count += len(skill_scores_map[job_id_str])
        skill_counts.append(count)
    data['num_skills'] = skill_counts
    
    df = pd.DataFrame(data)
    
    print(f"Created main analysis DataFrame with {len(df)} samples")
    
    # Compute detailed per-path statistics (with both ESCO and Decorte sources)
    print("Computing detailed per-path statistics...")
    df_path_stats = compute_detailed_path_statistics(
        aligned_job_ids, skill_scores_map, skill_idf, skill_freq, skill_genericness,
        decorte_idf=decorte_idf, decorte_freq=decorte_freq, decorte_genericness=decorte_genericness
    )
    
    print(f"Created path statistics DataFrame with {len(df_path_stats)} paths")
    
    return df, df_path_stats



# Styling Constants
LABEL_FONT = 16
SUBTITLE_FONT = 20
MAIN_TITLE_FONT = 22
TICK_FONT = 14
LEGEND_FONT = 14

def create_visualizations(
    df: pd.DataFrame,
    output_dir: str,
    figure_name: str = "pooling_analysis"
):
    """
    Idea 1: 1×2 Subplot — Skill Characteristics vs. Model Performance
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up the figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    plt.subplots_adjust(wspace=0.3)
    
    # Main Figure Title
    # fig.suptitle('Skill Characteristics Analysis', fontsize=MAIN_TITLE_FONT, fontweight='bold', y=0.98)
    
    # Color palette
    colors = {
        'text': '#1f77b4',   # Blue
        'skill': '#ff7f0e', # Orange
        'hybrid': '#2ca02c' # Green
    }
    
    # === LEFT PANEL: Genericness vs. CPP Confidence Scatter ===
    ax1 = axes[0]
    
    # Sample data for visualization (avoid overplotting)
    sample_size = min(2000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    
    for model_name, color in colors.items():
        # Scatter points
        ax1.scatter(
            df_sample['mean_genericness'],
            df_sample[f'confidence_{model_name}'],
            alpha=0.15,
            s=20,
            c=color,
            label=None # Don't label points to keep legend clean
        )
        
        # Trend lines
        mask = ~(np.isnan(df['mean_genericness']) | np.isnan(df[f'confidence_{model_name}']))
        if mask.sum() > 10:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                df.loc[mask, 'mean_genericness'],
                df.loc[mask, f'confidence_{model_name}']
            )
            
            # Plot line over full range
            x_range = np.linspace(df['mean_genericness'].min(), df['mean_genericness'].max(), 100)
            y_pred = slope * x_range + intercept
            
            label = f'{model_name.upper()} (r={r_value:.2f})'
            ax1.plot(x_range, y_pred, color=color, linewidth=3, linestyle='-', label=label)
            
    ax1.set_xlabel('Mean Skill Genericness (0=Specific, 1=Generic)', fontsize=LABEL_FONT, labelpad=10)
    ax1.set_ylabel('Model Confidence (Correct Target)', fontsize=LABEL_FONT, labelpad=10)
    ax1.set_title('Skill Characteristics vs. Model Performance', fontsize=SUBTITLE_FONT, fontweight='bold', pad=15)
    ax1.legend(fontsize=LEGEND_FONT)
    ax1.tick_params(labelsize=TICK_FONT)
    ax1.grid(True, alpha=0.3)

    # === RIGHT PANEL: Grouped Box Plots (Correct vs Incorrect) ===
    ax2 = axes[1]
    
    box_data = []
    positions = []
    box_colors = []
    labels = []
    
    # Grouped layout: Text (Corr/Inc), Skill (Corr/Inc), Hybrid (Corr/Inc)
    pos = 0
    stats_text = []

    for model_name, color in colors.items():
        # Correct
        correct_vals = df.loc[df[f'is_correct_{model_name}'], 'mean_genericness'].dropna()
        # Incorrect
        incorrect_vals = df.loc[~df[f'is_correct_{model_name}'], 'mean_genericness'].dropna()
        
        # Perform t-test
        if len(correct_vals) > 0 and len(incorrect_vals) > 0:
            t_stat, p_val = stats.ttest_ind(correct_vals, incorrect_vals, equal_var=False)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            stats_text.append(f"{model_name.upper()}: p={p_val:.1e} ({sig})")
        
        # Add to box data
        box_data.extend([correct_vals, incorrect_vals])
        positions.extend([pos - 0.2, pos + 0.2])
        box_colors.extend([color, color]) # Same color
        
        pos += 1.5 # Gap between models

    # Create boxplot
    bp = ax2.boxplot(box_data, positions=positions, patch_artist=True, widths=0.35, showfliers=False)
    
    # Style boxes
    for i, patch in enumerate(bp['boxes']):
        color = box_colors[i]
        patch.set_facecolor(color)
        patch.set_edgecolor('black')
        
        # Even indices (0, 2, 4) are Correct, Odd (1, 3, 5) are Incorrect
        if i % 2 == 0: # Correct
            patch.set_alpha(0.8)
        else: # Incorrect
            patch.set_alpha(0.3)

    # Add dummy legend for Correct/Incorrect
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    
    legend_elements = [
        Patch(facecolor='gray', alpha=0.8, label='Correct Prediction'),
        Patch(facecolor='gray', alpha=0.3, label='Incorrect Prediction'),
        Line2D([0], [0], color=colors['text'], lw=4, label='TEXT'),
        Line2D([0], [0], color=colors['skill'], lw=4, label='SKILL'),
        Line2D([0], [0], color=colors['hybrid'], lw=4, label='HYBRID'),
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=LEGEND_FONT)

    # X-Axis Labels
    ax2.set_xticks([0, 1.5, 3.0])
    ax2.set_xticklabels(['TEXT', 'SKILL', 'HYBRID'], fontsize=LABEL_FONT, fontweight='bold')
    
    ax2.set_ylabel('Mean Skill Genericness Distribution', fontsize=LABEL_FONT, labelpad=10)
    ax2.set_title('Genericness Distribution: Correct vs. Incorrect', fontsize=SUBTITLE_FONT, fontweight='bold', pad=15)
    ax2.tick_params(axis='y', labelsize=TICK_FONT)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # Add stats text
    plt.figtext(0.55, 0.02, " | ".join(stats_text), fontsize=12, ha='left')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    # Save
    output_path = os.path.join(output_dir, f"{figure_name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

def create_quartile_performance_analysis(df: pd.DataFrame, output_dir: str):
    """
    Idea 2: 1×2 Subplot — Quartile Performance Comparison
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    plt.subplots_adjust(wspace=0.3)
    
    colors = {
        'text': '#1f77b4',
        'skill': '#ff7f0e',
        'hybrid': '#2ca02c'
    }
    
    models = ['text', 'skill', 'hybrid']
    
    # Helper to plot bars
    def plot_quartile_bars(ax, metric_col, title, xlabel):
        try:
            # Create quartiles
            if df[metric_col].isna().all():
                 ax.text(0.5, 0.5, f"No data for {metric_col}", ha='center', va='center')
                 return

            df['temp_quartile'], bins = pd.qcut(df[metric_col].dropna(), q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], retbins=True)
            
            # Calculate Acc@1 per quartile per model
            quartiles = ['Q1', 'Q2', 'Q3', 'Q4']
            x = np.arange(len(quartiles))
            width = 0.25
            
            for i, model in enumerate(models):
                accs = []
                for q in quartiles:
                    mask = df['temp_quartile'] == q
                    if mask.sum() > 0:
                        acc = df.loc[mask, f'is_correct_{model}'].mean() * 100
                    else:
                        acc = 0
                    accs.append(acc)
                
                ax.bar(x + (i-1)*width, accs, width, label=model.upper(), color=colors[model])
            
            ax.set_xticks(x)
            ax.set_xticklabels(quartiles, fontsize=TICK_FONT)
            ax.set_ylabel('Top-1 Accuracy (%)', fontsize=LABEL_FONT, labelpad=10)
            ax.set_xlabel(xlabel, fontsize=LABEL_FONT, labelpad=10)
            ax.set_title(title, fontsize=SUBTITLE_FONT, fontweight='bold', pad=15)
            ax.legend(fontsize=LEGEND_FONT)
            ax.tick_params(labelsize=TICK_FONT)
            ax.grid(True, axis='y', alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f"Could not create plot:\n{e}", ha='center', va='center')

    # === LEFT PANEL: Skill Confidence Quartiles ===
    plot_quartile_bars(
        axes[0], 
        'mean_skill_score', 
        'Performance by Skill Confidence (Target)', 
        'Skill Confidence Quartiles (Q1=Low, Q4=High)'
    )
    
    # === RIGHT PANEL: Decorte Genericness Quartiles ===
    gen_col = 'mean_decorte_genericness' if 'mean_decorte_genericness' in df.columns else 'mean_genericness'
    source_name = "Decorte" if 'mean_decorte_genericness' in df.columns else "ESCO"
    
    plot_quartile_bars(
        axes[1],
        gen_col,
        f'Performance by Skill Genericness ({source_name})',
        f'Genericness Quartiles (Q1=Specific, Q4=Generic)'
    )
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, "quartile_performance_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")

def create_pooling_heatmap(df: pd.DataFrame, df_path_stats: pd.DataFrame, output_dir: str):
    """
    Idea 3: Single Heatmap — Pooling Strategy × Weight Concentration
    """
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Merge stats
    df_merged = df.merge(df_path_stats, left_on='sample_idx', right_on='path_idx', how='inner')
    
    print(f"Heatmap data merge: {len(df_merged)} samples (from {len(df)} and {len(df_path_stats)})")
    
    strategies = {
        'Mean': 'mean_effective_skills',
        'Weighted Mean': 'weighted_mean_effective_skills',
        'Weighted IDF': 'weighted_idf_effective_skills'
    }
    
    heatmap_data = [] # Rows: Strategies, Cols: Quartiles
    strategy_labels = []

    for strat_name, col_name in strategies.items():
        if col_name not in df_merged.columns:
            print(f"Missing column {col_name}")
            continue
            
        strategy_labels.append(strat_name)
        
        # Create quartiles for this strategy's concentration
        try:
            # Drop NaNs
            valid_mask = ~df_merged[col_name].isna()
            values = df_merged.loc[valid_mask, col_name]
            
            if len(values) == 0:
                heatmap_data.append([100]*4)
                continue

            # Robust binning: Use rank percentages to handle duplicates (force 4 bins)
            pct_ranks = values.rank(pct=True)
            
            # cut into 4 buckets
            qbinned = pd.cut(pct_ranks, bins=[0, 0.25, 0.5, 0.75, 1.0], labels=['Q1', 'Q2', 'Q3', 'Q4'], include_lowest=True)
            
            row = []
            for q in ['Q1', 'Q2', 'Q3', 'Q4']:
                indices_in_bin = values.index[qbinned == q]
                
                if len(indices_in_bin) > 0:
                    acc = df_merged.loc[indices_in_bin, 'is_correct_hybrid'].mean()
                else:
                    acc = np.nan
                row.append(acc)
            
            heatmap_data.append(row)
        except Exception as e:
            print(f"Error computing quartiles for {strat_name}: {e}")
            heatmap_data.append([np.nan]*4)
            
    heatmap_data = np.array(heatmap_data)
    
    # Plot heatmap
    if len(heatmap_data) > 0:
        sns.heatmap(
            heatmap_data, 
            annot=True, 
            fmt=".1%", 
            cmap="YlGnBu", 
            annot_kws={"size": 16},
            xticklabels=['Q1\n(Concentrated)', 'Q2', 'Q3', 'Q4\n(Diffuse)'],
            yticklabels=strategy_labels,
            ax=ax,
            cbar_kws={'label': 'Hybrid Model Top-1 Accuracy'}
        )
        
        # Adjust Colorbar Font
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=TICK_FONT)
        cbar.set_label('Hybrid Model Top-1 Accuracy', fontsize=LABEL_FONT, labelpad=15)
        
        ax.set_title('Impact of Pooling Strategy & Weight Concentration', fontsize=MAIN_TITLE_FONT, fontweight='bold', pad=20)
        ax.set_xlabel('Weight Concentration / Effective Skills\n(Binned by Rank Percentile)', fontsize=LABEL_FONT, labelpad=15)
        ax.set_ylabel('Pooling Strategy', fontsize=LABEL_FONT, labelpad=15)
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONT)
        # Rotate y-axis labels for readability if needed
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "pooling_strategy_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")


def create_pooling_strategy_analysis(
    df: pd.DataFrame,
    df_path_stats: pd.DataFrame,
    output_dir: str
):
    """
    Create visualizations comparing pooling strategies (mean, weighted_mean, weighted_idf).
    """
    # Merge path stats with main df
    df_merged = df.merge(df_path_stats, left_on='sample_idx', right_on='path_idx', how='left')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    colors = {
        'text': '#1f77b4',
        'skill': '#ff7f0e',
        'hybrid': '#2ca02c'
    }
    
    # === Panel 1: Effective Skills by Pooling Strategy ===
    ax1 = axes[0, 0]
    
    pooling_data = {
        'Mean': df_merged['mean_effective_skills'].dropna(),
        'Weighted Mean': df_merged['weighted_mean_effective_skills'].dropna(),
        'Weighted IDF': df_merged['weighted_idf_effective_skills'].dropna(),
    }
    
    bp1 = ax1.boxplot([pooling_data[k] for k in pooling_data.keys()], 
                      labels=pooling_data.keys(),
                      patch_artist=True)
    
    for patch, color in zip(bp1['boxes'], ['#1f77b4', '#ff7f0e', '#2ca02c']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('Effective Number of Skills\n(Weight Concentration)', fontsize=11)
    ax1.set_title('Skill Weight Concentration by Pooling Strategy', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # === Panel 2: Max Weight Distribution ===
    ax2 = axes[0, 1]
    
    for i, (name, col) in enumerate([
        ('Mean', 'mean_max_weight'),
        ('Weighted Mean', 'weighted_mean_max_weight'),
        ('Weighted IDF', 'weighted_idf_max_weight')
    ]):
        data = df_merged[col].dropna()
        ax2.hist(data, bins=50, alpha=0.6, label=name, density=True)
    
    ax2.set_xlabel('Maximum Weight (Skill Dominance)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('Weight Distribution by Pooling Strategy', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # === Panel 3: Effective Skills vs Accuracy (Hybrid) ===
    ax3 = axes[1, 0]
    
    for is_correct in [True, False]:
        mask = df_merged['is_correct_hybrid'] == is_correct
        data = df_merged.loc[mask, 'weighted_idf_effective_skills'].dropna()
        label = f"{'Correct' if is_correct else 'Incorrect'} (n={len(data)})"
        ax3.hist(data, bins=40, alpha=0.6, label=label, density=True)
    
    ax3.set_xlabel('Effective Number of Skills (Weighted IDF)', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('Weight Concentration vs Prediction Accuracy', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # === Panel 4: Median Frequency vs Rank ===
    ax4 = axes[1, 1]
    
    # Use log rank for better visualization
    df_plot = df_merged.copy()
    df_plot['log_rank'] = np.log10(df_plot['rank_hybrid'] + 1)
    
    scatter = ax4.scatter(
        df_plot['median_freq'],
        df_plot['log_rank'],
        c=df_plot['avg_score'],
        cmap='RdYlBu_r',
        alpha=0.5,
        s=15
    )
    
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Avg Skill Score', fontsize=10)
    
    ax4.set_xlabel('Median Skill Frequency (n_occupations)', fontsize=11)
    ax4.set_ylabel('Log₁₀(Rank + 1)', fontsize=11)
    ax4.set_title('Skill Frequency vs Prediction Rank', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "pooling_strategy_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.close()


def create_skill_confidence_analysis(
    df: pd.DataFrame,
    df_path_stats: pd.DataFrame,
    output_dir: str
):
    """
    Create additional visualization: Skill confidence score distribution.
    """
    # Merge path stats with main df
    df_merged = df.merge(df_path_stats, left_on='sample_idx', right_on='path_idx', how='left')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # === LEFT: Distribution of skill scores for correct vs incorrect predictions (Hybrid) ===
    ax1 = axes[0]
    
    for is_correct in [True, False]:
        mask = df['is_correct_hybrid'] == is_correct
        data = df.loc[mask, 'mean_skill_score'].dropna()
        label = f"{'Correct' if is_correct else 'Incorrect'} (n={len(data)})"
        ax1.hist(data, bins=50, alpha=0.7, label=label, density=True)
    
    ax1.set_xlabel('Mean Skill Prediction Score', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.set_title('Skill Score Distribution by Prediction Accuracy (Hybrid)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # === RIGHT: Scatter of num_skills vs rank ===
    ax2 = axes[1]
    
    # Use log rank for better visualization
    df_plot = df.copy()
    df_plot['log_rank_hybrid'] = np.log10(df_plot['rank_hybrid'] + 1)
    
    scatter = ax2.scatter(
        df_plot['num_skills'],
        df_plot['log_rank_hybrid'],
        c=df_plot['mean_genericness'],
        cmap='RdYlBu_r',
        alpha=0.5,
        s=15
    )
    
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label('Skill Genericness', fontsize=10)
    
    ax2.set_xlabel('Number of Skills in Career Path', fontsize=12)
    ax2.set_ylabel('Log₁₀(Rank + 1)', fontsize=12)
    ax2.set_title('Skills Count vs. Prediction Rank (Hybrid)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "skill_confidence_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.close()


def create_frequency_analysis(
    df: pd.DataFrame,
    df_path_stats: pd.DataFrame,
    output_dir: str
):
    """
    Create frequency-based analysis visualizations.
    """
    # Merge path stats with main df
    df_merged = df.merge(df_path_stats, left_on='sample_idx', right_on='path_idx', how='left')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    colors = {
        'text': '#1f77b4',
        'skill': '#ff7f0e',
        'hybrid': '#2ca02c'
    }
    
    # === Panel 1: Median Frequency by Correctness (Box Plot) ===
    ax1 = axes[0, 0]
    
    box_data = []
    labels = []
    box_colors = []
    
    for model_name, color in colors.items():
        for is_correct in [True, False]:
            mask = df_merged[f'is_correct_{model_name}'] == is_correct
            data = df_merged.loc[mask, 'median_freq'].dropna()
            box_data.append(data.values)
            status = "✓" if is_correct else "✗"
            labels.append(f'{model_name.capitalize()}\n{status}')
            box_colors.append(color)
    
    bp1 = ax1.boxplot(box_data, labels=labels, patch_artist=True)
    
    for patch, color in zip(bp1['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_ylabel('Median Skill Frequency (n_occupations)', fontsize=11)
    ax1.set_title('Skill Frequency by Prediction Correctness', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # === Panel 2: Min/Max Frequency Range ===
    ax2 = axes[0, 1]
    
    # For hybrid model
    correct_mask = df_merged['is_correct_hybrid'] == True
    incorrect_mask = df_merged['is_correct_hybrid'] == False
    
    for label, mask, color in [('Correct', correct_mask, '#2ca02c'), ('Incorrect', incorrect_mask, '#d62728')]:
        data = df_merged.loc[mask]
        ax2.scatter(
            data['min_freq'], 
            data['max_freq'],
            alpha=0.4, s=15, c=color, label=label
        )
    
    ax2.set_xlabel('Min Skill Frequency', fontsize=11)
    ax2.set_ylabel('Max Skill Frequency', fontsize=11)
    ax2.set_title('Skill Frequency Range (Hybrid Model)', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # === Panel 3: Frequency Std Dev vs Accuracy ===
    ax3 = axes[1, 0]
    
    for is_correct in [True, False]:
        mask = df_merged['is_correct_hybrid'] == is_correct
        data = df_merged.loc[mask, 'std_freq'].dropna()
        label = f"{'Correct' if is_correct else 'Incorrect'} (n={len(data)})"
        ax3.hist(data, bins=50, alpha=0.6, label=label, density=True)
    
    ax3.set_xlabel('Std Dev of Skill Frequency', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('Frequency Variance vs Prediction Accuracy', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # === Panel 4: Unique Skills vs Total Skills ===
    ax4 = axes[1, 1]
    
    scatter = ax4.scatter(
        df_merged['num_unique_skills'],
        df_merged['total_skills'],
        c=df_merged['is_correct_hybrid'].astype(int),
        cmap='RdYlGn',
        alpha=0.5,
        s=15
    )
    
    ax4.set_xlabel('Number of Unique Skills', fontsize=11)
    ax4.set_ylabel('Total Skill Occurrences', fontsize=11)
    ax4.set_title('Skill Diversity vs Repetition', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', alpha=0.7, label='Incorrect'),
        Patch(facecolor='#2ca02c', alpha=0.7, label='Correct')
    ]
    ax4.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "frequency_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.close()


def create_idf_comparison_visualization(
    df_path_stats: pd.DataFrame,
    esco_idf: Dict[str, float],
    decorte_idf: Dict[str, float],
    idf_comparison: Dict[str, any],
    output_dir: str
):
    """
    Create visualization comparing ESCO and Decorte IDF sources.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # === Panel 1: IDF Scatter (ESCO vs Decorte) ===
    ax1 = axes[0, 0]
    
    # Get common skills
    common_skills = set(esco_idf.keys()) & set(decorte_idf.keys())
    esco_vals = [esco_idf[s] for s in common_skills]
    decorte_vals = [decorte_idf[s] for s in common_skills]
    
    ax1.scatter(esco_vals, decorte_vals, alpha=0.3, s=10, c='#1f77b4')
    
    # Add diagonal line
    min_val = min(min(esco_vals), min(decorte_vals))
    max_val = max(max(esco_vals), max(decorte_vals))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x (perfect agreement)')
    
    # Add regression line
    if len(common_skills) > 10:
        slope, intercept, r_value, _, _ = stats.linregress(esco_vals, decorte_vals)
        x_line = np.linspace(min_val, max_val, 100)
        y_line = slope * x_line + intercept
        ax1.plot(x_line, y_line, 'g--', linewidth=2, 
                label=f'Regression (r={r_value:.3f})')
    
    ax1.set_xlabel('ESCO IDF (from taxonomy)', fontsize=11)
    ax1.set_ylabel('Decorte IDF (from dataset)', fontsize=11)
    ax1.set_title('IDF Source Comparison: ESCO vs Decorte', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # === Panel 2: IDF Distribution Comparison ===
    ax2 = axes[0, 1]
    
    ax2.hist(esco_vals, bins=50, alpha=0.6, label=f'ESCO (n={len(esco_vals)})', density=True)
    ax2.hist(decorte_vals, bins=50, alpha=0.6, label=f'Decorte (n={len(decorte_vals)})', density=True)
    
    ax2.set_xlabel('IDF Score', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title('IDF Distribution by Source', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # === Panel 3: Per-Path Genericness Comparison (ESCO vs Decorte) ===
    ax3 = axes[1, 0]
    
    # Check if Decorte columns exist
    if 'decorte_avg_genericness' in df_path_stats.columns:
        mask = (~df_path_stats['avg_genericness'].isna()) & (~df_path_stats['decorte_avg_genericness'].isna())
        df_plot = df_path_stats[mask]
        
        ax3.scatter(
            df_plot['avg_genericness'], 
            df_plot['decorte_avg_genericness'],
            alpha=0.4, s=15, c='#2ca02c'
        )
        
        # Add diagonal
        ax3.plot([0, 1], [0, 1], 'r--', linewidth=2, label='y=x')
        
        # Add regression
        if len(df_plot) > 10:
            slope, intercept, r_value, _, _ = stats.linregress(
                df_plot['avg_genericness'], df_plot['decorte_avg_genericness']
            )
            x_line = np.linspace(0, 1, 100)
            y_line = slope * x_line + intercept
            ax3.plot(x_line, y_line, 'b--', linewidth=2, 
                    label=f'Regression (r={r_value:.3f})')
        
        ax3.set_xlabel('ESCO Genericness (per path avg)', fontsize=11)
        ax3.set_ylabel('Decorte Genericness (per path avg)', fontsize=11)
        ax3.set_title('Per-Path Genericness: ESCO vs Decorte', fontsize=13, fontweight='bold')
        ax3.legend(loc='upper left')
    else:
        ax3.text(0.5, 0.5, 'Decorte data not available', ha='center', va='center', fontsize=12)
        ax3.set_title('Per-Path Genericness Comparison', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # === Panel 4: Frequency Distribution Comparison ===
    ax4 = axes[1, 1]
    
    if 'decorte_median_freq' in df_path_stats.columns:
        mask = (~df_path_stats['median_freq'].isna()) & (~df_path_stats['decorte_median_freq'].isna())
        df_plot = df_path_stats[mask]
        
        ax4.scatter(
            df_plot['median_freq'], 
            df_plot['decorte_median_freq'],
            alpha=0.4, s=15, c='#ff7f0e'
        )
        
        # Log scale for better visualization
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        
        ax4.set_xlabel('ESCO Median Frequency (n_occupations)', fontsize=11)
        ax4.set_ylabel('Decorte Median Frequency (n_jobs)', fontsize=11)
        ax4.set_title('Per-Path Median Frequency: ESCO vs Decorte', fontsize=13, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'Decorte data not available', ha='center', va='center', fontsize=12)
        ax4.set_title('Frequency Comparison', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "idf_source_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.close()


def compute_statistics_report(
    df: pd.DataFrame, 
    df_path_stats: pd.DataFrame, 
    output_dir: str, 
    idf_comparison: Optional[Dict] = None, 
    decorte_genericness: Optional[Dict[str, float]] = None,
    esco_genericness: Optional[Dict[str, float]] = None
):
    """
    Generate a comprehensive textual statistics report.
    """
    # Merge path stats with main df
    df_merged = df.merge(df_path_stats, left_on='sample_idx', right_on='path_idx', how='left')
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("POOLING METHOD ANALYSIS - COMPREHENSIVE STATISTICAL REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Overall metrics
    report_lines.append("--- OVERALL METRICS ---")
    for model in ['text', 'skill', 'hybrid']:
        acc = df[f'is_correct_{model}'].mean() * 100
        mrr = (1.0 / df[f'rank_{model}']).mean()
        median_rank = df[f'rank_{model}'].median()
        report_lines.append(f"  {model.upper():8s}: Acc@1 = {acc:.2f}%, MRR = {mrr:.4f}, Median Rank = {median_rank:.0f}")
    report_lines.append("")
    
    # Skill genericness statistics
    report_lines.append("--- SKILL GENERICNESS STATISTICS ---")
    report_lines.append(f"  Mean genericness: {df['mean_genericness'].mean():.4f} (std: {df['mean_genericness'].std():.4f})")
    report_lines.append(f"  Mean skill score: {df['mean_skill_score'].mean():.4f} (std: {df['mean_skill_score'].std():.4f})")
    report_lines.append(f"  Mean num skills: {df['num_skills'].mean():.2f} (std: {df['num_skills'].std():.2f})")
    report_lines.append("")
    
    # Detailed per-path frequency statistics
    report_lines.append("--- DETAILED FREQUENCY STATISTICS (Per Career Path) ---")
    report_lines.append("  [ESCO Taxonomy Source]")
    for stat in ['min_freq', 'median_freq', 'max_freq', 'avg_freq']:
        values = df_merged[stat].dropna()
        report_lines.append(f"  {stat:12s}: mean={values.mean():.2f}, std={values.std():.2f}, median={values.median():.2f}")
    
    if 'decorte_min_freq' in df_merged.columns:
        report_lines.append("")
        report_lines.append("  [Decorte Dataset Source]")
        for stat in ['decorte_min_freq', 'decorte_median_freq', 'decorte_max_freq', 'decorte_avg_freq']:
            values = df_merged[stat].dropna()
            if len(values) > 0:
                report_lines.append(f"  {stat:20s}: mean={values.mean():.2f}, std={values.std():.2f}, median={values.median():.2f}")
    report_lines.append("")
    
    # Pooling strategy comparison
    report_lines.append("--- POOLING STRATEGY ANALYSIS ---")
    for strategy in ['mean', 'weighted_mean', 'weighted_idf']:
        eff_col = f'{strategy}_effective_skills'
        if eff_col in df_merged.columns:
            values = df_merged[eff_col].dropna()
            report_lines.append(f"  {strategy.upper():15s}:")
            report_lines.append(f"    Effective skills: mean={values.mean():.2f}, std={values.std():.2f}, median={values.median():.2f}")
    report_lines.append("")
    
    # Correlation analysis
    report_lines.append("--- CORRELATION ANALYSIS ---")
    for model in ['text', 'skill', 'hybrid']:
        # Correlation between genericness and confidence
        mask = ~(np.isnan(df['mean_genericness']) | np.isnan(df[f'confidence_{model}']))
        if mask.sum() > 10:
            r, p = stats.pearsonr(
                df.loc[mask, 'mean_genericness'],
                df.loc[mask, f'confidence_{model}']
            )
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            report_lines.append(f"  {model.upper():8s} confidence vs genericness: r = {r:.4f} (p = {p:.4e}) {sig}")
    report_lines.append("")
    
    # Genericness by correctness (t-tests)
    report_lines.append("--- GENERICNESS BY PREDICTION CORRECTNESS (t-tests) ---")
    for model in ['text', 'skill', 'hybrid']:
        correct = df.loc[df[f'is_correct_{model}'], 'mean_genericness'].dropna()
        incorrect = df.loc[~df[f'is_correct_{model}'], 'mean_genericness'].dropna()
        
        if len(correct) > 10 and len(incorrect) > 10:
            t_stat, p_val = stats.ttest_ind(correct, incorrect)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            report_lines.append(f"  {model.upper():8s}: Correct mean = {correct.mean():.4f}, Incorrect mean = {incorrect.mean():.4f}")
            report_lines.append(f"            t = {t_stat:.3f}, p = {p_val:.4e} {sig}")
    report_lines.append("")
    
    # Frequency by correctness (t-tests)
    report_lines.append("--- MEDIAN FREQUENCY BY PREDICTION CORRECTNESS (t-tests) ---")
    for model in ['text', 'skill', 'hybrid']:
        correct = df_merged.loc[df_merged[f'is_correct_{model}'], 'median_freq'].dropna()
        incorrect = df_merged.loc[~df_merged[f'is_correct_{model}'], 'median_freq'].dropna()
        
        if len(correct) > 10 and len(incorrect) > 10:
            t_stat, p_val = stats.ttest_ind(correct, incorrect)
            sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
            report_lines.append(f"  {model.upper():8s}: Correct mean = {correct.mean():.2f}, Incorrect mean = {incorrect.mean():.2f}")
            report_lines.append(f"            t = {t_stat:.3f}, p = {p_val:.4e} {sig}")
    report_lines.append("")
    
    # Impact of skill score quality
    report_lines.append("--- SKILL SCORE QUALITY IMPACT (Quartiles) ---")
    try:
        df['skill_score_quantile'], bins = pd.qcut(df['mean_skill_score'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'], retbins=True, duplicates='drop')
        
        for i, q in enumerate(['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)']):
            if i >= len(bins) - 1: break
            subset = df[df['skill_score_quantile'] == q]
            if len(subset) > 0:
                report_lines.append(f"  {q} [{bins[i]:.4f} - {bins[i+1]:.4f}]:")
                for model in ['text', 'skill', 'hybrid']:
                    acc = subset[f'is_correct_{model}'].mean() * 100
                    mrr = (1.0 / subset[f'rank_{model}']).mean()
                    report_lines.append(f"    {model.upper():8s}: Acc@1 = {acc:.2f}%, MRR = {mrr:.4f}")
    except ValueError as e:
        report_lines.append(f"  Could not compute quartiles: {e}")
    report_lines.append("")
    
    # Impact of skill genericness (ESCO)
    report_lines.append("--- SKILL GENERICNESS IMPACT (Quartiles - using ESCO) ---")
    try:
        df['genericness_quantile'], bins = pd.qcut(df['mean_genericness'].dropna(), q=4, labels=['Q1 (Specific)', 'Q2', 'Q3', 'Q4 (Generic)'], retbins=True, duplicates='drop')
        
        for i, q in enumerate(['Q1 (Specific)', 'Q2', 'Q3', 'Q4 (Generic)']):
            if i >= len(bins) - 1: break
            subset = df[df['genericness_quantile'] == q]
            if len(subset) > 0:
                report_lines.append(f"  {q} [{bins[i]:.4f} - {bins[i+1]:.4f}]:")
                for model in ['text', 'skill', 'hybrid']:
                    acc = subset[f'is_correct_{model}'].mean() * 100
                    mrr = (1.0 / subset[f'rank_{model}']).mean()
                    report_lines.append(f"    {model.upper():8s}: Acc@1 = {acc:.2f}%, MRR = {mrr:.4f}")
    except ValueError as e:
         report_lines.append(f"  Could not compute quartiles: {e}")
    report_lines.append("")

    # Impact of skill genericness (Decorte)
    if 'mean_decorte_genericness' in df.columns:
        report_lines.append("--- SKILL GENERICNESS IMPACT (Quartiles - using Decorte Dataset) ---")
        try:
            df['decorte_genericness_quantile'], bins = pd.qcut(df['mean_decorte_genericness'].dropna(), q=4, labels=['Q1 (Specific)', 'Q2', 'Q3', 'Q4 (Generic)'], retbins=True, duplicates='drop')
            
            for i, q in enumerate(['Q1 (Specific)', 'Q2', 'Q3', 'Q4 (Generic)']):
                if i >= len(bins) - 1: break
                subset = df[df['decorte_genericness_quantile'] == q]
                if len(subset) > 0:
                    report_lines.append(f"  {q} [{bins[i]:.4f} - {bins[i+1]:.4f}]:")
                    for model in ['text', 'skill', 'hybrid']:
                        acc = subset[f'is_correct_{model}'].mean() * 100
                        mrr = (1.0 / subset[f'rank_{model}']).mean()
                        report_lines.append(f"    {model.upper():8s}: Acc@1 = {acc:.2f}%, MRR = {mrr:.4f}")
        except ValueError as e:
            report_lines.append(f"  Could not compute quartiles: {e}")
        report_lines.append("")

    # Impact of pooling weight concentration
    report_lines.append("--- POOLING WEIGHT CONCENTRATION IMPACT ---")
    if 'weighted_idf_effective_skills' in df_merged.columns:
        try:
            df_merged['eff_skills_quantile'], bins = pd.qcut(
                df_merged['weighted_idf_effective_skills'].dropna(), 
                q=4, labels=['Q1 (Concentrated)', 'Q2', 'Q3', 'Q4 (Dispersed)'],
                retbins=True, duplicates='drop'
            )
            
            for i, q in enumerate(['Q1 (Concentrated)', 'Q2', 'Q3', 'Q4 (Dispersed)']):
                if i >= len(bins) - 1: break
                subset = df_merged[df_merged['eff_skills_quantile'] == q]
                if len(subset) > 0:
                    report_lines.append(f"  {q} [{bins[i]:.4f} - {bins[i+1]:.4f}]:")
                    for model in ['text', 'skill', 'hybrid']:
                        acc = subset[f'is_correct_{model}'].mean() * 100
                        mrr = (1.0 / subset[f'rank_{model}']).mean()
                        report_lines.append(f"    {model.upper():8s}: Acc@1 = {acc:.2f}%, MRR = {mrr:.4f}")
        except ValueError as e:
             report_lines.append(f"  Could not compute quartiles: {e}")
    report_lines.append("")
    
    # IDF Source Comparison (if available)
    if idf_comparison:
        report_lines.append("--- IDF SOURCE COMPARISON (ESCO vs DECORTE) ---")
        report_lines.append("  This compares IDF scores computed from two different sources:")
        report_lines.append("    • ESCO: From ESCO taxonomy (official occupation-skill relations)")
        report_lines.append("    • Decorte: From actual job postings in the dataset")
        report_lines.append("")
        report_lines.append(f"  Common skills in both sources: {idf_comparison.get('n_common', 'N/A')}")
        report_lines.append(f"  Skills only in ESCO: {idf_comparison.get('n_esco_only', 'N/A')}")
        report_lines.append(f"  Skills only in Decorte: {idf_comparison.get('n_decorte_only', 'N/A')}")
        
        if 'pearson_r' in idf_comparison:
            r = idf_comparison['pearson_r']
            p = idf_comparison['pearson_p']
            rho = idf_comparison['spearman_rho']
            p_rho = idf_comparison['spearman_p']
            report_lines.append("")
            report_lines.append(f"  Correlation between sources:")
            report_lines.append(f"    Pearson r = {r:.4f} (p = {p:.4e}){'***' if p < 0.001 else ''}")
            report_lines.append(f"    Spearman ρ = {rho:.4f} (p = {p_rho:.4e}){'***' if p_rho < 0.001 else ''}")
            report_lines.append("")
            
            # Interpretation
            if r > 0.7:
                interpretation = "Strong agreement between ESCO and Decorte IDF scores."
            elif r > 0.4:
                interpretation = "Moderate agreement - some skills have different frequencies in real-world data."
            else:
                interpretation = "Weak agreement - real-world skill usage differs significantly from ESCO taxonomy."
            report_lines.append(f"  Interpretation: {interpretation}")
        
        # Decorte-based frequency statistics
        if 'decorte_median_freq' in df_path_stats.columns:
            report_lines.append("")
            report_lines.append("  Decorte Frequency Statistics (per career path):")
            for stat in ['decorte_min_freq', 'decorte_median_freq', 'decorte_max_freq', 'decorte_avg_freq']:
                values = df_merged[stat].dropna()
                if len(values) > 0:
                    report_lines.append(f"    {stat:20s}: mean={values.mean():.2f}, std={values.std():.2f}")

    # Top/Bottom generic skills from ESCO
    if esco_genericness:
        report_lines.append("")
        report_lines.append("--- EXTREME SKILLS (ESCO-based Genericness / Taxonomy) ---")
        
        # Load skill names (loaded once)
        if 'uri_to_name' not in locals():
            try:
                skills_df = pd.read_csv(ESCO_SKILLS_FILE)
                uri_to_name = dict(zip(skills_df['conceptUri'], skills_df['preferredLabel']))
            except Exception as e:
                print(f"Warning: Could not load ESCO skill names: {e}")
                uri_to_name = {}

        # Sort skills by ESCO genericness
        sorted_skills_esco = sorted(esco_genericness.items(), key=lambda x: x[1], reverse=True)
        
        report_lines.append("  Top 10 Most Generic (Most Common in Taxonomy) Skills:")
        for uri, gen_score in sorted_skills_esco[:10]:
            name = uri_to_name.get(uri, uri)
            report_lines.append(f"    {gen_score:.4f}: {name}")

        report_lines.append("")
        report_lines.append("  Top 10 Least Generic (Rare in Taxonomy) Skills:")
        for uri, gen_score in sorted_skills_esco[-10:]:
            name = uri_to_name.get(uri, uri)
            report_lines.append(f"    {gen_score:.4f}: {name}")

    # Top/Bottom generic skills from Decorte
    if decorte_genericness:
        report_lines.append("")
        report_lines.append("--- EXTREME SKILLS (Decorte-based Genericness / Dataset) ---")
        
        # Load skill names (if not already loaded)
        if 'uri_to_name' not in locals():
            try:
                skills_df = pd.read_csv(ESCO_SKILLS_FILE)
                uri_to_name = dict(zip(skills_df['conceptUri'], skills_df['preferredLabel']))
            except Exception as e:
                print(f"Warning: Could not load ESCO skill names: {e}")
                uri_to_name = {}

        # Sort skills by genericness (high genericness = common)
        sorted_skills = sorted(decorte_genericness.items(), key=lambda x: x[1], reverse=True)
        
        report_lines.append("  Top 10 Most Generic (Most Common) Skills:")
        for uri, gen_score in sorted_skills[:10]:
            name = uri_to_name.get(uri, uri)
            report_lines.append(f"    {gen_score:.4f}: {name}")
            
        report_lines.append("")
        report_lines.append("  Top 10 Least Generic (Rare) Skills:")
        for uri, gen_score in sorted_skills[-10:]:
            name = uri_to_name.get(uri, uri)
            report_lines.append(f"    {gen_score:.4f}: {name}")

    report_lines.append("=" * 80)
    
    # Save report
    report_path = os.path.join(output_dir, "statistics_report.txt")
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    print(f"Saved report: {report_path}")
    
    # Also print to stdout
    print('\n'.join(report_lines))


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze pooling method effects on CPP")
    
    parser.add_argument("--text_scores", type=str, required=True,
                       help="Path to text-only CPP scores pickle file")
    parser.add_argument("--skill_scores", type=str, required=True,
                       help="Path to skill-only CPP scores pickle file")
    parser.add_argument("--hybrid_scores", type=str, required=True,
                       help="Path to hybrid CPP scores pickle file")
    parser.add_argument("--master_skill_file", type=str, required=True,
                       help="Path to fused_predictions.json with skill scores per job_id")
    parser.add_argument("--master_job_data", type=str, default=None,
                       help="Path to decorte_master_3.csv (optional, for additional metadata)")
    parser.add_argument("--esco_taxonomy_file", type=str, default=ESCO_TAXONOMY_FILE,
                       help="Path to ESCO occupation-skill relations CSV")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for figures and reports")
    parser.add_argument("--alpha", type=float, default=1.0,
                       help="Exponent for confidence score in weighted_idf pooling")
    parser.add_argument("--beta", type=float, default=1.0,
                       help="Exponent for IDF score in weighted_idf pooling")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("POOLING METHOD ANALYSIS FOR CPP")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load prediction data
    print("\n[1/6] Loading prediction scores...")
    print(f"  > Text:   {args.text_scores}")
    d_text = load_prediction_data(args.text_scores)
    print(f"    Loaded {d_text['scores'].shape[0]} samples")
    
    print(f"  > Skill:  {args.skill_scores}")
    d_skill = load_prediction_data(args.skill_scores)
    print(f"    Loaded {d_skill['scores'].shape[0]} samples")
    
    print(f"  > Hybrid: {args.hybrid_scores}")
    d_hybrid = load_prediction_data(args.hybrid_scores)
    print(f"    Loaded {d_hybrid['scores'].shape[0]} samples")
    
    # Load skill scores and compute IDF/frequency
    print("\n[2/7] Loading skill scores and IDF data...")
    print(f"  > Skill scores: {args.master_skill_file}")
    skill_scores_map = load_skill_scores(args.master_skill_file)
    print(f"    Loaded skills for {len(skill_scores_map)} job_ids")
    
    # ESCO Taxonomy-based IDF (static, from official ESCO database)
    print(f"\n  > Computing IDF from ESCO taxonomy: {args.esco_taxonomy_file}")
    skill_idf = compute_skill_idf_from_taxonomy(args.esco_taxonomy_file)
    skill_genericness = compute_skill_genericness(skill_idf)
    skill_freq = compute_skill_frequency_from_taxonomy(args.esco_taxonomy_file)
    print(f"    Computed ESCO IDF/frequency for {len(skill_idf)} skills")
    df_master = load_master_job_data(args.master_job_data)
    
    # Decorte Dataset-based IDF (dynamic, from actual job postings)
    decorte_idf, decorte_freq = compute_skill_idf_from_dataset(
        skill_scores_map, source_name="Decorte dataset (fused_predictions.json)", master_df=df_master
    )
    decorte_genericness = compute_skill_genericness_from_idf(decorte_idf, source_prefix="decorte")
    print(f"    Computed Decorte IDF/frequency for {len(decorte_idf)} skills")
    
    # Compare IDF sources
    idf_comparison = compare_idf_sources(skill_idf, decorte_idf)
    
    # Align predictions
    print("\n[3/7] Aligning predictions across models...")
    aligned_scores, aligned_targets, master_labels, aligned_job_ids = align_predictions(
        d_text, d_skill, d_hybrid
    )
    
    # Create analysis DataFrames
    print("\n[4/7] Building analysis DataFrames...")
    df, df_path_stats = create_analysis_dataframe(
        aligned_scores, aligned_targets, aligned_job_ids,
        skill_scores_map, skill_idf, skill_freq, skill_genericness,
        decorte_idf=decorte_idf, decorte_freq=decorte_freq, decorte_genericness=decorte_genericness
    )
    
    # Save the DataFrames for later analysis
    df_path = os.path.join(args.output_dir, "analysis_dataframe.csv")
    df.to_csv(df_path, index=False)
    print(f"  Saved main DataFrame to: {df_path}")
    
    df_path_stats_path = os.path.join(args.output_dir, "path_statistics.csv")
    df_path_stats.to_csv(df_path_stats_path, index=False)
    print(f"  Saved path statistics to: {df_path_stats_path}")
    
    # Create visualizations and report
    print("\n[5/7] Creating visualizations...")
    create_visualizations(df, args.output_dir)
    create_quartile_performance_analysis(df, args.output_dir)
    create_pooling_heatmap(df, df_path_stats, args.output_dir)
    
    # Legacy visualizations (replaced by new requests)
    # create_skill_confidence_analysis(df, df_path_stats, args.output_dir)
    # create_pooling_strategy_analysis(df, df_path_stats, args.output_dir)
    # create_frequency_analysis(df, df_path_stats, args.output_dir)
    
    # Create IDF comparison visualization (Keeping as it might be useful, or comment out if strictly replacing)
    # The user said "Replace them with the current ones", suggesting broad replacement. 
    # But IDF comparison is quite specific. I'll leave it but commented out to be safe/clean as per "Replace them" instruction.
    # print("\n[6/7] Creating IDF source comparison...")
    # create_idf_comparison_visualization(
    #     df_path_stats, skill_idf, decorte_idf, idf_comparison, args.output_dir
    # )
    
    print("\n[7/7] Generating statistical report...")
    print("\n[7/7] Generating statistical report...")
    compute_statistics_report(
        df, df_path_stats, args.output_dir, 
        idf_comparison=idf_comparison, 
        decorte_genericness=decorte_genericness,
        esco_genericness=skill_genericness
    )
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
