"""
Script to quantify the information overlap between career text embeddings (v_C)
and aggregated skill embeddings (h_C).

Computes pairwise cosine similarity between v_C and h_C for test samples
across different pooling strategies.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
from tqdm import tqdm
import torch
import random
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
# Add src directory to sys.path to allow "from cpp import ..."
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

try:
    from src.cpp.data_classes import Data
    from src.cpp.utils import SEP_TOKEN
    from src.cpp.data_loaders import (
        load_job_skill_data_by_id,
        precompute_target_embeddings,
        precompute_input_embeddings_with_job_ids,
        load_precomputed_skill_embeddings,
        #filter_repetitive_samples_with_job_ids,
    )
    
    # Mock optuna to avoid import error from train_cpp_enhanced_v3
    if 'optuna' not in sys.modules:
        import types
        sys.modules['optuna'] = types.ModuleType('optuna')

    # Import V3 specific functions
    from src.cpp.train_cpp_enhanced_v3 import (
        build_last_job_skill_embeddings,
        load_skill_mappings,  # V3 helper
        load_skill_descriptions, # V3 helper
    )
except ImportError as e:
    print(f"Error: Required modules not found. {e}")
    sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Quantify Embedding Overlap")
    
    # Data paths (defaults matching train_cpp_enhanced_v2.py where possible)
    parser.add_argument("--data_type", type=str, default="decorte")
    parser.add_argument("--skill_scores_file", type=str, required=True,
                       help="Path to JSON file with skill scores keyed by job_id")
    parser.add_argument("--esco_skills_file", type=str, default="data/esco_datasets/skills_en.csv")
    parser.add_argument("--esco_taxonomy_file", type=str, 
                       default="/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/occupationSkillRelations_en.csv")
    parser.add_argument("--vocab_dir", type=str, default="data/processed/master_datasets_2/")

    
    # Encoder configuration
    parser.add_argument("--encoder_text", type=str, default="ElenaSenger/career-path-representation-mpnet-decorte")
    parser.add_argument("--encoder_skill", type=str, default="")
    parser.add_argument("--skill_embeddings_dir", type=str, default=None,
                       help="Path to directory with precomputed skill embeddings")
    
    # Cache
    parser.add_argument("--embeddings_cache_dir", type=str, 
                       default="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/embeddings")
    parser.add_argument("--force_recompute", action='store_true')

    # Analysis output
    parser.add_argument("--output_dir", type=str, default="src/error_analysis/embeddings/results")
    
    # Parameters for IDF calculation and filtering
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--skill_confidence_threshold", type=float, default=None)
    parser.add_argument("--filter_repetitive", action='store_true')
    parser.add_argument("--no_subspans", action='store_true')
    parser.add_argument("--eval_clean_test", action='store_true')
    parser.add_argument("--use_text_description", action='store_true')
    parser.add_argument("--use_skill_description", action='store_true')
    parser.add_argument("--seed", type=int, default=42)

    # V3 (Concatenation) Arguments
    parser.add_argument("--method", type=str, default="v2", choices=["v2", "v3", "both"],
                       help="Methods to analyze: 'v2' (pooling), 'v3' (concatenation), or 'both'.")
    parser.add_argument("--top_k_skills", type=int, default=10, 
                       help="[V3] Max number of skills per job.")
    parser.add_argument("--skill_selection_strategy", type=str, default="top_k",
                       choices=["top_k", "stratified"],
                       help="[V3] Strategy to select skills.")
    parser.add_argument("--scoring_mode", type=str, default="idf_only",
                       choices=["idf_only", "scores_only", "weighted"],
                       help="[V3] Scoring mode for skill selection.")
    parser.add_argument("--importance_weight", type=float, default=0.5,
                       help="[V3] Weight for per-job scores in weighted scoring (0-1).")
    parser.add_argument("--skill_scores_json", type=str, default=None,
                        help="[V3] Path to fused_predictions.json (alias for skill_scores_file if not provided)")
    
    # Task 3 and Task 5 arguments
    parser.add_argument("--run_task3", action='store_true',
                       help="Run Task 3: Target Proximity Analysis")
    parser.add_argument("--run_task5", action='store_true',
                       help="Run Task 5: Correctness Pattern Analysis (requires prediction files)")
    parser.add_argument("--predictions_text", type=str, default=None,
                       help="Path to text-only model predictions .pkl file for Task 5")
    parser.add_argument("--predictions_skill", type=str, default=None,
                       help="Path to skill-only model predictions .pkl file for Task 5")
    parser.add_argument("--predictions_hybrid", type=str, default=None,
                       help="Path to hybrid model predictions .pkl file for Task 5")
    
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_cosine_stats(v_c, h_c):
    """
    Compute pairwise cosine similarity between v_c and h_c rows.
    Returns statistics dict.
    """
    # Validate dimensions
    logger.info(f"    Embedding Shapes: v_C (Text)={v_c.shape}, h_C (Skill)={h_c.shape}")
    if v_c.shape[1] != h_c.shape[1]:
        logger.error(f"⚠️  Dimension mismatch in compute_cosine_stats: Text dim={v_c.shape[1]} vs Skill dim={h_c.shape[1]}")
        logger.error("    Direct cosine similarity cannot be computed between different embedding spaces.")
        logger.error("    Please ensure both encoders produce embeddings of the same dimension.")
        return None, None

    # Normalize vectors to use dot product as cosine similarity
    v_c_norm = v_c / (np.linalg.norm(v_c, axis=1, keepdims=True) + 1e-8)
    h_c_norm = h_c / (np.linalg.norm(h_c, axis=1, keepdims=True) + 1e-8)
    
    # Pairwise cosine similarity is the diagonal of the dot product matrix
    # But we can compute it more efficiently as element-wise multiplication sum
    similarities = np.sum(v_c_norm * h_c_norm, axis=1)
    
    stats = {
        "mean": float(np.mean(similarities)),
        "std": float(np.std(similarities)),
        "min": float(np.min(similarities)),
        "max": float(np.max(similarities)),
        "p25": float(np.percentile(similarities, 25)),
        "p50": float(np.median(similarities)),
        "p75": float(np.percentile(similarities, 75)),
        "p90": float(np.percentile(similarities, 90)),
        "p99": float(np.percentile(similarities, 99)),
    }
    return similarities, stats

def compute_target_proximity_stats(v_c, h_c, target_embeddings_array, target_labels_list, Y_target_dict):
    """
    Task 3: Compare distance to target occupation.
    
    For each sample:
    - sim_text = cosine_similarity(v_C, v_target)
    - sim_skill = cosine_similarity(h_C, v_target)
    - delta = sim_text - sim_skill
    
    Returns: dict with statistics, arrays for visualization, and statistical test results
    """
    logger.info("  📊 Task 3: Target Proximity Analysis")
    
    n_samples = len(v_c)
    sim_text_to_target = np.zeros(n_samples)
    sim_skill_to_target = np.zeros(n_samples)
    
    # Normalize embeddings
    v_c_norm = v_c / (np.linalg.norm(v_c, axis=1, keepdims=True) + 1e-8)
    h_c_norm = h_c / (np.linalg.norm(h_c, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarities to target for each sample
    for i, target_label in enumerate(target_labels_list):
        if target_label in Y_target_dict:
            target_emb = Y_target_dict[target_label]
            target_norm = target_emb / (np.linalg.norm(target_emb) + 1e-8)
            
            sim_text_to_target[i] = np.dot(v_c_norm[i], target_norm)
            sim_skill_to_target[i] = np.dot(h_c_norm[i], target_norm)
    
    delta = sim_text_to_target - sim_skill_to_target
    
    # Compute statistics
    stats_text = {
        "mean": float(np.mean(sim_text_to_target)),
        "std": float(np.std(sim_text_to_target)),
        "min": float(np.min(sim_text_to_target)),
        "max": float(np.max(sim_text_to_target)),
        "median": float(np.median(sim_text_to_target)),
    }
    
    stats_skill = {
        "mean": float(np.mean(sim_skill_to_target)),
        "std": float(np.std(sim_skill_to_target)),
        "min": float(np.min(sim_skill_to_target)),
        "max": float(np.max(sim_skill_to_target)),
        "median": float(np.median(sim_skill_to_target)),
    }
    
    stats_delta = {
        "mean": float(np.mean(delta)),
        "std": float(np.std(delta)),
        "min": float(np.min(delta)),
        "max": float(np.max(delta)),
        "median": float(np.median(delta)),
    }
    
    # Paired statistical test (Wilcoxon signed-rank test)
    stat_result = stats.wilcoxon(sim_text_to_target, sim_skill_to_target, alternative='two-sided')
    
    logger.info(f"    Text→Target: Mean={stats_text['mean']:.4f}, Std={stats_text['std']:.4f}")
    logger.info(f"    Skill→Target: Mean={stats_skill['mean']:.4f}, Std={stats_skill['std']:.4f}")
    logger.info(f"    Delta (Text-Skill): Mean={stats_delta['mean']:.4f}, Std={stats_delta['std']:.4f}")
    logger.info(f"    Wilcoxon test: statistic={stat_result.statistic:.2f}, p={stat_result.pvalue:.4e}")
    
    return {
        "stats_text": stats_text,
        "stats_skill": stats_skill,
        "stats_delta": stats_delta,
        "statistical_test": {
            "test": "wilcoxon_signed_rank",
            "statistic": float(stat_result.statistic),
            "pvalue": float(stat_result.pvalue),
        },
        "arrays": {
            "sim_text_to_target": sim_text_to_target,
            "sim_skill_to_target": sim_skill_to_target,
            "delta": delta,
        }
    }

def load_predictions(predictions_file):
    """
    Load prediction scores from pickle file.
    
    Expected format: dict with keys like 'scores', 'predictions', 'targets', 'job_ids'
    or numpy array of scores.
    
    Returns: dict with predictions data
    """
    if not predictions_file or not os.path.exists(predictions_file):
        return None
    
    try:
        with open(predictions_file, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"  ✓ Loaded predictions from {os.path.basename(predictions_file)}")
        return data
    except Exception as e:
        logger.warning(f"  ⚠️ Failed to load predictions from {predictions_file}: {e}")
        return None

def compute_top1_correctness(predictions_data, target_labels_list, Y_target_dict, label=None):
    """
    Compute top-1 prediction correctness from scores.
    
    Args:
        predictions_data: dict with 'scores' (NxM similarity matrix) and 'target_labels'
        target_labels_list: list of true target labels for each sample
        Y_target_dict: dict mapping labels to embeddings (not used, kept for compatibility)
    
    Returns: boolean array of correctness
    """
    if predictions_data is None:
        return None
    
    # Extract scores matrix and target labels
    if isinstance(predictions_data, dict):
        if 'scores' in predictions_data:
            scores = predictions_data['scores']
            # Use target_labels from the predictions file (correct order!)
            if 'target_labels' in predictions_data:
                label_order = predictions_data['target_labels']
                logger.info(f"    Using target_labels from predictions file ({len(label_order)} targets)")
            else:
                logger.warning("  ⚠️ No 'target_labels' in predictions dict, falling back to sorted Y_target_dict")
                label_order = sorted(Y_target_dict.keys())
        elif 'similarities' in predictions_data:
            scores = predictions_data['similarities']
            if 'target_labels' in predictions_data:
                label_order = predictions_data['target_labels']
            else:
                label_order = sorted(Y_target_dict.keys())
        else:
            logger.warning("  ⚠️ Predictions dict has no 'scores' or 'similarities' key")
            return None
    elif isinstance(predictions_data, np.ndarray):
        scores = predictions_data
        # Fallback to sorted order if only array provided
        label_order = sorted(Y_target_dict.keys())
        logger.warning("  ⚠️ Predictions is numpy array, using sorted Y_target_dict for label order")
    else:
        logger.warning(f"  ⚠️ Unknown predictions format: {type(predictions_data)}")
        return None
    
    # Compute top-1 predictions
    top1_indices = np.argmax(scores, axis=1)
    top1_labels = [label_order[idx] for idx in top1_indices]
    
    # Compute correctness
    correctness = np.array([pred == true for pred, true in zip(top1_labels, target_labels_list)])
    
    accuracy = np.mean(correctness)
    logger.info(f"    Top-1 Accuracy of {label}: {accuracy:.4f} ({np.sum(correctness)}/{len(correctness)})")
    
    # Sanity check: warn if accuracy is suspiciously low
    if accuracy < 0.05:
        logger.warning(f"  ⚠️ WARNING: Accuracy is very low ({accuracy:.2%}). This might indicate a label ordering mismatch.")
        logger.warning(f"     Expected ~18.6% for text-only model. Please verify target_labels alignment.")
    
    return correctness

def analyze_correctness_patterns(v_c, h_c, target_embeddings_array, target_labels_list, 
                                  Y_target_dict, correctness_labels, text_skill_similarities,
                                  model_name="model"):
    """
    Task 5: Analyze patterns by prediction correctness.
    
    Split samples by correct/incorrect predictions and compare:
    - Text-skill similarity (from Task 1)
    - Text-to-target similarity (from Task 3)
    - Skill-to-target similarity (from Task 3)
    
    Returns: dict with group statistics and p-values
    """
    logger.info(f"  📊 Task 5: Correctness Pattern Analysis ({model_name})")
    
    if correctness_labels is None:
        logger.warning("  ⚠️ No correctness labels provided, skipping Task 5")
        return None
    
    # Split by correctness
    correct_mask = correctness_labels
    incorrect_mask = ~correctness_labels
    
    n_correct = np.sum(correct_mask)
    n_incorrect = np.sum(incorrect_mask)
    
    logger.info(f"    Correct: {n_correct}, Incorrect: {n_incorrect}")
    
    if n_correct == 0 or n_incorrect == 0:
        logger.warning("  ⚠️ One group is empty, cannot perform comparison")
        return None
    
    # Compute target proximities
    v_c_norm = v_c / (np.linalg.norm(v_c, axis=1, keepdims=True) + 1e-8)
    h_c_norm = h_c / (np.linalg.norm(h_c, axis=1, keepdims=True) + 1e-8)
    
    sim_text_to_target = np.zeros(len(v_c))
    sim_skill_to_target = np.zeros(len(v_c))
    
    for i, target_label in enumerate(target_labels_list):
        if target_label in Y_target_dict:
            target_emb = Y_target_dict[target_label]
            target_norm = target_emb / (np.linalg.norm(target_emb) + 1e-8)
            sim_text_to_target[i] = np.dot(v_c_norm[i], target_norm)
            sim_skill_to_target[i] = np.dot(h_c_norm[i], target_norm)
    
    # Compute statistics for each group
    metrics = {
        "text_skill_sim": text_skill_similarities,
        "text_to_target": sim_text_to_target,
        "skill_to_target": sim_skill_to_target,
    }
    
    results = {}
    for metric_name, metric_values in metrics.items():
        correct_vals = metric_values[correct_mask]
        incorrect_vals = metric_values[incorrect_mask]
        
        # Mann-Whitney U test
        stat_result = stats.mannwhitneyu(correct_vals, incorrect_vals, alternative='two-sided')
        
        results[metric_name] = {
            "correct": {
                "mean": float(np.mean(correct_vals)),
                "std": float(np.std(correct_vals)),
                "median": float(np.median(correct_vals)),
                "n": int(n_correct),
            },
            "incorrect": {
                "mean": float(np.mean(incorrect_vals)),
                "std": float(np.std(incorrect_vals)),
                "median": float(np.median(incorrect_vals)),
                "n": int(n_incorrect),
            },
            "test": {
                "name": "mann_whitney_u",
                "statistic": float(stat_result.statistic),
                "pvalue": float(stat_result.pvalue),
            }
        }
        
        logger.info(f"    {metric_name}:")
        logger.info(f"      Correct: {results[metric_name]['correct']['mean']:.4f} ± {results[metric_name]['correct']['std']:.4f}")
        logger.info(f"      Incorrect: {results[metric_name]['incorrect']['mean']:.4f} ± {results[metric_name]['incorrect']['std']:.4f}")
        logger.info(f"      p-value: {results[metric_name]['test']['pvalue']:.4e}")
    
    return {
        "model_name": model_name,
        "n_correct": int(n_correct),
        "n_incorrect": int(n_incorrect),
        "metrics": results,
        "arrays": {
            "correct_mask": correct_mask,
            "incorrect_mask": incorrect_mask,
        }
    }

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def generate_visualizations(split_name, config_name, task1_similarities, task3_results, task5_results, output_dir):
    """
    Generate thesis-quality visualizations with large font sizes.
    """
    # 1. Global styling to match your butterfly plot example
    sns.set_style("whitegrid")
    plt.rcParams['figure.dpi'] = 300
    # Update tick label sizes globally for consistency
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    
    LABEL_FONT = 16
    TITLE_FONT = 20
    
    viz_dir = os.path.join(output_dir, "visualizations", split_name)
    os.makedirs(viz_dir, exist_ok=True)
    
    # Task 1: Histogram
    if task1_similarities is not None:
        fig, ax = plt.subplots(figsize=(10, 7)) # Slightly larger for readability
        ax.hist(task1_similarities, bins=50, edgecolor='black', alpha=0.7, color='#3498db')
        
        ax.set_xlabel('Text-Skill Cosine Similarity', fontsize=LABEL_FONT, labelpad=12)
        ax.set_ylabel('Frequency', fontsize=LABEL_FONT, labelpad=12)
        ax.set_title(f'Similarity Distribution\n({split_name})', fontsize=TITLE_FONT, fontweight='bold', pad=20)
        
        mean_val = np.mean(task1_similarities)
        ax.axvline(mean_val, color='#e74c3c', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_val:.3f}')
        
        ax.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"task1_histogram_{config_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Task 3: Scatter plot
    if task3_results is not None:
        arrays = task3_results['arrays']
        fig, ax = plt.subplots(figsize=(9, 9)) # Square aspect ratio for scatter
        ax.scatter(arrays['sim_text_to_target'], arrays['sim_skill_to_target'], 
                  alpha=0.5, s=30, edgecolors='none', color='#2c3e50')
        
        min_val = min(arrays['sim_text_to_target'].min(), arrays['sim_skill_to_target'].min())
        max_val = max(arrays['sim_text_to_target'].max(), arrays['sim_skill_to_target'].max())
        ax.plot([min_val, max_val], [min_val, max_val], '#e74c3c', linestyle='--', label='y=x (equal)', linewidth=2)
        
        ax.set_xlabel('Text $\\rightarrow$ Target Similarity', fontsize=LABEL_FONT, labelpad=12)
        ax.set_ylabel('Skill $\\rightarrow$ Target Similarity', fontsize=LABEL_FONT, labelpad=12)
        ax.set_title(f'Target Proximity Comparison', fontsize=TITLE_FONT, fontweight='bold', pad=20)
        
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f"task3_scatter_{config_name}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Task 5: Box plots
    if task5_results is not None:
        for model_name, model_results in task5_results.items():
            if model_results is None or 'metrics' not in model_results: continue
            
            metrics = model_results['metrics']
            metric_names = list(metrics.keys())
            
            # Adjust figsize based on number of subplots
            fig, axes = plt.subplots(1, len(metric_names), figsize=(7 * len(metric_names), 8))
            if len(metric_names) == 1: axes = [axes]
            
            for ax, metric_name in zip(axes, metric_names):
                if 'arrays' in model_results:
                    correct_mask = model_results['arrays']['correct_mask']
                    incorrect_mask = model_results['arrays']['incorrect_mask']
                    
                    # Logic to fetch values (same as yours)
                    if metric_name == "text_skill_sim": metric_vals = task1_similarities
                    elif metric_name == "text_to_target": metric_vals = task3_results['arrays']['sim_text_to_target']
                    elif metric_name == "skill_to_target": metric_vals = task3_results['arrays']['sim_skill_to_target']
                    else: continue
                    
                    data = [metric_vals[correct_mask], metric_vals[incorrect_mask]]
                    labels = ['Correct', 'Incorrect']
                    
                    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
                    for patch, color in zip(bp['boxes'], ['#27ae60', '#e74c3c']): # Matching your butterfly colors
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                    
                    ax.set_ylabel(metric_name.replace('_', ' ').title(), fontsize=LABEL_FONT)
                    ax.tick_params(axis='both', which='major', labelsize=14)
                    
                    p_val = metrics[metric_name]["test"]["pvalue"]
                    ax.set_title(f'p = {p_val:.2e}', fontsize=16, pad=10)
            
            plt.suptitle(f'Patterns: {model_name}', fontsize=TITLE_FONT, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, f"task5_boxplots_{config_name}_{model_name}.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close()

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Setup logging
    os.makedirs(args.output_dir, exist_ok=True)
    logger.add(os.path.join(args.output_dir, "analysis.log"), rotation="10 MB")
    logger.info("Starting Embedding Overlap Analysis")
    logger.info(f"Arguments: {vars(args)}")

    # 1. Load Data
    logger.info("Loading Data...")
    data = Data(DATA_TYPE=args.data_type, ONLY_TITLES=not args.use_text_description, 
                consider_subspans=not args.no_subspans, LOAD_CLEAN_TEST=args.eval_clean_test)
    
    # We mainly need Test set, but Train/Val are needed for IDF if not using taxonomy file
    # However, load_job_skill_data_by_id usually takes train_val_job_ids set.
    if args.eval_clean_test:
        (train_pairs, train_job_ids), (val_pairs, val_job_ids), (test_pairs, test_job_ids), (test_clean_pairs, test_clean_job_ids) = data.get_data_with_job_ids(stage='transformation_finetuning', include_clean_test=True)
    else:
        (train_pairs, train_job_ids), (val_pairs, val_job_ids), (test_pairs, test_job_ids) = data.get_data_with_job_ids(stage='transformation_finetuning', include_clean_test=False)
        test_clean_pairs = []
        test_clean_job_ids = []

    # Filter repetitive if requested
    #if args.filter_repetitive:
    #    logger.info("Filtering repetitive samples...")
    #    train_pairs, train_job_ids = filter_repetitive_samples_with_job_ids(train_pairs, train_job_ids)
    #    val_pairs, val_job_ids = filter_repetitive_samples_with_job_ids(val_pairs, val_job_ids)
    #    test_pairs, test_job_ids = filter_repetitive_samples_with_job_ids(test_pairs, test_job_ids)
    #    if args.eval_clean_test:
    #         test_clean_pairs, test_clean_job_ids = filter_repetitive_samples_with_job_ids(test_clean_pairs, test_clean_job_ids)

    # Collect train+val job ids for IDF
    train_val_job_ids_set = set()
    for job_ids in train_job_ids + val_job_ids:
        train_val_job_ids_set.update(job_ids)

    # 2. Load Encoders
    logger.info("Loading Encoders...")
    encoder_text = SentenceTransformer(args.encoder_text)
    
    precomputed_skill_embedding_map = None
    if args.skill_embeddings_dir:
        logger.info(f"Loading precomputed skill embeddings from: {args.skill_embeddings_dir}")
        precomputed_skill_embedding_map = load_precomputed_skill_embeddings(args.skill_embeddings_dir)
        encoder_skill = None # Not needed
        encoder_skill_name = "precomputed"
    elif args.encoder_skill:
        logger.info(f"Using separate skill encoder: {args.encoder_skill}")
        encoder_skill = SentenceTransformer(args.encoder_skill)
        encoder_skill_name = args.encoder_skill.split('/')[-1]
    else:
        logger.info("Using same encoder for skills")
        encoder_skill = encoder_text
        encoder_skill_name = args.encoder_text.split('/')[-1]

    # 3. Precompute Target Embeddings (Using dummy labels to establish Y_target_dict for filtering)
    # We need Y_target_dict because precompute_input_embeddings... filters based on it.
    # So we must compute it for all possible targets in our splits.
    all_target_labels = sorted(set([t for _, t in train_pairs + val_pairs + test_pairs + test_clean_pairs]))
    logger.info("Precomputing target embeddings (for filtering purposes)...")
    Y_target_dict, _ = precompute_target_embeddings(
        encoder_text, 
        all_target_labels, 
        cache_dir=args.embeddings_cache_dir,
        encoder_name=args.encoder_text.split('/')[-1],
        force_recompute=args.force_recompute
    )

    # Define Pooling Strategies to Evaluate
    pooling_strategies = ["mean", "weighted_mean", "weighted_idf"]
    log_pooling_options = [False, True]
    
    results = []

    # 4. Iterate and Analyze
    # We will use the TEST set for analysis.
    # Note: We can reuse the same v_C (text history embeddings) for all strategies on the same split.
    
    # Let's standardize on using the "test" split for analysis. 
    # Can also run on "test_clean" if needed, but let's start with standard "test".
    
    # Check if we should analyze clean test as well
    splits_to_analyze = [("test", test_pairs, test_job_ids)]
    if args.eval_clean_test:
        splits_to_analyze.append(("test_clean", test_clean_pairs, test_clean_job_ids))

    # Determine methods to run
    run_v2 = args.method in ["v2", "both"]
    run_v3 = args.method in ["v3", "both"]

    for split_name, current_pairs, current_job_ids in splits_to_analyze:
        logger.info(f"Analyzing split: {split_name} ({len(current_pairs)} samples)")
        
        # Cache v_C to avoid recomputing it for every pooling strategy
        # We'll get it from the first run (either v2 or v3)
        v_C_cached = None

        # --- V2 Analysis ---
        if run_v2:
            for pool_strat in pooling_strategies:
                for use_log_pool in log_pooling_options:
                    config_name = f"v2_{pool_strat}_{'logpool' if use_log_pool else 'flat'}"
                    logger.info(f"  Configuration: {config_name}")
                    logger.info(f"    Pooling: {pool_strat}, LogPool: {use_log_pool}")
                    logger.info(f"    Text Description: {args.use_text_description}, Skill Description: {args.use_skill_description}")

                    # Load Skill Data (re-load because IDF calculation might depend on params)
                    job_skill_map, esco_skill_text_map, _ = load_job_skill_data_by_id(
                        skill_scores_file=args.skill_scores_file,
                        esco_skills_file=args.esco_skills_file,
                        skill_properties_file=None, # Removed per user request
                        pooling_strategy=pool_strat,
                        alpha=args.alpha,
                        beta=args.beta,
                        train_val_job_ids=train_val_job_ids_set,
                        esco_taxonomy_file=args.esco_taxonomy_file
                    )

                    # Skill Confidence Filtering (V2 specific here, or shared?)
                    if args.skill_confidence_threshold is not None:
                         filtered_job_skill_map = {}
                         for jid, skills in job_skill_map.items():
                             filtered_skills = [s for s in skills if s.get('score', 0.0) >= args.skill_confidence_threshold]
                             filtered_job_skill_map[jid] = filtered_skills
                         job_skill_map = filtered_job_skill_map

                    _, _, h_text, h_skill = precompute_input_embeddings_with_job_ids(
                        data_pairs=current_pairs,
                        job_ids_list=current_job_ids,
                        Y_target_dict=Y_target_dict,
                        encoder_text=encoder_text,
                        encoder_skill=encoder_skill,
                        job_skill_map=job_skill_map,
                        esco_skill_text_map=esco_skill_text_map,
                        use_skill_description=args.use_skill_description,
                        pooling_strategy=pool_strat,
                        alpha=args.alpha,
                        beta=args.beta,
                        use_text_history=True, 
                        use_skill_text=True,
                        use_skill_path_log_pooling=use_log_pool,
                        skill_path_alpha_decay=0.5,
                        cache_dir=args.embeddings_cache_dir,
                        encoder_skill_name=encoder_skill_name,
                        force_recompute=args.force_recompute,
                        split_name=f"{split_name}_{config_name}",
                        precomputed_skill_embedding_map=precomputed_skill_embedding_map
                    )
                    
                    if h_text is None or h_skill is None:
                        logger.warning(f"Got None embeddings for {config_name}. Skipping.")
                        continue

                    if v_C_cached is None:
                        v_C_cached = h_text
                    
                    similarities, stats = compute_cosine_stats(v_C_cached, h_skill)
                    
                    if stats is None:
                        continue
                    
                    row = {
                        "split": split_name,
                        "method": "v2",
                        "config": config_name, 
                        "pooling_strategy": pool_strat,
                        "log_pooling": use_log_pool,
                        **stats
                    }
                    results.append(row)
                    logger.info(f"  Stats: Mean={stats['mean']:.4f}, Std={stats['std']:.4f}")

        # --- V3 Analysis ---
        if run_v3:
            config_name = f"v3_top{args.top_k_skills}_{args.scoring_mode}"
            if args.scoring_mode == "weighted":
                 config_name += f"_w{args.importance_weight}"
            logger.info(f"  Configuration: {config_name}")
            logger.info(f"    Top K: {args.top_k_skills}, Strategy: {args.skill_selection_strategy}")
            logger.info(f"    Scoring: {args.scoring_mode}, Weight: {args.importance_weight}")
            logger.info(f"    Text Description: {args.use_text_description}, Skill Description: {args.use_skill_description}")

            # Load Skill Data for V3
            # V3 uses load_skill_mappings + load_skills_by_job_id (but we need to figure out exactly what)
            # Actually train_cpp_enhanced_v3 calls load_job_and_skill_data which calls load_job_skill_data_by_id!
            # But the key is `load_skill_mappings` and `job_skill_map` preparation might be different?
            # V3 uses `cap_skills_per_job_lexicographic` etc on TOP of `job_skill_map`.
            # BUT `build_last_job_skill_embeddings` takes `job_skill_map` directly.
            # AND `train_cpp_enhanced_v3` calls `load_job_and_skill_data` which calls `load_job_skill_data_by_id` with `pooling_strategy="weighted_idf"`.
            # So we should do the same: Load with weighted_idf to get IDF scores.
            
            # Using skill_scores_file argument or checking for separate json?
            # User wants to use `skill_scores_file` or `skill_scores_json`? 
            # Ideally use `skill_scores_file` if suitable.
            
            # Re-load data with 'weighted_idf' to ensure IDF scores are present
            job_skill_map_v3, _, _ = load_job_skill_data_by_id(
                skill_scores_file=args.skill_scores_file, 
                esco_skills_file=args.esco_skills_file,
                skill_properties_file=None, # Removed
                pooling_strategy="weighted_idf", 
                alpha=1.0, 
                beta=1.0,
                train_val_job_ids=train_val_job_ids_set,
                esco_taxonomy_file=args.esco_taxonomy_file
            )

            # We also need skill_desc_map for descriptions (V3 helper)
            # User requested using esco_skills_file (CSV) instead of json properties file
            skill_desc_map = load_skill_descriptions(args.esco_skills_file)

            # NOTE: We are NOT implementing the full V3 "cap_skills" logic here because
            # `build_last_job_skill_embeddings` simply TAKES the job_skill_map.
            # In `train_cpp_enhanced_v3`, it applies `cap_skills_...` BEFORE calling `build...`.
            # We strictly need to replicate that capping/selection if we want true "V3".
            # The User asked for "skill_selection_strategy", "scoring_mode" arguments which implies we MUST implement selection.
            
            # Import Capping functions - they are in src.cpp.skill_pooling
            from src.cpp.skill_pooling import (
                cap_skills_per_job_lexicographic, 
                cap_skills_per_job_stratified,
                cap_skills_per_job_by_score,
                calculate_idf_scores_by_job_id
            )
            
            # Recalculate/Enrich with Weighted IDF if needed
            if args.scoring_mode == "weighted":
                logger.info(f"  Calculating weighted IDF (weight={args.importance_weight})...")
                job_skill_map_v3 = calculate_idf_scores_by_job_id(
                    job_skill_map_v3, 
                    use_job_scores=True, 
                    importance_weight=args.importance_weight
                )

            # Apply Selection/Capping
            logger.info("  Applying V3 skill selection/capping...")
            
            if args.skill_selection_strategy == "top_k":
                 if args.scoring_mode == "scores_only":
                      # Use pure score-based capping
                      job_skill_map_v3 = cap_skills_per_job_by_score(
                          job_skill_map_v3,
                          max_skills_per_job=args.top_k_skills,
                          skill_desc_map=skill_desc_map
                      )
                 else:
                      # Use lexicographic (IDF or Weighted IDF)
                      job_skill_map_v3 = cap_skills_per_job_lexicographic(
                          job_skill_map_v3,
                          max_skills_per_job=args.top_k_skills,
                          skill_desc_map=skill_desc_map,
                          use_weighted_idf=(args.scoring_mode == "weighted")
                      )
                      
            elif args.skill_selection_strategy == "stratified":
                 # Map scoring_mode to score_source
                 score_source_map = {
                     "idf_only": "idf",
                     "weighted": "weighted_idf",
                     "scores_only": "score"
                 }
                 score_source = score_source_map.get(args.scoring_mode, "auto")
                 
                 job_skill_map_v3 = cap_skills_per_job_stratified(
                     job_skill_map_v3,
                     max_skills_per_job=args.top_k_skills,
                     use_weighted_idf=(args.scoring_mode == "weighted"),
                     score_source=score_source
                 )

            # Build Embeddings
            # We need device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            h_skill_v3 = build_last_job_skill_embeddings(
                data_pairs=current_pairs,
                job_ids_list=current_job_ids,
                job_skill_map=job_skill_map_v3,
                skill_desc_map=skill_desc_map,
                encoder_skill=encoder_skill,
                include_skill_descriptions=args.use_skill_description, # Assuming shared arg
                device=device,
                cache_dir=args.embeddings_cache_dir,
                encoder_name=encoder_skill_name,
                force_recompute=args.force_recompute
            )

            # Get v_C (Text History) if not already cached
            # We can use precompute_input_embeddings... just for text history if needed
            if v_C_cached is None:
                 # Minimal call to get just text history
                 _, _, v_C_cached, _ = precompute_input_embeddings_with_job_ids(
                    data_pairs=current_pairs,
                    job_ids_list=current_job_ids,
                    Y_target_dict=Y_target_dict,
                    encoder_text=encoder_text,
                    encoder_skill=encoder_skill, # Dummy if not needed
                    job_skill_map={}, # Empty, not needed for text history
                    esco_skill_text_map={},
                    pooling_strategy="mean", # Dummy
                    use_text_history=True,
                    use_skill_text=False, # Disable skill text here
                    cache_dir=args.embeddings_cache_dir,
                    encoder_skill_name=None, # Avoid skill cache
                    force_recompute=False, # Use existing cache for text
                    split_name=f"{split_name}_v3_base" # Suffix for text cache matching? Actually text cache hash based on content.
                )

            # Compute stats
            similarities, stats = compute_cosine_stats(v_C_cached, h_skill_v3)
            
            if stats is not None:
                row = {
                    "split": split_name,
                    "method": "v3",
                    "config": config_name,
                    "pooling_strategy": "concatenation",
                    **stats
                }
                results.append(row)
                logger.info(f"  Stats: Mean={stats['mean']:.4f}, Std={stats['std']:.4f}")


    # Load predictions for Task 5 if requested
    predictions_dict = {}
    if args.run_task5:
        logger.info("\nLoading predictions for Task 5...")
        if args.predictions_text:
            predictions_dict['text'] = load_predictions(args.predictions_text)
        if args.predictions_skill:
            predictions_dict['skill'] = load_predictions(args.predictions_skill)
        if args.predictions_hybrid:
            predictions_dict['hybrid'] = load_predictions(args.predictions_hybrid)
        
        if not predictions_dict:
            logger.warning("  ⚠️ No valid predictions loaded. Task 5 will be skipped.")
            args.run_task5 = False
    
    # Run Tasks 3 and 5 for each split/config combination
    task3_results_all = []
    task5_results_all = []
    
    if args.run_task3 or args.run_task5:
        logger.info("\n" + "="*80)
        logger.info("Running Extended Analysis (Tasks 3 & 5)")
        logger.info("="*80)
        
        for split_name, current_pairs, current_job_ids in splits_to_analyze:
            logger.info(f"\nAnalyzing split: {split_name}")
            
            # Extract target labels
            target_labels_list = [target for _, target in current_pairs]
            
            # We need to recompute embeddings for this split
            # For simplicity, let's use v3 method if available, otherwise v2 with weighted_idf
            if run_v3:
                config_name = f"v3_top{args.top_k_skills}_{args.scoring_mode}"
                if args.scoring_mode == "weighted":
                    config_name += f"_w{args.importance_weight}"
                
                # Recompute embeddings (reuse cached if available)
                job_skill_map_v3, _, _ = load_job_skill_data_by_id(
                    skill_scores_file=args.skill_scores_file,
                    esco_skills_file=args.esco_skills_file,
                    skill_properties_file=None,
                    pooling_strategy="weighted_idf",
                    alpha=1.0,
                    beta=1.0,
                    train_val_job_ids=train_val_job_ids_set,
                    esco_taxonomy_file=args.esco_taxonomy_file
                )
                
                skill_desc_map = load_skill_descriptions(args.esco_skills_file)
                
                from src.cpp.skill_pooling import (
                    cap_skills_per_job_lexicographic,
                    cap_skills_per_job_stratified,
                    cap_skills_per_job_by_score,
                    calculate_idf_scores_by_job_id
                )
                
                if args.scoring_mode == "weighted":
                    job_skill_map_v3 = calculate_idf_scores_by_job_id(
                        job_skill_map_v3,
                        use_job_scores=True,
                        importance_weight=args.importance_weight
                    )
                
                if args.skill_selection_strategy == "top_k":
                    if args.scoring_mode == "scores_only":
                        job_skill_map_v3 = cap_skills_per_job_by_score(
                            job_skill_map_v3,
                            max_skills_per_job=args.top_k_skills,
                            skill_desc_map=skill_desc_map
                        )
                    else:
                        job_skill_map_v3 = cap_skills_per_job_lexicographic(
                            job_skill_map_v3,
                            max_skills_per_job=args.top_k_skills,
                            skill_desc_map=skill_desc_map,
                            use_weighted_idf=(args.scoring_mode == "weighted")
                        )
                elif args.skill_selection_strategy == "stratified":
                    score_source_map = {
                        "idf_only": "idf",
                        "weighted": "weighted_idf",
                        "scores_only": "score"
                    }
                    score_source = score_source_map.get(args.scoring_mode, "auto")
                    job_skill_map_v3 = cap_skills_per_job_stratified(
                        job_skill_map_v3,
                        max_skills_per_job=args.top_k_skills,
                        use_weighted_idf=(args.scoring_mode == "weighted"),
                        score_source=score_source
                    )
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                h_skill_v3 = build_last_job_skill_embeddings(
                    data_pairs=current_pairs,
                    job_ids_list=current_job_ids,
                    job_skill_map=job_skill_map_v3,
                    skill_desc_map=skill_desc_map,
                    encoder_skill=encoder_skill,
                    include_skill_descriptions=args.use_skill_description,
                    device=device,
                    cache_dir=args.embeddings_cache_dir,
                    encoder_name=encoder_skill_name,
                    force_recompute=args.force_recompute
                )
                
                _, _, v_C_text, _ = precompute_input_embeddings_with_job_ids(
                    data_pairs=current_pairs,
                    job_ids_list=current_job_ids,
                    Y_target_dict=Y_target_dict,
                    encoder_text=encoder_text,
                    encoder_skill=encoder_skill,
                    job_skill_map={},
                    esco_skill_text_map={},
                    pooling_strategy="mean",
                    use_text_history=True,
                    use_skill_text=False,
                    cache_dir=args.embeddings_cache_dir,
                    encoder_skill_name=None,
                    force_recompute=False,
                    split_name=f"{split_name}_v3_base"
                )
                
                h_text = v_C_text
                h_skill = h_skill_v3
                
            else:
                # Use v2 with weighted_idf
                config_name = "v2_weighted_idf_flat"
                job_skill_map, esco_skill_text_map, _ = load_job_skill_data_by_id(
                    skill_scores_file=args.skill_scores_file,
                    esco_skills_file=args.esco_skills_file,
                    skill_properties_file=None,
                    pooling_strategy="weighted_idf",
                    alpha=args.alpha,
                    beta=args.beta,
                    train_val_job_ids=train_val_job_ids_set,
                    esco_taxonomy_file=args.esco_taxonomy_file
                )
                
                _, _, h_text, h_skill = precompute_input_embeddings_with_job_ids(
                    data_pairs=current_pairs,
                    job_ids_list=current_job_ids,
                    Y_target_dict=Y_target_dict,
                    encoder_text=encoder_text,
                    encoder_skill=encoder_skill,
                    job_skill_map=job_skill_map,
                    esco_skill_text_map=esco_skill_text_map,
                    use_skill_description=args.use_skill_description,
                    pooling_strategy="weighted_idf",
                    alpha=args.alpha,
                    beta=args.beta,
                    use_text_history=True,
                    use_skill_text=True,
                    use_skill_path_log_pooling=False,
                    skill_path_alpha_decay=0.5,
                    cache_dir=args.embeddings_cache_dir,
                    encoder_skill_name=encoder_skill_name,
                    force_recompute=args.force_recompute,
                    split_name=f"{split_name}_extended_analysis",
                    precomputed_skill_embedding_map=precomputed_skill_embedding_map
                )
            
            # Compute Task 1 similarities for use in Task 5
            task1_similarities, _ = compute_cosine_stats(h_text, h_skill)
            
            # Task 3: Target Proximity Analysis
            task3_result = None
            if args.run_task3:
                task3_result = compute_target_proximity_stats(
                    h_text, h_skill, None, target_labels_list, Y_target_dict
                )
                task3_results_all.append({
                    "split": split_name,
                    "config": config_name,
                    **task3_result
                })
            
            # Task 5: Correctness Pattern Analysis
            task5_result_dict = {}
            if args.run_task5:
                for model_name, pred_data in predictions_dict.items():
                    if pred_data is not None:
                        correctness = compute_top1_correctness(pred_data, target_labels_list, Y_target_dict, label=model_name)
                        if correctness is not None:
                            task5_result = analyze_correctness_patterns(
                                h_text, h_skill, None, target_labels_list,
                                Y_target_dict, correctness, task1_similarities,
                                model_name=model_name
                            )
                            if task5_result is not None:
                                task5_result_dict[model_name] = task5_result
                                task5_results_all.append({
                                    "split": split_name,
                                    "config": config_name,
                                    **task5_result
                                })
            
            # Generate visualizations
            if args.run_task3 or args.run_task5:
                generate_visualizations(
                    split_name, config_name,
                    task1_similarities if (args.run_task3 or args.run_task5) else None,
                    task3_result,
                    task5_result_dict if task5_result_dict else None,
                    args.output_dir
                )


    # Save Results
    df_results = pd.DataFrame(results)
    output_csv = os.path.join(args.output_dir, "overlap_stats.csv")
    df_results.to_csv(output_csv, index=False)
    logger.info(f"Saved results to {output_csv}")
    
    # Print Summary Table
    print("\n=== Embedding Overlap Analysis Results ===")
    print(df_results.to_markdown(index=False, floatfmt=".4f"))

if __name__ == "__main__":
    main()

    # Save Task 3 results if available
    if 'task3_results_all' in locals() and task3_results_all:
        task3_output = os.path.join(args.output_dir, "task3_target_proximity.json")
        with open(task3_output, 'w') as f:
            task3_to_save = []
            for result in task3_results_all:
                result_copy = result.copy()
                if 'arrays' in result_copy:
                    del result_copy['arrays']
                task3_to_save.append(result_copy)
            json.dump(task3_to_save, f, indent=2)
        logger.info(f"Saved Task 3 results to {task3_output}")
    
    # Save Task 5 results if available
    if 'task5_results_all' in locals() and task5_results_all:
        task5_output = os.path.join(args.output_dir, "task5_correctness_patterns.json")
        with open(task5_output, 'w') as f:
            task5_to_save = []
            for result in task5_results_all:
                result_copy = result.copy()
                if 'arrays' in result_copy:
                    del result_copy['arrays']
                task5_to_save.append(result_copy)
            json.dump(task5_to_save, f, indent=2)
        logger.info(f"Saved Task 5 results to {task5_output}")
    
    # Enhanced reporting for Task 3
    if 'task3_results_all' in locals() and task3_results_all:
        print("\n" + "="*80)
        print("=== Task 3: Target Proximity ===")
        print("="*80)
        for result in task3_results_all:
            print(f"\nSplit: {result['split']}, Config: {result['config']}")
            print(f"\n| Metric | Text→Target | Skill→Target | Delta |")
            print(f"|--------|-------------|--------------|-------|")
            print(f"| Mean   | {result['stats_text']['mean']:.4f} | {result['stats_skill']['mean']:.4f} | {result['stats_delta']['mean']:.4f} |")
            print(f"| Std    | {result['stats_text']['std']:.4f} | {result['stats_skill']['std']:.4f} | {result['stats_delta']['std']:.4f} |")
            print(f"| Median | {result['stats_text']['median']:.4f} | {result['stats_skill']['median']:.4f} | {result['stats_delta']['median']:.4f} |")
            print(f"\nWilcoxon signed-rank test: statistic={result['statistical_test']['statistic']:.2f}, p={result['statistical_test']['pvalue']:.4e}")
            if result['stats_delta']['mean'] > 0:
                print("Interpretation: Text embeddings are significantly closer to targets.")
            else:
                print("Interpretation: Skill embeddings are significantly closer to targets.")
    
    # Enhanced reporting for Task 5
    if 'task5_results_all' in locals() and task5_results_all:
        print("\n" + "="*80)
        print("=== Task 5: Correctness Patterns ===")
        print("="*80)
        for result in task5_results_all:
            print(f"\nSplit: {result['split']}, Config: {result['config']}, Model: {result['model_name']}")
            print(f"\n| Metric | Correct (n={result['n_correct']}) | Incorrect (n={result['n_incorrect']}) | p-value |")
            print(f"|--------|----------------|------------------|---------|")
            for metric_name, metric_data in result['metrics'].items():
                print(f"| {metric_name} | {metric_data['correct']['mean']:.4f} | {metric_data['incorrect']['mean']:.4f} | {metric_data['test']['pvalue']:.4e} |")
