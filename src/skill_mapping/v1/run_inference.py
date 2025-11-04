# run_inference.py

import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

import faiss
import numpy as np
import pandas as pd
import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Import our custom components
from skill_mapping.v1.data import HIER_COL_MAP, build_job_text, build_skill_text, build_skill_lookups
from skill_mapping.v1.indexing import load_category_model, merge_hybrid_scores
from skill_mapping.v1.utils import normalize_embeddings, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description=f'Run 2-step inference for Hierarchical Skill Prediction')
    # --- File Paths ---
    parser.add_argument("--job_data_path", type=str, required=True, 
                        help="Path to the job dataset (e.g., decorte_test.csv)")
    parser.add_argument("--esco_data_path", type=str, 
                        default="data/processed/master_datasets/master_complete_hierarchy_w_occ.csv")
    parser.add_argument("--cat_model_ckpt", type=str, required=True, 
                        help="Path to your trained CategoryPredictor .pt file")
    parser.add_argument("--out_dir", type=str, default="results/")

    # --- Model Checkpoints ---
    parser.add_argument("--base_encoder_ckpt", type=str, default="all-MiniLM-L6-v2",
                        help="Base encoder for the CategoryPredictor")
    parser.add_argument("--skill_encoder_ckpt", type=str, default="BAAI/bge-large-en-v1.5",
                        help="SOTA Skill Encoder (e.g., from pjmathematician or BGE)")
    
    # --- Model & Pipeline Config ---
    parser.add_argument("--hier_level", type=int, default=1, 
                        help="Hierarchy level (0-3) to use for re-ranking.")
    parser.add_argument("--hidden_dim", type=int, default=None, 
                        help="Hidden dim of the base_encoder. If None, inferred.")  
    parser.add_argument("--text_fields", type=str, default="title+desc")
    parser.add_argument("--is_structured", action="store_true")

    # --- Inference Config ---
    parser.add_argument("--top_k", type=int, default=50, 
                        help="Number of skills to retrieve from FAISS before re-ranking")
    parser.add_argument("--cat_threshold", type=float, default=0.1,
                        help="Minimum category probability to consider for re-ranking")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str)

    return parser.parse_args()

# -----------------------------
# 1. EVALUATION METRICS
# -----------------------------

def average_precision_at_k(
    predicted_labels: List[str], 
    actual_labels: Set[str], 
    k: int
) -> float:
    """Calculates Average Precision (AP) @ k."""
    if not actual_labels:
        return 0.0

    hits = 0
    precision_sum = 0.0
    
    for i in range(min(len(predicted_labels), k)):
        if predicted_labels[i] in actual_labels:
            hits += 1
            precision_at_i = hits / (i + 1)
            precision_sum += precision_at_i
            
    return precision_sum / min(len(actual_labels), k)

def evaluate_predictions(
    all_predictions: Dict[str, List[Tuple[str, float]]],
    ground_truth_map: Dict[str, Set[str]],
    k_list: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Computes Precision@k, Recall@k, and MAP@k for all queries.
    """
    metrics = {f"P@{k}": [] for k in k_list}
    metrics.update({f"R@{k}": [] for k in k_list})
    metrics.update({f"MAP@{k}": [] for k in k_list})

    job_texts = list(all_predictions.keys())
    
    for job_text in job_texts:
        actual_labels = ground_truth_map.get(job_text)
        if not actual_labels or len(actual_labels) == 0:
            continue # Skip if no ground truth skills

        pred_tuples = all_predictions.get(job_text, [])
        predicted_labels = [label for label, score in pred_tuples]
        
        total_relevant = len(actual_labels)

        for k in k_list:
            top_k_preds = predicted_labels[:k]
            hits = len(set(top_k_preds) & actual_labels)
            
            # Precision@k
            p_at_k = hits / k
            metrics[f"P@{k}"].append(p_at_k)
            
            # Recall@k
            r_at_k = hits / total_relevant
            metrics[f"R@{k}"].append(r_at_k)
            
            # MAP@k
            ap_at_k = average_precision_at_k(predicted_labels, actual_labels, k)
            metrics[f"MAP@{k}"].append(ap_at_k)

    # Calculate final mean metrics
    final_metrics = {key: np.mean(val) for key, val in metrics.items()}
    final_metrics["num_queries_evaluated"] = len(metrics["P@5"])
    
    return final_metrics

# -----------------------------
# 2. DATA LOADING
# -----------------------------

def load_ground_truth(
    job_df: pd.DataFrame, 
    esco_df: pd.DataFrame, 
    args: argparse.Namespace
) -> Tuple[Dict[str, Set[str]], List[str]]:
    """
    Loads the ground truth by merging the job_df with the esco_df.
    Returns a map of {job_text: set_of_true_skill_labels}
    """
    logger.info(f"Building ground truth from {len(job_df)} jobs...")
    
    # Reset index to ensure we can track original job indices
    job_df_with_idx = job_df.reset_index().rename(columns={'index': 'original_job_idx'})
    
    # Use the column names you specified
    merged_df = job_df_with_idx.merge(
        esco_df,
        left_on="esco_id",
        right_on="occupationUri",
        how="left",
        suffixes=("_job", "_skill")
    )
    
    ground_truth_map = {}
    job_texts_to_run = []
    
    # Group by the original job index to get one sample per job
    for job_idx, skills_for_job in merged_df.groupby('original_job_idx'):
        if skills_for_job.empty:
            continue
        
        first_row = skills_for_job.iloc[0]
        
        # Build the job text exactly as it will be fed to the models
        job_text = build_job_text(
            first_row, 
            text_fields=args.text_fields, 
            is_structured=args.is_structured
        )
        
        # Get unique skill labels from all *skills* linked to this job
        true_skills = set(
            skills_for_job["skillLabel"].dropna().astype(str).unique()
        )
        
        ground_truth_map[job_text] = true_skills
        job_texts_to_run.append(job_text)

    logger.success(f"Built ground truth for {len(ground_truth_map)} jobs.")
    return ground_truth_map, job_texts_to_run

# -----------------------------
# 3. FAISS & PIPELINE SETUP
# -----------------------------

def build_faiss_index(
    skill_encoder: SentenceTransformer,
    all_skill_labels: List[str],
    all_skill_texts: List[str],
    device: str
) -> Tuple[faiss.Index, Dict[int, str]]:
    """
    Encodes all skill texts and builds a FAISS index.
    Returns the index and a map from {faiss_id -> skill_label}.
    """
    logger.info(f"Building FAISS index for {len(all_skill_labels)} skills...")
    
    skill_emb = skill_encoder.encode(
        all_skill_texts,
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=32,
        device=device
    ).cpu().numpy()
    
    skill_emb = normalize_embeddings(skill_emb).astype(np.float32)
    hidden_dim = skill_emb.shape[1]

    index = faiss.IndexFlatIP(hidden_dim)
    index.add(skill_emb)
    
    # Map from FAISS index ID (0, 1, 2...) to the skill's *label*
    id_to_label_map = {i: label for i, label in enumerate(all_skill_labels)}
    
    logger.success(f"FAISS index built with {index.ntotal} skills.")
    return index, id_to_label_map

# -----------------------------
# 4. MAIN INFERENCE SCRIPT
# -----------------------------

def main(args):
    set_seed(args.seed)
    device = torch.device(args.device)
    
    if args.run_name:
        run_name = args.run_name
    else:
        run_name = (
            f"results_{Path(args.job_data_path).stem}_"
            f"{args.skill_encoder_ckpt.split('/')[-1]}_"
            f"lvl{args.hier_level}"
        )
        
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    logger.add(Path(args.out_dir) / f"{run_name}.log")
    logger.info(f"Starting run: {run_name}")
    logger.info(f"Arguments: {vars(args)}")

    # --- 1. Load ESCO Data & Lookups ---
    logger.info(f"Loading ESCO master data from {args.esco_data_path}")
    esco_df = pd.read_csv(args.esco_data_path)
    
    # Get the category vocabulary for the level we trained on
    cat_col = HIER_COL_MAP[args.hier_level]
    all_cat_labels = esco_df[cat_col].dropna().astype(str).unique().tolist()
    cat_idx2label = {i: label for i, label in enumerate(all_cat_labels)}
    
    # Get all skill texts, labels, and the skill-to-category map
    all_skill_labels, all_skill_texts, skill_to_cat_map = build_skill_lookups(
        esco_df=esco_df,
        hier_level=args.hier_level,
        text_fields=args.text_fields,
        is_structured=args.is_structured
    )

    # --- 2. Load Ground Truth Job Data ---
    job_df = pd.read_csv(args.job_data_path)
    ground_truth_map, job_texts_to_run = load_ground_truth(job_df, esco_df, args)

    # --- 3. Load Step 1: Category Model ---
    base_encoder = SentenceTransformer(args.base_encoder_ckpt, device=device)

    if args.hidden_dim is None:
        logger.info(f"hidden_dim not set. Inferring from {args.base_encoder_ckpt}...")
        args.hidden_dim = base_encoder.get_sentence_embedding_dimension()
        logger.info(f"Inferred hidden_dim: {args.hidden_dim}")

    category_model = load_category_model(
        ckpt_path=Path(args.cat_model_ckpt),
        base_encoder=base_encoder,
        hidden_dim=args.hidden_dim,
        categories_idx2str=cat_idx2label,
        device=device
    )

    # --- 4. Load Step 2: Skill SOTA Model & Build FAISS Index ---
    skill_encoder = SentenceTransformer(args.skill_encoder_ckpt, device=device)
    faiss_index, faiss_id_to_label_map = build_faiss_index(
        skill_encoder=skill_encoder,
        all_skill_labels=all_skill_labels,
        all_skill_texts=all_skill_texts,
        device=device
    )

    # --- 5. Run Hybrid Inference Pipeline ---
    logger.info(f"Running hybrid inference on {len(job_texts_to_run)} job texts...")
    all_final_predictions = {}
    
    # Process in batches for efficiency
    batch_size = args.batch_size
    for i in tqdm(range(0, len(job_texts_to_run), batch_size), desc="Hybrid Inference"):
        batch_job_texts = job_texts_to_run[i : i + batch_size]
        
        # Step 1: Get category probabilities
        cat_probs_batch = predict_categories(category_model, batch_job_texts, cat_idx2label)
        
        # Step 2: Get global skill similarities
        retrieved_skills_batch = retrieve_skills(
            batch_job_texts, skill_encoder, faiss_index, faiss_id_to_label_map, args.top_k
        )
        
        # Step 3: Merge them
        for j, job_text in enumerate(batch_job_texts):
            ranked_skills = merge_hybrid_scores(
                category_prob_map=cat_probs_batch[j],
                retrieved_skills=retrieved_skills_batch[j],
                skill_to_cat_map=skill_to_cat_map,
                category_threshold=args.cat_threshold
            )
            all_final_predictions[job_text] = ranked_skills

    # --- 6. Evaluate and Save Results ---
    logger.info("Evaluating predictions...")
    k_list_eval = [5, 10, 20, 50]
    final_metrics = evaluate_predictions(
        all_predictions=all_final_predictions,
        ground_truth_map=ground_truth_map,
        k_list=k_list_eval
    )
    
    logger.success("--- EVALUATION RESULTS ---")
    print(json.dumps(final_metrics, indent=2))
    
    # Save predictions and metrics
    out_path_preds = Path(args.out_dir) / f"{run_name}_predictions.json"
    out_path_metrics = Path(args.out_dir) / f"{run_name}_metrics.json"
    
    # Convert predictions to a serializable format
    serializable_preds = {
        job: [{"skill": s, "score": r} for s, r in skills] 
        for job, skills in all_final_predictions.items()
    }
    
    with open(out_path_preds, 'w') as f:
        json.dump(serializable_preds, f, indent=2)
    with open(out_path_metrics, 'w') as f:
        json.dump(final_metrics, f, indent=2)
        
    logger.success(f"Saved predictions to {out_path_preds}")
    logger.success(f"Saved metrics to {out_path_metrics}")


if __name__ == "__main__":

    args = parse_args()
    
    # --- Quick Check ---
    if "bge" in args.skill_encoder_ckpt.lower() and args.text_fields == "title+desc":
        logger.warning("BGE models were trained with structured queries. "
                       "Consider adding --is_structured for the skill encoder.")
    if "bge-large" in args.base_encoder_ckpt.lower() and args.hidden_dim != 1024:
        logger.warning(f"Base encoder {args.base_encoder_ckpt} likely has hidden_dim=1024, not {args.hidden_dim}")

    main(args)