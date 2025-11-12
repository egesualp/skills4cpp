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

import time  # NEW: For timing
import datetime  # NEW: For timestamp
import csv  # NEW: For CSV logging
import os  # NEW: For checking file existence

# Import our custom components
from skill_mapping.v1.data import HIER_COL_MAP, build_job_text, build_skill_text, build_skill_lookups
from skill_mapping.v1.model import build_encoder
from skill_mapping.v1.indexing import load_category_model, merge_hybrid_scores, predict_categories, retrieve_skills
from skill_mapping.v1.utils import normalize_embeddings, set_seed

def parse_args():
    parser = argparse.ArgumentParser(description=f'Run 2-step inference for Hierarchical Skill Prediction')
    # --- File Paths ---
    parser.add_argument("--job_data_path", type=str, required=True, 
                        help="Path to the job dataset (e.g., decorte_test.csv)")
    parser.add_argument("--esco_data_path", type=str, 
                        default="data/processed/master_datasets_2/master_complete_hierarchy_w_occ.csv")
    parser.add_argument("--cat_model_ckpt", type=str, 
                        help="Path to your trained CategoryPredictor .pt file")
    parser.add_argument("--out_dir", type=str, default="results/")

    # --- Model Checkpoints ---
    parser.add_argument("--base_encoder_ckpt", type=str, default="all-MiniLM-L6-v2",
                        help="Base encoder for the CategoryPredictor")
    parser.add_argument("--skill_encoder_ckpt", type=str, default="BAAI/bge-large-en-v1.5",
                        help="SOTA Skill Encoder (e.g., from pjmathematician or BGE)")
    
    # --- Model & Pipeline Config ---
    parser.add_argument("--hier_level", type=int, default=None, 
                        help="Hierarchy level (0-3) to use for re-ranking.")
    parser.add_argument("--hidden_dim", type=int, default=None, 
                        help="Hidden dim of the base_encoder. If None, inferred.")  
    parser.add_argument("--text_fields", type=str, default="title+desc")
    parser.add_argument("--use_expanded_corpus", action="store_true",
                        help="Expands alternative labels as new rows, then mapped back to canonical label during indexing.")
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
    alt_to_canonical_map: Dict[str, str], # <-- NEW ARGUMENT
    k_list: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Computes Precision@k, Recall@k, and MAP@k for all queries.
    """
    metrics = {f"P@{k}": [] for k in k_list}
    metrics.update({f"R@{k}": [] for k in k_list})
    metrics.update({f"MAP@{k}": [] for k in k_list})

    job_texts = list(all_predictions.keys())

    num_queries_evaluated = 0
    
    for job_text in job_texts:
        actual_labels = ground_truth_map.get(job_text)
        if not actual_labels or len(actual_labels) == 0: continue
        
        num_queries_evaluated += 1
        predicted_scores = all_predictions.get(job_text, [])
        total_relevant = len(actual_labels)

        # --- This is the new normalization logic ---
        normalized_predicted_labels = []
        seen_canonical_labels = set()
        for alt_label, score in predicted_scores:
            # Map the retrieved label (which could be an alt) back to its canonical form
            canonical_label = alt_to_canonical_map.get(alt_label, alt_label)
            
            # Add to list only if we haven't seen this canonical skill before
            if canonical_label not in seen_canonical_labels:
                normalized_predicted_labels.append(canonical_label)
                seen_canonical_labels.add(canonical_label)
        # --- End of new logic ---

        for k in k_list:
            # Use the new normalized list for evaluation
            top_k_preds = normalized_predicted_labels[:k]
            
            hits = len(set(top_k_preds) & actual_labels)
            metrics[f"P@{k}"].append(hits / k)
            metrics[f"R@{k}"].append(hits / total_relevant)
            
            # Pass the full normalized list to MAP@k
            metrics[f"MAP@{k}"].append(
                average_precision_at_k(normalized_predicted_labels, actual_labels, k)
            )

    final_metrics = {key: np.mean(val) if val else 0.0 for key, val in metrics.items()}
    final_metrics["num_queries_evaluated"] = num_queries_evaluated
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
# 4. NEW: CSV LOGGING
# -----------------------------

def log_results_to_csv(csv_path: Path, run_name: str, duration: float, process: str,
                       args: argparse.Namespace, metrics: Dict[str, float]):
    """Appends a single row of results to a master CSV file."""
    try:
        # 1. Create the flat dictionary for the row
        log_row = {}
        log_row['run_name'] = run_name
        log_row['timestamp'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        log_row['duration_seconds'] = round(duration, 2)
        log_row['process'] = process
        
        # Add all args
        log_row.update(vars(args))
        
        # Add all metrics
        log_row.update(metrics)
        
        # Sanitize values (e.g., convert Path objects to str, None to empty str)
        for key, val in log_row.items():
            if isinstance(val, Path):
                log_row[key] = str(val)
            elif val is None:
                log_row[key] = ""
        
        # 2. Get all fieldnames (robustly)
        current_fieldnames = sorted(log_row.keys())
        existing_fieldnames = []
        
        if csv_path.is_file():
            with open(csv_path, 'r', newline='', encoding='utf-8') as f_read:
                reader = csv.reader(f_read)
                try:
                    existing_fieldnames = next(reader)
                except StopIteration:
                    pass # File is empty

        # Merge headers, adding new ones to the end
        final_fieldnames = list(existing_fieldnames)
        for f in current_fieldnames:
            if f not in final_fieldnames:
                final_fieldnames.append(f)
                
        # 3. Write/Append to the file
        with open(csv_path, 'a', newline='', encoding='utf-8') as f_write:
            # Use restval='' to fill in any missing columns from previous runs
            writer = csv.DictWriter(f_write, fieldnames=final_fieldnames, restval='')
            
            if not existing_fieldnames:
                writer.writeheader() # Write header only if file was new/empty
                
            writer.writerow(log_row)
            
        logger.success(f"Results successfully appended to {csv_path}")
        
    except Exception as e:
        logger.warning(f"Failed to append results to master CSV log: {e}")

# -----------------------------
# 5. MAIN INFERENCE SCRIPT
# -----------------------------

def main(args):
    start_time = time.time()  # NEW: Start timer
    set_seed(args.seed)
    device = torch.device(args.device)
    
    # --- NEW NAMING LOGIC ---
    run_hybrid_mode = args.cat_model_ckpt is not None
    process = "HYBRID" if run_hybrid_mode else "BASELINE"
    corpus_flag = "expanded" if args.use_expanded_corpus else "combined"
    test_data_name = Path(args.job_data_path).stem
    skill_encoder_name = args.skill_encoder_ckpt.split('/')[-1]

    if args.run_name:
        run_name = args.run_name
    else:
        if run_hybrid_mode:
            # Get the category model's name from its checkpoint file
            # e.g., "cat_probe_MiniLM_L1"
            cat_model_name = Path(args.cat_model_ckpt).stem 
            run_name = (
                f"results_{process}_{test_data_name}_"
                f"{cat_model_name}_reranks_{skill_encoder_name}_"
                f"{corpus_flag}"
            )
        else:
            # Baseline name is simple
            run_name = (
                f"results_{process}_{test_data_name}_"
                f"{skill_encoder_name}_{corpus_flag}"
            )
    # --- END NEW NAMING LOGIC ---
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    logger.add(Path(args.out_dir) / f"{run_name}.log")
    logger.info(f"Starting run: {run_name}")
    logger.info(f"Arguments: {vars(args)}")

    # --- 1. Load ESCO Data & Lookups ---
    logger.info(f"Loading ESCO master data from {args.esco_data_path}")
    esco_df = pd.read_csv(args.esco_data_path)

    if not run_hybrid_mode:
        esco_df = esco_df[
            ['occupationUri', 'skillUri', 'occupationLabel', 'occupationDescription',
            'skillLabel', 'skillAltLabels', 'description']].drop_duplicates()


    
    # Get all skill texts, labels, and the skill-to-category map
    # --- MODIFIED: build_skill_lookups now returns 4 items ---
    all_skill_labels, all_skill_texts, skill_to_cat_map, alt_to_canonical_map = build_skill_lookups(
        esco_df=esco_df, hier_level=args.hier_level, 
        text_fields=args.text_fields, is_structured=args.is_structured,
        use_expanded_corpus=args.use_expanded_corpus # <-- Pass the new arg
    )
    # --- END MODIFICATION ---

    # --- 2. Load Ground Truth Job Data ---
    job_df = pd.read_csv(args.job_data_path)
    ground_truth_map, job_texts_to_run = load_ground_truth(job_df, esco_df, args)

    # --- 3. Load Step 1: Category Model ---
    category_model = None
    if run_hybrid_mode:
        if args.base_encoder_ckpt is None:
            logger.error("Must provide --base_encoder_ckpt when running in hybrid mode.")
            return
        base_encoder = build_encoder(args.base_encoder_ckpt, device=device)

        if args.hidden_dim is None:
            logger.info(f"hidden_dim not set. Inferring from {args.base_encoder_ckpt}...")
            args.hidden_dim = base_encoder.get_sentence_embedding_dimension()
            logger.info(f"Inferred hidden_dim: {args.hidden_dim}")

        cat_col = HIER_COL_MAP[args.hier_level]
        all_cat_labels = esco_df[cat_col].dropna().astype(str).unique().tolist()
        cat_idx2label = {i: label for i, label in enumerate(all_cat_labels)}

        category_model = load_category_model(
            ckpt_path=Path(args.cat_model_ckpt),
            base_encoder=base_encoder,
            hidden_dim=args.hidden_dim,
            categories_idx2str=cat_idx2label,
            device=device
        )

    # --- 4. Load Step 2: Skill SOTA Model & Build FAISS Index ---
    logger.info(f"Loading SOTA skill encoder: {args.skill_encoder_ckpt}")
    skill_encoder = build_encoder(args.skill_encoder_ckpt, device=device)
    faiss_index, faiss_id_to_label_map = build_faiss_index(
        skill_encoder=skill_encoder,
        all_skill_labels=all_skill_labels,
        all_skill_texts=all_skill_texts,
        device=device
    )

    # --- 5. Run Hybrid Inference Pipeline ---
    logger.info(f"Running hybrid inference on {len(job_texts_to_run)} job texts...")
    all_predictions = {}    
    
    # Process in batches for efficiency
    batch_size = args.batch_size
    desc = "Hybrid Inference" if run_hybrid_mode else "Baseline Inference"
    for i in tqdm(range(0, len(job_texts_to_run), batch_size), desc=desc):
        batch_job_texts = job_texts_to_run[i : i + batch_size]
        
        # --- Step 1: Always retrieve baseline skills ---
        retrieved_skills_batch = retrieve_skills(
            batch_job_texts, skill_encoder, faiss_index, faiss_id_to_label_map, args.top_k
        )
        
        # --- Step 2: Optionally run hybrid re-ranking ---
        if run_hybrid_mode:
            cat_probs_batch = predict_categories(category_model, batch_job_texts, cat_idx2label)
        
        for j, job_text in enumerate(batch_job_texts):
            if run_hybrid_mode:
            # Always save the baseline result
                ranked_skills = merge_hybrid_scores(
                    category_prob_map=cat_probs_batch[j],
                    retrieved_skills=retrieved_skills_batch[j],
                    skill_to_cat_map=skill_to_cat_map,
                    category_threshold=args.cat_threshold
                )
                all_predictions[job_text] = ranked_skills
            
            else:
                all_predictions[job_text] = retrieved_skills_batch[j]


    # --- 6. Evaluate and Save Results ---
    logger.info(f"Evaluating {process} predictions...")
       
    
    k_list_eval = [5, 10, 20, 50, 20000]
    final_metrics = evaluate_predictions(
        all_predictions=all_predictions,
        ground_truth_map=ground_truth_map,
        alt_to_canonical_map=alt_to_canonical_map,
        k_list=k_list_eval
    )
    
    logger.success(f"--- {process} EVALUATION RESULTS ---")
    print(json.dumps(final_metrics, indent=2))
    
    # Save predictions and metrics
    out_path_preds = Path(args.out_dir) / f"{run_name}_predictions.json"
    out_path_metrics = Path(args.out_dir) / f"{run_name}_metrics.json"
    
    # Convert predictions to a serializable format
    serializable_preds = {
        job: [{"skill": s, "score": r} for s, r in skills] 
        for job, skills in all_predictions.items()
    }
    
    with open(out_path_preds, 'w') as f:
        json.dump(serializable_preds, f, indent=2)
    with open(out_path_metrics, 'w') as f:
        json.dump(final_metrics, f, indent=2)
        
    logger.success(f"Saved predictions to {out_path_preds}")
    logger.success(f"Saved metrics to {out_path_metrics}")

    # --- NEW: Append to master CSV log ---
    duration_seconds = time.time() - start_time
    csv_path = Path(args.out_dir) / "master_results.csv"
    log_results_to_csv(csv_path, run_name, duration_seconds, process, args, final_metrics)


if __name__ == "__main__":

    args = parse_args()
    
    # --- Quick Check ---
    if "bge" in args.skill_encoder_ckpt.lower() and args.text_fields == "title+desc":
        logger.warning("BGE models were trained with structured queries. "
                       "Consider adding --is_structured for the skill encoder.")
    if "bge-large" in args.base_encoder_ckpt.lower() and args.hidden_dim != 1024:
        logger.warning(f"Base encoder {args.base_encoder_ckpt} likely has hidden_dim=1024, not {args.hidden_dim}")

    main(args)