# run_clef_sanity_check.py
import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import faiss
import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm
import time
import ast  # NEW: For parsing string lists
import skill_mapping.v1.utils as utils

from sentence_transformers import SentenceTransformer

# Import our robust encoder factory from model.py
# (Adjust path if necessary)
#try:
#    from skill_mapping.v1.model import build_encoder
#except ImportError:
#    logger.error("Could not import build_encoder from model.py. Please ensure model.py is accessible.")
#    exit(1)

CORPUS_PATH = "data/talent_clef/TaskB/validation/corpus_elements"
QUERIES_PATH = "data/talent_clef/TaskB/validation/queries"  
QRELS_PATH = "data/talent_clef/TaskB/validation/qrels.tsv"
# -----------------------------------------------------------------
# 1. REUSED COMPONENTS FROM run_inference.py
# (These are identical to our main script, which is the point of the test)
# -----------------------------------------------------------------

def average_precision_at_k(predicted_labels: List[str], actual_labels: Set[str], k: int) -> float:
    # ... (Unchanged) ...
    if not actual_labels: return 0.0
    hits, precision_sum = 0, 0.0
    for i in range(min(len(predicted_labels), k)):
        if predicted_labels[i] in actual_labels:
            hits += 1
            precision_sum += hits / (i + 1)
    return precision_sum / min(len(actual_labels), k)

def evaluate_predictions(all_predictions: Dict[str, List[Tuple[str, float]]],
                         ground_truth_map: Dict[str, Set[str]],
                         alt_to_canonical_map: Dict[str, str],
                         k_list: List[int] = [5, 10, 20]) -> Dict[str, float]:
    # ... (Unchanged) ...
    # This logic is *exactly* what we need to test
    metrics = {f"P@{k}": [] for k in k_list}
    metrics.update({f"R@{k}": [] for k in k_list})
    metrics.update({f"MAP@{k}": [] for k in k_list})
    
    num_queries_evaluated = 0
    for job_text_id in all_predictions.keys():
        actual_labels = ground_truth_map.get(job_text_id)
        if not actual_labels or len(actual_labels) == 0: continue
        
        num_queries_evaluated += 1
        predicted_scores = all_predictions.get(job_text_id, [])
        total_relevant = len(actual_labels)

        normalized_predicted_labels = []
        seen_canonical_labels = set()
        for alt_label_id, score in predicted_scores:
            # Map the retrieved alias_id back to its canonical c_id
            canonical_label_id = alt_to_canonical_map.get(alt_label_id, alt_label_id)
            if canonical_label_id not in seen_canonical_labels:
                normalized_predicted_labels.append(canonical_label_id)
                seen_canonical_labels.add(canonical_label_id)

        for k in k_list:
            top_k_preds = normalized_predicted_labels[:k]
            hits = len(set(top_k_preds) & actual_labels)
            
            metrics[f"P@{k}"].append(hits / k)
            metrics[f"R@{k}"].append(hits / total_relevant)
            metrics[f"MAP@{k}"].append(
                average_precision_at_k(normalized_predicted_labels, actual_labels, k)
            )

    final_metrics = {key: np.mean(val) if val else 0.0 for key, val in metrics.items()}
    final_metrics["num_queries_evaluated"] = num_queries_evaluated
    return final_metrics

def build_faiss_index(skill_encoder, 
                      all_skill_ids: List[str], 
                      all_skill_texts: List[str], 
                      device: str) -> Tuple[faiss.Index, Dict[int, str]]:
    # ... (Unchanged) ...
    logger.info(f"Building FAISS index for {len(all_skill_ids)} skills...")
    skill_emb = skill_encoder.encode(
        all_skill_texts, convert_to_tensor=True, show_progress_bar=True, 
        batch_size=32, device=device
    ).cpu().numpy()
    
    skill_emb = utils.normalize_embeddings(skill_emb).astype(np.float32)
    index = faiss.IndexFlatIP(skill_emb.shape[1])
    index.add(skill_emb)
    # Map the FAISS integer ID back to our unique alias_id
    id_to_label_map = {i: label for i, label in enumerate(all_skill_ids)}
    logger.success(f"FAISS index built with {index.ntotal} skills.")
    return index, id_to_label_map

@torch.no_grad()
def retrieve_skills(job_texts: List[str], 
                    skill_encoder, 
                    faiss_index: faiss.Index, 
                    id_map: Dict[int, str], 
                    top_k: int) -> List[List[Tuple[str, float]]]:
    # ... (Unchanged) ...
    actual_k = min(top_k, faiss_index.ntotal)
    
    job_emb = skill_encoder.encode(
        job_texts, 
        show_progress_bar=False,
        device=skill_encoder.device
    )
    D, I = faiss_index.search(job_emb.astype(np.float32), actual_k)    
    results = []
    for i in range(len(job_texts)):
        batch_results = []
        for j in range(actual_k):
            faiss_id = I[i, j]
            
            # This check is redundant now but good practice
            if faiss_id == -1:
                continue 
                
            batch_results.append((id_map[faiss_id], float(D[i, j])))
        results.append(batch_results)
    return results

# ---------------------------------------------------
# 2. CORRECTED TALENTCLEF DATA LOADERS
# ---------------------------------------------------

def load_clef_queries(path: str) -> Dict[str, str]:
    """Loads queries.tsv (q_id, jobtitle) into a dict: {q_id -> jobtitle}"""
    try:
        df = pd.read_csv(path, sep='\\t', engine='python')
    except pd.errors.ParserError:
        # Fallback if the first row is not a header
        df = pd.read_csv(path, sep='\\t', engine='python', header=None, names=['q_id', 'jobtitle'])
        
    return dict(zip(df['q_id'].astype(str), df['jobtitle'].astype(str)))


def load_clef_qrels(path: str) -> Dict[str, Set[str]]:
    """Loads qrels.tsv into a dict: {query_id -> {doc_id, doc_id, ...}}"""
    df = pd.read_csv(path, sep='\\t', header=None, names=['query-id', 'unused1', 'corpus-id', 'unused2'])
    qrels = {}
    for _, row in df.iterrows():
        q_id = str(row['query-id'])
        doc_id = str(row['corpus-id'])
        if q_id not in qrels:
            qrels[q_id] = set()
        qrels[q_id].add(doc_id)
    return qrels

def load_clef_taskB_corpus(corpus_path: str) -> Tuple[List[str], List[str], Dict[str, str]]:
    """
    Loads and "explodes" the Task B corpus (corpus_elements) file.
    This file has 'c_id' and 'skill_aliases' (as a string list).
    """
    logger.info(f"Loading and expanding CLEF Task B corpus from {corpus_path}...")
    try:
        corpus_df = pd.read_csv(corpus_path, sep='\\t', engine='python')
    except Exception as e:
        logger.error(f"Failed to load corpus file: {e}")
        return [], [], {}

    all_skill_ids = []       # Will store unique alias IDs (e.g., "c_id_alias_0")
    all_skill_texts = []     # Will store the alias texts
    alt_to_canonical_map = {} # Maps {alias_id -> c_id}

    for _, row in corpus_df.iterrows():
        c_id = str(row['c_id'])
        aliases_str = str(row['skill_aliases'])

        try:
            # Use ast.literal_eval to safely parse the string list
            aliases = ast.literal_eval(aliases_str)
            if not isinstance(aliases, list):
                aliases = [str(aliases)]
        except Exception:
            # Handle parsing errors (e.g., 'nan', empty string)
            continue

        if not aliases:
            continue

        # Create a new corpus entry for *every* alias
        for i, alias_text in enumerate(aliases):
            alias_text = str(alias_text).strip()
            if not alias_text:
                continue
            
            # Create a unique ID for this specific alias
            alias_id = f"{c_id}_alias_{i}"
            
            all_skill_ids.append(alias_id)
            all_skill_texts.append(alias_text)
            
            # Map this unique alias ID back to its canonical c_id
            alt_to_canonical_map[alias_id] = c_id

    logger.info(f"Total items in expanded corpus: {len(all_skill_ids)} (from {len(corpus_df)} canonical skills)")
    logger.info(f"Total items in alias map: {len(alt_to_canonical_map)}")
    return all_skill_ids, all_skill_texts, alt_to_canonical_map

# ---------------------------------------------------
# 3. MAIN SCRIPT
# ---------------------------------------------------

def main(args):
    utils.set_seed(args.seed)
    device = torch.device(args.device)
    
    run_name = f"sanity_check_{args.skill_encoder_ckpt.split('/')[-1]}_expanded"
    Path("logs").mkdir(exist_ok=True)
    logger.add(f"logs/{run_name}.log")
    
    logger.info("Loading TalentCLEF data...")
    queries = load_clef_queries(QUERIES_PATH)
    ground_truth_map = load_clef_qrels(QRELS_PATH)
    
    # Load and expand the Task B corpus
    all_skill_ids, all_skill_texts, alt_to_canonical_map = load_clef_taskB_corpus(CORPUS_PATH)

    job_texts_to_run = list(queries.values())
    job_text_ids_to_run = list(queries.keys()) 
    
    if not all_skill_ids:
        logger.error("Corpus is empty. Check --corpus_path and file format.")
        return

    logger.info(f"Loaded {len(queries)} queries, {len(all_skill_ids)} expanded corpus docs, and {len(ground_truth_map)} qrels.")
    
    logger.info(f"Loading SOTA encoder using build_encoder: {args.skill_encoder_ckpt}")
    #skill_encoder = build_encoder(
    #    encoder_name_or_path=args.skill_encoder_ckpt,
    #    device=device
    #)

    model_raw = SentenceTransformer(args.skill_encoder_ckpt, device=device)
    skill_encoder = SentenceTransformer(modules=[model_raw[0], model_raw[1]], device=device)
    
    faiss_index, faiss_id_to_label_map = build_faiss_index(
        skill_encoder=skill_encoder,
        all_skill_ids=all_skill_ids,
        all_skill_texts=all_skill_texts,
        device=device
    )
    
    logger.info(f"Running inference on {len(job_texts_to_run)} queries...")
    all_predictions = {}
    batch_size = args.batch_size
    
    for i in tqdm(range(0, len(job_texts_to_run), batch_size), desc="Sanity Check Inference"):
        batch_job_texts = job_texts_to_run[i : i + batch_size]
        batch_job_ids = job_text_ids_to_run[i : i + batch_size]
        
        retrieved_skills_batch = retrieve_skills(
            batch_job_texts, skill_encoder, faiss_index, faiss_id_to_label_map, args.top_k
        )
        
        for j, job_id in enumerate(batch_job_ids):
            # We map q_id (job_id) to the list of (alias_id, score)
            all_predictions[job_id] = retrieved_skills_batch[j]

    logger.info("Evaluating predictions using our custom alias-normalization...")
    final_metrics = evaluate_predictions(
        all_predictions, 
        ground_truth_map, 
        alt_to_canonical_map,
        k_list=[1, 3, 5, 10, 100, 18672]  # <-- MODIFIED
    )
    
    logger.success("--- TALENTCLEF SANITY CHECK RESULTS ---")
    print(json.dumps(final_metrics, indent=2))
    
    out_path = Path("results/")
    out_path.mkdir(exist_ok=True)
    with open(out_path / f"{run_name}_metrics.json", 'w') as f:
        json.dump(final_metrics, f, indent=2)
    logger.success(f"Metrics saved to {out_path / f'{run_name}_metrics.json'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Corrected paths based on your vanilla script
    parser.add_argument("--skill_encoder_ckpt", type=str, 
                        default="TechWolf/JobBERT-v2",
                        help="SOTA encoder for skill retrieval")
                        # "TechWolf/JobBERT-v2"
                        # "pj-mathematician/JobSkillGTE-7b-lora"
    
    parser.add_argument("--top_k", type=int, default=19000, 
                        help="k for FAISS (must be high for alias normalization)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    main(args)