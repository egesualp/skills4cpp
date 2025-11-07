"""
Robust Information Retrieval Evaluation Script

This script manually performs encoding to support complex models 
(like instruction-tuned or custom-architecture models) and then
uses the InformationRetrievalEvaluator just for its .compute_metrics() method.
"""

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from pathlib import Path
from datetime import datetime
import numpy as np # Import numpy

# ==================== Configuration ====================
RUN_NAME = "manual_eval_jobbert"
CORPUS_PATH = "data/talent_clef/TaskA/validation/english/corpus_elements"
QUERIES_PATH = "data/talent_clef/TaskA/validation/english/queries"  
QRELS_PATH = "data/talent_clef/TaskA/validation/english/qrels.tsv"

# --- Switch Model Here ---
MODEL_ID = "TechWolf/JobBERT-v3"
# MODEL_ID = "pj-mathematician/JobGTE-7b-Lora" 

# --- Set Prompt Here ---
# Set to None for models like JobBERT
QUERY_PROMPT = None 
# Set for instruction models like GTE
# QUERY_PROMPT = "Given a job title, retrieve similar job titles: "

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_FILE = "results/ir_eval_results.csv"
BATCH_SIZE = 128 # Adjust based on your VRAM

# ========================================================
# (MetricsCapturingEvaluator class remains the same)
# ========================================================

class MetricsCapturingEvaluator(InformationRetrievalEvaluator):
    """
    We only use this class as a container for the .compute_metrics() method.
    We will NOT be using its __call__ method.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured_metrics = {}

    def compute_and_capture(self, scores):
        """A helper to run compute_metrics and store the result."""
        # Ensure scores are in the format compute_metrics expects
        # (which is List[Dict] for older libraries)
        
        # Check if scores is already in List[Dict] format
        first_val = next(iter(scores.values()))
        if isinstance(first_val, list) and isinstance(first_val[0], dict):
            # Format is already correct
            pass
        elif isinstance(first_val, dict):
             # Format is Dict[str, Dict[str, float]], convert it
            print("  [Converting scores to List[Dict] format for evaluator...]")
            scores = {
                qid: [{"corpus_id": cid, "score": s} for cid, s in c_scores.items()]
                for qid, c_scores in scores.items()
            }
        
        print("  [Calculating metrics...]")
        self.captured_metrics = self.compute_metrics(scores)
        
        # Clean up numpy types for logging
        self.captured_metrics = self._clean_metrics(self.captured_metrics)
        
        return self.captured_metrics

    def _clean_metrics(self, metrics_dict):
        """Recursively convert numpy types to standard Python types."""
        cleaned = {}
        if not isinstance(metrics_dict, dict):
            return metrics_dict

        for key, value in metrics_dict.items():
            if isinstance(value, dict):
                cleaned[key] = self._clean_metrics(value)
            elif isinstance(value, (np.float64, np.float32)):
                cleaned[key] = float(value)
            elif isinstance(value, (np.int64, np.int32)):
                cleaned[key] = int(value)
            else:
                cleaned[key] = value
        return cleaned


def save_results(run_name, model_id, pairs_path, esco_path, metrics, num_queries, num_corpus):
    """Save evaluation results to CSV file (append mode)."""
    results_path = Path(RESULTS_FILE)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    results_row = {
        "run_name": run_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": model_id,
        "pairs_path": pairs_path,
        "esco_path": esco_path,
        "num_queries": num_queries,
        "num_corpus": num_corpus,
    }
    results_row.update(metrics)
    
    # Flatten any nested metrics for CSV
    flat_row = {}
    for key, value in results_row.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat_row[f"{key}_{sub_key}"] = sub_value
        else:
            flat_row[key] = value

    df = pd.DataFrame([flat_row])
    
    if results_path.exists():
        df.to_csv(results_path, mode='a', header=False, index=False)
        print(f"\n✅ Results appended to {results_path}")
    else:
        df.to_csv(results_path, mode='w', header=True, index=False)
        print(f"\n✅ Results saved to {results_path} (new file created)")


def main():
    print("=" * 60)
    print("Manual Information Retrieval Evaluation")
    print("=" * 60)
    
    # ==================== Load Data ====================
    print(f"\n[1/5] Loading data...")
    queries_df = pd.read_csv(QUERIES_PATH, sep='\t', dtype={'q_id': str})
    corpus_df = pd.read_csv(CORPUS_PATH, sep='\t', dtype={'c_id': str})
    qrels_df = pd.read_csv(QRELS_PATH, sep='\t', header=None, 
                          names=['q_id', 'iter', 'c_id', 'relevance'],
                          dtype={'q_id': str, 'c_id': str})
    
    print(f"  ✓ Loaded {len(queries_df)} queries")
    print(f"  ✓ Loaded {len(corpus_df)} corpus elements")
    print(f"  ✓ Loaded {len(qrels_df)} relevance judgments")
    
    # ==================== Prepare Data ====================
    print(f"\n[2/5] Preparing data...")
    
    # --- Use INT keys for queries, STR keys for corpus ---
    unique_query_ids = sorted(queries_df['q_id'].unique())
    query_id_mapping = {orig_id: i for i, orig_id in enumerate(unique_query_ids)}
    
    queries = {query_id_mapping[row['q_id']]: str(row['jobtitle']) 
              for _, row in queries_df.iterrows() 
              if row['q_id'] in query_id_mapping}
    
    corpus = {str(row['c_id']): str(row['jobtitle']) 
             for _, row in corpus_df.iterrows()}
    
    relevant_docs = {}
    for _, row in qrels_df.iterrows():
        orig_q_id = row['q_id']
        mapped_q_id = query_id_mapping.get(orig_q_id) # This is an INT
        c_id = str(row['c_id']) 
        
        if row['relevance'] > 0 and mapped_q_id is not None:
            if mapped_q_id not in relevant_docs:
                relevant_docs[mapped_q_id] = set() 
            if c_id in corpus:
                relevant_docs[mapped_q_id].add(c_id)
    
    queries = {qid: text for qid, text in queries.items() if qid in relevant_docs}
    
    print(f"  ✓ Queries (filtered): {len(queries)}")
    print(f"  ✓ Corpus: {len(corpus)}")
    print(f"  ✓ Relevant docs: {len(relevant_docs)}")

    # ==================== Load Model ====================
    print(f"\n[3/5] Loading model: {MODEL_ID} on {DEVICE}")
    model = SentenceTransformer(MODEL_ID, device=DEVICE)
    print(f"  ✓ Model loaded")

    # ==================== Manual Encoding ====================
    print(f"\n[4/5] Manually encoding queries and corpus...")
    
    # Prepare query texts
    query_texts = [queries[qid] for qid in queries.keys()]
    if QUERY_PROMPT:
        print(f"  ✓ Applying query prompt: '{QUERY_PROMPT}...'")
        query_texts = [f"{QUERY_PROMPT}{text}" for text in query_texts]

    # Prepare corpus texts
    corpus_texts = list(corpus.values())
    
    # Encode
    query_embeddings = model.encode(
        query_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True
    )
    
    corpus_embeddings = model.encode(
        corpus_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=True
    )
    
    print(f"  ✓ Query embeddings shape: {query_embeddings.shape}")
    print(f"  ✓ Corpus embeddings shape: {corpus_embeddings.shape}")

    # ==================== Calculate Metrics ====================
    print(f"\n[5/5] Computing scores and metrics...")

    # Compute scores
    scores_matrix = torch.mm(query_embeddings, corpus_embeddings.T).cpu().numpy()

    # Map back to original IDs
    query_ids_list = list(queries.keys())
    corpus_ids_list = list(corpus.keys())
    
    scores = {}
    for i, qid in enumerate(query_ids_list):
        scores[qid] = [
            {"corpus_id": corpus_ids_list[j], "score": float(scores_matrix[i, j])} 
            for j in range(len(corpus_ids_list))
        ]

    # Use the evaluator class *only* to compute metrics
    ir_evaluator = MetricsCapturingEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs
    )
    
    captured_metrics = ir_evaluator.compute_and_capture(scores)
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    
    print("\nDetailed Metrics:")
    for key, value in sorted(captured_metrics.items()):
        print(f"  {key}: {value}")
    
    # ==================== Save Results ====================
    print(f"\n[6/6] Saving results...")
    save_results(
        run_name=RUN_NAME,
        model_id=MODEL_ID,
        pairs_path=f"{QUERIES_PATH}+{CORPUS_PATH}+{QRELS_PATH}",
        esco_path="N/A",
        metrics=captured_metrics,
        num_queries=len(queries),
        num_corpus=len(corpus),
    )
    

if __name__ == "__main__":
    main()