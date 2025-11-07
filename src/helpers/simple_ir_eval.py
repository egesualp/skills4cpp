"""
Simple Information Retrieval Evaluation Script

This script demonstrates how to use sentence_transformers' InformationRetrievalEvaluator
for evaluating job-to-ESCO title matching. It's designed to be standalone and easy to understand.

Usage:
    python -m src.helpers.simple_ir_eval
"""

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from pathlib import Path
from datetime import datetime


# ==================== Configuration ====================
RUN_NAME = "decorte_test_lean"
CORPUS_PATH = "data/talent_clef/TaskA/validation/english/corpus_elements"
QUERIES_PATH = "data/talent_clef/TaskA/validation/english/queries"  
QRELS_PATH = "data/talent_clef/TaskA/validation/english/qrels.tsv"
MODEL_ID =  "pj-mathematician/JobGTE-7b-Lora"    # Can use any sentence-transformer model
DEVICE = "cpu"                                          # or "cpu"
TOP_K = 10                                               # Top-K for metrics
RESULTS_FILE = "results/ir_eval_results.csv"             # Results will be appended here


class MetricsCapturingEvaluator(InformationRetrievalEvaluator):
    """Extended evaluator that captures metrics for saving."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.captured_metrics = {}
        self.use_fallback = False

    def _has_router_issue(self, model):
        """Check if model has a Router that might cause issues."""
        try:
            # Check if model has Router module
            for module in model.modules():
                if module.__class__.__name__ == 'Router':
                    return True
            return False
        except Exception:
            return False

    def __call__(self, model, output_path=None, epoch=-1, steps=-1, *args, **kwargs):
        # Check if we need fallback for models with Router issues (e.g., JobBERT)
        if self._has_router_issue(model):
            self.use_fallback = True

        if self.use_fallback:
            # Fallback: Manual encoding for models with Router issues.
            # We use the attributes prepared by the parent __init__ method
            # (self.queries, self.corpus, self.queries_ids, self.corpus_ids)
            # to ensure consistency.
            print("  [Using fallback encoding for Router-based model]")

            corpus_embeddings = model.encode(
                self.corpus,  # Use the list of texts from parent
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor=True,
                normalize_embeddings=True,
                prompt_name=None,
            )

            query_embeddings = model.encode(
                self.queries,  # Use the list of texts from parent
                batch_size=self.batch_size,
                show_progress_bar=self.show_progress_bar,
                convert_to_tensor=True,
                normalize_embeddings=True,
                prompt_name=None,
            )

            # Compute similarity scores
            scores_matrix = torch.mm(query_embeddings, corpus_embeddings.T).cpu().numpy()

            # Format scores as dict using IDs from parent class
            # Note: compute_metrics expects integer keys for query IDs and list of dict values
            scores = {}
            for i, qid in enumerate(self.queries_ids):
                # Convert query ID to integer for compute_metrics compatibility
                query_idx = int(qid) if qid.isdigit() else i
                # Format as list of dicts with corpus_id and score keys (expected by compute_metrics)
                scores[query_idx] = [
                    {"corpus_id": self.corpus_ids[j], "score": float(scores_matrix[i, j])} 
                    for j in range(len(self.corpus_ids))
                ]


            # Compute metrics
            self.captured_metrics = self.compute_metrics(scores)

            # Print results
            print(f"\n{self.name} Results:")
            for key, value in sorted(self.captured_metrics.items()):
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

            return self.captured_metrics.get('map@100', 0.0)

        else:
            # Default: Use native InformationRetrievalEvaluator
            try:
                main_score = super().__call__(model, output_path, epoch, steps, *args, **kwargs)

                # Capture metrics by re-computing (native evaluator doesn't expose them)
                # This ensures we can save all metrics for all models.
                corpus_embeddings = model.encode(
                    self.corpus,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                )

                query_embeddings = model.encode(
                    self.queries,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                )

                scores_matrix = torch.mm(query_embeddings, corpus_embeddings.T).cpu().numpy()

                scores = {}
                for i, qid in enumerate(self.queries_ids):
                    scores[qid] = {self.corpus_ids[j]: float(scores_matrix[i, j]) for j in range(len(self.corpus_ids))}

                self.captured_metrics = self.compute_metrics(scores)

                return main_score

            except ValueError as e:
                if "No route found" in str(e):
                    # Router error detected, switch to fallback
                    print(f"\n  [Router error detected: {e}]")
                    print("  [Switching to fallback encoding...]")
                    self.use_fallback = True
                    return self.__call__(model, output_path, epoch, steps, *args, **kwargs)
                else:
                    raise


def save_results(run_name, model_id, pairs_path, esco_path, metrics, num_queries, num_corpus):
    """Save evaluation results to CSV file (append mode)."""
    
    # Create results directory if it doesn't exist
    results_path = Path(RESULTS_FILE)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare results row
    results_row = {
        "run_name": run_name,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_id": model_id,
        "pairs_path": pairs_path,
        "esco_path": esco_path,
        "num_queries": num_queries,
        "num_corpus": num_corpus,
    }
    
    # Add all metrics
    results_row.update(metrics)
    
    # Convert to DataFrame
    df = pd.DataFrame([results_row])
    
    # Append to CSV (create with header if doesn't exist)
    if results_path.exists():
        df.to_csv(results_path, mode='a', header=False, index=False)
        print(f"\n✅ Results appended to {results_path}")
    else:
        df.to_csv(results_path, mode='w', header=True, index=False)
        print(f"\n✅ Results saved to {results_path} (new file created)")


def main():
    print("=" * 60)
    print("Simple Information Retrieval Evaluation")
    print("=" * 60)
    
    # ==================== Load Data ====================
    print(f"\n[1/4] Loading data...")
    
    # Load queries
    queries_df = pd.read_csv(QUERIES_PATH, sep='\t')
    print(f"  ✓ Loaded {len(queries_df)} queries")
    
    # Load corpus
    corpus_df = pd.read_csv(CORPUS_PATH, sep='\t')
    print(f"  ✓ Loaded {len(corpus_df)} corpus elements")
    
    # Load relevance judgments
    qrels_df = pd.read_csv(QRELS_PATH, sep='\t', header=None, 
                          names=['q_id', 'iter', 'c_id', 'relevance'])
    print(f"  ✓ Loaded {len(qrels_df)} relevance judgments")
    
    # ==================== Prepare Data for Evaluator ====================
    print(f"\n[2/4] Preparing data for InformationRetrievalEvaluator...")
    
    # Create ID mappings to ensure consecutive IDs starting from 0
    # InformationRetrievalEvaluator expects query IDs to be consecutive integers (as strings)
    unique_query_ids = sorted(queries_df['q_id'].unique())
    query_id_mapping = {orig_id: i for i, orig_id in enumerate(unique_query_ids)}
    
    print(f"  ✓ Query ID mapping: {len(query_id_mapping)} queries ({min(unique_query_ids)} → {max(unique_query_ids)})")
    
    # Format as dictionaries required by InformationRetrievalEvaluator
    # queries: {query_id: query_text} - using integer keys as strings
    queries = {str(query_id_mapping[row['q_id']]): str(row['jobtitle']) 
              for _, row in queries_df.iterrows()}
    
    # corpus: {corpus_id: corpus_text}  
    corpus = {str(row['c_id']): str(row['jobtitle']) 
             for _, row in corpus_df.iterrows()}
    
    # relevant_docs: {query_id: set of relevant corpus_ids}
    # Group by query_id and collect all relevant corpus_ids using mapped IDs
    relevant_docs = {}
    for _, row in qrels_df.iterrows():
        orig_q_id = row['q_id']
        mapped_q_id = query_id_mapping.get(orig_q_id)
        c_id = str(row['c_id'])
        relevance = row['relevance']
        
        # Only include relevant documents (relevance > 0) and valid query IDs
        if relevance > 0 and mapped_q_id is not None:
            mapped_q_id_str = str(mapped_q_id)
            if mapped_q_id_str not in relevant_docs:
                relevant_docs[mapped_q_id_str] = set()
            relevant_docs[mapped_q_id_str].add(c_id)
    
    print(f"  ✓ Queries: {len(queries)}")
    print(f"  ✓ Corpus: {len(corpus)}")
    print(f"  ✓ Relevant docs: {len(relevant_docs)}")
    print(f"  ✓ Sample query IDs: {list(queries.keys())[:5]}")
    print(f"  ✓ Sample corpus IDs: {list(corpus.keys())[:5]}")
    print(f"  ✓ Sample relevant docs: {list(relevant_docs.keys())[:5]}")
    
    # ==================== Load Model ====================
    print(f"\n[3/4] Loading model: {MODEL_ID}")
    model = SentenceTransformer(MODEL_ID, device=DEVICE)
    print(f"  ✓ Model loaded")
    
    # ==================== Run Evaluation ====================
    print(f"\n[4/4] Running evaluation...")
    
    # Create the evaluator (use custom class to capture metrics)
    ir_evaluator = MetricsCapturingEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="decorte_test_lean",
        show_progress_bar=True,
        batch_size=128,
    )
    
    # Run evaluation (this will encode queries and corpus, then compute metrics)
    main_metric = ir_evaluator(model)
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"\nMain metric (MAP@{TOP_K}): {main_metric:.4f}")
    print("\nNote: The evaluator prints detailed metrics above including:")
    print("  - Accuracy@k, Precision@k, Recall@k")
    print("  - MRR (Mean Reciprocal Rank)")
    print("  - NDCG (Normalized Discounted Cumulative Gain)")
    print("  - MAP (Mean Average Precision)")
    
    # ==================== Save Results ====================
    print(f"\n[5/5] Saving results...")
    save_results(
        run_name=RUN_NAME,
        model_id=MODEL_ID,
        pairs_path=f"{QUERIES_PATH}+{CORPUS_PATH}+{QRELS_PATH}",  # Combined path info
        esco_path="N/A",  # Not applicable for this format
        metrics=ir_evaluator.captured_metrics,
        num_queries=len(queries),
        num_corpus=len(corpus),
    )
    

if __name__ == "__main__":
    main()

