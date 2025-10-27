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
RUN_NAME = "decorte_test_techwolf"
PAIRS_PATH = "data/title_pairs/decorte_test_pairs.csv"  # Columns: raw_title, esco_title, esco_id
ESCO_PATH = "data/occupations_en_expanded.csv"           # Columns: conceptUri, preferredLabel
MODEL_ID =  "pj-mathematician/JobGTE-7b-Lora"    # Can use any sentence-transformer model
DEVICE = "cuda"                                          # or "cpu"
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
            scores = {}
            for i, qid in enumerate(self.queries_ids):
                scores[qid] = {self.corpus_ids[j]: float(scores_matrix[i, j]) for j in range(len(self.corpus_ids))}

            # Compute metrics
            self.captured_metrics = self.compute_metrics(scores)

            # Print results
            print(f"\n{self.name} Results:")
            for key, value in sorted(self.captured_metrics.items()):
                print(f"  {key}: {value:.4f}")

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
    
    # Load pairs (job titles and their gold ESCO IDs)
    pairs_df = pd.read_csv(PAIRS_PATH)
    job_titles = pairs_df["raw_title"].astype(str).tolist()
    gold_esco_ids = pairs_df["esco_id"].astype(str).tolist()
    print(f"  ✓ Loaded {len(job_titles)} job title pairs")
    
    # Load ESCO corpus (all possible ESCO titles)
    esco_df = pd.read_csv(ESCO_PATH)
    esco_df = esco_df.rename(columns={"preferredLabel": "esco_title", "conceptUri": "esco_id"})
    
    # Remove duplicates (keep first occurrence)
    esco_df = esco_df.drop_duplicates(subset=["esco_id"], keep="first")
    
    esco_ids = esco_df["esco_id"].astype(str).tolist()
    esco_titles = esco_df["esco_title"].astype(str).tolist()
    print(f"  ✓ Loaded {len(esco_ids)} unique ESCO titles")
    
    # ==================== Prepare Data for Evaluator ====================
    print(f"\n[2/4] Preparing data for InformationRetrievalEvaluator...")
    
    # Format as dictionaries required by InformationRetrievalEvaluator
    # queries: {query_id: query_text}
    queries = {str(i): job_title for i, job_title in enumerate(job_titles)}
    
    # corpus: {corpus_id: corpus_text}
    corpus = {esco_id: esco_title for esco_id, esco_title in zip(esco_ids, esco_titles)}
    
    # relevant_docs: {query_id: set of relevant corpus_ids}
    relevant_docs = {str(i): {gold_id} for i, gold_id in enumerate(gold_esco_ids)}
    
    print(f"  ✓ Queries: {len(queries)}")
    print(f"  ✓ Corpus: {len(corpus)}")
    print(f"  ✓ Relevant docs: {len(relevant_docs)}")
    
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
        name="decorte_techwolf",
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
        pairs_path=PAIRS_PATH,
        esco_path=ESCO_PATH,
        metrics=ir_evaluator.captured_metrics,
        num_queries=len(queries),
        num_corpus=len(corpus),
    )
    

if __name__ == "__main__":
    main()

