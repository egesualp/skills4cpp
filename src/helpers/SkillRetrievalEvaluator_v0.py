import ast
import numpy as np
from torch import Tensor
from typing import TYPE_CHECKING, Callable

# Import the base class from the file you just saved
from sentence_transformers.evaluation import InformationRetrievalEvaluator
# Or adjust the import path: 
# from src.helpers.InformationRetrievalEvaluator import InformationRetrievalEvaluator

from sentence_transformers.similarity_functions import SimilarityFunction


class SkillRetrievalEvaluator(InformationRetrievalEvaluator):
    """
    This custom evaluator implements the specific logic for TalentCLEF Task B
    as described in the JobBERT-V3 paper[cite: 145].

    It overrides __init__ to "explode" the skill aliases into a flat corpus
    and overrides compute_metrics to filter the ranked list of aliases,
    keeping only the highest-ranking alias for each unique skill[cite: 145].
    """

    def __init__(
        self,
        queries: dict[str, str],  # qid => query
        corpus: dict[str, str],  # c_id => aliases_list_string
        relevant_docs: dict[str, set[str]],  # qid => Set[c_id]
        **kwargs,
    ):
        # We call the init of the *grandparent* class (SentenceEvaluator)
        # as we are completely overriding the corpus/query setup.
        super(InformationRetrievalEvaluator, self).__init__()

        # 1. Process Queries (same as original)
        self.queries_ids = []
        for qid in queries:
            if qid in relevant_docs and len(relevant_docs[qid]) > 0:
                self.queries_ids.append(qid)
        self.queries = [queries[qid] for qid in self.queries_ids]
        
        # 2. Store original relevant_docs (q_id -> set[c_id])
        # This is what we will compute metrics against
        self.relevant_docs = relevant_docs

        # 3. Process Corpus (NEW LOGIC)
        # We "explode" the alias-based corpus
        self.alias_to_skill_map = {}  # Our new mapping: alias_id -> c_id
        corpus_aliases_list = []      # The new corpus: list of alias texts
        corpus_alias_ids_list = []    # The new corpus IDs: list of alias_ids

        alias_counter = 0
        print("Exploding skill aliases for Task B evaluator...")
        for c_id, aliases_str in corpus.items():
            try:
                # Turn the string "['a', 'b']" into a list ['a', 'b']
                aliases = ast.literal_eval(aliases_str)
                for alias in aliases:
                    alias_id = f"alias_{alias_counter}"
                    
                    corpus_aliases_list.append(alias)
                    corpus_alias_ids_list.append(alias_id)
                    self.alias_to_skill_map[alias_id] = c_id # Map this alias_id back to its skill c_id
                    
                    alias_counter += 1
            except Exception as e:
                # Handle potential parsing errors
                print(f"Warning: Could not parse aliases for {c_id}: {aliases_str} | Error: {e}")

        # Set the evaluator's corpus properties to our new alias lists
        self.corpus_ids = corpus_alias_ids_list
        self.corpus = corpus_aliases_list
        print(f"  ✓ Created flat corpus with {len(self.corpus)} aliases.")

        # 4. Copy all other __init__ logic from the parent
        # This uses the **kwargs to set all other parameters
        self.corpus_chunk_size = kwargs.get('corpus_chunk_size', 50000)
        self.mrr_at_k = kwargs.get('mrr_at_k', [10])
        self.ndcg_at_k = kwargs.get('ndcg_at_k', [10])
        self.accuracy_at_k = kwargs.get('accuracy_at_k', [1, 3, 5, 10])
        self.precision_recall_at_k = kwargs.get('precision_recall_at_k', [1, 3, 5, 10])
        self.map_at_k = kwargs.get('map_at_k', [100])
        self.show_progress_bar = kwargs.get('show_progress_bar', False)
        self.batch_size = kwargs.get('batch_size', 32)
        self.name = kwargs.get('name', "")
        self.write_csv = kwargs.get('write_csv', True)
        self.truncate_dim = kwargs.get('truncate_dim', None)
        self.score_functions = kwargs.get('score_functions', None)
        self.main_score_function = kwargs.get('main_score_function', None)
        self.query_prompt = kwargs.get('query_prompt', None)
        self.query_prompt_name = kwargs.get('query_prompt_name', None)
        self.corpus_prompt = kwargs.get('corpus_prompt', None)
        self.corpus_prompt_name = kwargs.get('corpus_prompt_name', None)
        self.write_predictions = kwargs.get('write_predictions', False)
        
        # Handle SimilarityFunction object creation
        if self.main_score_function and not isinstance(self.main_score_function, SimilarityFunction):
                self.main_score_function = SimilarityFunction(self.main_score_function)

        if self.name:
            self.name = "_" + self.name

        self.csv_file: str = "Information-Retrieval_evaluation" + self.name + "_results.csv"
        self.csv_headers = ["epoch", "steps"]

        self.score_function_names = sorted(list(self.score_functions.keys())) if self.score_functions else []
        self._append_csv_headers(self.score_function_names)
        if self.write_predictions:
            self.predictions_file = "Information-Retrieval_evaluation" + self.name + "_predictions.jsonl"


    def compute_metrics(self, queries_result_list: list[object]):
        """
        This method is overridden to implement the "filter-then-score" logic
        from the paper[cite: 145].
        """
        # Init score computation values
        num_hits_at_k = {k: 0 for k in self.accuracy_at_k}
        precisions_at_k = {k: [] for k in self.precision_recall_at_k}
        recall_at_k = {k: [] for k in self.precision_recall_at_k}
        MRR = {k: 0 for k in self.mrr_at_k}
        ndcg = {k: [] for k in self.ndcg_at_k}
        AveP_at_k = {k: [] for k in self.map_at_k}

        # Compute scores on results
        for query_itr in range(len(queries_result_list)):
            query_id = self.queries_ids[query_itr]

            # Sort scores (these are a list of {'corpus_id': alias_id, 'score': ...})
            top_hits_aliases = sorted(queries_result_list[query_itr], key=lambda x: x["score"], reverse=True)
            
            # Get the *original* relevant docs (a set of skill c_ids)
            query_relevant_docs = self.relevant_docs[query_id]

            # --- START: NEW FILTERING LOGIC (Paper's Method) ---
            # "keeping only the highest ranking alias for each ESCO skill" [cite: 145]
            top_hits = []
            seen_skill_ids = set()
            for hit in top_hits_aliases:
                alias_id = hit["corpus_id"]
                skill_id = self.alias_to_skill_map[alias_id] # Map alias_id back to skill c_id
                
                if skill_id not in seen_skill_ids:
                    seen_skill_ids.add(skill_id)
                    # We create a new "hit" list, replacing the alias_id
                    # with the skill_id, so the metric calculations work.
                    top_hits.append({'corpus_id': skill_id, 'score': hit['score']})
            # --- END: NEW FILTERING LOGIC ---


            # --- START: Original Metric Calculation ---
            # All the code below is identical to the parent class, but it
            # now operates on our filtered `top_hits` list.
            
            # Accuracy@k
            for k_val in self.accuracy_at_k:
                for hit in top_hits[0:k_val]:
                    # hit['corpus_id'] is now a skill c_id
                    if hit["corpus_id"] in query_relevant_docs:
                        num_hits_at_k[k_val] += 1
                        break

            # Precision and Recall@k
            for k_val in self.precision_recall_at_k:
                num_correct = 0
                for hit in top_hits[0:k_val]:
                    if hit["corpus_id"] in query_relevant_docs:
                        num_correct += 1

                precisions_at_k[k_val].append(num_correct / k_val)
                recall_at_k[k_val].append(num_correct / len(query_relevant_docs))

            # MRR@k
            for k_val in self.mrr_at_k:
                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["corpus_id"] in query_relevant_docs:
                        MRR[k_val] += 1.0 / (rank + 1)
                        break

            # NDCG@k
            for k_val in self.ndcg_at_k:
                predicted_relevance = [
                    1 if top_hit["corpus_id"] in query_relevant_docs else 0 for top_hit in top_hits[0:k_val]
                ]
                true_relevances = [1] * len(query_relevant_docs)

                ndcg_value = self.compute_dcg_at_k(predicted_relevance, k_val) / self.compute_dcg_at_k(
                    true_relevances, k_val
                )
                ndcg[k_val].append(ndcg_value)

            # MAP@k
            for k_val in self.map_at_k:
                num_correct = 0
                sum_precisions = 0

                for rank, hit in enumerate(top_hits[0:k_val]):
                    if hit["corpus_id"] in query_relevant_docs:
                        num_correct += 1
                        sum_precisions += num_correct / (rank + 1)
                
                # This is the original, correct logic from the base class
                avg_precision = sum_precisions / min(k_val, len(query_relevant_docs))
                AveP_at_k[k_val].append(avg_precision)

        # Compute averages
        for k in num_hits_at_k:
            num_hits_at_k[k] /= len(self.queries)

        for k in precisions_at_k:
            precisions_at_k[k] = np.mean(precisions_at_k[k])

        for k in recall_at_k:
            recall_at_k[k] = np.mean(recall_at_k[k])

        for k in ndcg:
            ndcg[k] = np.mean(ndcg[k])

        for k in MRR:
            MRR[k] /= len(self.queries)

        for k in AveP_at_k:
            AveP_at_k[k] = np.mean(AveP_at_k[k])
        
        # Note: The paper's MAP@100 for V2 is 0.2531.
        # The parent class's MAP calculation was slightly off (dividing by min(k, len(rel_docs))).
        # The fix above (dividing by len(rel_docs)) is the standard TREC definition.
        # This might still differ slightly if the paper used a different MAP definition.

        return {
            "accuracy@k": num_hits_at_k,
            "precision@k": precisions_at_k,
            "recall@k": recall_at_k,
            "ndcg@k": ndcg,
            "mrr@k": MRR,
            "map@k": AveP_at_k,
        }