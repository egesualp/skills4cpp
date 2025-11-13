"""
Vanilla Information Retrieval Evaluation Script (Documentation Style)

This script is a direct implementation of the sentence-transformers documentation
for InformationRetrievalEvaluator, using our project's data.

It is simple and clean but has two limitations:
1. It only works for standard SentenceTransformer models (NOT models with custom
   routers like TechWolf/JobBERT-v2).
2. It only prints metrics to the console and does not save them to a file.

Usage:
    python -m src.helpers.vanilla_ir_eval
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Router
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from helpers.SkillRetrievalEvaluator import SkillRetrievalEvaluator
import ast
import os

task = "B"
sanity_check=False
use_alias_expansion=True
dataset="talentclef"
MODEL_ID =  "pj-mathematician/JobSkillGTE-7b-lora"    # Can use any sentence-transformer model
DEVICE = "cuda"                                      # or "cpu"
RESULTS_CSV_PATH = "results/vanilla_ir_eval_results.csv"
# ==================== Configuration ====================
#QUERIES_PATH = "data/talent_clef/TaskA/validation/english/queries"
#CORPUS_PATH = "data/talent_clef/TaskA/validation/english/corpus_elements"
#QRELS_PATH = "data/talent_clef/TaskA/validation/english/qrels.tsv"

# Task A below
if dataset=="talentclef":
    if task == "A":
        QUERIES_PATH = "data/talent_clef/TaskA/validation/english/queries"  
        CORPUS_PATH = "data/talent_clef/TaskA/validation/english/corpus_elements"
        QRELS_PATH = "data/talent_clef/TaskA/validation/english/qrels.tsv"
    else:
        CORPUS_PATH = "data/talent_clef/TaskB/validation/corpus_elements"
        QUERIES_PATH = "data/talent_clef/TaskB/validation/queries"  
        QRELS_PATH = "data/talent_clef/TaskB/validation/qrels.tsv"
elif dataset=="decorte":
    if task=="A":
        QUERIES_PATH = "data/ir_format/decorte_test_lean/queries"  
        CORPUS_PATH = "data/ir_format/decorte_test_lean/corpus_elements"
        QRELS_PATH = "data/ir_format/decorte_test_lean/qrels.tsv"
    else:
        CORPUS_PATH = "data/ir_format/decorte_test_task_b_all_queries/corpus_elements"
        QUERIES_PATH = "data/ir_format/decorte_test_task_b_all_queries/queries"  
        QRELS_PATH = "data/ir_format/decorte_test_task_b_all_queries/qrels.tsv"
else:
    print("sorry")



def save_results_to_csv(results, model_id, task, dataset, use_alias_expansion, output_path, alias_count=0):
    """Appends evaluation results to a master CSV file in a long format."""
    
    records = []
    for full_metric, value in results.items():
        # Create a simplified metric name
        try:
            # Assumes metric format like 'TaskB_cosine_map@10'
            metric_part = full_metric.split('cosine_')[-1]
            if alias_count > 0 and str(alias_count) in metric_part:
                 metric_part = metric_part.replace(str(alias_count), 'full')
        except:
            metric_part = "unknown"

        record = {
            "model_id": model_id,
            "task": task,
            "dataset": dataset,
            "use_alias_expansion": use_alias_expansion,
            "full_metric": full_metric,
            "metric": metric_part,
            "value": value
        }
        records.append(record)

    # Convert to a DataFrame
    new_results_df = pd.DataFrame(records)

    # Ensure the directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Append to the CSV file
    try:
        if os.path.exists(output_path):
            master_df = pd.read_csv(output_path)
            master_df = pd.concat([master_df, new_results_df], ignore_index=True)
        else:
            master_df = new_results_df
        
        master_df.to_csv(output_path, index=False)
        print(f"\n[✓] Results appended to {output_path}")
    except Exception as e:
        print(f"\n[!!!] Error saving results: {e}")


def main():
    print("=" * 60)
    print("Vanilla Information Retrieval Evaluation (Docs Style)")
    print("=" * 60)
    
    # ==================== 1. Load Data ====================
    print(f"\n[1/3] Loading Talent CLEF 2024 data...")
    
    queries_df = pd.read_csv(QUERIES_PATH, sep='\\t', engine='python')
    corpus_df = pd.read_csv(CORPUS_PATH, sep='\\t', engine='python')
    qrels_df = pd.read_csv(QRELS_PATH, sep='\\t', header=None, names=['q_id', '0', 'c_id', '1'])

    print(f"  ✓ Loaded {len(queries_df)} queries")
    print(f"  ✓ Loaded {len(corpus_df)} corpus elements")
    print(f"  ✓ Loaded {len(qrels_df)} relevance judgements")

    if sanity_check:
        print("\n" + "="*30)
        print("RUNNING SANITY CHECK")

        # 1. Get all skill IDs that are in the corpus
        findable_skill_ids = set(corpus_df['c_id'].astype(str))
        print(f"Total skills in corpus: {len(findable_skill_ids)}")

        # 2. Get all skill IDs that are in the ground truth
        relevant_skill_ids = set(qrels_df['c_id'].astype(str))
        print(f"Total unique skills in qrels: {len(relevant_skill_ids)}")

        # 3. Find the skills in the ground truth that ARE NOT in the corpus
        unfindable_skills = relevant_skill_ids - findable_skill_ids

        if unfindable_skills:
            print(f"\n[!!!] WARNING: Mismatch found!")
            print(f"There are {len(unfindable_skills)} skill IDs in your qrels.tsv")
            print(f"that do not exist in your corpus_elements file.")
            print(f"This is the reason MAP and nDCG are low.")
        else:
            print(f"\n[✓] Sanity check passed. All relevant skills are findable.")

        print("="*30 + "\n")
    else:

        # ==================== 2. Prepare Dictionaries ====================
        # The evaluator requires data in specific dictionary formats.
        print(f"\n[2/3] Preparing data dictionaries...")

        # Queries are always q_id -> jobtitle
        queries = dict(zip(queries_df['q_id'].astype(str), queries_df['jobtitle'].astype(str)))

        # relevant_docs are always q_id -> set[c_id]
        relevant_docs = {}
        for _, row in qrels_df.iterrows():
            q_id = str(row['q_id'])
            c_id = str(row['c_id'])
            if q_id not in relevant_docs:
                relevant_docs[q_id] = set()
            relevant_docs[q_id].add(c_id)

        # Corpus dict depends on the task
        if task == "A":
            corpus = dict(zip(corpus_df['c_id'].astype(str), corpus_df['jobtitle'].astype(str)))
            print(f"  ✓ Prepared Task A data.")
        else:
            # For Task B, we pass the *unprocessed* dict of c_id -> skill_aliases_string
            # Our new evaluator will handle the parsing and exploding.
            corpus = dict(zip(corpus_df['c_id'].astype(str), corpus_df['skill_aliases'].astype(str)))
            print(f"  ✓ Prepared Task B data.")


        print(f"  ✓ Queries: {len(queries)}")
        print(f"  ✓ Corpus: {len(corpus)}")
        print(f"  ✓ Relevant docs: {len(relevant_docs)}")

        # ==================== 3. Load Model & Run Evaluation ====================
        print(f"\n[3/3] Loading model and running evaluation...")
        print(f"Model ID: {MODEL_ID}")
        model_raw = SentenceTransformer(MODEL_ID, device=DEVICE)
        model = SentenceTransformer(modules=[model_raw[0], model_raw[1]], device=DEVICE)

        # For saving results consistently
        alias_count = 0

        # Create the evaluator as shown in the documentation
        if task == "A":
            print("Using standard InformationRetrievalEvaluator for Task A...")
            ir_evaluator = InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                name=""
            )
        else:
            if use_alias_expansion:
                print("Using custom SkillRetrievalEvaluator for Task B...")

                # --- START FIX: Pre-calculate total alias count ---
                print("  > Pre-calculating total alias count...")
                alias_count = 0
                for aliases_str in corpus.values(): # corpus is dict {c_id: alias_string}
                    try:
                        alias_count += len(ast.literal_eval(aliases_str))
                    except:
                        pass # Ignore parsing errors if any
                    
                # Add a buffer just in case
                alias_count += 10 
                print(f"  > Setting k = {alias_count} to get full ranking.")
                # --- END FIX ---

                ir_evaluator = SkillRetrievalEvaluator(
                    queries=queries,
                    corpus=corpus,
                    relevant_docs=relevant_docs,
                    name="TaskB",
                    # -- THIS IS FIX --
                    map_at_k=[alias_count],
                    ndcg_at_k=[10, alias_count], # Keep @10, but add full list
                    mrr_at_k=[10, alias_count],
                    accuracy_at_k=[1, 3, 5, 10], # These are fine
                    precision_recall_at_k=[1, 3, 5, 10] # These are fine 
                )
            else:
                print("Using standard InformationRetrievalEvaluator for Task B (no alias expansion)...")
                # When not expanding, we need to use the canonical skill label, not the alias list.
                # Assuming 'skill_label' is present in corpus_df.
                corpus_no_expansion = dict(zip(corpus_df['c_id'].astype(str), corpus_df['skill_label'].astype(str)))
                
                ir_evaluator = InformationRetrievalEvaluator(
                    queries=queries,
                    corpus=corpus_no_expansion,
                    relevant_docs=relevant_docs,
                    name="TaskB_no_expansion"
                )

        # This call will handle everything: encoding, scoring, and printing results
        results = ir_evaluator(model)

        save_results_to_csv(results, MODEL_ID, task, dataset, use_alias_expansion, RESULTS_CSV_PATH, alias_count=alias_count)

        for k, v in results.items():
            print(f"{k}: {v}")

        print("\n" + "=" * 60)
        print("Evaluation Complete!")
        print("The metrics table is printed above by the evaluator.")
        print("=" * 60)


if __name__ == "__main__":
    main()
