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


# ==================== Configuration ====================
#QUERIES_PATH = "data/talent_clef/TaskA/validation/english/queries"
#CORPUS_PATH = "data/talent_clef/TaskA/validation/english/corpus_elements"
#QRELS_PATH = "data/talent_clef/TaskA/validation/english/qrels.tsv"

CORPUS_PATH = "data/talent_clef/TaskA/validation/english/corpus_elements"
QUERIES_PATH = "data/talent_clef/TaskA/validation/english/queries"  
QRELS_PATH = "data/talent_clef/TaskA/validation/english/qrels.tsv"
MODEL_ID =  "pj-mathematician/JobGTE-7b-Lora"    # Can use any sentence-transformer model
DEVICE = "cuda"                                      # or "cpu"


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

    # ==================== 2. Prepare Dictionaries ====================
    # The evaluator requires data in specific dictionary formats.
    print(f"\n[2/3] Preparing data dictionaries...")

    queries = dict(zip(queries_df['q_id'].astype(str), queries_df['jobtitle'].astype(str)))
    corpus = dict(zip(corpus_df['c_id'].astype(str), corpus_df['jobtitle'].astype(str)))
    
    relevant_docs = {}
    for _, row in qrels_df.iterrows():
        q_id = str(row['q_id'])
        c_id = str(row['c_id'])
        if q_id not in relevant_docs:
            relevant_docs[q_id] = set()
        relevant_docs[q_id].add(c_id)

    print(f"  ✓ Queries: {len(queries)}")
    print(f"  ✓ Corpus: {len(corpus)}")
    print(f"  ✓ Relevant docs: {len(relevant_docs)}")

    # ==================== 3. Load Model & Run Evaluation ====================
    print(f"\n[3/3] Loading model and running evaluation...")
    print(f"Model ID: {MODEL_ID}")
    model = SentenceTransformer(MODEL_ID, device=DEVICE)

    # Create the evaluator as shown in the documentation
    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        name="",
    )
    
    # This call will handle everything: encoding, scoring, and printing results
    results = ir_evaluator(model)

    for k, v in results.items():
        print(f"{k}: {v}")
    
    print("\\n" + "=" * 60)
    print("Evaluation Complete!")
    print("The metrics table is printed above by the evaluator.")
    print("=" * 60)


if __name__ == "__main__":
    main()
