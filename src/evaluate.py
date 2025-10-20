import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from box import Box
from loguru import logger

from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers import util


logger.remove()
logger.add(sys.stderr, format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}")

from .config import load_config
from .utils import load_esco_titles, load_pairs
from .indexing import build_or_load_faiss_index, search_faiss_index
from .metrics import map_esco_id_to_row, METRICS, load_skills_per_occupation, compute_skill_coverage
from .model import BiEncoder


class AsymmetricInformationRetrievalEvaluator(InformationRetrievalEvaluator):
    """
    A custom Information Retrieval Evaluator for asymmetric models where the query and
    corpus are encoded using different methods of the model. This evaluator is designed
    to work with the custom BiEncoder class, which has distinct `encode_job` and
    `encode_esco` methods.

    This class overrides the `__call__` method of the base `InformationRetrievalEvaluator`
    to use these specific encoding functions instead of a generic `encode` method. This
    allows for correct handling of models with asymmetric architectures, such as those
    with different projection heads for queries and documents.
    """
    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1, *args, **kwargs) -> float:
        if epoch != -1:
            out_txt = f" after epoch {epoch}:"
        else:
            out_txt = ":"

        logger.info(f"Asymmetric Information Retrieval Evaluation on {self.name} dataset{out_txt}")

        corpus_embeddings = model.encode_esco(
            list(self.corpus.values()),
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            normalize=True,
            convert_to_numpy=True
        )

        query_embeddings = model.encode_job(
            list(self.queries.values()),
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            normalize=True,
            convert_to_numpy=True
        )
        
        # The information_retrieval method from the base class can be reused.
        # It handles the search (including FAISS) and returns scores in the required format.
        self.corpus_embeddings = corpus_embeddings
        scores = self.information_retrieval(query_embeddings, self.corpus_embeddings)

        # The compute_metrics and file logging logic can also be reused from the base class.
        # We'll just call the necessary methods.
        metrics = self.compute_metrics(scores)
        
        ndcg = metrics.get(f"ndcg@{self.top_k}", 0)
        _map = metrics.get(f"map@{self.top_k}", 0)
        recall = metrics.get(f"recall@{self.top_k}", 0)
        precision = metrics.get(f"precision@{self.top_k}", 0)

        metrics_log = {
            f"ndcg@{self.top_k}": ndcg,
            f"map@{self.top_k}": _map,
            f"recall@{self.top_k}": recall,
            f"precision@{self.top_k}": precision,
        }
        
        logger.info("Metrics computed using sentence-transformers evaluator:")
        pretty_print_metrics(metrics_log)

        if output_path is not None:
            csv_path = Path(output_path) / "st_eval_results.csv"
            is_new_file = not csv_path.exists()
            
            with open(csv_path, "a", encoding="utf-8") as f:
                if is_new_file:
                    f.write(f"epoch,steps,ndcg@{self.top_k},map@{self.top_k},recall@{self.top_k},precision@{self.top_k}\n")
                f.write(f"{epoch},{steps},{ndcg},{_map},{recall},{precision}\n")
        
        return _map

def seed_all(seed: int):
    """Seed all random number generators."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pretty_print_metrics(metrics: dict):
    """Prints a compact table of metrics."""
    df = pd.DataFrame([metrics])
    df = df.round(4)
    logger.info("\n" + df.to_string(index=False))


def main(cfg: Box):
    """Main evaluation script."""
    t_start = time.monotonic()
    seed_all(cfg.project.seed)
    
    # Create a unique model identifier for caching from the model path/name.
    model_id = Path(cfg.model.hf_id).name
    if cfg.eval.get("ckpt_path"):
        model_id += "_" + Path(cfg.eval.ckpt_path).stem

    cache_dir = Path(cfg.eval.cache_dir) / model_id

    if cfg.eval.get("clean_cache"):
        logger.info(f"Cleaning cache directory: {cache_dir}")
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)

    if not cfg.eval.get("run_name"):
        raise ValueError("Please specify a run_name in your config file.")
    
    results_dir = Path("experiments/results") / cfg.eval.run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load model
    logger.info("Loading model...")
    model_cfg = cfg.model
    # For raw sentence-transformer models, there's no projection head.
    if not cfg.eval.get("ckpt_path"):
        logger.info("No checkpoint path provided, using raw sentence-transformer. Setting proj_dim to None.")
        model_cfg.proj_dim = None

    model = BiEncoder(model_cfg, device).to(device)

    if cfg.eval.get("ckpt_path"):
        logger.info(f"Loading model checkpoint from {cfg.eval.ckpt_path}")
        ckpt = torch.load(cfg.eval.ckpt_path, map_location=device)
        model.load_state_dict(ckpt)
    model.eval()

    # If sentence-transformer evaluator is enabled, run it and exit.
    if cfg.eval.get("use_st_evaluator"):
        logger.info("Using sentence-transformers InformationRetrievalEvaluator.")
        
        # 1. Load data
        lowercase_setting = cfg.data.get("lowercase")
        lowercase_esco_flag = lowercase_setting in ["both", "esco"]
        esco_ids, esco_titles = load_esco_titles(
            cfg.data.esco_path, lowercase=lowercase_esco_flag
        )
        
        lowercase_pairs_flag = lowercase_setting in ["both", "pairs"]
        pairs = load_pairs(
            cfg.data.pairs_path,
            lowercase_raw=lowercase_pairs_flag,
            lowercase_esco=lowercase_esco_flag,
        )
        
        # 2. Prepare data for the evaluator
        corpus = {str(esco_id): title for esco_id, title in zip(esco_ids, esco_titles)}
        queries = {str(i): p["job_title"] for i, p in enumerate(pairs)}
        
        ground_truth_col = cfg.data.get("ground_truth_col", "esco_id")
        relevant_docs = {str(i): {str(p[ground_truth_col])} for i, p in enumerate(pairs)}

        evaluator = AsymmetricInformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name=cfg.eval.get("evaluator_name", "st-eval"),
            top_k=cfg.eval.topk,
            batch_size=cfg.eval.batch_size,
            show_progress_bar=True,
        )

        evaluator(model, output_path=str(results_dir))
        
        # Compute skill coverage metrics if requested
        if cfg.eval.get("compute_skill_coverage", True):
            logger.info("Computing skill coverage metrics for sentence-transformers evaluator...")
            skills_path = cfg.data.get("skills_path", "data/skills_per_occupations.csv")
            
            if Path(skills_path).exists():
                try:
                    # Reuse embeddings from the evaluator to avoid duplicate encoding
                    logger.info("Reusing embeddings from evaluator for skill coverage...")
                    esco_emb = evaluator.corpus_embeddings
                    job_emb = evaluator.query_embeddings
                    
                    # Compute rankings
                    scores = job_emb @ esco_emb.T
                    I = np.argsort(scores, axis=1)[:, ::-1]
                    
                    # Map gold IDs to rows
                    gold_ids = [p[ground_truth_col] for p in pairs]
                    gold_rows, _ = map_esco_id_to_row(gold_ids, esco_ids)
                    
                    # Compute skill coverage
                    skills_by_occupation = load_skills_per_occupation(skills_path)
                    skill_coverage_metrics = compute_skill_coverage(
                        I, gold_rows, esco_ids, skills_by_occupation,
                        ks=(1, 3, 5, 10)
                    )
                    
                    logger.info("Skill coverage metrics:")
                    pretty_print_metrics(skill_coverage_metrics)
                    
                    # Save skill coverage metrics
                    with open(results_dir / "skill_coverage_metrics.json", "w") as f:
                        json.dump(skill_coverage_metrics, f, indent=2)
                    
                except Exception as e:
                    logger.warning(f"Could not compute skill coverage metrics: {e}")
            else:
                logger.warning(f"Skills data file not found at {skills_path}. Skipping skill coverage metrics.")
        
        logger.info("Evaluation with sentence-transformers evaluator finished.")
        return


    # 2. Load and encode ESCO titles
    logger.info("Loading ESCO titles...")
    lowercase_setting = cfg.data.get("lowercase")
    lowercase_esco_flag = lowercase_setting in ["both", "esco"]
    esco_ids, esco_titles = load_esco_titles(
        cfg.data.esco_path, lowercase=lowercase_esco_flag
    )

    emb_path = cache_dir / "esco_emb.npy"
    ids_path = cache_dir / "esco_ids.json"

    if cfg.eval.save_embeddings and emb_path.exists() and ids_path.exists():
        logger.info("Loading cached ESCO embeddings...")
        esco_emb = np.load(emb_path)
        with open(ids_path) as f:
            cached_ids = json.load(f)
        if cached_ids == esco_ids:
            assert esco_emb.dtype == np.float32
        else:
            # Re-encode if IDs don't match
            logger.info("ESCO IDs mismatch, re-encoding...")
            esco_emb = model.encode_esco(
                esco_titles,
                batch_size=cfg.eval.batch_size,
                normalize=True,
                show_progress_bar=True,
            )
            np.save(emb_path, esco_emb)
            with open(ids_path, "w") as f:
                json.dump(esco_ids, f)
    else:
        logger.info("Encoding ESCO titles...")
        esco_emb = model.encode_esco(
            esco_titles,
            batch_size=cfg.eval.batch_size,
            normalize=True,
            show_progress_bar=True,
        )
        if cfg.eval.save_embeddings:
            np.save(emb_path, esco_emb)
            with open(ids_path, "w") as f:
                json.dump(esco_ids, f)

    assert esco_emb.dtype == np.float32

    # 3. Build or load FAISS index
    if cfg.eval.use_faiss:
        logger.info("Building or loading FAISS index...")
        index = build_or_load_faiss_index(
            esco_emb, cache_dir / "index.faiss", use_cache=cfg.eval.save_embeddings
        )

    # 4. Load pairs and encode job titles
    if cfg.eval.get("identity_test"):
        logger.info("Performing identity test. Using ESCO titles as queries.")
        job_texts = esco_titles
        gold_ids = esco_ids
    else:
        logger.info("Loading evaluation pairs...")
        lowercase_pairs_flag = lowercase_setting in ["both", "pairs"]
        pairs = load_pairs(
            cfg.data.pairs_path,
            lowercase_raw=lowercase_pairs_flag,
            lowercase_esco=lowercase_esco_flag,
        )
        job_texts = [p["job_title"] for p in pairs]
        
        # Use the configured ground truth column
        ground_truth_col = cfg.data.get("ground_truth_col", "esco_id")
        gold_ids = [p[ground_truth_col] for p in pairs]

    logger.info("Encoding job titles...")
    t_encode_start = time.monotonic()
    job_emb = model.encode_job(
        job_texts,
        batch_size=cfg.eval.batch_size,
        normalize=True,
        show_progress_bar=True,
    )
    t_encode_end = time.monotonic()
    assert job_emb.dtype == np.float32

    # 5. Retrieve top-k
    logger.info("Retrieving top-k candidates...")
    if cfg.eval.use_faiss:
        # Use exact FAISS flat index to retrieve full ranking for exact MAP/MRR
        k = len(esco_ids)
        _, I = search_faiss_index(index, job_emb, k)
    else:
        # Cosine similarity here
        scores = job_emb @ esco_emb.T
        # Get the full ranking
        I = np.argsort(scores, axis=1)[:, ::-1]

    # 6. Compute metrics
    logger.info("Computing metrics...")
    
    # Select the correct list for mapping based on the ground truth column
    ground_truth_col = cfg.data.get("ground_truth_col", "esco_id")
    if ground_truth_col == "esco_id":
        id_list_for_mapping = esco_ids
    else:
        id_list_for_mapping = esco_titles

    gold_rows, coverage = map_esco_id_to_row(gold_ids, id_list_for_mapping)

    if coverage < 0.95:
        logger.warning(
            f"Low coverage ({coverage:.2f}). "
            "This might indicate an ESCO version mismatch between your pairs and titles."
        )

    metrics = {}
    for metric_fn in METRICS:
        metrics.update(metric_fn(I, gold_rows))
    
    # 6a. Compute skill coverage metrics (only if ground truth is esco_id)
    if ground_truth_col == "esco_id" and cfg.eval.get("compute_skill_coverage", True):
        logger.info("Loading skills data and computing skill coverage metrics...")
        skills_path = cfg.data.get("skills_path", "data/skills_per_occupations.csv")
        
        if Path(skills_path).exists():
            try:
                skills_by_occupation = load_skills_per_occupation(skills_path)
                skill_coverage_metrics = compute_skill_coverage(
                    I, gold_rows, esco_ids, skills_by_occupation,
                    ks=(1, 3, 5, 10)
                )
                metrics.update(skill_coverage_metrics)
                logger.info(f"Skill coverage metrics computed: {list(skill_coverage_metrics.keys())}")
            except Exception as e:
                logger.warning(f"Could not compute skill coverage metrics: {e}")
        else:
            logger.warning(f"Skills data file not found at {skills_path}. Skipping skill coverage metrics.")

    N_eval = len(job_texts)
    encode_ms_per_query = (t_encode_end - t_encode_start) * 1000 / N_eval

    metrics.update({
        "coverage": coverage,
        "N_eval": N_eval,
        "encode_ms_per_query": encode_ms_per_query,
    })

    # 7. Add run details to metrics
    run_details = {
        "model_id": model_id,
        "data_path": cfg.data.pairs_path,
        "proj_dim": cfg.model.get("proj_dim"),
        "topk": cfg.eval.topk,
        "use_faiss": cfg.eval.use_faiss,
        "normalize_embeddings": True,  # Hardcoded in encoding calls
    }
    metrics["run_details"] = run_details


    # 8. Print and save results
    logger.info("Evaluation results:")
    pretty_print_metrics(metrics)

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if cfg.eval.save_predictions:
        logger.info("Saving predictions...")
        predictions = []
        for i in range(len(job_texts)):
            # Note: We save only top-k predictions regardless of I's size
            preds = [esco_ids[j] for j in I[i, :cfg.eval.topk]]
            
            # Use the configured ground truth column for saving predictions
            ground_truth_col = cfg.data.get("ground_truth_col", "esco_id")
            gold_id = gold_ids[i]
            
            predictions.append(
                {
                    "job_title": job_texts[i],
                    f"gold_{ground_truth_col}": gold_id,
                    "predicted_esco_ids": preds,
                }
            )
        with open(results_dir / "predictions.jsonl", "w") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")

    logger.info(f"Total evaluation time: {time.monotonic() - t_start:.2f}s")
    logger.info(f"Results saved to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a BiEncoder model.")
    parser.add_argument("--cfg", type=str, required=True, help="Path to config file.")
    args = parser.parse_args()
    config = load_config(args.cfg)
    main(config)

