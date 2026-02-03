import os
from typing import Dict, List, Tuple, Any
import sys
import json

import argparse
import random
import torch
from loguru import logger
from datetime import datetime

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from datasets import Dataset, load_dataset
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import util
from transformers import TrainerCallback

from src.cpp.skill_pooling import (
    load_skill_mappings,
    load_skill_descriptions,
    calculate_idf_scores,
    calculate_idf_scores_by_job_id,
    cap_skills_per_job_lexicographic,
    cap_skills_per_job_stratified,
    cap_skills_per_job_by_score,
    load_skills_by_job_id,
    load_raw_esco_taxonomy,
)
from src.cpp.utils import SEP_TOKEN
from src.cpp.data_classes import Data
default_output_dir = f"dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/skill-idf-finetune-{int(datetime.now().timestamp())}"


# Explicitly select the first GPU unless overridden externally
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Configure logging - stdout only initially, file handler added in main()
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time}</green> | <level>{message}</level>",
    level="INFO"
)


def setup_file_logging(output_dir: str) -> None:
    """Setup file logging to the output directory."""
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "training.log")
    logger.add(
        log_file,
        format="{time} | {level} | {message}",
        level="DEBUG",
        rotation="10 MB",
        retention="7 days",
        enqueue=True
    )
    logger.info(f"Logging to file: {log_file}")


class CustomTrainer(SentenceTransformerTrainer):
    """
    Custom trainer that logs how many data points have been processed.
    """

    def log(self, logs: dict, *args, **kwargs):
        logs["data_points_seen"] = (
            self.state.global_step * self.args.per_device_train_batch_size
        )
        super().log(logs, *args, **kwargs)


def _skills_to_doc(
    skills: List[Dict[str, Any]],
    skill_desc_map: Dict[str, Dict[str, str]],
    include_descriptions: bool = True,
) -> str:
    """
    Convert a list of skills into the document format.
    
    If include_descriptions is True:
        "skill: ... \\n description: ...<SEP>skill: ..."
    If include_descriptions is False:
        "skill_1<SEP>skill_2..."
    """
    segments = []
    for skill in skills:
        uri = skill.get("skillUri")
        skill_meta = skill_desc_map.get(uri, {})
        name = skill_meta.get("name") or skill.get("skill") or ""
        if include_descriptions:
            description = skill_meta.get("description") or ""
            segments.append(f"skill: {name} \n description: {description}")
        else:
            segments.append(name)
    return SEP_TOKEN.join(segments)


def print_sample_debug(
    data_pairs: List[Tuple[str, str]],
    job_ids_list: List[List[str]],
    job_skill_map: Dict[str, List[Dict[str, Any]]],
    skill_desc_map: Dict[str, Dict[str, str]],
    skill_pairs: List[Tuple[str, str]],
    scoring_mode: str,
    include_skill_descriptions: bool,
    sample_idx: int = 0,
) -> None:
    """
    Print detailed debug information for a sample to verify the pipeline.
    
    Shows: job_id, original job info, skills with scores, final doc1/doc2.
    """
    logger.info("=" * 80)
    logger.info("SAMPLE DEBUG OUTPUT")
    logger.info("=" * 80)
    
    if sample_idx >= len(data_pairs):
        logger.warning(f"Sample index {sample_idx} out of range (max: {len(data_pairs)-1})")
        return
    
    doc1_original, doc2 = data_pairs[sample_idx]
    job_ids = job_ids_list[sample_idx] if sample_idx < len(job_ids_list) else []
    
    logger.info(f"\n[1] ORIGINAL DATA (sample {sample_idx}):")
    logger.info(f"    Job IDs in career path: {job_ids}")
    logger.info(f"    Number of jobs: {len(job_ids)}")
    
    # Show original doc1 (truncated)
    doc1_preview = doc1_original[:250] + "..." + doc1_original[-250:] if len(doc1_original) > 500 else doc1_original
    logger.info(f"    Original doc1 (career history):\n{doc1_preview}")
    last_job = doc1_original.split(SEP_TOKEN)[-1]
    logger.info(f"    Last job: {last_job}")
    
    logger.info(f"\n[2] TARGET OCCUPATION (doc2):")
    logger.info(f"    {doc2}")
    
    # Get the last job_id (the one we use for skills)
    if job_ids:
        last_job_id = str(job_ids[-1]) if job_ids[-1] is not None else None
        logger.info(f"\n[3] LAST JOB (used for skill lookup):")
        logger.info(f"    Job ID: {last_job_id}")
        
        if last_job_id and last_job_id in job_skill_map:
            skills = job_skill_map[last_job_id]
            logger.info(f"    Number of skills (after capping): {len(skills)}")
            logger.info(f"\n[4] TOP-K SKILLS (scoring_mode={scoring_mode}):")
            
            for i, skill in enumerate(skills[:10]):  # Show first 10
                uri = skill.get("skillUri", "")
                skill_meta = skill_desc_map.get(uri, {})
                name = skill_meta.get("name", skill.get("skill", "unknown"))
                
                # Show different score fields based on what's available
                score_info = []
                if "score" in skill:
                    score_info.append(f"pred_score={skill['score']:.4f}")
                if "idf" in skill:
                    score_info.append(f"idf={skill['idf']:.4f}")
                if "weighted_idf" in skill:
                    score_info.append(f"weighted_idf={skill['weighted_idf']:.4f}")
                
                score_str = ", ".join(score_info) if score_info else "no scores"
                logger.info(f"    [{i+1}] {name}")
                logger.info(f"        URI: {uri}")
                logger.info(f"        Scores: {score_str}")
            
            if len(skills) > 10:
                logger.info(f"    ... and {len(skills) - 10} more skills")
        else:
            logger.warning(f"    Job ID {last_job_id} not found in job_skill_map!")
    else:
        logger.warning("    No job_ids available for this sample")
    
    # Show the final skill_pairs output
    if sample_idx < len(skill_pairs):
        skill_doc1, skill_doc2 = skill_pairs[sample_idx]
        
        logger.info(f"\n[5] FINAL OUTPUT (skill-based pair):")
        logger.info(f"    Include descriptions: {include_skill_descriptions}")
        
        # Truncate for display
        doc1_display = skill_doc1[:1000] + "..." if len(skill_doc1) > 1000 else skill_doc1
        logger.info(f"\n    doc1 (skills of last job):\n{doc1_display}")
        logger.info(f"\n    doc2 (target occupation):\n{skill_doc2}")
        logger.info(f"\n    doc1 length: {len(skill_doc1)} chars")
        logger.info(f"    doc2 length: {len(skill_doc2)} chars")
    else:
        logger.warning(f"    Sample {sample_idx} not found in skill_pairs (possibly filtered out)")
    
    logger.info("=" * 80)


def _normalize_title(title: str) -> str:
    return str(title).strip().lower()


def _select_language_fields(language: str, group) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Returns (source_titles_for_skills, source_descriptions, esco_titles, esco_descriptions)
    """
    if language in ("en", "esco_100k"):
        titles = group["preferredLabel_en"].tolist()
        descriptions = group["description_en"].tolist()
        return titles, descriptions, titles, descriptions
    elif language == "en_free":
        titles_esco = group["preferredLabel_en"].tolist()
        descriptions_esco = group["description_en"].tolist()
        descriptions_free = group["new_job_description_en_occ"].tolist()
        return titles_esco, descriptions_free, titles_esco, descriptions_esco
    elif language == "en_free_cp":
        titles_esco = group["preferredLabel_en"].tolist()
        descriptions_esco = group["description_en"].tolist()
        descriptions_free = group["new_job_description_en_cp"].tolist()
        return titles_esco, descriptions_free, titles_esco, descriptions_esco
    else:
        raise ValueError(f"Unsupported language: {language}")


def _build_pairs_for_split(
    split,
    language: str,
    job_skill_map: Dict[str, List[Dict[str, Any]]],
    skill_desc_map: Dict[str, Dict[str, str]],
    include_skill_descriptions: bool = True,
) -> List[Tuple[str, str]]:
    """
    Build (doc1, doc2) pairs for a dataset split.
    doc1: top-k skills (already capped) of the last observed job
    doc2: next ESCO occupation (always includes description)
    """
    pairs: List[Tuple[str, str]] = []
    df = split.to_pandas()
    grouped = df.groupby("_id")

    for _, group in grouped:
        group = group.sort_values("experience_order")
        (
            titles_for_skills,
            _,
            target_titles_esco,
            target_desc_esco,
        ) = _select_language_fields(language, group)

        # need at least 2 jobs to form a transition
        if len(target_titles_esco) < 2:
            continue

        for idx in range(len(target_titles_esco) - 1):
            source_title_norm = _normalize_title(titles_for_skills[idx])
            skills = job_skill_map.get(source_title_norm)
            if not skills:
                continue

            doc1 = _skills_to_doc(skills, skill_desc_map, include_skill_descriptions)
            if not doc1:
                continue

            # Target job description is always included
            doc2 = f"esco role: {target_titles_esco[idx + 1]} \n description: {target_desc_esco[idx + 1]}"
            pairs.append((doc1, doc2))

    return pairs


def load_skill_transition_pairs(
    data_type: str,
    job_skill_map: Dict[str, List[Dict[str, Any]]],
    skill_desc_map: Dict[str, Dict[str, str]],
    include_skill_descriptions: bool = True,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Load the career-path dataset and build skill-based transition pairs.
    """
    if data_type == "karrierewege":
        dataset_name = "ElenaSenger/Karrierewege"
        language = "en"
    elif data_type == "karrierewege_occ":
        dataset_name = "ElenaSenger/Karrierewege_plus"
        language = "en_free"
    elif data_type == "karrierewege_100k":
        dataset_name = "ElenaSenger/Karrierewege_plus"
        language = "esco_100k"
    elif data_type == "karrierewege_cp":
        dataset_name = "ElenaSenger/Karrierewege_plus"
        language = "en_free_cp"
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")

    dataset = load_dataset(dataset_name)

    train_pairs = _build_pairs_for_split(dataset["train"], language, job_skill_map, skill_desc_map, include_skill_descriptions)
    val_pairs = _build_pairs_for_split(dataset["validation"], language, job_skill_map, skill_desc_map, include_skill_descriptions)
    test_pairs = _build_pairs_for_split(dataset["test"], language, job_skill_map, skill_desc_map, include_skill_descriptions)

    return train_pairs, val_pairs, test_pairs


def _build_pairs_from_job_ids(
    data_pairs: List[Tuple[str, str]],
    job_ids_list: List[List[str]],
    job_skill_map: Dict[str, List[Dict[str, Any]]],
    skill_desc_map: Dict[str, Dict[str, str]],
    include_skill_descriptions: bool = True,
) -> List[Tuple[str, str]]:
    """
    Build (doc1, doc2) pairs using job_ids to look up skills.
    
    Args:
        data_pairs: List of (doc1, doc2) pairs from Data class (minus_last applied)
        job_ids_list: List of job_id lists, one per data pair
        job_skill_map: Dict mapping job_id -> list of skills
        skill_desc_map: Dict mapping skillUri -> {name, description}
        include_skill_descriptions: Whether to include skill descriptions in doc1
    
    Returns:
        List of (skill_doc, target_doc) pairs
    """
    pairs: List[Tuple[str, str]] = []
    missing_job_ids = 0
    missing_skills = 0
    
    for (doc1, doc2), job_ids in zip(data_pairs, job_ids_list):
        if not job_ids:
            missing_job_ids += 1
            continue
        
        # Use the last job_id in the list (most recent job before target)
        last_job_id = job_ids[-1] if job_ids else None
        if last_job_id is None:
            missing_job_ids += 1
            continue
        
        last_job_id_str = str(last_job_id)
        skills = job_skill_map.get(last_job_id_str)
        if not skills:
            missing_skills += 1
            continue
        
        skill_doc = _skills_to_doc(skills, skill_desc_map, include_skill_descriptions)
        if not skill_doc:
            continue
        
        pairs.append((skill_doc, doc2))
    
    if missing_job_ids > 0:
        logger.warning(f"Missing job_ids for {missing_job_ids} samples")
    if missing_skills > 0:
        logger.warning(f"Missing skills for {missing_skills} job_ids")
    
    return pairs


def load_skill_transition_pairs_with_job_ids(
    data_type: str,
    job_skill_map: Dict[str, List[Dict[str, Any]]],
    skill_desc_map: Dict[str, Dict[str, str]],
    include_skill_descriptions: bool = True,
    consider_subspans: bool = True,
    return_debug_info: bool = False,
) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Load dataset using Data class and build skill-based transition pairs using job_ids.
    
    This function uses the Data class which provides job_ids for each sample,
    allowing direct lookup of skills from fused_predictions.json by job_id.
    
    Args:
        data_type: Dataset type (decorte, decorte_esco, karrierewege, etc.)
        job_skill_map: Dict mapping job_id -> list of skills (from load_skills_by_job_id)
        skill_desc_map: Dict mapping skillUri -> {name, description}
        include_skill_descriptions: Whether to include skill descriptions in doc1
        consider_subspans: Whether to consider subspans for data augmentation
        return_debug_info: If True, also return raw pairs and job_ids for debugging
    
    Returns:
        If return_debug_info is False:
            Tuple of (train_pairs, val_pairs, test_pairs)
        If return_debug_info is True:
            Tuple of (train_pairs, val_pairs, test_pairs, debug_info)
            where debug_info = (raw_train_pairs, raw_train_job_ids)
    """
    logger.info(f"Loading data using Data class with job_ids (data_type={data_type})...")
    
    data = Data(
        DATA_TYPE=data_type,
        consider_subspans=consider_subspans,
    )
    
    # Get data with job_ids, using transformation_finetuning stage (applies minus_last)
    (train_pairs, train_job_ids), (val_pairs, val_job_ids), (test_pairs, test_job_ids) = \
        data.get_data_with_job_ids(stage="transformation_finetuning")
    
    logger.info(f"Loaded {len(train_pairs)} train, {len(val_pairs)} val, {len(test_pairs)} test samples")
    
    # Build skill-based pairs using job_ids
    train_skill_pairs = _build_pairs_from_job_ids(
        train_pairs, train_job_ids, job_skill_map, skill_desc_map, include_skill_descriptions
    )
    val_skill_pairs = _build_pairs_from_job_ids(
        val_pairs, val_job_ids, job_skill_map, skill_desc_map, include_skill_descriptions
    )
    test_skill_pairs = _build_pairs_from_job_ids(
        test_pairs, test_job_ids, job_skill_map, skill_desc_map, include_skill_descriptions
    )
    
    if return_debug_info:
        debug_info = (train_pairs, train_job_ids)
        return train_skill_pairs, val_skill_pairs, test_skill_pairs, debug_info
    
    return train_skill_pairs, val_skill_pairs, test_skill_pairs


def construct_evaluator(valid_pairs: List[Tuple[str, str]]):
    """
    Build an EmbeddingSimilarityEvaluator with in-batch negatives.
    """
    valid_input_pairs = []
    for a, p in valid_pairs:
        valid_input_pairs.append(InputExample(texts=[a, p], label=1))
        r_a, r_p = random.choice(valid_pairs)
        valid_input_pairs.append(InputExample(texts=[a, r_p], label=0))

    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        valid_input_pairs, write_csv=False, show_progress_bar=False
    )

    return evaluator


def analyze_token_lengths(
    model: SentenceTransformer,
    pairs: List[Tuple[str, str]],
    desc: str,
) -> None:
    """
    Analyze token lengths of inputs to check for truncation.
    """
    if not pairs:
        return

    logger.info(f"Analyzing token lengths for {desc} ({len(pairs)} pairs)...")
    max_len = model.max_seq_length
    tokenizer = model.tokenizer

    doc1_truncated = 0
    doc2_truncated = 0
    doc1_lengths = []
    doc2_lengths = []

    # Sample if too large to avoid taking too much time
    sample_size = 5000
    if len(pairs) > sample_size:
        logger.info(f"Sampling {sample_size} pairs for analysis...")
        pairs_to_analyze = random.sample(pairs, sample_size)
    else:
        pairs_to_analyze = pairs

    for doc1, doc2 in pairs_to_analyze:
        # Tokenize without truncation to get real length
        # Note: SentenceTransformer usually adds special tokens, so we should account for that
        # by checking how it tokenizes. But using tokenizer directly gives a good estimate.
        # We need to use the tokenizer call that mimics ST's encode but just for length.
        # ST's encode uses: self.tokenize(sentences)
        
        # We can use tokenizer directly. 
        # For BERT-like models: [CLS] text [SEP] -> length is len(ids)
        t1 = tokenizer(doc1, truncation=False, padding=False)["input_ids"]
        t2 = tokenizer(doc2, truncation=False, padding=False)["input_ids"]
        
        doc1_lengths.append(len(t1))
        doc2_lengths.append(len(t2))

        if len(t1) > max_len:
            doc1_truncated += 1
        if len(t2) > max_len:
            doc2_truncated += 1

    avg_len1 = sum(doc1_lengths) / len(doc1_lengths)
    avg_len2 = sum(doc2_lengths) / len(doc2_lengths)
    
    logger.info(f"--- Token Length Analysis ({desc}) ---")
    logger.info(f"Model max length: {max_len}")
    logger.info(f"Doc1 (Anchor) - Avg Len: {avg_len1:.1f}, Max Len: {max(doc1_lengths)}")
    logger.info(f"Doc1 Truncated: {doc1_truncated}/{len(pairs_to_analyze)} ({doc1_truncated/len(pairs_to_analyze)*100:.2f}%)")
    logger.info(f"Doc2 (Positive) - Avg Len: {avg_len2:.1f}, Max Len: {max(doc2_lengths)}")
    logger.info(f"Doc2 Truncated: {doc2_truncated}/{len(pairs_to_analyze)} ({doc2_truncated/len(pairs_to_analyze)*100:.2f}%)")
    
    if WANDB_AVAILABLE and wandb.run is not None:
        wandb.log({
            f"{desc}/doc1_truncation_rate": doc1_truncated/len(pairs_to_analyze),
            f"{desc}/doc2_truncation_rate": doc2_truncated/len(pairs_to_analyze),
            f"{desc}/doc1_avg_len": avg_len1,
            f"{desc}/doc2_avg_len": avg_len2,
        })


def evaluate_retrieval(
    model: SentenceTransformer,
    eval_pairs: List[Tuple[str, str]],
    candidate_texts: List[str] | None = None,
    batch_size: int = 32,
    k_values: Tuple[int, int] = (5, 10),
    desc: str = "test",
) -> Dict[str, float]:
    """
    Evaluate retrieval performance by comparing each doc1 embedding against a candidate pool of doc2 embeddings.

    Metrics:
    - Recall@k for k in k_values
    - Mean Reciprocal Rank (MRR)
    """
    if not eval_pairs:
        logger.warning(f"No evaluation pairs for {desc}; skipping retrieval evaluation.")
        return {}

    # Build corpus of unique doc2 texts.
    # If candidate_texts is provided, it is used as the retrieval candidate pool (e.g. train+val+test).
    # Otherwise, we fall back to only doc2 texts observed in eval_pairs.
    if candidate_texts is not None:
        corpus_texts = list(dict.fromkeys(candidate_texts))
    else:
        corpus_texts = list(dict.fromkeys([p for _, p in eval_pairs]))
    corpus_index = {text: idx for idx, text in enumerate(corpus_texts)}

    # Align queries (doc1) with indices of their true doc2 in the corpus
    query_texts: List[str] = []
    true_indices: List[int] = []
    for a, p in eval_pairs:
        idx = corpus_index.get(p)
        if idx is None:
            continue
        query_texts.append(a)
        true_indices.append(idx)

    if not query_texts:
        logger.warning(f"No valid query-doc2 pairs for {desc}; skipping retrieval evaluation.")
        return {}

    # Encode corpus once
    corpus_embeddings = model.encode(
        corpus_texts,
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=False,
    )

    ranks: List[int] = []
    total_queries = len(query_texts)

    for start_idx in range(0, total_queries, batch_size):
        end_idx = min(start_idx + batch_size, total_queries)
        batch_queries = query_texts[start_idx:end_idx]

        query_embeddings = model.encode(
            batch_queries,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

        # Cosine similarity between each query and all corpus embeddings
        cos_scores = util.cos_sim(query_embeddings, corpus_embeddings)  # [B, C]

        for i in range(cos_scores.size(0)):
            true_idx = true_indices[start_idx + i]
            scores_row = cos_scores[i]

            # Rank is 1 + number of items with strictly higher similarity
            sorted_indices = torch.argsort(scores_row, descending=True)
            match_pos = (sorted_indices == true_idx).nonzero(as_tuple=False)
            if match_pos.numel() == 0:
                continue
            rank = int(match_pos[0].item()) + 1
            ranks.append(rank)

    if not ranks:
        logger.warning(f"No ranks could be computed for {desc}; skipping metrics.")
        return {}

    ranks_tensor = torch.tensor(ranks, dtype=torch.float32)
    metrics: Dict[str, float] = {}
    for k in k_values:
        metrics[f"R@{k}"] = float((ranks_tensor <= k).float().mean().item())
    metrics["MRR"] = float((1.0 / ranks_tensor).mean().item())

    logger.info(f"Retrieval evaluation ({desc}) on {len(ranks)} queries:")
    logger.info(f"  Candidate pool size: {len(corpus_texts)} unique doc2 texts")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    return metrics


class RetrievalEvalCallback(TrainerCallback):
    """
    HuggingFace Trainer callback to run retrieval evaluation after each epoch.
    """

    def __init__(
        self,
        eval_pairs: List[Tuple[str, str]],
        candidate_texts: List[str] | None,
        batch_size: int,
    ):
        super().__init__()
        self.eval_pairs = eval_pairs
        self.candidate_texts = candidate_texts
        self.batch_size = batch_size

    def on_epoch_end(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is None:
            return control

        epoch_num = int(state.epoch) if state.epoch is not None else -1
        logger.info(f"Running retrieval evaluation after epoch {epoch_num}...")
        metrics = evaluate_retrieval(
            model,
            self.eval_pairs,
            candidate_texts=self.candidate_texts,
            batch_size=self.batch_size,
            k_values=(5, 10),
            desc=f"test (epoch {epoch_num})",
        )
        
        # Push to wandb if available
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log({f"eval_epoch/{k}": v for k, v in metrics.items()}, step=state.global_step)
            
        return control


def fine_tune(train_pairs, valid_pairs, test_pairs, model, training_config, candidate_texts: List[str] | None = None):
    """
    Train a SentenceTransformer on the provided pairs using MNRL.
    """
    train_dataset = Dataset.from_dict(
        {
            "anchor": [a for a, _ in train_pairs],
            "positive": [p for _, p in train_pairs],
        }
    )
    eval_dataset = Dataset.from_dict(
        {
            "anchor": [a for a, p in valid_pairs],
            "positive": [p for a, p in valid_pairs],
        }
    )

    loss = losses.MultipleNegativesRankingLoss(model, scale=20.0)

    report_to = (
        ["wandb"]
        if WANDB_AVAILABLE and getattr(training_config, "use_wandb", False)
        else []
    )

    eval_save_steps = max(
        1,
        int(
            training_config.epoch_eval_frac
            * len(train_dataset)
            / training_config.batch_size
        ),
    )
    logger.info(f"Eval and save steps interval: {eval_save_steps}")

    args = SentenceTransformerTrainingArguments(
        output_dir=training_config.output_dir,
        num_train_epochs=training_config.epochs,
        per_device_train_batch_size=training_config.batch_size,
        per_device_eval_batch_size=training_config.batch_size,
        learning_rate=training_config.learning_rate,
        fp16=not training_config.disable_fp16,
        dataloader_num_workers=training_config.dataloader_num_workers,
        bf16=False,
        max_grad_norm=1.0,
        eval_strategy="steps",
        eval_steps=eval_save_steps,
        report_to=report_to,
        save_strategy="steps",
        save_steps=eval_save_steps,
        save_total_limit=2,
        logging_steps=eval_save_steps,
        run_name=getattr(training_config, "run_name", "skill-idf"),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Configure LR scheduler based on CLI / training config (e.g. "linear")
    args = args.set_lr_scheduler(name=training_config.scheduler, warmup_ratio=0.05)
    args = args.set_training(
        learning_rate=training_config.learning_rate,
        batch_size=training_config.batch_size,
        weight_decay=0,
        num_epochs=training_config.epochs,
        max_steps=-1,
        gradient_accumulation_steps=training_config.gradient_accumulation_steps,
        seed=42,
        gradient_checkpointing=False,
    )

    evaluator = construct_evaluator(valid_pairs)

    callbacks = []
    test_strategy = getattr(training_config, "test_strategy", "final")
    if test_strategy == "epoch" and test_pairs:
        callbacks.append(
            RetrievalEvalCallback(
                eval_pairs=test_pairs,
                candidate_texts=candidate_texts,
                batch_size=training_config.batch_size,
            )
        )

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        args=args,
        evaluator=evaluator,
        callbacks=callbacks,
    )

    if WANDB_AVAILABLE and getattr(training_config, "use_wandb", False):
        wandb.watch(model, log="all", log_freq=eval_save_steps)

    trainer.train()
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune embeddings on last-job skills (IDF top-k) -> next ESCO occupation."
    )
    parser.add_argument(
        "--data_type", 
        type=str, 
        default="karrierewege_100k",
        choices=["karrierewege", "karrierewege_occ", "karrierewege_100k", "karrierewege_cp",
                 "decorte", "decorte_esco"],
        help="Dataset type. Use 'decorte' or 'decorte_esco' for job_id-based skill lookup."
    )
    parser.add_argument(
        "--job_title_skills_csv", 
        type=str, 
        default=None,
        help="Path to CSV with job_title -> skills mapping. Required for idf_only mode with karrierewege datasets unless --raw_esco_dir is provided."
    )
    parser.add_argument("--skills_csv", type=str, required=False, help="Required unless --raw_esco_dir is provided.")
    parser.add_argument("--raw_esco_dir", type=str, default=None, help="Path to directory containing raw ESCO taxonomy files.")
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--output_dir", type=str, default=default_output_dir)
    parser.add_argument("--top_k_skills", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epoch_eval_frac", type=float, default=0.1)
    parser.add_argument("--disable_fp16", action="store_true")
    parser.add_argument("--scheduler", type=str, default="linear")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument(
        "--test_strategy",
        type=str,
        default="final",
        choices=["final", "epoch", "none"],
        help="When to run retrieval evaluation: 'final', 'epoch', or 'none'.",
    )
    parser.add_argument(
        "--test_base_model",
        action="store_true",
        help="If set, evaluate the base encoder before fine-tuning.",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="skill-idf-finetune",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity name (optional)",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="skill-idf",
        help="Run name for logging / W&B",
    )
    parser.add_argument(
        "--dataloader_num_workers", type=int, default=0 
    )
    parser.add_argument(
        "--no_skill_descriptions",
        action="store_true",
        help="Exclude skill descriptions from doc1. Encodes as 'skill: skill_1 [SEP] skill: skill_2...'. "
             "Target job description is always included.",
    )
    parser.add_argument(
        "--skill_selection_strategy",
        type=str,
        default="top_k",
        choices=["top_k", "stratified"],
        help="Strategy to select skills: 'top_k' (highest score/IDF) or 'stratified' (sample from high/mid/low tiers).",
    )
    parser.add_argument(
        "--stratified_sampling_basis",
        type=str,
        default="scoring_mode",
        choices=["scoring_mode", "idf_only"],
        help="When --skill_selection_strategy=stratified, choose what score source defines the strata. "
             "'scoring_mode' (default) uses the current scoring_mode behavior (scores_only→score, weighted→weighted_idf, idf_only→idf). "
             "'idf_only' always stratifies by IDF (even if scoring_mode is scores_only/weighted).",
    )
    parser.add_argument(
        "--skill_scores_json",
        type=str,
        default=None,
        help="Path to fused_predictions.json with skill predictions per job_id. "
             "Required for 'decorte' data_type and 'scores_only'/'weighted' scoring modes.",
    )
    parser.add_argument(
        "--importance_weight",
        type=float,
        default=0.5,
        help="Weight for per-job scores in weighted scoring (0-1). "
             "0.0 = only IDF, 1.0 = only per-job scores. Default: 0.5. "
             "Only used with --scoring_mode weighted.",
    )
    parser.add_argument(
        "--scoring_mode",
        type=str,
        default="scores_only",
        choices=["idf_only", "scores_only", "weighted"],
        help="How to score/rank skills for selection: "
             "'idf_only' = use only IDF (score=1 for ESCO-based datasets), "
             "'scores_only' = use only per-job prediction scores (decorte only), "
             "'weighted' = combine IDF and per-job scores. Default: scores_only",
    )
    parser.add_argument(
        "--consider_subspans",
        action="store_true",
        help="Consider all subspans of length >= 2 for data augmentation (Data class feature).",
    )
    parser.add_argument(
        "--print_sample",
        action="store_true",
        help="Print a detailed debug sample showing job_id, skills, scores, and final doc1/doc2.",
    )
    parser.add_argument(
        "--save_model",
        action='store_true',
        help="Save or not save basically."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup file logging to output directory
    setup_file_logging(args.output_dir)
    
    # Datasets with free-text titles -> skills are PREDICTED with scores per job_id
    # These use job_id-based skill loading from fused_predictions.json
    FREE_TEXT_DATASETS = ("decorte", "karrierewege_occ", "karrierewege_cp")
    
    # Datasets with ESCO titles -> skills come from ESCO taxonomy (score=1 for all)
    # These use job_title-based skill loading from CSV
    ESCO_TITLE_DATASETS = ("decorte_esco", "karrierewege", "karrierewege_100k")
    
    uses_predicted_skills = args.data_type in FREE_TEXT_DATASETS
    uses_esco_taxonomy = args.data_type in ESCO_TITLE_DATASETS
    
    # Validate dataset type
    if not uses_predicted_skills and not uses_esco_taxonomy:
        raise ValueError(f"Unknown data_type: {args.data_type}")
    
    # Validate required arguments based on mode
    if uses_predicted_skills:
        # Free-text datasets require skill_scores_json for predicted skills
        if not args.skill_scores_json:
            raise ValueError(f"--skill_scores_json is required for {args.data_type} (free-text dataset with predicted skills)")
    elif not args.raw_esco_dir:
        # ESCO datasets require job_title_skills_csv and skills_csv for taxonomy skills if no raw dir
        if not args.job_title_skills_csv:
            raise ValueError(f"--job_title_skills_csv is required for {args.data_type} (ESCO-based dataset) if --raw_esco_dir is not provided")
        if not args.skills_csv:
            raise ValueError(f"--skills_csv is required for {args.data_type} if --raw_esco_dir is not provided")

    if WANDB_AVAILABLE and args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=args.run_name,
        )
        logger.info(f"W&B logging enabled for run: {args.run_name}")
    elif args.use_wandb and not WANDB_AVAILABLE:
        logger.warning("wandb is not installed; proceeding without W&B logging.")

    logger.info(f"Scoring mode: {args.scoring_mode}, Data type: {args.data_type}")
    logger.info(f"Dataset type: {'FREE-TEXT (predicted skills)' if uses_predicted_skills else 'ESCO (taxonomy skills)'}")
    if args.raw_esco_dir and uses_esco_taxonomy:
        logger.info(f"Loading ESCO taxonomy directly from {args.raw_esco_dir}...")
        job_skill_map, skill_desc_map = load_raw_esco_taxonomy(args.raw_esco_dir)
    else:
        logger.info("Loading skill descriptions...")
        skill_desc_map = load_skill_descriptions(args.skills_csv) if args.skills_csv else {}

    # Determine whether to include skill descriptions
    include_skill_descriptions = not args.no_skill_descriptions
    
    # ==================================================================
    # FREE-TEXT MODE (decorte, karrierewege_occ, karrierewege_cp)
    # Skills are predicted per job_id with scores from fused_predictions.json
    # ==================================================================
    if uses_predicted_skills:
        logger.info(f"Using job_id-based skill loading for {args.data_type} (free-text titles)...")
        
        if args.scoring_mode == "scores_only":
            # Load skills directly by job_id from fused_predictions.json
            logger.info(f"Loading skills by job_id from {args.skill_scores_json}...")
            job_skill_map = load_skills_by_job_id(args.skill_scores_json)
            
            # Apply top-k or stratified selection based on prediction scores
            if args.skill_selection_strategy == "stratified":
                # Allow overriding the stratification basis to IDF even in scores_only mode
                stratified_score_source = "score"
                added_idf_for_stratification_only = False
                if args.stratified_sampling_basis == "idf_only":
                    logger.info(
                        "Computing IDF scores for stratified sampling override "
                        "(scoring_mode=scores_only, stratified_sampling_basis=idf_only)..."
                    )
                    job_skill_map = calculate_idf_scores_by_job_id(
                        job_skill_map, use_job_scores=False
                    )
                    stratified_score_source = "idf"
                    # Important: scores_only mode historically does NOT attach IDF,
                    # which keeps pooling weights uniform (default=1.0). We therefore
                    # remove the temporary IDF field after stratified selection.
                    added_idf_for_stratification_only = True

                logger.info(
                    f"Stratified sampling basis: {args.stratified_sampling_basis} "
                    f"(effective score_source={stratified_score_source})"
                )
                logger.info(
                    f"Using stratified sampling (target: {args.top_k_skills} skills)..."
                )
                job_skill_map = cap_skills_per_job_stratified(
                    job_skill_map,
                    max_skills_per_job=args.top_k_skills,
                    score_source=stratified_score_source,
                )
                if added_idf_for_stratification_only:
                    for _, skills in job_skill_map.items():
                        for s in skills:
                            s.pop("idf", None)
            else:
                logger.info(f"Using top-k selection by score (k={args.top_k_skills})...")
                job_skill_map = cap_skills_per_job_by_score(
                    job_skill_map,
                    max_skills_per_job=args.top_k_skills,
                    skill_desc_map=skill_desc_map,
                )
        
        elif args.scoring_mode == "weighted":
            # Load skills by job_id, then compute IDF and combine with per-job scores
            logger.info(f"Loading skills by job_id from {args.skill_scores_json}...")
            job_skill_map = load_skills_by_job_id(args.skill_scores_json)
            
            # Calculate IDF and combine with per-job prediction scores
            # Note: We use the score already in each skill dict (per-job), NOT global aggregated scores
            logger.info(f"Computing weighted_idf with per-job scores (importance_weight={args.importance_weight})...")
            job_skill_map = calculate_idf_scores_by_job_id(
                job_skill_map,
                use_job_scores=True,
                importance_weight=args.importance_weight,
            )
            
            if args.skill_selection_strategy == "stratified":
                stratified_score_source = (
                    "idf" if args.stratified_sampling_basis == "idf_only" else "weighted_idf"
                )
                logger.info(
                    f"Stratified sampling basis: {args.stratified_sampling_basis} "
                    f"(effective score_source={stratified_score_source})"
                )
                logger.info(
                    f"Using stratified sampling (target: {args.top_k_skills} skills)..."
                )
                job_skill_map = cap_skills_per_job_stratified(
                    job_skill_map,
                    max_skills_per_job=args.top_k_skills,
                    score_source=stratified_score_source,
                )
            else:
                logger.info(f"Using top-k selection by weighted_idf (k={args.top_k_skills})...")
                job_skill_map = cap_skills_per_job_lexicographic(
                    job_skill_map,
                    max_skills_per_job=args.top_k_skills,
                    skill_desc_map=skill_desc_map,
                    use_weighted_idf=True,
                )
        
        else:  # idf_only with decorte
            # For idf_only with decorte, we still need skills by job_id
            # but we'll use IDF-only for ranking (ignore per-job prediction scores)
            if args.skill_scores_json:
                logger.info(f"Loading skills by job_id from {args.skill_scores_json}...")
                job_skill_map = load_skills_by_job_id(args.skill_scores_json)
            else:
                raise ValueError("--skill_scores_json is required for decorte datasets to get skills by job_id")
            
            logger.info("Computing IDF scores (idf_only mode, ignoring prediction scores)...")
            job_skill_map = calculate_idf_scores_by_job_id(job_skill_map, use_job_scores=False)
            
            if args.skill_selection_strategy == "stratified":
                logger.info(
                    f"Stratified sampling basis: {args.stratified_sampling_basis} "
                    "(effective score_source=idf)"
                )
                logger.info(f"Using stratified sampling by IDF (target: {args.top_k_skills} skills)...")
                job_skill_map = cap_skills_per_job_stratified(
                    job_skill_map,
                    max_skills_per_job=args.top_k_skills,
                    score_source="idf",
                )
            else:
                logger.info(f"Using top-k selection by IDF (k={args.top_k_skills})...")
                job_skill_map = cap_skills_per_job_lexicographic(
                    job_skill_map,
                    max_skills_per_job=args.top_k_skills,
                    skill_desc_map=skill_desc_map,
                    use_weighted_idf=False,
                )
        
        # Build pairs using job_ids
        if include_skill_descriptions:
            logger.info("Building skill-based transition pairs (with skill descriptions)...")
        else:
            logger.info("Building skill-based transition pairs (without skill descriptions)...")
        
        result = load_skill_transition_pairs_with_job_ids(
            args.data_type,
            job_skill_map,
            skill_desc_map,
            include_skill_descriptions,
            consider_subspans=args.consider_subspans,
            return_debug_info=args.print_sample,
        )
        
        if args.print_sample:
            train_pairs, val_pairs, test_pairs, (raw_train_pairs, raw_train_job_ids) = result
            # Print sample debug output
            print_sample_debug(
                data_pairs=raw_train_pairs,
                job_ids_list=raw_train_job_ids,
                job_skill_map=job_skill_map,
                skill_desc_map=skill_desc_map,
                skill_pairs=train_pairs,
                scoring_mode=args.scoring_mode,
                include_skill_descriptions=include_skill_descriptions,
                sample_idx=0,
            )
        else:
            train_pairs, val_pairs, test_pairs = result
    
    # ==================================================================
    # ESCO TITLE MODE (decorte_esco, karrierewege, karrierewege_100k)
    # Skills come from ESCO taxonomy with score=1 (no prediction needed)
    # ==================================================================
    else:
        logger.info(f"Using job_title-based skill loading for {args.data_type} (ESCO titles)...")
        logger.info("Note: ESCO-based datasets use taxonomy skills with score=1 (no prediction)")
        
        if not args.raw_esco_dir:
            logger.info("Loading skill mappings from CSV...")
            job_skill_map = load_skill_mappings(args.job_title_skills_csv)

        # For ESCO-based datasets, we only use IDF for ranking (scores are all 1.0)
        # weighted/scores_only modes don't apply since there are no prediction scores
        if args.scoring_mode != "idf_only":
            logger.warning(f"scoring_mode={args.scoring_mode} ignored for ESCO-based datasets. Using IDF only.")

        logger.info("Computing IDF scores...")
        job_skill_map = calculate_idf_scores(job_skill_map)
        
        if args.skill_selection_strategy == "stratified":
            if args.stratified_sampling_basis != "idf_only":
                logger.info(
                    f"Stratified sampling basis: {args.stratified_sampling_basis} ignored for ESCO-based datasets "
                    "(forcing score_source=idf)."
                )
            else:
                logger.info(
                    f"Stratified sampling basis: {args.stratified_sampling_basis} (effective score_source=idf)"
                )
            logger.info(f"Using stratified sampling by IDF (target: {args.top_k_skills} skills)...")
            job_skill_map = cap_skills_per_job_stratified(
                job_skill_map,
                max_skills_per_job=args.top_k_skills,
                score_source="idf",
            )
        else:
            logger.info(f"Using top-k selection by IDF (k={args.top_k_skills})...")
            job_skill_map = cap_skills_per_job_lexicographic(
                job_skill_map,
                max_skills_per_job=args.top_k_skills,
                skill_desc_map=skill_desc_map,
                use_weighted_idf=False,
            )

        if include_skill_descriptions:
            logger.info("Building skill-based transition pairs (with skill descriptions)...")
        else:
            logger.info("Building skill-based transition pairs (without skill descriptions)...")
        
        train_pairs, val_pairs, test_pairs = load_skill_transition_pairs(
            args.data_type, job_skill_map, skill_desc_map, include_skill_descriptions
        )

    logger.info(
        f"Pairs -> train: {len(train_pairs)}, "
        f"val: {len(val_pairs)}, test: {len(test_pairs)}"
    )
    if len(train_pairs) == 0:
        raise RuntimeError("No training pairs could be built. Check data_type and skill mappings.")

    logger.info("Loading Sentence Transformer model...")
    model = SentenceTransformer(args.model_name, trust_remote_code=True)

    # Analyze token lengths before training
    analyze_token_lengths(model, train_pairs, "train")
    if val_pairs:
        analyze_token_lengths(model, val_pairs, "validation")

    # Candidate pool for retrieval evaluation: use ALL doc2 texts from train+val+test for fair testing.
    # This ensures the test is not restricted to only test occupations.
    candidate_texts = list(
        dict.fromkeys([p for _, p in (train_pairs + val_pairs + test_pairs)])
    )
    if test_pairs:
        logger.info(
            f"Candidate pool (train+val+test) has {len(candidate_texts)} unique doc2 texts"
        )

    # If we run test evaluation at all, we always compute a baseline on the exact same test pairs.
    # This guarantees a fair baseline vs fine-tuned comparison.
    base_metrics: Dict[str, float] | None = None
    should_run_test_eval = args.test_strategy in ("final", "epoch")
    if test_pairs and (args.test_base_model or should_run_test_eval):
        logger.info("Running retrieval evaluation on base (unfine-tuned) model...")
        base_metrics = evaluate_retrieval(
            model,
            test_pairs,  # exact test doc_1/doc_2 pairs
            candidate_texts=candidate_texts,  # full candidate pool (train+val+test)
            batch_size=args.batch_size,
            k_values=(5, 10),
            desc="test (base model; full candidates)",
        )
        if WANDB_AVAILABLE and args.use_wandb:
            wandb.log(
                {
                    **{f"base_eval/{k}": v for k, v in base_metrics.items()},
                    "candidate_pool/unique_doc2": len(candidate_texts),
                    "splits/train_pairs": len(train_pairs),
                    "splits/val_pairs": len(val_pairs),
                    "splits/test_pairs": len(test_pairs),
                }
            )

    logger.info("Starting fine-tuning...")
    model = fine_tune(
        train_pairs,
        val_pairs,
        test_pairs,
        model,
        args,
        candidate_texts=candidate_texts if test_pairs else None,
    )

    if args.save_model:
        best_model_path = os.path.join(args.output_dir, "best-model")
        logger.info(f"Saving the best model to: {best_model_path}")
        model.save(best_model_path)
        logger.info("Model fine-tuning and saving completed.")
    else:
        logger.info("Model fine-tuning completed. Saving is not requested.")

    final_metrics: Dict[str, float] | None = None
    if args.test_strategy in ("final", "epoch") and test_pairs:
        logger.info("Running retrieval evaluation on best/final model...")
        final_metrics = evaluate_retrieval(
            model,
            test_pairs,
            candidate_texts=candidate_texts,
            batch_size=args.batch_size,
            k_values=(5, 10),
            desc="test (final model; full candidates)",
        )
        
        # Push to wandb if enabled
        if WANDB_AVAILABLE and args.use_wandb:
            wandb.log({f"final_eval/{k}": v for k, v in final_metrics.items()})
            logger.info("Final metrics pushed to wandb")

    # Save a single consolidated results.json in output_dir (baseline + fine-tuned + config).
    results_path = os.path.join(args.output_dir, "results.json")
    results_payload: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "output_dir": args.output_dir,
        "config": vars(args),
        "data_summary": {
            "num_train_pairs": len(train_pairs),
            "num_val_pairs": len(val_pairs),
            "num_test_pairs": len(test_pairs),
            "num_candidate_doc2_unique": len(candidate_texts),
        },
        "baseline": {
            "evaluated": base_metrics is not None,
            "metrics": base_metrics,
            "notes": "Evaluated on test doc_1/doc_2 pairs; retrieved against train+val+test candidate pool.",
        },
        "fine_tuned": {
            "evaluated": final_metrics is not None,
            "metrics": final_metrics,
            "notes": "Evaluated on test doc_1/doc_2 pairs; retrieved against train+val+test candidate pool.",
        },
    }
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    logger.info(f"Saved consolidated results to: {results_path}")

    if WANDB_AVAILABLE and args.use_wandb and wandb.run is not None:
        # Mirror key outputs into W&B run summary for easy comparison.
        if base_metrics is not None:
            for k, v in base_metrics.items():
                wandb.run.summary[f"base_eval/{k}"] = v
        if final_metrics is not None:
            for k, v in final_metrics.items():
                wandb.run.summary[f"final_eval/{k}"] = v
        wandb.run.summary["candidate_pool/unique_doc2"] = len(candidate_texts)
        wandb.run.summary["splits/train_pairs"] = len(train_pairs)
        wandb.run.summary["splits/val_pairs"] = len(val_pairs)
        wandb.run.summary["splits/test_pairs"] = len(test_pairs)

    if WANDB_AVAILABLE and args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

