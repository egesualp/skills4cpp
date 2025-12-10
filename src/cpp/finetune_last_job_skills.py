import os
from typing import Dict, List, Tuple, Any

import argparse
import random

from datasets import Dataset, load_dataset
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from src.cpp.skill_pooling import (
    load_skill_mappings,
    load_skill_descriptions,
    calculate_idf_scores,
    cap_skills_per_job_lexicographic,
)
from src.cpp.utils import SEP_TOKEN


# Explicitly select the first GPU unless overridden externally
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


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
) -> str:
    """
    Convert a list of skills into the document format:
    "skill: ... \\n description: ...<SEP>skill: ..."
    """
    segments = []
    for skill in skills:
        uri = skill.get("skillUri")
        skill_meta = skill_desc_map.get(uri, {})
        name = skill_meta.get("name") or skill.get("skill") or ""
        description = skill_meta.get("description") or ""
        segments.append(f"skill: {name} \n description: {description}")
    return SEP_TOKEN.join(segments)


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
) -> List[Tuple[str, str]]:
    """
    Build (doc1, doc2) pairs for a dataset split.
    doc1: top-k skills (already capped) of the last observed job
    doc2: next ESCO occupation
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

            doc1 = _skills_to_doc(skills, skill_desc_map)
            if not doc1:
                continue

            doc2 = f"esco role: {target_titles_esco[idx + 1]} \n description: {target_desc_esco[idx + 1]}"
            pairs.append((doc1, doc2))

    return pairs


def load_skill_transition_pairs(
    data_type: str,
    job_skill_map: Dict[str, List[Dict[str, Any]]],
    skill_desc_map: Dict[str, Dict[str, str]],
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

    train_pairs = _build_pairs_for_split(dataset["train"], language, job_skill_map, skill_desc_map)
    val_pairs = _build_pairs_for_split(dataset["validation"], language, job_skill_map, skill_desc_map)
    test_pairs = _build_pairs_for_split(dataset["test"], language, job_skill_map, skill_desc_map)

    return train_pairs, val_pairs, test_pairs


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


def fine_tune(train_pairs, valid_pairs, model, training_config):
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

    eval_save_steps = max(
        1,
        int(
            training_config.epoch_eval_frac
            * len(train_dataset)
            / training_config.batch_size
        ),
    )
    print(f"Eval and save steps interval: {eval_save_steps}")

    args = SentenceTransformerTrainingArguments(
        output_dir=training_config.output_dir,
        num_train_epochs=training_config.epochs,
        per_device_train_batch_size=training_config.batch_size,
        per_device_eval_batch_size=training_config.batch_size,
        learning_rate=training_config.learning_rate,
        fp16=not training_config.disable_fp16,
        bf16=False,
        max_grad_norm=1.0,
        eval_strategy="steps",
        eval_steps=eval_save_steps,
        save_strategy="steps",
        save_steps=eval_save_steps,
        save_total_limit=2,
        logging_steps=eval_save_steps,
        run_name="skill-idf",
        #report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    args = args.set_lr_scheduler(name="linear", warmup_ratio=0.05)
    args = args.set_training(
        learning_rate=training_config.learning_rate,
        batch_size=training_config.batch_size,
        weight_decay=0,
        num_epochs=training_config.epochs,
        max_steps=-1,
        gradient_accumulation_steps=1,
        seed=42,
        gradient_checkpointing=False,
    )

    evaluator = construct_evaluator(valid_pairs)

    trainer = CustomTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        args=args,
        evaluator=evaluator,
    )

    trainer.train()
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune embeddings on last-job skills (IDF top-k) -> next ESCO occupation."
    )
    parser.add_argument("--data_type", type=str, default="karrierewege_100k")
    parser.add_argument("--job_title_skills_csv", type=str, required=True)
    parser.add_argument("--skills_csv", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--output_dir", type=str, default="./models/skill-idf-finetune")
    parser.add_argument("--top_k_skills", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--epoch_eval_frac", type=float, default=0.1)
    parser.add_argument("--disable_fp16", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    print("Loading skill mappings and descriptions...")
    job_skill_map = load_skill_mappings(args.job_title_skills_csv)
    skill_desc_map = load_skill_descriptions(args.skills_csv)

    print("Computing IDF scores and capping to top-k skills...")
    job_skill_map = calculate_idf_scores(job_skill_map)
    job_skill_map = cap_skills_per_job_lexicographic(
        job_skill_map,
        max_skills_per_job=args.top_k_skills,
        skill_desc_map=skill_desc_map,
    )

    print("Building skill-based transition pairs...")
    train_pairs, val_pairs, test_pairs = load_skill_transition_pairs(
        args.data_type, job_skill_map, skill_desc_map
    )

    print(
        f"Pairs -> train: {len(train_pairs)}, "
        f"val: {len(val_pairs)}, test: {len(test_pairs)}"
    )
    if len(train_pairs) == 0:
        raise RuntimeError("No training pairs could be built. Check data_type and skill mappings.")

    print("Loading Sentence Transformer model...")
    model = SentenceTransformer(args.model_name)

    print("Starting fine-tuning...")
    model = fine_tune(train_pairs, val_pairs, model, args)

    best_model_path = os.path.join(args.output_dir, "best-model")
    print(f"Saving the best model to: {best_model_path}")
    model.save(best_model_path)
    print("Model fine-tuning and saving completed.")


if __name__ == "__main__":
    main()

