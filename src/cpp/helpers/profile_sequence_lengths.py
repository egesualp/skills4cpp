"""Profile token sequence lengths for last-job skills finetuning.

This script mirrors the data preparation pipeline in
`src.cpp.finetune_last_job_skills` and reports the distribution of
Transformer token lengths for the (doc1, doc2) pairs that are used for
training.

It answers questions like:
- How long are the skill-based documents (doc1)?
- How long are the ESCO occupation texts (doc2)?
- What fraction of examples exceed typical max_seq_length values
  (128 / 256 / 384 / 512)?

You can use these statistics to choose a safe `max_seq_length` or decide
whether to drop only the very longest sequences.
"""

import argparse
from typing import List, Tuple

import numpy as np
from transformers import AutoTokenizer

from src.cpp.skill_pooling import (
    load_skill_mappings,
    load_skill_descriptions,
    calculate_idf_scores,
    cap_skills_per_job_lexicographic,
)
from src.cpp.finetune_last_job_skills import load_skill_transition_pairs


def describe_distribution(name: str, values: List[int]) -> None:
    """Print basic distribution statistics for a list of integers."""
    if not values:
        print(f"{name}: no data")
        return

    arr = np.array(values, dtype=np.int32)
    print(f"{name}:")
    print(f"  count   = {arr.size}")
    print(f"  mean    = {arr.mean():.2f}")
    print(f"  std     = {arr.std():.2f}")
    print(f"  min     = {arr.min()}")
    print(f"  25%     = {np.percentile(arr, 25):.2f}")
    print(f"  50%     = {np.percentile(arr, 50):.2f}")
    print(f"  75%     = {np.percentile(arr, 75):.2f}")
    print(f"  90%     = {np.percentile(arr, 90):.2f}")
    print(f"  95%     = {np.percentile(arr, 95):.2f}")
    print(f"  99%     = {np.percentile(arr, 99):.2f}")
    print(f"  max     = {arr.max()}")
    print("")


def report_thresholds(name: str, values: List[int], thresholds: List[int]) -> None:
    if not values:
        return
    arr = np.array(values, dtype=np.int32)
    print(f"{name} (fraction of examples > threshold):")
    total = arr.size
    for t in thresholds:
        frac = float((arr > t).sum()) / total
        print(f"  > {t:4d}: {frac:.4f}")
    print("")


def profile_sequence_lengths(
    data_type: str,
    job_title_skills_csv: str,
    skills_csv: str,
    model_name: str,
    top_k_skills: int,
    max_samples: int,
    include_val_and_test: bool,
) -> None:
    """Profile token lengths for the last-job skills finetuning pairs."""

    print("Loading job->skill mappings and skill descriptions...")
    print(f"  job_title_skills_csv = {job_title_skills_csv}")
    print(f"  skills_csv           = {skills_csv}")

    job_skill_map = load_skill_mappings(job_title_skills_csv)
    skill_desc_map = load_skill_descriptions(skills_csv)

    print("Calculating IDF scores and applying top-K cap (same as training)...")
    job_skill_map = calculate_idf_scores(job_skill_map)
    job_skill_map = cap_skills_per_job_lexicographic(
        job_skill_map,
        max_skills_per_job=top_k_skills,
        skill_desc_map=skill_desc_map,
    )

    print("Loading career-path pairs (same as finetune_last_job_skills)...")
    train_pairs, val_pairs, test_pairs = load_skill_transition_pairs(
        data_type=data_type,
        job_skill_map=job_skill_map,
        skill_desc_map=skill_desc_map,
    )
    print(
        f"  Train: {len(train_pairs)}, Val: {len(val_pairs)}, "
        f"Test: {len(test_pairs)}"
    )

    if include_val_and_test:
        all_pairs: List[Tuple[str, str]] = (
            list(train_pairs) + list(val_pairs) + list(test_pairs)
        )
    else:
        all_pairs = list(train_pairs)

    if max_samples is not None and max_samples > 0:
        all_pairs = all_pairs[: max_samples]
    print(f"Profiling {len(all_pairs)} pairs...")

    print(f"Loading tokenizer for model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    doc1_lengths: List[int] = []
    doc2_lengths: List[int] = []
    pair_max_lengths: List[int] = []

    for idx, (doc1, doc2) in enumerate(all_pairs):
        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1} pairs...")

        tokens1 = tokenizer(doc1, truncation=False, padding=False)
        tokens2 = tokenizer(doc2, truncation=False, padding=False)
        len1 = len(tokens1["input_ids"])
        len2 = len(tokens2["input_ids"])

        doc1_lengths.append(len1)
        doc2_lengths.append(len2)
        pair_max_lengths.append(max(len1, len2))

    print("\n=== Sequence length statistics (in tokens) ===\n")
    describe_distribution("Doc1 (skills document) length", doc1_lengths)
    describe_distribution("Doc2 (ESCO occupation) length", doc2_lengths)
    describe_distribution("Max length per pair (max(doc1, doc2))", pair_max_lengths)

    thresholds = [64, 128, 192, 256, 320, 384, 512]
    report_thresholds("Doc1 (skills document)", doc1_lengths, thresholds)
    report_thresholds("Doc2 (ESCO occupation)", doc2_lengths, thresholds)
    report_thresholds("Max length per pair", pair_max_lengths, thresholds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile token sequence lengths for last-job skills "
            "finetuning inputs."
        )
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="karrierewege_100k",
        help=(
            "Dataset type (must match the values supported by "
            "load_skill_transition_pairs, e.g. karrierewege_100k)."
        ),
    )
    parser.add_argument(
        "--job_title_skills_csv",
        type=str,
        default=(
            "results/karrierewege_esco_100k_esco_ground_truth/"
            "job_title_skills_master.csv"
        ),
        help="Path to job title -> skills mapping CSV.",
    )
    parser.add_argument(
        "--skills_csv",
        type=str,
        default="data/esco_datasets/skills_en.csv",
        help="Path to ESCO skills CSV (used for skill descriptions).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help=(
            "Model name or path whose tokenizer should be used "
            "for measuring token lengths."
        ),
    )
    parser.add_argument(
        "--top_k_skills",
        type=int,
        default=20,
        help=(
            "Top-K skills per job to keep after IDF-based capping "
            "(must match training)."
        ),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=50000,
        help=(
            "Maximum number of pairs to profile. Set to a larger value "
            "or -1 to use all pairs."
        ),
    )
    parser.add_argument(
        "--include_val_and_test",
        action="store_true",
        help="If set, profile train + val + test instead of train only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("")

    max_samples = args.max_samples
    if max_samples is not None and max_samples < 0:
        max_samples = None

    profile_sequence_lengths(
        data_type=args.data_type,
        job_title_skills_csv=args.job_title_skills_csv,
        skills_csv=args.skills_csv,
        model_name=args.model_name,
        top_k_skills=args.top_k_skills,
        max_samples=max_samples,
        include_val_and_test=args.include_val_and_test,
    )


if __name__ == "__main__":
    main()




