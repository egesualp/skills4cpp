"""
Utility script to profile skill-path complexity and batching behaviour.

It answers questions like:
- How many jobs per career path?
- How many skills per job / per path?
- How many skill texts are encoded per training batch (given a batch_size)?

It reuses the same data loading and dataset classes as train_cpp_skills_v3.py,
so the statistics reflect exactly what the training script sees.
"""

import argparse
from typing import Dict, List, Any

import numpy as np

from src.cpp.data_classes import Data
from src.cpp.utils import SEP_TOKEN
from src.cpp.skill_dataset import (
    SkillBasedCareerPathDataset,
    ISCOGroupBatchSampler,
    collate_skill_batch,
)
from src.cpp.skill_pooling import (
    load_skill_mappings,
    load_skill_descriptions,
    load_occupation_isco_groups,
    calculate_idf_scores,
    create_target_occupation_map,
)


def describe_distribution(name: str, values: List[int]) -> None:
    if not values:
        print(f"{name}: no data")
        return
    arr = np.array(values)
    print(f"{name}:")
    print(f"  count   = {len(values)}")
    print(f"  mean    = {arr.mean():.2f}")
    print(f"  std     = {arr.std():.2f}")
    print(f"  min     = {arr.min()}")
    print(f"  25%     = {np.percentile(arr, 25):.2f}")
    print(f"  50%     = {np.percentile(arr, 50):.2f}")
    print(f"  75%     = {np.percentile(arr, 75):.2f}")
    print(f"  max     = {arr.max()}")
    print("")


def profile_dataset(
    data_type: str,
    job_title_skills_csv: str,
    skills_csv: str,
    occupations_csv: str,
    batch_size: int,
    max_paths: int = None,
) -> None:
    # 1) Load raw pairs (same as train_cpp_skills_v3)
    print("Loading career path data...")
    data = Data(DATA_TYPE=data_type, ONLY_TITLES=False)
    train_pairs, val_pairs, test_pairs = data.get_data(stage="embedding_finetuning")
    print(
        f"  Train: {len(train_pairs)}, Val: {len(val_pairs)}, "
        f"Test: {len(test_pairs)}"
    )

    all_pairs = train_pairs + val_pairs + test_pairs

    # 2) Load mappings / metadata
    print("\nLoading skill mappings and descriptions...")
    job_skill_map = load_skill_mappings(job_title_skills_csv)
    skill_desc_map = load_skill_descriptions(skills_csv)
    isco_map = load_occupation_isco_groups(occupations_csv)

    print("\nCalculating IDF scores...")
    job_skill_map = calculate_idf_scores(job_skill_map)

    print("\nCreating target occupation map...")
    target_occupation_map = create_target_occupation_map(all_pairs, isco_map)

    # 3) Build dataset (same as training)
    print("\nBuilding SkillBasedCareerPathDataset...")
    dataset = SkillBasedCareerPathDataset(
        data_pairs=train_pairs,
        job_skill_map=job_skill_map,
        target_occupation_map=target_occupation_map,
        sep_token=SEP_TOKEN,
    )
    n_samples = len(dataset)
    print(f"  Dataset size (train): {n_samples}")

    # Optionally limit number of paths for profiling
    max_idx = n_samples if max_paths is None else min(max_paths, n_samples)

    # 4) Path-level and job-level statistics
    jobs_per_path: List[int] = []
    skills_per_path: List[int] = []
    skills_per_job: List[int] = []

    print("\nCollecting per-path / per-job statistics...")
    for idx in range(max_idx):
        sample = dataset[idx]
        job_skills_list = sample["job_skills_list"]

        num_jobs = len(job_skills_list)
        jobs_per_path.append(num_jobs)

        path_skill_count = 0
        for job_skills in job_skills_list:
            num_skills_job = len(job_skills)
            skills_per_job.append(num_skills_job)
            path_skill_count += num_skills_job

        skills_per_path.append(path_skill_count)

    describe_distribution("Jobs per path (train)", jobs_per_path)
    describe_distribution("Skills per job (train)", skills_per_job)
    describe_distribution("Skills per path (train)", skills_per_path)

    # 5) Batch-level statistics (how many skill texts per batch)
    print("Building DataLoader to inspect batch-wise skill counts...")
    sampler = ISCOGroupBatchSampler(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    from torch.utils.data import DataLoader

    dummy_loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_skill_batch,
        num_workers=0,
    )

    batch_skill_counts: List[int] = []
    batch_jobs_counts: List[int] = []
    max_batches_to_profile = 50

    print(
        f"\nProfiling up to {max_batches_to_profile} batches with batch_size={batch_size}..."
    )
    for b_idx, batch in enumerate(dummy_loader):
        if b_idx >= max_batches_to_profile:
            break

        total_jobs = 0
        total_skills = 0
        for career_job_skills in batch["job_skills_list"]:
            total_jobs += len(career_job_skills)
            for job_skills in career_job_skills:
                total_skills += len(job_skills)

        batch_jobs_counts.append(total_jobs)
        batch_skill_counts.append(total_skills)

    describe_distribution("Jobs per batch", batch_jobs_counts)
    describe_distribution("Skill texts per batch (i.e., encoded skill descriptions)", batch_skill_counts)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile skill-path statistics for train_cpp_skills_v3."
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="karrierewege_100k",
        help="Dataset type (same as train_cpp_skills_v3)",
    )
    parser.add_argument(
        "--job_title_skills_csv",
        type=str,
        default="results/karrierewege_esco_100k_esco_ground_truth/job_title_skills_master.csv",
        help="Path to job title skills mapping CSV",
    )
    parser.add_argument(
        "--skills_csv",
        type=str,
        default="data/esco_datasets/skills_en.csv",
        help="Path to ESCO skills CSV",
    )
    parser.add_argument(
        "--occupations_csv",
        type=str,
        default="data/esco_datasets/occupations_en.csv",
        help="Path to ESCO occupations CSV",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Training batch size to simulate (per step, before grad accumulation).",
    )
    parser.add_argument(
        "--max_paths",
        type=int,
        default=50000,
        help="Optional cap on number of training paths to profile (for speed).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("")

    profile_dataset(
        data_type=args.data_type,
        job_title_skills_csv=args.job_title_skills_csv,
        skills_csv=args.skills_csv,
        occupations_csv=args.occupations_csv,
        batch_size=args.batch_size,
        max_paths=args.max_paths,
    )


if __name__ == "__main__":
    main()





