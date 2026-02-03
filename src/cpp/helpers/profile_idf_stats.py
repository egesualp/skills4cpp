"""
Profile IDF scores and their relationship to skill frequency and per-job caps.

This script uses the SAME loading and IDF logic as train_cpp_skills_v3.py:
- load_skill_mappings
- calculate_idf_scores

It answers:
- What is the distribution of IDF over unique skills?
- How many occupations does a typical skill appear in?
- For a job, how many skills fall into top IDF tiers (e.g., top 10, 20)?
"""

import argparse
from typing import Dict, List, Any, Tuple
from collections import defaultdict

import numpy as np

from src.cpp.skill_pooling import (
    load_skill_mappings,
    calculate_idf_scores,
)


def describe_distribution(name: str, values: List[float]) -> None:
    if not values:
        print(f"{name}: no data")
        return
    arr = np.array(values)
    print(f"{name}:")
    print(f"  count   = {len(values)}")
    print(f"  mean    = {arr.mean():.4f}")
    print(f"  std     = {arr.std():.4f}")
    print(f"  min     = {arr.min():.4f}")
    print(f"  25%     = {np.percentile(arr, 25):.4f}")
    print(f"  50%     = {np.percentile(arr, 50):.4f}")
    print(f"  75%     = {np.percentile(arr, 75):.4f}")
    print(f"  90%     = {np.percentile(arr, 90):.4f}")
    print(f"  95%     = {np.percentile(arr, 95):.4f}")
    print(f"  max     = {arr.max():.4f}")
    print("")


def compute_skill_idf_and_freq(
    job_skill_map: Dict[str, List[Dict[str, Any]]]
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """
    From the IDF-annotated job_skill_map, compute:
      - per-skill IDF
      - per-skill occupation frequency (in how many job titles it appears)
    """
    skill_idf: Dict[str, float] = {}
    skill_occ_freq: Dict[str, int] = defaultdict(int)

    for job_title, skills in job_skill_map.items():
        seen_uris = set()
        for s in skills:
            uri = s["skillUri"]
            # IDF already attached by calculate_idf_scores
            idf_val = float(s.get("idf", 0.0))
            skill_idf[uri] = idf_val

            if uri not in seen_uris:
                skill_occ_freq[uri] += 1
                seen_uris.add(uri)

    return skill_idf, skill_occ_freq


def analyze_idf(
    job_title_skills_csv: str,
    topk_candidates: List[int],
) -> None:
    print(f"Loading job->skill map from: {job_title_skills_csv}")
    job_skill_map = load_skill_mappings(job_title_skills_csv)
    n_jobs = len(job_skill_map)
    print(f"  Unique job titles: {n_jobs}")

    print("\nCalculating IDF scores (same as train_cpp_skills_v3)...")
    job_skill_map = calculate_idf_scores(job_skill_map)

    print("Computing per-skill IDF and occupation frequency...")
    skill_idf, skill_occ_freq = compute_skill_idf_and_freq(job_skill_map)
    print(f"  Unique skills (with IDF): {len(skill_idf)}")

    # 1) Global IDF distribution
    idf_values = list(skill_idf.values())
    describe_distribution("IDF over unique skills", idf_values)

    # 2) Occupation frequency distribution
    occ_freq_values = list(skill_occ_freq.values())
    describe_distribution("Occupation frequency per skill (number of job titles)", occ_freq_values)

    # 3) Relationship: show IDF for some frequency buckets
    #    (e.g., skills that appear in 1 job, 2-5 jobs, 6-20, >20)
    buckets = {
        "freq == 1": [],
        "freq 2-5": [],
        "freq 6-20": [],
        "freq > 20": [],
    }
    for uri, freq in skill_occ_freq.items():
        idf_val = skill_idf.get(uri, 0.0)
        if freq == 1:
            buckets["freq == 1"].append(idf_val)
        elif 2 <= freq <= 5:
            buckets["freq 2-5"].append(idf_val)
        elif 6 <= freq <= 20:
            buckets["freq 6-20"].append(idf_val)
        else:
            buckets["freq > 20"].append(idf_val)

    for name, vals in buckets.items():
        describe_distribution(f"IDF for skills with {name}", vals)

    # 4) Per-job: how many skills would be kept if we cap by top-K IDF?
    print("Analyzing per-job coverage for top-K-by-IDF caps...")
    for k in topk_candidates:
        kept_counts: List[int] = []
        total_skills_per_job: List[int] = []
        for job_title, skills in job_skill_map.items():
            total = len(skills)
            total_skills_per_job.append(total)
            if total == 0:
                kept_counts.append(0)
                continue

            # Sort this job's skills by IDF descending
            sorted_skills = sorted(
                skills,
                key=lambda s: s.get("idf", 0.0),
                reverse=True,
            )
            kept = min(k, total)
            kept_counts.append(kept)

        kept_arr = np.array(kept_counts)
        total_arr = np.array(total_skills_per_job)
        ratio_arr = kept_arr / np.maximum(total_arr, 1)

        print(f"Top-K IDF cap per job with K={k}:")
        print(f"  mean kept skills/job   = {kept_arr.mean():.2f}")
        print(f"  median kept skills/job = {np.percentile(kept_arr, 50):.2f}")
        print(f"  max kept skills/job    = {kept_arr.max():.0f}")
        print(f"  mean kept ratio        = {ratio_arr.mean():.2f}")
        print(f"  median kept ratio      = {np.percentile(ratio_arr, 50):.2f}")
        print("")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile IDF scores and top-K caps for skills."
    )
    parser.add_argument(
        "--job_title_skills_csv",
        type=str,
        default="results/karrierewege_esco_100k_esco_ground_truth/job_title_skills_master.csv",
        help="Path to job title skills mapping CSV (same as train_cpp_skills_v3).",
    )
    parser.add_argument(
        "--topk",
        type=str,
        default="10,20,30,50",
        help="Comma-separated list of K values for top-K-by-IDF per job analysis.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("")

    topk = [int(x) for x in args.topk.split(",") if x.strip()]
    analyze_idf(
        job_title_skills_csv=args.job_title_skills_csv,
        topk_candidates=topk,
    )


if __name__ == "__main__":
    main()











