#!/usr/bin/env python3
"""
Error analysis for category-based fusion / re-ranking.

Implements:
  - Check 1: For each job, how many L1 categories do its gold skills span?
  - Check 2: What % of top-K retrieved skills share L1 categories with gold skills?
  - Check 3: For the (e.g. 14) jobs whose scores are adjusted (threshold-pass),
             did they improve, worsen, or stay same? (AP@K comparison: base vs fused)

Notes:
  - Streams huge JSON files with ijson (recommended for multi-GB inputs).
  - Builds hierarchy mappings using chunked CSV reads to keep RAM bounded.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


def _require_ijson():
    try:
        import ijson  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Missing dependency 'ijson' for streaming JSON parsing. "
            "Install it (e.g. `pip install ijson`) and re-run."
        ) from e
    return ijson


@dataclass(frozen=True)
class Paths:
    base_scores_json: str
    fused_scores_json: str
    category_scores_json: str
    hierarchy_csv: str
    jobs_csv: str


def load_jobs_csv(jobs_csv: str) -> pd.DataFrame:
    df = pd.read_csv(jobs_csv, low_memory=False)
    df["job_id"] = df["job_id"].astype(str)
    if "split" in df.columns:
        df["split"] = df["split"].astype(str).str.lower()
    # Match the fuser's deduping (important for test-job counts)
    df = df.drop_duplicates(subset=["raw_title", "raw_description", "esco_id", "job_id"], keep="first")
    return df


def build_hierarchy_maps(
    hierarchy_csv: str,
    chunksize: int = 500_000,
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]], Dict[str, Set[str]]]:
    """
    Returns:
      - occ_to_l1: occupationUri -> set(level1_label)
      - occ_to_skills: occupationUri -> set(skillUri)
      - skill_to_l1: skillUri -> set(level1_label)
    """
    usecols = ["occupationUri", "skillUri", "level1_label"]
    occ_to_l1: Dict[str, Set[str]] = defaultdict(set)
    occ_to_skills: Dict[str, Set[str]] = defaultdict(set)
    skill_to_l1: Dict[str, Set[str]] = defaultdict(set)

    for chunk in pd.read_csv(
        hierarchy_csv,
        usecols=usecols,
        dtype=str,
        low_memory=False,
        chunksize=chunksize,
    ):
        # Drop rows missing required fields
        chunk = chunk.dropna(subset=["occupationUri", "skillUri", "level1_label"])
        if chunk.empty:
            continue

        # Normalize types
        occ = chunk["occupationUri"].astype(str).values
        skill = chunk["skillUri"].astype(str).values
        l1 = chunk["level1_label"].astype(str).values

        for o, s, c in zip(occ, skill, l1):
            if not o or not s or not c:
                continue
            occ_to_l1[o].add(c)
            occ_to_skills[o].add(s)
            skill_to_l1[s].add(c)

    return dict(occ_to_l1), dict(occ_to_skills), dict(skill_to_l1)


def describe_counts(counts: Iterable[int]) -> pd.Series:
    s = pd.Series(list(counts), dtype=float)
    if s.empty:
        return pd.Series(dtype=float)
    return s.describe(percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])


def ap_at_k(ranked_skill_uris: List[Optional[str]], gold_set: Set[str], k: int) -> float:
    if not gold_set:
        return float("nan")
    hits = 0
    sum_prec = 0.0
    for rank, su in enumerate(ranked_skill_uris[:k], start=1):
        if su in gold_set:
            hits += 1
            sum_prec += hits / rank
    return sum_prec / len(gold_set)


def load_adjusted_jobs(
    category_scores_json: str,
    jobs_df: pd.DataFrame,
    threshold: float,
    split: str,
) -> List[str]:
    with open(category_scores_json, "r") as f:
        category_scores = json.load(f)

    # Determine which jobs pass threshold: max P(category|job) >= threshold
    use_weighting: Dict[str, bool] = {}
    for job_id, cat_list in category_scores.items():
        max_prob = max((float(x.get("score", 0.0)) for x in cat_list), default=0.0)
        use_weighting[str(job_id)] = (max_prob >= threshold)

    if split != "all":
        job_ids = set(jobs_df.loc[jobs_df["split"] == split, "job_id"].astype(str).tolist())
    else:
        job_ids = set(jobs_df["job_id"].astype(str).tolist())

    adjusted = [jid for jid in job_ids if use_weighting.get(jid, False)]
    adjusted.sort(key=lambda x: int(x) if x.isdigit() else x)
    return adjusted


def stream_base_scores_overlap_and_extract(
    base_scores_json: str,
    topk: int,
    job_to_gold_l1: Dict[str, Set[str]],
    skill_to_l1: Dict[str, Set[str]],
    extract_job_ids: Set[str],
) -> Tuple[float, List[float], Dict[str, List[Optional[str]]], int, int]:
    """
    Streams the huge base similarity JSON and computes:
      - overall overlap% (topk)
      - per-job overlap% (topk) (only for jobs with gold L1 present)
      - extracted topk lists for extract_job_ids
      - seen_jobs
      - missing_gold_jobs (no gold L1)
    """
    ijson = _require_ijson()

    per_job_pct: List[float] = []
    base_topk_for_jobs: Dict[str, List[Optional[str]]] = {}

    total_overlap = 0
    total_retrieved = 0
    missing_gold_jobs = 0
    seen_jobs = 0

    current_job: Optional[str] = None
    current_item: Optional[dict] = None
    collected: List[Optional[str]] = []

    with open(base_scores_json, "rb") as f:
        for prefix, event, value in ijson.parse(f):
            if prefix == "" and event == "map_key":
                current_job = str(value)
                collected = []
                current_item = None
                continue

            if current_job is None:
                continue

            if prefix == f"{current_job}.item" and event == "start_map":
                current_item = {}
                continue

            if current_item is not None and prefix.startswith(f"{current_job}.item.") and event in ("string", "number"):
                field = prefix.split(".")[-1]
                current_item[field] = value
                continue

            if prefix == f"{current_job}.item" and event == "end_map":
                if len(collected) < topk:
                    collected.append(current_item.get("skill_uri"))
                current_item = None
                continue

            if prefix == current_job and event == "end_array":
                seen_jobs += 1

                # Extract topk for specific jobs (for Check 3)
                if current_job in extract_job_ids:
                    base_topk_for_jobs[current_job] = collected[:topk]

                gold_l1 = job_to_gold_l1.get(current_job)
                if not gold_l1:
                    missing_gold_jobs += 1
                else:
                    overlap = 0
                    for su in collected[:topk]:
                        cats = skill_to_l1.get(su) if su else None
                        if cats and (cats & gold_l1):
                            overlap += 1
                    per_job_pct.append(100.0 * overlap / topk)
                    total_overlap += overlap
                    total_retrieved += topk

                current_job = None

    overall_pct = 100.0 * total_overlap / max(total_retrieved, 1)
    return overall_pct, per_job_pct, base_topk_for_jobs, seen_jobs, missing_gold_jobs


def stream_fused_scores_extract(
    fused_scores_json: str,
    topk: int,
    extract_job_ids: Set[str],
) -> Dict[str, List[Optional[str]]]:
    ijson = _require_ijson()

    fused_topk_for_jobs: Dict[str, List[Optional[str]]] = {}

    current_job: Optional[str] = None
    current_item: Optional[dict] = None
    collected: List[Optional[str]] = []

    with open(fused_scores_json, "rb") as f:
        for prefix, event, value in ijson.parse(f):
            # job keys are under the "scores" map
            if prefix == "scores" and event == "map_key":
                current_job = str(value)
                collected = []
                current_item = None
                continue

            if current_job is None:
                continue

            if current_job not in extract_job_ids:
                if prefix == f"scores.{current_job}" and event == "end_array":
                    current_job = None
                continue

            if prefix == f"scores.{current_job}.item" and event == "start_map":
                current_item = {}
                continue

            if current_item is not None and prefix.startswith(f"scores.{current_job}.item.") and event in ("string", "number"):
                field = prefix.split(".")[-1]
                current_item[field] = value
                continue

            if prefix == f"scores.{current_job}.item" and event == "end_map":
                if len(collected) < topk:
                    collected.append(current_item.get("skill_uri"))
                current_item = None
                continue

            if prefix == f"scores.{current_job}" and event == "end_array":
                fused_topk_for_jobs[current_job] = collected[:topk]
                current_job = None
                if len(fused_topk_for_jobs) == len(extract_job_ids):
                    break

    return fused_topk_for_jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_scores_json",
        default="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/decorte_w_desc_3/similarity_scores.json",
    )
    parser.add_argument(
        "--fused_scores_json",
        default="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v2_bayesian_fuser/linear_h1_sum_100/best_fused_scores.json",
    )
    parser.add_argument(
        "--category_scores_json",
        default="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/category_model_h1_soft_deep_larger_val/decorte_w_desc_2_inference/category_scores.json",
    )
    parser.add_argument(
        "--hierarchy_csv",
        default="/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/processed/master_datasets_2/master_complete_hierarchy_w_occ.csv",
    )
    parser.add_argument(
        "--jobs_csv",
        default="/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/decorte_master_3.csv",
    )
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--hierarchy_chunksize", type=int, default=500_000)
    args = parser.parse_args()

    paths = Paths(
        base_scores_json=args.base_scores_json,
        fused_scores_json=args.fused_scores_json,
        category_scores_json=args.category_scores_json,
        hierarchy_csv=args.hierarchy_csv,
        jobs_csv=args.jobs_csv,
    )

    print("Loading jobs CSV (with fuser-style dedupe)...")
    jobs_df = load_jobs_csv(paths.jobs_csv)
    print(f"  deduped_jobs={len(jobs_df)} splits={jobs_df['split'].value_counts().to_dict() if 'split' in jobs_df.columns else 'N/A'}")

    print("Building hierarchy maps (chunked CSV read)...")
    occ_to_l1, occ_to_skills, skill_to_l1 = build_hierarchy_maps(paths.hierarchy_csv, chunksize=args.hierarchy_chunksize)
    print(f"  occ_to_l1={len(occ_to_l1)} occ_to_skills={len(occ_to_skills)} skill_to_l1={len(skill_to_l1)}")

    # Build job->gold mappings
    job_to_esco = dict(zip(jobs_df["job_id"].astype(str), jobs_df["esco_id"].astype(str)))
    all_job_ids = jobs_df["job_id"].astype(str).tolist()

    if args.split != "all":
        eval_job_ids = jobs_df.loc[jobs_df["split"] == args.split, "job_id"].astype(str).tolist()
    else:
        eval_job_ids = all_job_ids

    job_to_gold_l1 = {jid: occ_to_l1.get(job_to_esco.get(jid, ""), set()) for jid in eval_job_ids}

    # ----- Check 1 -----
    print("\nCheck 1: Category concentration in ground truth")
    counts = [len(job_to_gold_l1.get(jid, set())) for jid in eval_job_ids]
    print(describe_counts(counts).to_string())

    # ----- Check 3 (identify adjusted jobs) -----
    print("\nIdentifying adjusted jobs (threshold-pass) from category scores...")
    adjusted_job_ids = load_adjusted_jobs(paths.category_scores_json, jobs_df, threshold=args.threshold, split=args.split)
    print(f"  adjusted_{args.split}_jobs={len(adjusted_job_ids)} (threshold={args.threshold})")
    if len(adjusted_job_ids) <= 50:
        print(f"  adjusted_job_ids={adjusted_job_ids}")

    extract_job_ids = set(adjusted_job_ids)

    # ----- Check 2 (and base extraction for Check 3) -----
    print("\nCheck 2: Category overlap between retrieval and ground truth (streaming base scores)...")
    overall_pct, per_job_pct, base_topk, seen_jobs, missing_gold_jobs = stream_base_scores_overlap_and_extract(
        paths.base_scores_json,
        topk=args.topk,
        job_to_gold_l1={str(k): v for k, v in job_to_gold_l1.items()},
        skill_to_l1=skill_to_l1,
        extract_job_ids=extract_job_ids,
    )
    print(f"  jobs_seen_in_base_scores={seen_jobs}")
    print(f"  jobs_missing_gold_categories={missing_gold_jobs}")
    print(f"  overall_overlap_pct_top{args.topk}={overall_pct:.4f}")
    print("  per-job overlap% stats (jobs with gold only):")
    print(pd.Series(per_job_pct, dtype=float).describe(percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).to_string())

    # ----- Check 3 (fused extraction + AP@K compare) -----
    print("\nCheck 3: Adjusted jobs improve/worsen/same? (AP@K base vs fused)")
    fused_topk = stream_fused_scores_extract(paths.fused_scores_json, topk=args.topk, extract_job_ids=extract_job_ids)

    rows = []
    for jid in adjusted_job_ids:
        esco = job_to_esco.get(jid, "")
        gold = occ_to_skills.get(esco, set())
        ap_base = ap_at_k(base_topk.get(jid, []), gold, k=args.topk)
        ap_fused = ap_at_k(fused_topk.get(jid, []), gold, k=args.topk)
        delta = ap_fused - ap_base

        if np.isnan(delta) or abs(delta) < 1e-12:
            outcome = "same"
        elif delta > 0:
            outcome = "improve"
        else:
            outcome = "worsen"

        title = None
        try:
            title = jobs_df.loc[jobs_df["job_id"] == jid, "raw_title"].iloc[0]
        except Exception:
            title = None

        rows.append(
            {
                "job_id": jid,
                "raw_title": title,
                f"ap@{args.topk}_base": ap_base,
                f"ap@{args.topk}_fused": ap_fused,
                f"delta_ap@{args.topk}": delta,
                "outcome": outcome,
            }
        )

    df_cmp = pd.DataFrame(rows)
    if df_cmp.empty:
        print("  No adjusted jobs found for the requested split/threshold.")
        return

    print("\nOutcome counts:")
    print(df_cmp["outcome"].value_counts(dropna=False).to_string())

    print("\nPer-job details (sorted by outcome, then delta desc):")
    sort_cols = ["outcome", f"delta_ap@{args.topk}"]
    df_cmp = df_cmp.sort_values(sort_cols, ascending=[True, False])
    print(df_cmp.to_string(index=False))


if __name__ == "__main__":
    main()

