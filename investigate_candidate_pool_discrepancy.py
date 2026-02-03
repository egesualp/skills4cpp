#!/usr/bin/env python3
"""
Investigate discrepancy between:
  - "Unique doc_2 / targets" (often computed on *raw* dataset targets), and
  - "Candidate pool size" used in retrieval evaluation (computed on *filtered* skill-based pairs).

This script reproduces, for a given configuration:
  1) The set of unique target doc2 texts available in the underlying dataset transitions (RAW).
  2) The set of unique target doc2 texts that remain after building skill-based (doc1, doc2) pairs (FILTERED).
  3) A drop-reason breakdown (why raw examples/targets disappear) and the most frequent missing targets.
  4) A sanity check showing whether the logged candidate pool size matches "test-only" or "train+val+test".

Typical usage (free-text / predicted-skills mode):
  python investigate_candidate_pool_discrepancy.py \
    --data_type karrierewege_occ \
    --skill_scores_json /path/to/fused_predictions.json \
    --skills_csv /path/to/skills.csv \
    --scoring_mode scores_only \
    --top_k_skills 10

Typical usage (ESCO-title / taxonomy-skills mode):
  python investigate_candidate_pool_discrepancy.py \
    --data_type karrierewege_100k \
    --job_title_skills_csv /path/to/job_title_to_skills.csv \
    --skills_csv /path/to/skills.csv \
    --scoring_mode idf_only \
    --top_k_skills 10
"""

from __future__ import annotations

import argparse
import os
import sys
import re
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _repo_root() -> str:
    return os.path.abspath(os.path.dirname(__file__))


# Ensure imports like `from src...` work when run from repo root.
sys.path.insert(0, _repo_root())


from datasets import load_dataset  # noqa: E402

from src.cpp.data_classes import Data  # noqa: E402
from src.cpp.skill_pooling import (  # noqa: E402
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
from src.cpp.finetune_last_job_skills import (  # noqa: E402
    _skills_to_doc,
    _normalize_title,
    _select_language_fields,
)


ESCO_ROLE_RE = re.compile(r"esco role:\s*(.*?)\s*\n", re.IGNORECASE)


def extract_esco_role_title(doc2: str) -> str:
    m = ESCO_ROLE_RE.search(doc2 or "")
    return m.group(1).strip() if m else ""


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(items))


def describe_set_delta(a: set[str], b: set[str]) -> str:
    # a \ b
    return f"{len(a)} total; {len(a - b)} missing vs other; {len(a & b)} overlap"


@dataclass
class SplitStats:
    num_pairs: int
    unique_doc2_texts: int
    unique_doc2_titles: int


def split_stats(pairs: List[Tuple[str, str]]) -> SplitStats:
    doc2s = [p for _, p in pairs]
    titles = [extract_esco_role_title(p) for p in doc2s]
    return SplitStats(
        num_pairs=len(pairs),
        unique_doc2_texts=len(set(doc2s)),
        unique_doc2_titles=len(set(titles)),
    )


def dataset_spec_for_esco_mode(data_type: str) -> Tuple[str, str]:
    """
    Returns (hf_dataset_name, language_selector) matching finetune_last_job_skills.load_skill_transition_pairs().
    """
    if data_type == "karrierewege":
        return "ElenaSenger/Karrierewege", "en"
    if data_type == "karrierewege_occ":
        return "ElenaSenger/Karrierewege_plus", "en_free"
    if data_type == "karrierewege_100k":
        return "ElenaSenger/Karrierewege_plus", "esco_100k"
    if data_type == "karrierewege_cp":
        return "ElenaSenger/Karrierewege_plus", "en_free_cp"
    raise ValueError(f"Unsupported data_type for this helper: {data_type}")


def raw_doc2_texts_from_esco_dataset_split(split, language: str) -> List[str]:
    """
    Count *all possible* transition targets as doc2 texts (idx+1 experience) for each career.
    This ignores skill availability for doc1 entirely.
    """
    df = split.to_pandas()
    grouped = df.groupby("_id")
    doc2_texts: List[str] = []

    for _, group in grouped:
        group = group.sort_values("experience_order")
        (_, _, target_titles_esco, target_desc_esco) = _select_language_fields(language, group)
        if len(target_titles_esco) < 2:
            continue
        for idx in range(len(target_titles_esco) - 1):
            doc2 = f"esco role: {target_titles_esco[idx + 1]} \n description: {target_desc_esco[idx + 1]}"
            doc2_texts.append(doc2)

    return doc2_texts


def filter_pairs_job_id_mode(
    raw_pairs: List[Tuple[str, str]],
    raw_job_ids: List[List[Any]],
    job_skill_map: Dict[str, List[Dict[str, Any]]],
    skill_desc_map: Dict[str, Dict[str, str]],
    include_skill_descriptions: bool,
) -> Tuple[List[Tuple[str, str]], Counter[str], Counter[str]]:
    """
    Replicates the filtering logic used to build skill-based pairs from job_ids,
    but also returns:
      - drop_reasons: counts per drop reason
      - dropped_target_titles: counts of target titles (from doc2) that were dropped
    """
    kept: List[Tuple[str, str]] = []
    drop_reasons: Counter[str] = Counter()
    dropped_target_titles: Counter[str] = Counter()

    for (doc1, doc2), job_ids in zip(raw_pairs, raw_job_ids):
        target_title = extract_esco_role_title(doc2)

        if not job_ids:
            drop_reasons["missing_job_ids"] += 1
            dropped_target_titles[target_title] += 1
            continue

        last_job_id = job_ids[-1] if job_ids else None
        if last_job_id is None:
            drop_reasons["last_job_id_is_none"] += 1
            dropped_target_titles[target_title] += 1
            continue

        last_job_id_str = str(last_job_id)
        skills = job_skill_map.get(last_job_id_str)
        if not skills:
            drop_reasons["missing_skills_for_last_job_id"] += 1
            dropped_target_titles[target_title] += 1
            continue

        skill_doc = _skills_to_doc(skills, skill_desc_map, include_skill_descriptions)
        if not skill_doc:
            drop_reasons["empty_skill_doc"] += 1
            dropped_target_titles[target_title] += 1
            continue

        kept.append((skill_doc, doc2))

    return kept, drop_reasons, dropped_target_titles


def raw_data_job_id_mode(
    data_type: str, consider_subspans: bool
) -> Tuple[
    Tuple[List[Tuple[str, str]], List[List[Any]]],
    Tuple[List[Tuple[str, str]], List[List[Any]]],
    Tuple[List[Tuple[str, str]], List[List[Any]]],
]:
    """
    Loads raw (doc1, doc2) pairs + job_ids using the Data class
    exactly like finetune_last_job_skills.load_skill_transition_pairs_with_job_ids().
    """
    data = Data(DATA_TYPE=data_type, consider_subspans=consider_subspans)
    (train, train_job_ids), (val, val_job_ids), (test, test_job_ids) = data.get_data_with_job_ids(
        stage="transformation_finetuning"
    )
    return (train, train_job_ids), (val, val_job_ids), (test, test_job_ids)


def build_job_skill_map_and_desc_map(args) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, str]], bool, bool]:
    """
    Mirrors finetune_last_job_skills main() for building:
      - job_skill_map
      - skill_desc_map
    Returns (job_skill_map, skill_desc_map, uses_predicted_skills, uses_esco_taxonomy)
    """
    FREE_TEXT_DATASETS = ("decorte", "karrierewege_occ", "karrierewege_cp")
    ESCO_TITLE_DATASETS = ("decorte_esco", "karrierewege", "karrierewege_100k")

    uses_predicted_skills = args.data_type in FREE_TEXT_DATASETS
    uses_esco_taxonomy = args.data_type in ESCO_TITLE_DATASETS

    if not uses_predicted_skills and not uses_esco_taxonomy:
        raise ValueError(f"Unknown data_type: {args.data_type}")

    if uses_predicted_skills and not args.skill_scores_json:
        raise ValueError(f"--skill_scores_json is required for {args.data_type} (free-text dataset with predicted skills)")
    if uses_esco_taxonomy and (not args.raw_esco_dir):
        if not args.job_title_skills_csv:
            raise ValueError(f"--job_title_skills_csv is required for {args.data_type} (ESCO-based dataset) if --raw_esco_dir is not provided")
        if not args.skills_csv:
            raise ValueError(f"--skills_csv is required for {args.data_type} if --raw_esco_dir is not provided")

    # Skill descriptions
    if args.raw_esco_dir and uses_esco_taxonomy:
        job_skill_map, skill_desc_map = load_raw_esco_taxonomy(args.raw_esco_dir)
    else:
        skill_desc_map = load_skill_descriptions(args.skills_csv) if args.skills_csv else {}

    include_skill_descriptions = not args.no_skill_descriptions

    # Build job_skill_map for the selected mode
    if uses_predicted_skills:
        job_skill_map = load_skills_by_job_id(args.skill_scores_json)

        if args.scoring_mode == "scores_only":
            if args.skill_selection_strategy == "stratified":
                job_skill_map = cap_skills_per_job_stratified(
                    job_skill_map,
                    max_skills_per_job=args.top_k_skills,
                    use_weighted_idf=False,
                )
            else:
                job_skill_map = cap_skills_per_job_by_score(
                    job_skill_map,
                    max_skills_per_job=args.top_k_skills,
                    skill_desc_map=skill_desc_map,
                )

        elif args.scoring_mode == "weighted":
            job_skill_map = calculate_idf_scores_by_job_id(
                job_skill_map,
                use_job_scores=True,
                importance_weight=args.importance_weight,
            )
            if args.skill_selection_strategy == "stratified":
                job_skill_map = cap_skills_per_job_stratified(
                    job_skill_map,
                    max_skills_per_job=args.top_k_skills,
                    use_weighted_idf=True,
                )
            else:
                job_skill_map = cap_skills_per_job_lexicographic(
                    job_skill_map,
                    max_skills_per_job=args.top_k_skills,
                    skill_desc_map=skill_desc_map,
                    use_weighted_idf=True,
                )

        elif args.scoring_mode == "idf_only":
            job_skill_map = calculate_idf_scores_by_job_id(job_skill_map, use_job_scores=False)
            if args.skill_selection_strategy == "stratified":
                job_skill_map = cap_skills_per_job_stratified(
                    job_skill_map,
                    max_skills_per_job=args.top_k_skills,
                    use_weighted_idf=False,
                )
            else:
                job_skill_map = cap_skills_per_job_lexicographic(
                    job_skill_map,
                    max_skills_per_job=args.top_k_skills,
                    skill_desc_map=skill_desc_map,
                    use_weighted_idf=False,
                )
        else:
            raise ValueError(f"Unknown scoring_mode: {args.scoring_mode}")

    else:
        if not args.raw_esco_dir:
            job_skill_map = load_skill_mappings(args.job_title_skills_csv)

        # ESCO mode always uses IDF (scores are all 1)
        job_skill_map = calculate_idf_scores(job_skill_map)
        if args.skill_selection_strategy == "stratified":
            job_skill_map = cap_skills_per_job_stratified(
                job_skill_map,
                max_skills_per_job=args.top_k_skills,
                use_weighted_idf=False,
            )
        else:
            job_skill_map = cap_skills_per_job_lexicographic(
                job_skill_map,
                max_skills_per_job=args.top_k_skills,
                skill_desc_map=skill_desc_map,
                use_weighted_idf=False,
            )

    return job_skill_map, skill_desc_map, uses_predicted_skills, uses_esco_taxonomy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Investigate candidate pool discrepancy (1022 vs 600 etc.)")
    p.add_argument(
        "--data_type",
        type=str,
        required=True,
        choices=["karrierewege", "karrierewege_occ", "karrierewege_100k", "karrierewege_cp", "decorte", "decorte_esco"],
    )
    p.add_argument("--skills_csv", type=str, default=None, help="Skills CSV (for skill descriptions).")
    p.add_argument("--raw_esco_dir", type=str, default=None, help="Raw ESCO taxonomy directory (optional).")
    p.add_argument("--job_title_skills_csv", type=str, default=None, help="Job-title -> skills mapping CSV (ESCO mode).")
    p.add_argument("--skill_scores_json", type=str, default=None, help="fused_predictions.json (job_id -> skills with scores).")

    p.add_argument("--top_k_skills", type=int, default=10)
    p.add_argument("--skill_selection_strategy", type=str, default="top_k", choices=["top_k", "stratified"])
    p.add_argument("--no_skill_descriptions", action="store_true")
    p.add_argument("--consider_subspans", action="store_true")

    p.add_argument("--scoring_mode", type=str, default="scores_only", choices=["idf_only", "scores_only", "weighted"])
    p.add_argument("--importance_weight", type=float, default=0.5)

    p.add_argument("--write_debug_json", type=str, default=None, help="Optional path to write a debug JSON blob.")
    p.add_argument("--top_n_missing", type=int, default=30, help="How many missing targets to print.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    job_skill_map, skill_desc_map, uses_predicted_skills, uses_esco_taxonomy = build_job_skill_map_and_desc_map(args)
    include_skill_descriptions = not args.no_skill_descriptions

    debug: Dict[str, Any] = {
        "config": vars(args),
        "mode": {
            "uses_predicted_skills": uses_predicted_skills,
            "uses_esco_taxonomy": uses_esco_taxonomy,
            "include_skill_descriptions": include_skill_descriptions,
        },
        "counts": {},
        "drops": {},
        "missing_targets": {},
    }

    if uses_predicted_skills:
        # RAW: Data-class pairs (doc2 targets exist regardless of whether we have skills for the last job_id)
        (raw_train, raw_train_job_ids), (raw_val, raw_val_job_ids), (raw_test, raw_test_job_ids) = raw_data_job_id_mode(
            args.data_type, consider_subspans=args.consider_subspans
        )
        raw_all_pairs = raw_train + raw_val + raw_test
        raw_all_doc2 = [p for _, p in raw_all_pairs]
        raw_unique_doc2_texts = set(raw_all_doc2)
        raw_unique_titles = set(extract_esco_role_title(p) for p in raw_all_doc2)

        # FILTERED: apply the exact job_id -> skills filter
        train_pairs, train_drop_reasons, train_dropped_titles = filter_pairs_job_id_mode(
            raw_train, raw_train_job_ids, job_skill_map, skill_desc_map, include_skill_descriptions
        )
        val_pairs, val_drop_reasons, val_dropped_titles = filter_pairs_job_id_mode(
            raw_val, raw_val_job_ids, job_skill_map, skill_desc_map, include_skill_descriptions
        )
        test_pairs, test_drop_reasons, test_dropped_titles = filter_pairs_job_id_mode(
            raw_test, raw_test_job_ids, job_skill_map, skill_desc_map, include_skill_descriptions
        )
        drop_reasons = train_drop_reasons + val_drop_reasons + test_drop_reasons
        dropped_titles = train_dropped_titles + val_dropped_titles + test_dropped_titles

    else:
        # RAW: all possible transitions (targets) from the HF dataset, regardless of skill availability
        dataset_name, language = dataset_spec_for_esco_mode(args.data_type)
        ds = load_dataset(dataset_name)

        raw_doc2_train = raw_doc2_texts_from_esco_dataset_split(ds["train"], language)
        raw_doc2_val = raw_doc2_texts_from_esco_dataset_split(ds["validation"], language)
        raw_doc2_test = raw_doc2_texts_from_esco_dataset_split(ds["test"], language)

        raw_all_doc2 = raw_doc2_train + raw_doc2_val + raw_doc2_test
        raw_unique_doc2_texts = set(raw_all_doc2)
        raw_unique_titles = set(extract_esco_role_title(p) for p in raw_all_doc2)

        # FILTERED: replicate _build_pairs_for_split but track drops
        def filter_esco_split(split) -> Tuple[List[Tuple[str, str]], Counter[str], Counter[str]]:
            df = split.to_pandas()
            grouped = df.groupby("_id")
            kept_local: List[Tuple[str, str]] = []
            reasons: Counter[str] = Counter()
            dropped_target_titles_local: Counter[str] = Counter()

            for _, group in grouped:
                group = group.sort_values("experience_order")
                (titles_for_skills, _, target_titles_esco, target_desc_esco) = _select_language_fields(language, group)
                if len(target_titles_esco) < 2:
                    reasons["too_short_history"] += 1
                    continue
                for idx in range(len(target_titles_esco) - 1):
                    source_title_norm = _normalize_title(titles_for_skills[idx])
                    target_doc2 = f"esco role: {target_titles_esco[idx + 1]} \n description: {target_desc_esco[idx + 1]}"
                    target_title = extract_esco_role_title(target_doc2)

                    skills = job_skill_map.get(source_title_norm)
                    if not skills:
                        reasons["missing_skills_for_source_title"] += 1
                        dropped_target_titles_local[target_title] += 1
                        continue

                    doc1 = _skills_to_doc(skills, skill_desc_map, include_skill_descriptions)
                    if not doc1:
                        reasons["empty_skill_doc"] += 1
                        dropped_target_titles_local[target_title] += 1
                        continue

                    kept_local.append((doc1, target_doc2))

            return kept_local, reasons, dropped_target_titles_local

        train_pairs, train_drop_reasons, train_dropped_titles = filter_esco_split(ds["train"])
        val_pairs, val_drop_reasons, val_dropped_titles = filter_esco_split(ds["validation"])
        test_pairs, test_drop_reasons, test_dropped_titles = filter_esco_split(ds["test"])

        drop_reasons = train_drop_reasons + val_drop_reasons + test_drop_reasons
        dropped_titles = train_dropped_titles + val_dropped_titles + test_dropped_titles

    # Build candidate pools the way retrieval eval is supposed to
    filtered_all_pairs = train_pairs + val_pairs + test_pairs
    filtered_all_doc2 = [p for _, p in filtered_all_pairs]
    filtered_unique_doc2_texts = set(filtered_all_doc2)
    filtered_unique_titles = set(extract_esco_role_title(p) for p in filtered_all_doc2)

    # Candidate pools
    candidate_pool_full = unique_preserve_order(filtered_all_doc2)  # intended: train+val+test unique doc2
    candidate_pool_test_only = unique_preserve_order([p for _, p in test_pairs])

    # Missing targets (by TITLE) and (by DOC2 TEXT)
    missing_titles = sorted(raw_unique_titles - filtered_unique_titles)
    missing_doc2_texts = sorted(raw_unique_doc2_texts - filtered_unique_doc2_texts)

    print("=" * 100)
    print("CONFIG")
    print("=" * 100)
    print(json.dumps({"data_type": args.data_type, **debug["mode"]}, indent=2))

    print("\n" + "=" * 100)
    print("RAW (dataset transitions) vs FILTERED (skill-based pairs)")
    print("=" * 100)
    print(f"RAW unique doc2 texts:  {len(raw_unique_doc2_texts)}")
    print(f"RAW unique doc2 titles: {len(raw_unique_titles)}")
    print(f"FILTERED unique doc2 texts:  {len(filtered_unique_doc2_texts)}")
    print(f"FILTERED unique doc2 titles: {len(filtered_unique_titles)}")

    print("\n" + "=" * 100)
    print("SPLIT STATS (FILTERED)")
    print("=" * 100)
    print(f"train: {split_stats(train_pairs)}")
    print(f"val:   {split_stats(val_pairs)}")
    print(f"test:  {split_stats(test_pairs)}")

    print("\n" + "=" * 100)
    print("CANDIDATE POOL SANITY CHECKS")
    print("=" * 100)
    print(f"Candidate pool (train+val+test unique doc2 texts): {len(candidate_pool_full)}")
    print(f"Candidate pool (test-only unique doc2 texts):      {len(candidate_pool_test_only)}")
    if len(candidate_pool_test_only) != len(candidate_pool_full):
        print("NOTE: If your log shows the test-only number as 'Candidate pool size',")
        print("      then retrieval eval is likely NOT using the full candidate pool argument.")

    print("\n" + "=" * 100)
    print("DROP REASONS (FILTERING)")
    print("=" * 100)
    for k, v in drop_reasons.most_common():
        print(f"{k}: {v}")

    print("\n" + "=" * 100)
    print("MISSING TARGETS SUMMARY")
    print("=" * 100)
    print(f"Missing titles (RAW - FILTERED): {len(missing_titles)}")
    print(f"Missing doc2 texts (RAW - FILTERED): {len(missing_doc2_texts)}")

    # Show the most frequent dropped target titles (where we tracked them)
    if dropped_titles:
        print("\nTop dropped target titles (by frequency in dropped examples):")
        for title, cnt in dropped_titles.most_common(args.top_n_missing):
            print(f"- {cnt:6d}  {title}")

    # Also show a few examples of missing doc2 texts (truncated)
    if missing_doc2_texts:
        print("\nExample missing doc2 texts (first 10; truncated to 200 chars):")
        for s in missing_doc2_texts[:10]:
            s_trunc = s if len(s) <= 200 else (s[:200] + "...")
            print(f"- {s_trunc}")

    debug["counts"] = {
        "raw_unique_doc2_texts": len(raw_unique_doc2_texts),
        "raw_unique_doc2_titles": len(raw_unique_titles),
        "filtered_unique_doc2_texts": len(filtered_unique_doc2_texts),
        "filtered_unique_doc2_titles": len(filtered_unique_titles),
        "candidate_pool_full_unique_doc2_texts": len(candidate_pool_full),
        "candidate_pool_test_only_unique_doc2_texts": len(candidate_pool_test_only),
        "missing_titles": len(missing_titles),
        "missing_doc2_texts": len(missing_doc2_texts),
    }
    debug["drops"] = {
        "drop_reasons": dict(drop_reasons),
        "dropped_target_titles_top": dropped_titles.most_common(args.top_n_missing),
    }
    debug["missing_targets"] = {
        "missing_titles_sample": missing_titles[: min(200, len(missing_titles))],
        "missing_doc2_texts_sample": missing_doc2_texts[: min(50, len(missing_doc2_texts))],
    }

    if args.write_debug_json:
        out_path = os.path.abspath(args.write_debug_json)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(debug, f, indent=2, ensure_ascii=False)
        print(f"\nWrote debug JSON to: {out_path}")


if __name__ == "__main__":
    main()

