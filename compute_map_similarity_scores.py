#!/usr/bin/env python3
"""
Compute MAP@K from a huge similarity score JSON file.

Expected formats
----------------
1) Scores file (pretty-printed huge JSON object):
   {
     "0": [
       {"skill_uri": "http://data.europa.eu/esco/skill/...", "score": 0.88, "rank": 1},
       ...
     ],
     "1": [...],
     ...
   }

2) Job_id -> occupationUri mapping (CSV):
   /.../decorte_master_2.csv
   columns include: job_id, esco_id (occupationUri)

3) Ground truth occupation -> skills (CSV):
   /.../occupationSkillRelations_en.csv
   columns: occupationUri, relationType, skillType, skillUri
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from heapq import nsmallest
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple


class CharStream:
    """Buffered character stream for large files (faster than read(1))."""

    def __init__(self, f, chunk_size: int = 1 << 20):
        self._f = f
        self._chunk_size = chunk_size
        self._buf = ""
        self._i = 0
        self._eof = False

    def _fill(self, min_available: int = 1) -> None:
        if self._eof:
            return
        if (len(self._buf) - self._i) >= min_available:
            return
        # Compact buffer
        if self._i > 0:
            self._buf = self._buf[self._i :]
            self._i = 0
        chunk = self._f.read(self._chunk_size)
        if chunk == "":
            self._eof = True
            return
        self._buf += chunk

    def peek(self) -> str:
        self._fill(1)
        if self._i >= len(self._buf):
            return ""
        return self._buf[self._i]

    def get(self) -> str:
        self._fill(1)
        if self._i >= len(self._buf):
            return ""
        ch = self._buf[self._i]
        self._i += 1
        return ch

    def skip_ws(self) -> None:
        while True:
            ch = self.peek()
            if ch and ch.isspace():
                self.get()
                continue
            return

    def skip_ws_and_commas(self) -> None:
        while True:
            ch = self.peek()
            if ch == ",":
                self.get()
                continue
            if ch and ch.isspace():
                self.get()
                continue
            return


def _expect(cs: CharStream, expected: str) -> None:
    got = cs.get()
    if got != expected:
        raise ValueError(f"Expected {expected!r}, got {got!r}")


def _read_json_string(cs: CharStream) -> str:
    """Read a JSON string token and return its decoded Python string."""
    _expect(cs, '"')
    out_chars: List[str] = []
    escape = False
    while True:
        ch = cs.get()
        if ch == "":
            raise ValueError("Unexpected EOF while reading JSON string")
        if escape:
            # Keep escapes; we'll decode via json.loads on the full string at the end.
            out_chars.append("\\" + ch)
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            break
        out_chars.append(ch)
    # Decode escapes safely by re-wrapping as a JSON string.
    return json.loads('"' + "".join(out_chars) + '"')


def _read_json_array_text(cs: CharStream) -> str:
    """Read a JSON array token and return its raw text (including brackets)."""
    ch0 = cs.peek()
    if ch0 != "[":
        raise ValueError(f"Expected '[' to start array, got {ch0!r}")

    out: List[str] = []
    depth = 0
    in_str = False
    escape = False

    while True:
        ch = cs.get()
        if ch == "":
            raise ValueError("Unexpected EOF while reading JSON array")
        out.append(ch)

        if in_str:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_str = False
            continue

        # Not in string
        if ch == '"':
            in_str = True
            continue
        if ch == "[":
            depth += 1
            continue
        if ch == "]":
            depth -= 1
            if depth == 0:
                break
            continue

    return "".join(out)


def iter_scores_from_top_level_object(path: str) -> Iterator[Tuple[str, List[dict]]]:
    """
    Stream a huge JSON file structured as a single top-level object:
      { "job_id": [ ... ], "job_id2": [ ... ], ... }
    Yields (job_id, list_of_items) without loading the whole file.
    """
    with open(path, "r", encoding="utf-8") as f:
        cs = CharStream(f)
        cs.skip_ws()
        _expect(cs, "{")

        while True:
            cs.skip_ws_and_commas()
            ch = cs.peek()
            if ch == "}":
                cs.get()
                break
            if ch == "":
                raise ValueError("Unexpected EOF while reading top-level object")

            key = _read_json_string(cs)
            cs.skip_ws()
            _expect(cs, ":")
            cs.skip_ws()
            arr_text = _read_json_array_text(cs)

            try:
                items = json.loads(arr_text)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse array for key={key!r}: {e}") from e

            if not isinstance(items, list):
                raise ValueError(f"Expected list for key={key!r}, got {type(items)}")
            yield key, items


def load_job_to_occupation_uri(path: str) -> Dict[str, str]:
    job_to_occ: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        if "job_id" not in reader.fieldnames or "esco_id" not in reader.fieldnames:
            raise ValueError(
                f"Expected columns job_id and esco_id in {path}, got {reader.fieldnames}"
            )
        for row in reader:
            job_id = str(row["job_id"]).strip()
            occ = str(row["esco_id"]).strip()
            if not job_id or not occ:
                continue
            job_to_occ[job_id] = occ
    return job_to_occ


def load_ground_truth(
    path: str,
    allowed_relation_types: Set[str],
) -> Dict[str, Set[str]]:
    occ_to_skills: Dict[str, Set[str]] = defaultdict(set)
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"occupationUri", "relationType", "skillUri"}
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Expected columns {sorted(required)} in {path}, got {reader.fieldnames}"
            )
        for row in reader:
            rel = str(row["relationType"]).strip()
            if rel not in allowed_relation_types:
                continue
            occ = str(row["occupationUri"]).strip()
            skill = str(row["skillUri"]).strip()
            if occ and skill:
                occ_to_skills[occ].add(skill)
    return dict(occ_to_skills)


def _is_non_decreasing_ranks(items: Sequence[dict], sample: int = 200) -> bool:
    last = -math.inf
    for x in items[:sample]:
        r = x.get("rank", None)
        if r is None:
            return False
        try:
            r = int(r)
        except Exception:
            return False
        if r < last:
            return False
        last = r
    return True


def iter_topk_skill_uris(items: Sequence[dict], k: int) -> Iterator[str]:
    """
    Yield up to k skill URIs, in rank order, de-duplicated by URI.
    Falls back to rank-based selection if the list order isn't monotonic by 'rank'.
    """
    if k <= 0:
        return

    # If the list is already in increasing rank order, just iterate.
    if _is_non_decreasing_ranks(items):
        ordered = items
    else:
        # Take n smallest by rank without sorting the whole list.
        def _rank(x: dict) -> int:
            r = x.get("rank", 10**18)
            try:
                return int(r)
            except Exception:
                return 10**18

        ordered = nsmallest(k, items, key=_rank)

    seen: Set[str] = set()
    yielded = 0
    for x in ordered:
        if yielded >= k:
            break
        uri = x.get("skill_uri", None)
        if not uri:
            continue
        uri = str(uri)
        if uri in seen:
            continue
        seen.add(uri)
        yielded += 1
        yield uri


def average_precision_at_k(preds: Iterable[str], gt: Set[str], k: int) -> float:
    """AP@k with binary relevance; denominator is |gt| (standard IR AP)."""
    if not gt:
        return float("nan")
    hits = 0
    sum_prec = 0.0
    for i, uri in enumerate(preds, start=1):
        if i > k:
            break
        if uri in gt:
            hits += 1
            sum_prec += hits / i
    return sum_prec / len(gt)


@dataclass
class EvalStats:
    processed: int = 0
    missing_job_mapping: int = 0
    missing_ground_truth: int = 0
    empty_ground_truth: int = 0
    evaluated: int = 0
    map1000_sum: float = 0.0
    map3000_sum: float = 0.0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scores",
        required=True,
        help="Path to similarity_scores.json (top-level object: job_id -> list of skill candidates).",
    )
    ap.add_argument(
        "--mapping",
        required=True,
        help="Path to decorte_master_2.csv (must contain columns: job_id, esco_id).",
    )
    ap.add_argument(
        "--ground-truth",
        required=True,
        help="Path to occupationSkillRelations_en.csv.",
    )
    ap.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[1000, 3000],
        help="Cutoffs K to compute MAP@K for (default: 1000 3000).",
    )
    ap.add_argument(
        "--relation-types",
        nargs="+",
        default=["essential", "optional"],
        help="Which relationType values to include in ground truth (default: essential optional).",
    )
    ap.add_argument(
        "--max-jobs",
        type=int,
        default=0,
        help="If >0, stop after this many jobs (useful for quick testing).",
    )
    ap.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="Print progress every N processed job_ids (default: 200).",
    )
    args = ap.parse_args(argv)

    ks = sorted(set(args.k))
    if ks != [1000, 3000]:
        # Script still supports arbitrary ks, but prints the common ones nicely.
        pass
    k1000 = 1000 if 1000 in ks else ks[0]
    k3000 = 3000 if 3000 in ks else ks[-1]
    max_k = max(ks)

    allowed_relation_types = {str(x).strip() for x in args.relation_types if str(x).strip()}
    if not allowed_relation_types:
        raise ValueError("No --relation-types provided after stripping.")

    job_to_occ = load_job_to_occupation_uri(args.mapping)
    occ_to_skills = load_ground_truth(args.ground_truth, allowed_relation_types)

    stats = EvalStats()

    for job_id, items in iter_scores_from_top_level_object(args.scores):
        stats.processed += 1

        occ = job_to_occ.get(str(job_id), None)
        if not occ:
            stats.missing_job_mapping += 1
            continue

        gt = occ_to_skills.get(occ, None)
        if gt is None:
            stats.missing_ground_truth += 1
            continue
        if len(gt) == 0:
            stats.empty_ground_truth += 1
            continue

        preds = list(iter_topk_skill_uris(items, max_k))
        ap1000 = average_precision_at_k(preds, gt, k1000)
        ap3000 = average_precision_at_k(preds, gt, k3000)

        # Should never be NaN here because gt non-empty, but keep it safe.
        if not math.isnan(ap1000) and not math.isnan(ap3000):
            stats.evaluated += 1
            stats.map1000_sum += ap1000
            stats.map3000_sum += ap3000

        if args.progress_every > 0 and (stats.processed % args.progress_every) == 0:
            denom = max(1, stats.evaluated)
            print(
                f"processed={stats.processed} evaluated={stats.evaluated} "
                f"MAP@{k1000}={stats.map1000_sum/denom:.6f} MAP@{k3000}={stats.map3000_sum/denom:.6f}",
                file=sys.stderr,
            )

        if args.max_jobs and stats.processed >= args.max_jobs:
            break

    if stats.evaluated == 0:
        print("No evaluable queries found (evaluated=0). Check mappings/ground truth.", file=sys.stderr)
        return 2

    map1000 = stats.map1000_sum / stats.evaluated
    map3000 = stats.map3000_sum / stats.evaluated

    print(f"MAP@{k1000}: {map1000:.8f}")
    print(f"MAP@{k3000}: {map3000:.8f}")
    print("---")
    print(f"processed_job_ids: {stats.processed}")
    print(f"evaluated_job_ids: {stats.evaluated}")
    print(f"missing_job_mapping: {stats.missing_job_mapping}")
    print(f"missing_ground_truth: {stats.missing_ground_truth}")
    print(f"empty_ground_truth: {stats.empty_ground_truth}")
    print(f"relation_types_used: {sorted(allowed_relation_types)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

