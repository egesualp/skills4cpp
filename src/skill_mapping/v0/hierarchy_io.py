# hierarchy_io.py
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import json, csv, pathlib

@dataclass(frozen=True)
class ESCOGroup:
    id: str
    label: str
    pillar_id: str | None = None
    pillar_label: str | None = None
    parent_id: str | None = None

@dataclass(frozen=True)
class ESCOSkill:
    id: str
    label: str
    group_ids: tuple[str, ...]
    pillar_ids: tuple[str, ...] = ()

def _read_json(p): 
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def load_esco_hierarchy(
    csv_path: str,                 # skills_to_group_pilar.csv
    group2label_json: str,         # group2label.json
    group2parent_json: str,        # group2parent.json   (may be missing for roots)
    group2pillar_json: str,        # group2pillar.json
    skill2group_json: str,         # skill2group.json    (skill -> [groups])
    skill2pillar_json: str | None = None,  # optional
):
    # --- 1) Load basic mappings
    group2label  = _read_json(group2label_json)            # {group_id: group_label}
    group2parent = _read_json(group2parent_json)           # {group_id: parent_group_id or null}
    group2pillar = _read_json(group2pillar_json)           # {group_id: pillar_id}
    skill2group  = _read_json(skill2group_json)            # {skill_id: [group_id, ...]}
    skill2pillar = _read_json(skill2pillar_json) if skill2pillar_json else {}

    # --- 2) From CSV, collect human-readable labels (skill & pillar labels)
    skill_label = {}
    pillar_label = {}
    group_label_from_csv = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row["skill_id"].strip()
            gid = row["group_id"].strip()
            skill_label[sid] = row["skill_label"].strip()
            group_label_from_csv[gid] = row["group_label"].strip()
            pid = row["pillar_id"].strip()
            if pid:
                pillar_label[pid] = row["pillar_label"].strip()

    # Backfill group labels if JSON has them; CSV wins when present
    for gid, glab in group2label.items():
        group_label_from_csv.setdefault(gid, glab)

    # --- 3) Build group objects (Level-2 classes)
    groups: dict[str, ESCOGroup] = {}
    for gid, glab in group_label_from_csv.items():
        pid = group2pillar.get(gid)
        groups[gid] = ESCOGroup(
            id=gid,
            label=glab,
            pillar_id=pid,
            pillar_label=pillar_label.get(pid),
            parent_id=group2parent.get(gid)
        )

    # --- 4) Build skills with multi-parent groups
    skills: dict[str, ESCOSkill] = {}
    for sid, gids in skill2group.items():
        gids = tuple(sorted(set(gids)))
        skills[sid] = ESCOSkill(
            id=sid,
            label=skill_label.get(sid, sid.split("/")[-1].replace("-", " ")),
            group_ids=gids,
            pillar_ids=tuple(sorted(set(
                [groups[g].pillar_id for g in gids if g in groups] + (skill2pillar.get(sid, []) or [])
            )))
        )

    # --- 5) class_to_skills & skill_texts
    class_to_skills = defaultdict(set)
    for s in skills.values():
        for g in s.group_ids:
            class_to_skills[g].add(s.id)

    # --- 6) siblings (for hard negatives)
    # sibling groups = groups sharing the same parent
    parent_to_children = defaultdict(list)
    for gid, g in groups.items():
        parent_to_children[g.parent_id].append(gid)  # parent_id may be None for roots
    siblings = {gid: set(parent_to_children.get(groups[gid].parent_id, [])) - {gid} for gid in groups}

    # --- 7) Pillar helpers (optional)
    pillar_to_groups = defaultdict(set)
    for gid, g in groups.items():
        if g.pillar_id:
            pillar_to_groups[g.pillar_id].add(gid)

    # --- 8) Friendly text dicts
    skill_texts  = {s.id: s.label for s in skills.values()}
    class_texts  = {g.id: g.label for g in groups.values()}
    pillar_texts = {pid: plab for pid, plab in pillar_label.items()}

    return {
        "groups": groups,                       # dict[str, ESCOGroup]
        "skills": skills,                       # dict[str, ESCOSkill]
        "class_to_skills": {k: sorted(v) for k, v in class_to_skills.items()},
        "siblings": {k: sorted(v) for k, v in siblings.items()},
        "pillar_to_groups": {k: sorted(v) for k, v in pillar_to_groups.items()},
        "skill_texts": skill_texts,
        "class_texts": class_texts,
        "pillar_texts": pillar_texts,
    }

# candidates.py
import random
from typing import Iterable

def candidate_skills_for_topk(
    topk_classes: list[tuple[str, float]],  # [(class_id, prob), ...]
    class_to_skills: dict[str, list[str]],
    extra_per_class: int | None = None,    # if you want to cap per class
) -> list[str]:
    cand = set()
    for cid, _ in topk_classes:
        skills = class_to_skills.get(cid, [])
        if extra_per_class and len(skills) > extra_per_class:
            skills = random.sample(skills, extra_per_class)
        cand.update(skills)
    return sorted(cand)

def sample_hard_negatives(
    gold_skills: set[str],
    pos_class_ids: Iterable[str],
    class_to_skills: dict[str, list[str]],
    siblings: dict[str, list[str]],
    n_from_same: int = 16,
    n_from_siblings: int = 16,
) -> list[str]:
    same_pool = set()
    sib_pool  = set()
    for cid in pos_class_ids:
        same_pool.update(class_to_skills.get(cid, []))
        for sib in siblings.get(cid, []):
            sib_pool.update(class_to_skills.get(sib, []))
    same_pool.difference_update(gold_skills)
    sib_pool.difference_update(gold_skills)

    negs = []
    if same_pool:
        negs += random.sample(list(same_pool), min(n_from_same, len(same_pool)))
    if sib_pool:
        negs += random.sample(list(sib_pool), min(n_from_siblings, len(sib_pool)))
    return negs
