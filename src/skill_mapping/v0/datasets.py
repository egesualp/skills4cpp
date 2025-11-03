# datasets.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Iterable, Tuple, Optional
import csv, json, random
from collections import defaultdict
import torch
from torch.utils.data import Dataset

# ---------- tiny CSV reader for your jobs ----------
@dataclass
class JobExample:
    job_id: str
    text: str                      # title or title + description
    group_ids: List[str]           # gold Level-2 class ids (can be empty)
    skill_ids: List[str]           # gold skill ids (can be empty)

def load_jobs_csv(
    path: str,
    title_col: str = "title",
    desc_col: Optional[str] = "description",
    group_col: str = "group_ids",   # e.g. "id1;id2"
    skill_col: str = "skill_ids",   # e.g. "id1;id2;..."
    sep: str = ";",
    id_col: str = "id",
    join_title_desc: bool = True,
) -> List[JobExample]:
    jobs = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            title = (row.get(title_col) or "").strip()
            desc  = (row.get(desc_col) or "").strip() if desc_col else ""
            text = f"{title} [SEP] {desc}" if join_title_desc and desc else title
            groups = [x for x in (row.get(group_col) or "").split(sep) if x]
            skills = [x for x in (row.get(skill_col) or "").split(sep) if x]
            jobs.append(JobExample(
                job_id=row.get(id_col, str(len(jobs))),
                text=text,
                group_ids=groups,
                skill_ids=skills
            ))
    return jobs

def load_title_pairs_csv(
    path: str,
    raw_title_col: str = "raw_title",
    raw_desc_col: str = "raw_description",
    esco_id_col: str = "esco_id",
    skill2group: Optional[Dict[str, List[str]]] = None,
    join_title_desc: bool = True,
    sep: str = " [SEP] ",
    id_col: Optional[str] = None,
) -> List[JobExample]:
    """
    Load job data from title_pairs_desc CSV files (DECORTE, Karrierewege+).
    
    These datasets contain columns like raw_title, raw_description, esco_title, 
    esco_description, and esco_id. This function:
    - Combines raw_title and raw_description into a single text field
    - Uses esco_id as the gold skill ID
    - Optionally maps esco_id to ESCO group/category IDs via skill2group mapping
    
    Args:
        path: Path to the CSV file
        raw_title_col: Column name for raw job title (default: "raw_title")
        raw_desc_col: Column name for raw job description (default: "raw_description")
        esco_id_col: Column name for ESCO skill ID (default: "esco_id")
        skill2group: Optional mapping from skill IDs to group IDs. Can be loaded 
                     from skill2group.json via hierarchy_io.load_skill2group()
        join_title_desc: Whether to combine title and description (default: True)
        sep: Separator between title and description (default: " [SEP] ")
        id_col: Optional column name for job ID. If None, uses row index.
    
    Returns:
        List of JobExample objects with:
        - job_id: str (from id_col or row index)
        - text: str (combined raw_title and raw_description)
        - group_ids: List[str] (mapped from esco_id via skill2group, or empty)
        - skill_ids: List[str] (contains esco_id)
    
    Example:
        >>> import json
        >>> # Load skill-to-group mapping
        >>> with open("data/processed/skill2group.json") as f:
        ...     skill2group = json.load(f)
        >>> # Load job data with group mapping
        >>> jobs = load_title_pairs_csv(
        ...     "data/title_pairs_desc/decorte_train_pairs.csv",
        ...     skill2group=skill2group
        ... )
    """
    jobs = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for idx, row in enumerate(r):
            raw_title = (row.get(raw_title_col) or "").strip()
            raw_desc = (row.get(raw_desc_col) or "").strip()
            esco_id = (row.get(esco_id_col) or "").strip()
            
            # Combine title and description
            if join_title_desc and raw_desc:
                text = f"{raw_title}{sep}{raw_desc}"
            else:
                text = raw_title
            
            # Extract skill IDs (single esco_id per row)
            skill_ids = [esco_id] if esco_id else []
            
            # Map skill to groups if hierarchy provided
            group_ids = []
            if skill2group and esco_id:
                # skill2group can be dict[skill_id -> group_id] or dict[skill_id -> list[group_id]]
                mapped = skill2group.get(esco_id, [])
                if isinstance(mapped, str):
                    group_ids = [mapped]
                elif isinstance(mapped, list):
                    group_ids = mapped
            
            # Get job_id
            if id_col and id_col in row:
                job_id = row[id_col]
            else:
                job_id = str(idx)
            
            jobs.append(JobExample(
                job_id=job_id,
                text=text,
                group_ids=group_ids,
                skill_ids=skill_ids
            ))
    
    return jobs

# ---------- helpers from your hierarchy ----------
# Expect the dictionary returned by load_esco_hierarchy(...)
Hierarchy = Dict[str, object]

def build_index_maps(H: Hierarchy):
    # groups/classes
    class_ids = sorted(H["class_texts"].keys())
    class2idx = {cid: i for i, cid in enumerate(class_ids)}
    # skills
    skill_ids = sorted(H["skill_texts"].keys())
    skill2idx = {sid: i for i, sid in enumerate(skill_ids)}
    return class_ids, class2idx, skill_ids, skill2idx

# ---------- STEP 1: Class (category) dataset ----------
class ClassDataset(Dataset):
    """
    Yields: dict(text=str, target=[class_idx, ...])
    Use BCEWithLogitsLoss for multi-label, or CE if you choose single label.
    """
    def __init__(self, jobs: List[JobExample], class2idx: Dict[str, int]):
        self.jobs = jobs
        self.class2idx = class2idx

    def __len__(self): return len(self.jobs)

    def __getitem__(self, i: int):
        j = self.jobs[i]
        cls_idx = [self.class2idx[c] for c in j.group_ids if c in self.class2idx]
        return {"text": j.text, "class_idx": cls_idx, "job_id": j.job_id}

def class_collate(batch, tokenizer=None, n_classes: int = None):
    texts = [b["text"] for b in batch]
    # labels (multi-hot) if n_classes provided, else variable-length indices
    class_targets = None
    if n_classes is not None:
        class_targets = torch.zeros(len(batch), n_classes, dtype=torch.float32)
        for i, b in enumerate(batch):
            for c in b["class_idx"]:
                class_targets[i, c] = 1.0
    if tokenizer is None:
        return {"texts": texts, "labels": class_targets, "job_ids": [b["job_id"] for b in batch]}
    tok = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    tok["labels"] = class_targets
    tok["job_ids"] = [b["job_id"] for b in batch]
    return tok

# ---------- STEP 2: Skill-ranking dataset with hard negatives ----------
def _sample_hard_negs(
    gold_skills: Iterable[str],
    gold_groups: Iterable[str],
    class_to_skills: Dict[str, List[str]],
    siblings: Dict[str, List[str]],
    n_from_same: int = 16,
    n_from_sibs: int = 16,
) -> List[str]:
    gold_skills = set(gold_skills)
    same_pool, sib_pool = set(), set()
    for gid in gold_groups:
        same_pool.update(class_to_skills.get(gid, []))
        for sib in siblings.get(gid, []):
            sib_pool.update(class_to_skills.get(sib, []))
    same_pool -= gold_skills
    sib_pool  -= gold_skills
    negs = []
    if same_pool:
        negs += random.sample(list(same_pool), min(n_from_same, len(same_pool)))
    if sib_pool:
        negs += random.sample(list(sib_pool), min(n_from_sibs, len(sib_pool)))
    return negs

class SkillRankingDataset(Dataset):
    """
    Yields: dict(text, pos_skill_idx, neg_skill_idx, gold_group_idx)
    For each job, we re-sample negatives on the fly (optional).
    """
    def __init__(
        self,
        jobs: List[JobExample],
        H: Hierarchy,
        class2idx: Dict[str, int],
        skill2idx: Dict[str, int],
        n_negs_same: int = 16,
        n_negs_sibs: int = 16,
        resample_each_call: bool = True,
    ):
        self.jobs = jobs
        self.class2idx = class2idx
        self.skill2idx = skill2idx
        self.c2s = H["class_to_skills"]          # dict[class_id -> list[skill_id]]
        self.siblings = H["siblings"]            # dict[class_id -> list[class_id]]
        self.n_negs_same = n_negs_same
        self.n_negs_sibs = n_negs_sibs
        self.resample_each_call = resample_each_call

        # cache a first set of negatives
        self._cached_negs = [self._sample_negs(j) for j in self.jobs]

    def _sample_negs(self, j: JobExample) -> List[str]:
        return _sample_hard_negs(
            gold_skills=j.skill_ids,
            gold_groups=j.group_ids,
            class_to_skills=self.c2s,
            siblings=self.siblings,
            n_from_same=self.n_negs_same,
            n_from_sibs=self.n_negs_sibs,
        )

    def __len__(self): return len(self.jobs)

    def __getitem__(self, i: int):
        j = self.jobs[i]
        if self.resample_each_call:
            neg_skill_ids = self._sample_negs(j)
        else:
            neg_skill_ids = self._cached_negs[i]

        pos_idx = [self.skill2idx[s] for s in j.skill_ids if s in self.skill2idx]
        neg_idx = [self.skill2idx[s] for s in neg_skill_ids if s in self.skill2idx]
        grp_idx = [self.class2idx[g] for g in j.group_ids if g in self.class2idx]

        return {
            "text": j.text,
            "pos_idx": pos_idx,
            "neg_idx": neg_idx,
            "group_idx": grp_idx,
            "job_id": j.job_id
        }

def skill_collate(batch, tokenizer=None):
    texts = [b["text"] for b in batch]
    pos = [torch.tensor(b["pos_idx"], dtype=torch.long) for b in batch]
    neg = [torch.tensor(b["neg_idx"], dtype=torch.long) for b in batch]
    groups = [torch.tensor(b["group_idx"], dtype=torch.long) for b in batch]
    job_ids = [b["job_id"] for b in batch]

    if tokenizer is None:
        return {"texts": texts, "pos": pos, "neg": neg, "groups": groups, "job_ids": job_ids}

    tok = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    tok.update({"pos": pos, "neg": neg, "groups": groups, "job_ids": job_ids})
    return tok
