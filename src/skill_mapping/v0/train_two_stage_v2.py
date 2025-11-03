#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
playground_two_stage_pjmat.py
Two-stage ESCO skill prediction with TalentCLEF-style tricks.

Stage 1: job -> ESCO group(s) (class)
Stage 2: job -> skills (inside top predicted groups)

Main features you can toggle:
- Contrastive job↔skill (baseline)
- Optional guide model distillation (GIST-style)
- Optional contrastive job↔group head
- Optional asymmetric projection on job branch
- Optional LoRA fine-tuning
- Alias handling at inference
- Cosine LR + early stopping on MAP

We assume:
- hierarchy_io.load_esco_hierarchy already exists
- datasets.load_jobs_csv etc. already exist
"""

from __future__ import annotations
import argparse, os, math, random, json, copy
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from sentence_transformers import SentenceTransformer, InputExample, util

from hierarchy_io import load_esco_hierarchy
from datasets import (
    load_jobs_csv,
    build_index_maps,
    JobExample,
)

# -------------------------
# Static hierarchy file paths
# -------------------------
HIER_CSV = "data/processed/skill_to_group_pillar.csv"
GROUP2LABEL_JSON = "data/processed/group2label.json"
GROUP2PARENT_JSON = "data/processed/group2parent.json"
GROUP2PILLAR_JSON = "data/processed/group2pillar.json"
SKILL2GROUP_JSON = "data/processed/skill2group.json"
SKILL2PILLAR_JSON = "data/processed/skill2pillar.json"


# ========= utilities =========

def set_seed(s: int = 42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def chunked(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]

def normalize(t: torch.Tensor) -> torch.Tensor:
    return F.normalize(t, p=2, dim=-1)

# ========= text builders =========
# we are *not* doing text enrichment (no LLM desc concatenation) per user request

def build_job_text(j: JobExample) -> str:
    # load_jobs_csv already formats title [SEP] description
    return j.text

def build_skill_text(skill_id: str,
                     skill_texts: Dict[str, str]) -> str:
    # canonical label only (no enrichment)
    return skill_texts.get(
        skill_id,
        skill_id.split("/")[-1].replace("-", " ")
    )

def build_group_text(group_id: str,
                     class_texts: Dict[str, str]) -> str:
    return class_texts.get(
        group_id,
        group_id.split("/")[-1].replace("-", " ")
    )

# ========= alias handling (inference time) =========

def load_alias_map(path: Optional[str]) -> Dict[str, List[str]]:
    # alias_map: canonical_skill_id -> [alias strings]
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def score_skill_with_aliases(
    st: SentenceTransformer,
    job_emb: np.ndarray,                     # [d]
    canonical_text: str,
    alias_list: List[str],
) -> float:
    # job_emb is already normalized
    texts = [canonical_text] + alias_list
    emb_alias = st.encode(texts,
                          normalize_embeddings=True,
                          show_progress_bar=False)
    # cosine = dot since normalized
    return float(np.max(emb_alias @ job_emb))

# ========= dataset prep for training pairs =========

def build_pairs_job_skill(
    jobs: List[JobExample],
    skill_texts: Dict[str,str],
    max_pos_per_job: int,
) -> List[Tuple[str, str]]:
    pairs = []
    for j in jobs:
        jt = build_job_text(j)
        # limit positives so we don't overweight huge occupations
        pos_skills = j.skill_ids[:max_pos_per_job] if max_pos_per_job else j.skill_ids
        for sid in pos_skills:
            stxt = build_skill_text(sid, skill_texts)
            pairs.append((jt, stxt))
    return pairs

def build_pairs_job_group(
    jobs: List[JobExample],
    class_texts: Dict[str,str],
    max_pos_per_job: int,
) -> List[Tuple[str, str]]:
    pairs = []
    for j in jobs:
        jt = build_job_text(j)
        pos_groups = j.group_ids[:max_pos_per_job] if max_pos_per_job else j.group_ids
        for gid in pos_groups:
            gtxt = build_group_text(gid, class_texts)
            pairs.append((jt, gtxt))
    return pairs

# ========= model wrappers =========

class AsymmetricProjector(nn.Module):
    """
    Optional linear projection on job embeddings only
    (TechWolf-style asymmetry / JobBERT-style projection).
    Applied *during training* and optionally inference,
    but you can also drop it at inference if you want base space.
    """
    def __init__(self, dim, proj_dim=None):
        super().__init__()
        if proj_dim is None:
            proj_dim = dim
        self.linear = nn.Linear(dim, proj_dim, bias=False)

    def forward(self, job_emb: torch.Tensor) -> torch.Tensor:
        return normalize(self.linear(job_emb))

def maybe_wrap_lora(st: SentenceTransformer, use_lora: bool):
    """
    Placeholder hook. Real LoRA integration would wrap st.auto_model
    with PEFT. We keep a stub so you can extend.
    """
    if use_lora:
        print("[INFO] LoRA flag is on, but adapter injection is stubbed. Add PEFT here.")
    return st

# ========= loss builders =========

def multiple_negatives_ranking_loss(
    job_emb: torch.Tensor,    # [B,d]
    skill_emb: torch.Tensor,  # [B,d] positive aligned (same index)
) -> torch.Tensor:
    """
    Standard MultipleNegativesRankingLoss:
    - Cosine similarity of each job to all skills in batch.
    - Cross-entropy where diagonal is target.
    """
    job_emb = normalize(job_emb)
    skill_emb = normalize(skill_emb)
    logits = job_emb @ skill_emb.T  # [B,B]
    targets = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, targets)

def distillation_loss(
    student_job: torch.Tensor,     # [B,d]
    student_skill: torch.Tensor,   # [B,d]
    teacher_job: torch.Tensor,     # [B,d]
    teacher_skill: torch.Tensor,   # [B,d]
    tau: float = 0.07,
) -> torch.Tensor:
    """
    GIST / guide-model style: match similarity structure of teacher.
    We compute pairwise sims in student vs teacher batches.
    This is a simplified stand-in for full GISTEmbed.
    """
    s_j = normalize(student_job)
    s_s = normalize(student_skill)
    t_j = normalize(teacher_job)
    t_s = normalize(teacher_skill)

    # student similarities
    s_logits = (s_j @ s_s.T) / tau        # [B,B]
    # teacher similarities
    with torch.no_grad():
        t_logits = (t_j @ t_s.T) / tau    # [B,B]
        t_probs = F.softmax(t_logits, dim=1)

    # KL divergence between teacher soft targets and student
    log_probs = F.log_softmax(s_logits, dim=1)
    kd = F.kl_div(log_probs, t_probs, reduction="batchmean")
    return kd

# ========= training loops =========

def train_contrastive(
    st_student: SentenceTransformer,
    train_pairs: List[Tuple[str,str]],
    dev_pairs: List[Tuple[str,str]],
    batch_size: int,
    epochs: int,
    lr: float,
    device: str,
    use_asym_proj: bool,
    guide_model_name: Optional[str],
    patience_epochs: int,
):
    """
    Train student encoder with:
    - MultipleNegativesRankingLoss (always)
    - optional distillation (guide/teacher model)
    - cosine LR schedule
    - early stopping on dev MAP
    """

    # optional teacher
    teacher = None
    if guide_model_name:
        teacher = SentenceTransformer(guide_model_name, device=device)
        teacher.eval()

    # optional asym projection on the student job branch
    proj = None
    if use_asym_proj:
        dim = st_student.get_sentence_embedding_dimension()
        proj = AsymmetricProjector(dim).to(device)
        params = list(st_student.parameters()) + list(proj.parameters())
    else:
        params = list(st_student.parameters())

    optimizer = AdamW(params, lr=lr, weight_decay=0.01)

    # simple cosine schedule (no warmup for now)
    steps_per_epoch = max(1, len(train_pairs) // batch_size)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * steps_per_epoch)

    def batchify(pairs):
        random.shuffle(pairs)
        for chunk in chunked(pairs, batch_size):
            if len(chunk) < 2:
                continue  # need at least 2 for in-batch negatives
            jobs_txt = [j for (j, _) in chunk]
            skills_txt = [s for (_, s) in chunk]
            yield jobs_txt, skills_txt

    best_map = -1.0
    best_state = None
    bad_epochs = 0

    for ep in range(1, epochs+1):
        st_student.train()
        if proj: proj.train()

        for jobs_txt, skills_txt in batchify(train_pairs):
            # encode student
            job_emb_stu = st_student.encode(
                jobs_txt, convert_to_tensor=True, device=device,
                show_progress_bar=False, normalize_embeddings=False
            )
            skill_emb_stu = st_student.encode(
                skills_txt, convert_to_tensor=True, device=device,
                show_progress_bar=False, normalize_embeddings=False
            )

            if proj:
                job_emb_stu = proj(job_emb_stu)  # project only job side
            else:
                job_emb_stu = normalize(job_emb_stu)
            skill_emb_stu = normalize(skill_emb_stu)

            loss_main = multiple_negatives_ranking_loss(job_emb_stu, skill_emb_stu)

            loss_total = loss_main

            # optional guide-model KD
            if teacher is not None:
                with torch.no_grad():
                    job_emb_tea = teacher.encode(
                        jobs_txt, convert_to_tensor=True, device=device,
                        show_progress_bar=False, normalize_embeddings=False
                    )
                    skill_emb_tea = teacher.encode(
                        skills_txt, convert_to_tensor=True, device=device,
                        show_progress_bar=False, normalize_embeddings=False
                    )
                if proj:
                    job_emb_kd = proj(job_emb_stu.detach())  # proj(student) is already applied above
                else:
                    job_emb_kd = job_emb_stu.detach()
                # distillation wants both student and teacher
                loss_kd = distillation_loss(
                    job_emb_kd, skill_emb_stu.detach(),
                    job_emb_tea, skill_emb_tea,
                    tau=0.07,
                )
                loss_total = loss_total + loss_kd

            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()

        # ---- dev eval after each epoch ----
        cur_map = evaluate_map(dev_pairs, st_student, proj, device, k=10)

        print(f"[epoch {ep}] dev MAP@10={cur_map:.4f}  best={best_map:.4f}")

        if cur_map > best_map + 1e-4:
            best_map = cur_map
            bad_epochs = 0
            # save best state dicts in memory
            best_state = {
                "student": copy.deepcopy(st_student.state_dict()),
                "proj": copy.deepcopy(proj.state_dict()) if proj else None,
            }
        else:
            bad_epochs += 1
            if bad_epochs >= patience_epochs:
                print("[early stop]")
                break

    # restore best
    if best_state:
        st_student.load_state_dict(best_state["student"])
        if proj and best_state["proj"] is not None:
            proj.load_state_dict(best_state["proj"])

    return st_student, proj


def evaluate_map(
    dev_pairs: List[Tuple[str,str]],
    st_model: SentenceTransformer,
    proj: Optional[AsymmetricProjector],
    device: str,
    k: int = 10,
) -> float:
    """
    Approx MAP@k using the dev_pairs batch-level retrieval setting.
    We'll approximate by treating each pair as (query job -> positives [that skill]),
    and everything else in that batch as negatives.
    """
    # We'll do a few random mini-batches to approximate.
    if len(dev_pairs) < 2:
        return 0.0

    sample_pairs = random.sample(dev_pairs, min(256, len(dev_pairs)))
    # group by job_text
    by_job = defaultdict(list)
    for jt, stxt in sample_pairs:
        by_job[jt].append(stxt)

    job_texts = list(by_job.keys())
    all_skill_texts = list({stxt for _, stxt in sample_pairs})

    with torch.no_grad():
        st_model.eval()
        job_emb = st_model.encode(job_texts,
                                  convert_to_tensor=True,
                                  device=device,
                                  show_progress_bar=False,
                                  normalize_embeddings=False)
        skill_emb = st_model.encode(all_skill_texts,
                                    convert_to_tensor=True,
                                    device=device,
                                    show_progress_bar=False,
                                    normalize_embeddings=False)
        if proj:
            job_emb = proj(job_emb)
        job_emb = normalize(job_emb)
        skill_emb = normalize(skill_emb)

        sims = job_emb @ skill_emb.T  # [J,S]
        sims_np = sims.cpu().numpy()

    # MAP per job
    total_ap = 0.0
    for j_idx, jt in enumerate(job_texts):
        pos_set = set(by_job[jt])
        # rank all skills
        order = np.argsort(-sims_np[j_idx])
        hits = 0
        ap = 0.0
        denom = min(len(pos_set), k) if len(pos_set) else 1
        for rank_pos, si in enumerate(order[:k], start=1):
            skill_text = all_skill_texts[si]
            if skill_text in pos_set:
                hits += 1
                ap += hits / rank_pos
        total_ap += (ap / denom if denom > 0 else 0.0)

    return total_ap / max(1, len(job_texts))


# ========= Stage 1 prediction (groups) =========

def build_group_prototypes(
    st_model: SentenceTransformer,
    class_ids: List[str],
    class_texts: Dict[str,str],
    class_to_skills: Dict[str, List[str]],
    skill_texts: Dict[str,str],
    mode: str,
    device: str,
):
    """
    mode = "label" -> encode group label only
    mode = "mean_skills" -> mean of member skill embeddings
    """
    st_model.eval()
    dim = st_model.get_sentence_embedding_dimension()

    if mode == "label":
        gtxts = [build_group_text(g, class_texts) for g in class_ids]
        gemb = st_model.encode(gtxts, normalize_embeddings=True,
                               convert_to_tensor=True, device=device,
                               show_progress_bar=False)
        return gemb.cpu().numpy()

    # mean_skills
    proto = []
    for gid in class_ids:
        sids = class_to_skills.get(gid, [])
        if not sids:
            proto.append(np.zeros((dim,), dtype=np.float32))
            continue
        stxts = [build_skill_text(s, skill_texts) for s in sids]
        memb = st_model.encode(stxts, normalize_embeddings=True,
                               convert_to_tensor=True, device=device,
                               show_progress_bar=False)
        proto.append(memb.mean(dim=0).cpu().numpy())
    proto = np.stack(proto, axis=0)
    # normalize rows
    proto = util.normalize_embeddings(torch.tensor(proto)).cpu().numpy()
    return proto

def predict_groups(
    st_model: SentenceTransformer,
    jobs: List[JobExample],
    class_ids: List[str],
    group_proto: np.ndarray,
    topk: int,
    proj: Optional[AsymmetricProjector],
    device: str,
) -> List[List[Tuple[str,float]]]:
    job_texts = [build_job_text(j) for j in jobs]
    st_model.eval()
    with torch.no_grad():
        job_emb = st_model.encode(job_texts,
                                  normalize_embeddings=False,
                                  convert_to_tensor=True,
                                  device=device,
                                  show_progress_bar=False)
        if proj:
            job_emb = proj(job_emb)
        job_emb = normalize(job_emb)                # [B,d]
        job_emb = job_emb.cpu().numpy()

    sims = job_emb @ group_proto.T                 # [B,G], cosine because normalized
    results = []
    for i in range(sims.shape[0]):
        idx = np.argsort(-sims[i])[:topk]
        # convert topk sims into pseudo-probs by softmax over just topk
        logits = sims[i, idx]
        probs = np.exp(logits - logits.max())
        probs = probs / (probs.sum() + 1e-9)
        results.append([
            (class_ids[k], float(p))
            for k,p in zip(idx, probs.tolist())
        ])
    return results

# ========= final ranking with gating and alias =========

def rank_skills_two_stage(
    st_model: SentenceTransformer,
    jobs: List[JobExample],
    top_groups: List[List[Tuple[str,float]]],
    class_to_skills: Dict[str,List[str]],
    skill_ids: List[str],
    skill_texts: Dict[str,str],
    alias_map: Dict[str,List[str]],
    proj: Optional[AsymmetricProjector],
    topk_skills: int,
    device: str,
):
    st_model.eval()
    with torch.no_grad():
        job_emb = st_model.encode(
            [build_job_text(j) for j in jobs],
            normalize_embeddings=False,
            convert_to_tensor=True,
            device=device,
            show_progress_bar=False
        )
        if proj:
            job_emb = proj(job_emb)
        job_emb = normalize(job_emb)  # [B,d]
        job_emb_np = job_emb.cpu().numpy()

    # cache canonical skill embeddings once
    canonical_text = {sid: build_skill_text(sid, skill_texts) for sid in skill_ids}
    skill_emb = st_model.encode(
        [canonical_text[s] for s in skill_ids],
        normalize_embeddings=True,
        convert_to_tensor=False,
        device=device,
        show_progress_bar=False
    )  # np.ndarray [S,d]
    sid2idx = {s:i for i,s in enumerate(skill_ids)}

    results = []
    for b, job in enumerate(jobs):
        # candidate set from predicted groups
        cand_idx = sorted({
            sid2idx[s]
            for (g,_) in top_groups[b]
            for s in class_to_skills.get(g, [])
            if s in sid2idx
        })
        if not cand_idx:
            results.append([])
            continue

        # cosine
        cos = skill_emb[cand_idx] @ job_emb_np[b]  # [N]

        # class-weighted merge
        merged = np.zeros_like(cos)
        for (g, p) in top_groups[b]:
            merged += p * cos

        # alias boost (max over aliases)
        # we overwrite merged[i] with max(job↔canonical, job↔alias_i)
        for i_local, sidx in enumerate(cand_idx):
            sid = skill_ids[sidx]
            aliases = alias_map.get(sid, [])
            if aliases:
                merged[i_local] = max(
                    merged[i_local],
                    score_skill_with_aliases(
                        st_model,
                        job_emb_np[b],
                        canonical_text[sid],
                        aliases
                    )
                )

        order = np.argsort(-merged)[:topk_skills]
        ranked = [(skill_ids[cand_idx[i]], float(merged[i])) for i in order]
        results.append(ranked)

    return results

# ========= metrics =========

def recall_at_k(ranked_ids: List[List[str]], gold_sets: List[set], k: int) -> float:
    hits = sum(int(len(set(preds[:k]) & gold)>0) for preds, gold in zip(ranked_ids, gold_sets))
    return hits / max(1, len(ranked_ids))

def mrr_at_k(ranked_ids: List[List[str]], gold_sets: List[set], k: int) -> float:
    total = 0.0
    for preds, gold in zip(ranked_ids, gold_sets):
        rr = 0.0
        for rank, sid in enumerate(preds[:k], 1):
            if sid in gold:
                rr = 1.0/rank
                break
        total += rr
    return total / max(1, len(ranked_ids))

def map_at_k(ranked_ids: List[List[str]], gold_sets: List[set], k: int) -> float:
    total = 0.0
    for preds, gold in zip(ranked_ids, gold_sets):
        ap = 0.0
        hits = 0
        denom = min(len(gold), k) if len(gold)>0 else 1
        for rank, sid in enumerate(preds[:k], 1):
            if sid in gold:
                hits += 1
                ap += hits / rank
        total += (ap/denom if denom>0 else 0.0)
    return total / max(1, len(ranked_ids))

def evaluate_full(dev_jobs: List[JobExample], ranked: List[List[Tuple[str,float]]], k: int=10):
    ranked_ids = [[sid for sid,_ in lst] for lst in ranked]
    gold_sets = [set(j.skill_ids) for j in dev_jobs]
    return (
        recall_at_k(ranked_ids, gold_sets, k),
        mrr_at_k(ranked_ids, gold_sets, k),
        map_at_k(ranked_ids, gold_sets, k)
    )

# ========= main =========

def main():
    ap = argparse.ArgumentParser()
    # data / hierarchy
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--dev_csv", required=True)
    ap.add_argument("--hier_csv", default=HIER_CSV)
    ap.add_argument("--group2label", default=GROUP2LABEL_JSON)
    ap.add_argument("--group2parent", default=GROUP2PARENT_JSON)
    ap.add_argument("--group2pillar", default=GROUP2PILLAR_JSON)
    ap.add_argument("--skill2group", default=SKILL2GROUP_JSON)
    ap.add_argument("--skill2pillar", default=SKILL2PILLAR_JSON)
    ap.add_argument("--alias_map_json", default=None)

    # model
    ap.add_argument("--student_model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--guide_model", default=None, help="teacher model id for KD (e.g. sentence-transformers/all-MiniLM-L12-v2)")
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--use_asym_proj", action="store_true")

    # training
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--patience_epochs", type=int, default=2)
    ap.add_argument("--max_pos_per_job", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # stage-1 config
    ap.add_argument("--group_proto_mode", choices=["label","mean_skills"], default="mean_skills")
    ap.add_argument("--topk_groups", type=int, default=3)
    ap.add_argument("--topk_skills", type=int, default=10)

    # optional: also train job↔group contrastive (Stage-1 awareness)
    ap.add_argument("--train_group_head", action="store_true")

    args = ap.parse_args()
    set_seed(args.seed)

    # --- load hierarchy ---
    H = load_esco_hierarchy(
        csv_path=args.hier_csv,
        group2label_json=args.group2label,
        group2parent_json=args.group2parent,
        group2pillar_json=args.group2pillar,
        skill2group_json=args.skill2group,
        skill2pillar_json=args.skill2pillar
    )
    class_ids, class2idx, skill_ids, skill2idx = build_index_maps(H)
    class_texts = H["class_texts"]
    skill_texts = H["skill_texts"]
    c2s = H["class_to_skills"]

    alias_map = load_alias_map(args.alias_map_json)

    # --- load train/dev jobs ---
    # train_csv / dev_csv must already have columns:
    # raw_title, raw_description, group_ids (semicolon list), skill_ids (semicolon list)
    train_jobs = load_jobs_csv(args.train_csv,
                               title_col="raw_title",
                               desc_col="raw_description",
                               group_col="group_ids",
                               skill_col="skill_ids",
                               id_col="id")
    dev_jobs   = load_jobs_csv(args.dev_csv,
                               title_col="raw_title",
                               desc_col="raw_description",
                               group_col="group_ids",
                               skill_col="skill_ids",
                               id_col="id")

    # --- build pairs for Stage-2 job↔skill (always) ---
    train_pairs_js = build_pairs_job_skill(train_jobs, skill_texts, args.max_pos_per_job)
    dev_pairs_js   = build_pairs_job_skill(dev_jobs,   skill_texts, args.max_pos_per_job)

    # --- optional pairs for Stage-1 job↔group (train_group_head) ---
    # We won't train a separate model; we'll just merge these
    if args.train_group_head:
        train_pairs_jg = build_pairs_job_group(train_jobs, class_texts, args.max_pos_per_job)
        dev_pairs_jg   = build_pairs_job_group(dev_jobs,   class_texts, args.max_pos_per_job)
        # union them (simple way: treat group text as more skills)
        train_pairs = train_pairs_js + train_pairs_jg
        dev_pairs   = dev_pairs_js   + dev_pairs_jg
    else:
        train_pairs = train_pairs_js
        dev_pairs   = dev_pairs_js

    # --- load student model (+ optional LoRA stub) ---
    st_student = SentenceTransformer(args.student_model, device=args.device)
    st_student = maybe_wrap_lora(st_student, args.use_lora)

    # --- train contrastively (with KD / asym proj / cosine sched / early stop on MAP) ---
    st_trained, proj = train_contrastive(
        st_student=st_student,
        train_pairs=train_pairs,
        dev_pairs=dev_pairs,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        use_asym_proj=args.use_asym_proj,
        guide_model_name=args.guide_model,
        patience_epochs=args.patience_epochs,
    )

    # --- Stage 1 inference: predict groups via prototypes ---
    group_proto = build_group_prototypes(
        st_trained,
        class_ids,
        class_texts,
        c2s,
        skill_texts,
        mode=args.group_proto_mode,
        device=args.device
    )
    dev_group_preds = predict_groups(
        st_trained,
        dev_jobs,
        class_ids,
        group_proto,
        topk=args.topk_groups,
        proj=proj,
        device=args.device
    )

    # --- Stage 2 inference: rank skills inside predicted groups (alias-aware) ---
    ranked = rank_skills_two_stage(
        st_trained,
        dev_jobs,
        dev_group_preds,
        c2s,
        skill_ids,
        skill_texts,
        alias_map,
        proj,
        topk_skills=args.topk_skills,
        device=args.device
    )

    # --- final metrics ---
    r5, mrr10, map10 = evaluate_full(dev_jobs, ranked, k=10)
    print(f"\n[FINAL DEV] Recall@5={r5:.3f}  MRR@10={mrr10:.3f}  MAP@10={map10:.3f}")

    # pretty print a few examples
    for i, (j, lst) in enumerate(zip(dev_jobs[:3], ranked[:3])):
        print(f"\nJOB {i}: {j.text[:140]}...")
        for sid, sc in lst[:5]:
            print(f"  {sid} | {skill_texts[sid]} | score={sc:.3f}")

if __name__ == "__main__":
    main()
