# train_two_stage.py
from __future__ import annotations
import argparse, math, os, json, random
from typing import List, Dict, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

from hierarchy_io import load_esco_hierarchy
from datasets import (
    load_jobs_csv, build_index_maps,
    ClassDataset, class_collate,
    SkillRankingDataset, skill_collate,
    JobExample
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

# -------------------------
# Utilities
# -------------------------
def set_seed(s: int = 42):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def pooling_mean(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1)
    masked = last_hidden_state * mask
    return masked.sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)

# -------------------------
# Models
# -------------------------
class TextEncoder(nn.Module):
    def __init__(self, hf_id: str):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(hf_id)
        self.hidden = self.backbone.config.hidden_size

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        emb = pooling_mean(out.last_hidden_state, attention_mask)  # [B, d]
        return F.normalize(emb, p=2, dim=-1)

class ClassHead(nn.Module):
    def __init__(self, d_in: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):  # x: [B, d]
        return self.net(x)  # logits [B, C]

class SkillScorer(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2*d_in, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, job_emb: torch.Tensor, skill_emb: torch.Tensor):  # [B,d], [N,d] or [B,N,d]
        # supports batched gather: if skill_emb is [B, N, d]
        if skill_emb.dim() == 2:  # [N,d]
            b = job_emb.size(0)
            n = skill_emb.size(0)
            job_tiled = job_emb.unsqueeze(1).expand(b, n, -1)
            skill = skill_emb.unsqueeze(0).expand(b, n, -1)
        else:  # [B,N,d]
            job_tiled = job_emb.unsqueeze(1).expand_as(skill_emb)
            skill = skill_emb
        x = torch.cat([job_tiled, skill], dim=-1)         # [B,N,2d]
        logits = self.net(x)                              # [B,N,1]
        return logits.squeeze(-1)                         # [B,N]

# -------------------------
# Evaluation Metrics
# -------------------------
def recall_at_k(ranked_ids: List[List[str]], gold_sets: List[set], k: int) -> float:
    hits = 0
    for preds, gold in zip(ranked_ids, gold_sets):
        hits += int(len(set(preds[:k]) & gold) > 0)
    return hits / max(1, len(ranked_ids))

def mrr_at_k(ranked_ids: List[List[str]], gold_sets: List[set], k: int) -> float:
    total = 0.0
    for preds, gold in zip(ranked_ids, gold_sets):
        rr = 0.0
        for rank, sid in enumerate(preds[:k], 1):
            if sid in gold:
                rr = 1.0 / rank
                break
        total += rr
    return total / max(1, len(ranked_ids))

def map_at_k(ranked_ids: List[List[str]], gold_sets: List[set], k: int) -> float:
    total = 0.0
    for preds, gold in zip(ranked_ids, gold_sets):
        num_hits, ap = 0, 0.0
        for rank, sid in enumerate(preds[:k], 1):
            if sid in gold:
                num_hits += 1
                ap += num_hits / rank
        if len(gold) > 0:
            ap = ap / min(len(gold), k)
        total += ap
    return total / max(1, len(ranked_ids))

# -------------------------
# Training helpers
# -------------------------
def train_class_head(encoder, class_head, tokenizer, train_loader, dev_loader, n_classes, device, epochs=3, lr=5e-5):
    encoder.train(); class_head.train()
    opt = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": lr},
        {"params": class_head.parameters(), "lr": 5*lr}
    ], weight_decay=0.01)
    total_steps = epochs * len(train_loader)
    sch = get_linear_schedule_with_warmup(opt, int(0.06*total_steps), total_steps)
    bceloss = nn.BCEWithLogitsLoss()

    for ep in range(1, epochs+1):
        for batch in train_loader:
            tok = tokenizer(batch["texts"], padding=True, truncation=True, return_tensors="pt").to(device)
            labels = batch["labels"].to(device)  # [B, C] multi-hot
            job_emb = encoder(**tok)            # [B, d]
            logits = class_head(job_emb)        # [B, C]
            loss = bceloss(logits, labels)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(class_head.parameters(), 1.0)
            opt.step(); sch.step()

        # quick dev log-loss
        encoder.eval(); class_head.eval()
        with torch.no_grad():
            total = 0.0; steps = 0
            for batch in dev_loader:
                tok = tokenizer(batch["texts"], padding=True, truncation=True, return_tensors="pt").to(device)
                labels = batch["labels"].to(device)
                logits = class_head(encoder(**tok))
                total += nn.BCEWithLogitsLoss()(logits, labels).item()
                steps += 1
        print(f"[Class] epoch {ep} dev loss: {total/max(1,steps):.4f}")
        encoder.train(); class_head.train()

def build_skill_emb_matrix(skill_ids: List[str], skill_texts: Dict[str,str], tokenizer, encoder, device) -> torch.Tensor:
    encoder.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(skill_ids), 256):
            texts = [skill_texts[sid] for sid in skill_ids[i:i+256]]
            tok = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
            e = encoder(**tok)  # [B, d]
            embs.append(e.cpu())
    mat = torch.cat(embs, dim=0)  # [S, d]
    return F.normalize(mat, p=2, dim=-1)

def train_skill_scorer(encoder, scorer, tokenizer, skill_emb_mat, skill_ids, train_loader, dev_loader, device, epochs=3, lr=5e-4):
    encoder.train(); scorer.train()
    opt = torch.optim.AdamW([
        {"params": encoder.parameters(), "lr": 2e-5},
        {"params": scorer.parameters(), "lr": lr}
    ], weight_decay=0.01)

    bceloss = nn.BCEWithLogitsLoss()

    for ep in range(1, epochs+1):
        for batch in train_loader:
            # Encode jobs
            tok = tokenizer(batch["texts"], padding=True, truncation=True, return_tensors="pt").to(device)
            job_emb = encoder(**tok)  # [B, d]

            # Build candidate matrices per sample (positives + negatives)
            max_c = max(len(p)+len(n) for p,n in zip(batch["pos"], batch["neg"]))
            all_logits = []
            all_targets = []
            for b_idx, (pos_idx, neg_idx) in enumerate(zip(batch["pos"], batch["neg"])):
                pos = pos_idx.tolist(); neg = neg_idx.tolist()
                idxs = pos + neg
                targets = torch.tensor([1]*len(pos) + [0]*len(neg), dtype=torch.float32, device=device)

                skills = skill_emb_mat[idxs].to(device)     # [C, d]
                logits = scorer(job_emb[b_idx:b_idx+1], skills.unsqueeze(0)).squeeze(0)  # [C]
                all_logits.append(logits)
                all_targets.append(targets)

            # pad to same length for batch loss
            padded_logits, padded_targets = [], []
            for lg, tg in zip(all_logits, all_targets):
                pad = max_c - lg.size(0)
                if pad > 0:
                    lg = torch.cat([lg, torch.full((pad,), -10.0, device=device)])   # strong negative bias
                    tg = torch.cat([tg, torch.zeros(pad, device=device)])
                padded_logits.append(lg)
                padded_targets.append(tg)
            logits_b = torch.stack(padded_logits, dim=0)
            targets_b = torch.stack(padded_targets, dim=0)
            loss = bceloss(logits_b, targets_b)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(scorer.parameters(), 1.0)
            opt.step()

        # quick dev evaluation (ranking with pos+neg only)
        r5, mrr10, map10 = evaluate_skill_dev(encoder, scorer, tokenizer, skill_emb_mat, skill_ids, dev_loader, device, topk=10)
        print(f"[Skill] epoch {ep} dev  R@5 {r5:.3f}  MRR@10 {mrr10:.3f}  MAP@10 {map10:.3f}")

def evaluate_skill_dev(encoder, scorer, tokenizer, skill_emb_mat, skill_ids, dev_loader, device, topk=10):
    encoder.eval(); scorer.eval()
    ranked_ids, gold_sets = [], []
    with torch.no_grad():
        for batch in dev_loader:
            tok = tokenizer(batch["texts"], padding=True, truncation=True, return_tensors="pt").to(device)
            job_emb = encoder(**tok)  # [B,d]
            # score only (pos+neg) subset to keep dev fast
            for b_idx, (pos_idx, neg_idx) in enumerate(zip(batch["pos"], batch["neg"])):
                idxs = pos_idx.tolist() + neg_idx.tolist()
                skills = skill_emb_mat[idxs].to(device).unsqueeze(0)  # [1,C,d]
                logits = scorer(job_emb[b_idx:b_idx+1], skills).squeeze(0)  # [C]
                probs = torch.sigmoid(logits)
                order = torch.argsort(probs, descending=True).tolist()
                top = [skill_ids[idxs[i]] for i in order[:topk]]
                ranked_ids.append(top)
                gold_sets.append(set(skill_ids[i] for i in pos_idx.tolist()))
    return (
        recall_at_k(ranked_ids, gold_sets, 5),
        mrr_at_k(ranked_ids, gold_sets, topk),
        map_at_k(ranked_ids, gold_sets, topk),
    )

# -------------------------
# Inference (full ranking within predicted classes)
# -------------------------
def predict_skills_for_jobs(
    jobs: List[JobExample],
    encoder, class_head, scorer, tokenizer,
    class_ids, class_texts, class_to_skills,
    skill_ids, skill_texts, skill_emb_mat,
    device, topk_classes=3, topk_skills=10, beta=0.1
) -> List[List[Tuple[str, float]]]:
    encoder.eval(); class_head.eval(); scorer.eval()
    results = []
    with torch.no_grad():
        for j in jobs:
            tok = tokenizer([j.text], padding=True, truncation=True, return_tensors="pt").to(device)
            job_emb = encoder(**tok)  # [1,d]
            # class probs
            logits = class_head(job_emb)                # [1,C]
            probs = torch.sigmoid(logits).squeeze(0)    # [C]
            topc = torch.topk(probs, k=topk_classes)
            top_class_idxs = topc.indices.tolist()
            top_class_probs = topc.values.tolist()

            # candidate skills from union of top classes
            cand_skill_idxs = set()
            for ci in top_class_idxs:
                cid = class_ids[ci]
                cand_skill_idxs.update([skill_ids.index(s) for s in class_to_skills.get(cid, []) if s in skill_ids])
            if not cand_skill_idxs:
                results.append([])
                continue

            cand_skill_idxs = sorted(list(cand_skill_idxs))
            skills = skill_emb_mat[cand_skill_idxs].to(device).unsqueeze(0)  # [1,N,d]
            logits_s = scorer(job_emb, skills).squeeze(0)                    # [N]
            probs_s = torch.sigmoid(logits_s).cpu()

            # class-weighted merge (sum)
            merged = probs_s.clone() * 0
            for ci, pc in zip(top_class_idxs, top_class_probs):
                # (optional) class-conditional — here we approximate by reusing same logits
                merged += pc * probs_s

            # small global prior via cosine (beta)
            # cosine between job_emb and skill_emb
            cos = (job_emb @ skills.squeeze(0).T).squeeze(0).cpu()           # [N]
            final = merged + beta * (cos.clamp(min=0) / (cos.abs().max()+1e-6))

            order = torch.argsort(final, descending=True).tolist()[:topk_skills]
            ranked = [(skill_ids[cand_skill_idxs[i]], float(final[i])) for i in order]
            results.append(ranked)
    return results

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_id", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--dev_csv", required=True)
    ap.add_argument("--epochs_class", type=int, default=3)
    ap.add_argument("--epochs_skill", type=int, default=3)
    ap.add_argument("--batch_class", type=int, default=64)
    ap.add_argument("--batch_skill", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # 1) Load hierarchy
    H = load_esco_hierarchy(
        csv_path=HIER_CSV,
        group2label_json=GROUP2LABEL_JSON,
        group2parent_json=GROUP2PARENT_JSON,
        group2pillar_json=GROUP2PILLAR_JSON,
        skill2group_json=SKILL2GROUP_JSON,
        skill2pillar_json=SKILL2PILLAR_JSON
    )
    class_ids, class2idx, skill_ids, skill2idx = build_index_maps(H)
    class_texts = H["class_texts"]; skill_texts = H["skill_texts"]; c2s = H["class_to_skills"]

    # 2) Load jobs (DECORTE / KW+ CSVs mapped to esco_id; group_ids can be derived with skill2group)
    train_jobs = load_jobs_csv(args.train_csv, title_col="raw_title", desc_col="raw_description",
                               group_col="group_ids", skill_col="skill_ids", id_col="id")
    dev_jobs   = load_jobs_csv(args.dev_csv,   title_col="raw_title", desc_col="raw_description",
                               group_col="group_ids", skill_col="skill_ids", id_col="id")

    # 3) Datasets & loaders
    ds_cls_train = ClassDataset(train_jobs, class2idx)
    ds_cls_dev   = ClassDataset(dev_jobs,   class2idx)

    ds_sk_train = SkillRankingDataset(train_jobs, H, class2idx, skill2idx, n_negs_same=24, n_negs_sibs=24, resample_each_call=True)
    ds_sk_dev   = SkillRankingDataset(dev_jobs,   H, class2idx, skill2idx, n_negs_same=24, n_negs_sibs=24, resample_each_call=False)

    tokenizer = AutoTokenizer.from_pretrained(args.hf_id, use_fast=True)

    dl_cls_train = DataLoader(ds_cls_train, batch_size=args.batch_class, shuffle=True,
                              collate_fn=lambda b: class_collate(b, tokenizer=None, n_classes=len(class_ids)))
    dl_cls_dev   = DataLoader(ds_cls_dev,   batch_size=args.batch_class, shuffle=False,
                              collate_fn=lambda b: class_collate(b, tokenizer=None, n_classes=len(class_ids)))

    dl_sk_train  = DataLoader(ds_sk_train,  batch_size=args.batch_skill, shuffle=True,
                              collate_fn=lambda b: skill_collate(b, tokenizer=None))
    dl_sk_dev    = DataLoader(ds_sk_dev,    batch_size=args.batch_skill, shuffle=False,
                              collate_fn=lambda b: skill_collate(b, tokenizer=None))

    # 4) Models
    encoder = TextEncoder(args.hf_id).to(device)
    class_head = ClassHead(encoder.hidden, len(class_ids)).to(device)
    scorer = SkillScorer(encoder.hidden).to(device)

    # 5) Step-1: train class head
    print(">> Training Step-1 (category)...")
    train_class_head(encoder, class_head, tokenizer, dl_cls_train, dl_cls_dev, len(class_ids), device, epochs=args.epochs_class)

    # 6) Pre-compute skill embeddings
    print(">> Encoding skill embeddings...")
    skill_emb_mat = build_skill_emb_matrix(skill_ids, skill_texts, tokenizer, encoder, device)  # [S,d]

    # 7) Step-2: train skill scorer
    print(">> Training Step-2 (skills)...")
    train_skill_scorer(encoder, scorer, tokenizer, skill_emb_mat, skill_ids, dl_sk_train, dl_sk_dev, device, epochs=args.epochs_skill)

    # 8) Final dev ranking (within predicted classes)
    print(">> Final dev ranking within predicted classes...")
    preds = predict_skills_for_jobs(
        dev_jobs[:128],  # sample for quick check
        encoder, class_head, scorer, tokenizer,
        class_ids, class_texts, c2s,
        skill_ids, skill_texts, skill_emb_mat,
        device, topk_classes=3, topk_skills=10, beta=0.1
    )
    # quick print
    for i, (j, pr) in enumerate(zip(dev_jobs[:5], preds[:5])):
        print(f"\nJOB {i}  TEXT: {j.text[:120]}...")
        for sid, score in pr[:5]:
            print(f"  {sid}  {skill_texts[sid]}  score={score:.3f}")

if __name__ == "__main__":
    main()
