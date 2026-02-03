# PDR: Job Title → ESCO Mapping Pipeline (Karrierewege/Decorte)

*Last updated: 2025-09-23 (Europe/Berlin)*

---

## 1) Overview

**Goal.** Build a **config-driven retrieval system** that maps noisy job titles (and short job snippets) from **Karrierewege** and **Decorte** to **ESCO occupations (EN)**. Use **English–English** matching approaches inspired by top **TalentCLEF Task A** systems. Evaluate primarily on our datasets; **optionally** evaluate **zero-shot** on TalentCLEF EN–EN.

**Key idea.** Train a **bi-encoder** (contrastive) + optional **asymmetric heads** or **hybrid loss**, and keep a **linear-T** baseline. Optional **cross-encoder reranker** for reordering top-K.

**Primary outputs.**

* A single CLI & YAML-driven pipeline that can run: `linear`, `contrastive`, `hybrid`, `asymmetric`, optional `rerank`.
* ESCO index (FAISS) + checkpoints + metrics.
* Repro script + results tables.

---

## 2) Scope & Non-Goals

**In scope**

* Job title ↔ ESCO mapping (EN–EN).
* Methods: linear transformation **T**, bi-encoder **InfoNCE**, **hybrid loss**, **asymmetric projections**, **reranking**.
* Datasets: Karrierewege, Decorte (EN subsets).
* Metrics: **Recall@k**, **MAP@10**, **MRR**, latency.
* Evaluation on TalentCLEF **without re-training** (report only).

**Out of scope**

* Cross-lingual training.
* Next-job prediction (sequence models).
* Skill extraction/classification.

---

## 3) User Stories

* **Data Scientist**: “I want a script that takes job titles and returns the top-K ESCO titles with scores.”
* **Researcher**: “I want to compare several training strategies with identical evaluation and a consistent ESCO index.”
* **Engineer**: “I need a single CLI and YAML config to reproduce runs and export artifacts.”

---

## 4) Datasets & Inputs

* **Karrierewege (EN)**: Work experiences with mapped ESCO IDs/titles. Use **per-experience** titles (not whole CV).
* **Decorte (EN)**: Work histories with ESCO labels (smaller).
* **ESCO (EN)**: Canonical occupation titles (+ optional descriptions), version-locked (e.g., v1.2.0).

**Training views we build**

* **Job→ESCO positives**: `(job_title_text, esco_title_text, esco_id)`.
* **Job↔Job positives** (weak supervision): any two job titles sharing the same ESCO id.

**Negatives**

* In-batch negatives (default).
* Hard negatives: same macro group / high lexical similarity but different ESCO.

**File formats**

```json
// job→esco JSONL
{"job_title": "Senior Data Analyst", "esco_id": "1234.56", "esco_title": "Data analyst"}

// job↔job JSONL
{"job_title_a": "Frontend Developer", "job_title_b": "Web UI Engineer", "esco_id": "7890.12"}

// ESCO titles JSONL
{"esco_id": "1234.56", "title": "Data analyst"}
```

---

## 5) Tasks

1. **Normalization (primary)** — Given a job title, retrieve ESCO titles.
2. **Title similarity (aux)** — Given a job title, retrieve semantically equivalent job titles (TalentCLEF-style).

---

## 6) System Architecture

```
+-------------------+             +-----------------------+
|  Data Loaders     |  pairs -->  |  Trainers (strategy)  |
|  (KW, Decorte,    |-------------|  - linear_T           |
|   ESCO)           |             |  - contrastive        |
+-------------------+             |  - hybrid             |
                                  |  - asymmetric         |
                                  +-----------+-----------+
                                              |
                           (encoder/head/T)   v
                                      +--------------+
                                      |  Indexing    |  (ESCO embs)
 Query (job) --> Encode(+T/head) -->  |  FAISS IP    | --> top-K ESCO + scores
                                      +--------------+
                                              |
                                              v
                                       +-------------+
                                       |   Reranker  | (optional cross-encoder)
                                       +-------------+
```

---

## 7) Methods

### 7.1 Linear Transformation **T** (baseline)

* Embed jobs `A ∈ ℝ^{N×d}` and ESCOs `B ∈ ℝ^{N×d}` with a frozen encoder.
* Solve **least squares**: `T* = argmin_T ||A T − B||_F^2` via `np.linalg.lstsq(A, B)`, `T ∈ ℝ^{d×d}`.
* Inference: `a_t = a @ T`; rank ESCO by cosine between `a_t` and ESCO embeddings.
* Diagnostics: MSE/RMSE on train pairs; `||T − I||_F` and normalized Frobenius.

### 7.2 Contrastive Bi-Encoder (**InfoNCE**)

* Two towers share base encoder (or not), project to `proj_dim` and L2-normalize.
* Loss: **MultipleNegativesRankingLoss** (in-batch negatives).
* **Multitask (recommended)**: `L = λ * InfoNCE(job↔ESCO) + (1−λ) * InfoNCE(job↔job)`, e.g., `λ=0.5`.

### 7.3 Hybrid Loss (classification + contrastive)

* Add small classifier over a **sampled ESCO set** (softmax cross-entropy).
* `L = α*InfoNCE + (1−α)*CE`, α∈[0.5,0.9]. Helps calibration for fixed taxonomy.

### 7.4 Asymmetric Projections

* Separate heads: `z_job = MLP_job(enc(job))`, `z_esco = MLP_esco(enc(esco))`.
* Motivation: job texts are longer/noisier than short ESCO titles.

### 7.5 Cross-Encoder Reranker (optional)

* Take top-K candidates from bi-encoder; re-score `(job, esco)` with a cross-encoder (e.g., `bert-base`), pointwise or pairwise margin loss.
* Pros: boosts R@1; Cons: higher latency. Use only for final ranking.

---

## 8) Retrieval & Indexing

* **Embedding**: Use the same encoder/heads as in training. L2-normalize.
* **Index**: FAISS `IndexFlatIP` (cosine via IP on normalized vectors).
* **Candidate set**: ESCO EN titles (version-locked), possibly deduped + synonyms.
* **Latency**: For ~10k ESCO titles, flat IP is fine. For >100k, consider IVF or HNSW.

---

## 9) Metrics

* **Recall@k**: `mean_q [ 1{gold ∈ top-k(q)} ]`.
* **MAP@10**: For each `q`, AP over top-10, then mean.
* **MRR**: `mean_q [ 1 / rank_of_first_relevant ]`.
* **Latency**: ms/query for encode + search (+ rerank).

**Note:** For Job↔Job evaluation, “relevant” = titles sharing the same ESCO id.

---

## 10) Configuration (YAML)

**Contrastive example**

```yaml
seed: 42
device: cuda
strategy: contrastive           # linear|contrastive|hybrid|asymmetric
encoder:
  name: intfloat/e5-base        # or gte-large-en-v1.5
  proj_dim: 256
loss:
  type: info_nce                # info_nce|hybrid
  temperature: 0.05
  alpha: 0.7                    # only for hybrid
data:
  source: karrierewege          # decorte|karrierewege
  train_pairs_path: data/kw/train_job_esco.jsonl
  dev_pairs_path:   data/kw/dev_job_esco.jsonl
  jobjob_pairs_path: data/kw/train_job_job.jsonl
  esco_titles_path: data/esco/esco_titles_en.jsonl
train:
  batch_size: 64
  epochs: 2
  lr_encoder: 2e-5
  lr_heads: 1e-3
  lambda_multitask: 0.5
negatives:
  mined: true
  mined_topk: 50
eval:
  k: [1,5,10]
reranker:
  enabled: false
  model: cross-encoder/ms-marco-MiniLM-L-6-v2
artifacts_dir: runs/exp001
```

**Linear-T example**

```yaml
seed: 42
device: cuda
strategy: linear
encoder:
  name: intfloat/e5-base
data:
  source: decorte
  train_pairs_path: data/decorte/train_job_esco.jsonl
  dev_pairs_path:   data/decorte/dev_job_esco.jsonl
  esco_titles_path: data/esco/esco_titles_en.jsonl
train: {}
eval:
  k: [1,5,10]
artifacts_dir: runs/linear_decorte
```

---

## 11) CLI

```bash
# Train (strategy from YAML)
python -m src.cli --cfg configs/kw_contrastive.yaml --mode train

# Build ESCO index for current encoder/heads
python -m src.cli --cfg configs/kw_contrastive.yaml --mode index

# Evaluate retrieval on dev/test
python -m src.cli --cfg configs/kw_contrastive.yaml --mode eval

# Optional rerank top-K
python -m src.cli --cfg configs/kw_contrastive.yaml --mode rerank
```

---

## 12) Module Layout (initial tree)

```
project_root/
  configs/
    kw_contrastive.yaml
    decorte_linear.yaml
  data/
    kw/  # symlink or path to processed Karrierewege pairs
    decorte/
    esco/
  runs/
  scripts/
    reproduce.sh
  src/
    cli.py
    config.py
    data/
      datasets.py      # loaders for job↔ESCO, job↔job
      sampling.py      # hard negative mining
      utils.py         # text normalize, dedupe
    models/
      encoders.py      # HF/SBERT loaders, pooling
      heads.py         # symmetric/asymmetric MLP heads
    trainers/
      train_linear.py  # lstsq T, diagnostics
      train_contrastive.py # InfoNCE / hybrid loop
    indexing/
      faiss_index.py   # build/load/search
    eval/
      metrics.py       # R@k, MAP@10, MRR
      evaluate.py
    rerank/
      cross_encoder.py
  tests/
    test_shapes.py
    test_metrics.py
  README.md
  PDR.md
```

---

## 13) Training Details

**General**

* Optimizer: AdamW; wd=0.01 for heads, 0–0.01 for encoder.
* Scheduler: linear decay; warmup 5%.
* Mixed precision (fp16/bf16) on GPU (2×A100 40GB available).
* Early stopping on dev MAP@10.

**Contrastive**

* Batch 64 (grad-accum if needed).
* Temperature τ=0.05 (tune 0.03–0.1).
* Multitask ratio λ=0.5 (ablate 0.3, 0.7).

**Hybrid**

* Sampled ESCO softmax head over 2–5k classes (or sampled per-batch).
* α=0.7 typical.

**Asymmetric**

* 1–2 layer MLPs with GELU to `proj_dim` (e.g., 256).

**Linear-T**

* Same encoder as used to embed ESCO titles; store `T.npy`.
* Report Frobenius and normalized Frobenius (diagnostics).

---

## 14) Negative Sampling & Augmentation

* **In-batch negatives** by default.
* **Hard negatives**:

  * Lexical similarity (BM25) but different ESCO.
  * Same ESCO **macro group** but different leaf id.
* **Augmentation (optional)**:

  * Paraphrase job titles with an LLM (low rate, e.g., p=0.3).
  * Keep ESCO titles intact (canonical).

---

## 15) Evaluation Protocol

* **Internal**:

  * **Job→ESCO**: R@1/5/10, MAP@10, MRR.
  * **Job↔Job**: MAP@10 using job–job pairs (positives share ESCO).
* **TalentCLEF EN–EN (eval-only)**:

  * Embed job queries + candidate job titles with **job tower**.
  * Compute MAP (no re-training). Report domain transfer.

**Controls**

* Fix ESCO version and candidate set across runs.
* Normalize embeddings before FAISS (cosine via IP).
* Same encoder/heads at index build and query time.

---

## 16) Reproducibility & Artifacts

* **Seed everything** (PyTorch, numpy, Python).
* Save:

  * Config YAML + git hash.
  * Encoder checkpoint, projection heads, `T.npy`.
  * ESCO index + id map.
  * Metrics JSON + markdown.
* Provide `scripts/reproduce.sh` with exact commands.

**Example `scripts/reproduce.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

CFG=${1:-configs/kw_contrastive.yaml}

python -m src.cli --cfg $CFG --mode train
python -m src.cli --cfg $CFG --mode index
python -m src.cli --cfg $CFG --mode eval
# optional rerank
# python -m src.cli --cfg $CFG --mode rerank
```

---

## 17) Risks & Mitigations

* **Label noise** (wrong ESCO): contrastive + job↔job multitask; median-of-means on metrics.
* **Domain shift** (vs TalentCLEF): report both internal and TalentCLEF metrics; optional paraphrase aug.
* **Overfitting**: early stopping; clean held-out split.
* **ESCO version drift**: lock and validate IDs at load time.

---

## 18) Two-Week Plan (thesis-oriented)

**W1**

* D1: Scaffold, loaders, tiny mock run; tests (shapes/metrics).
* D2: Linear-T baseline end-to-end (index+eval).
* D3–D4: Contrastive (InfoNCE) on 50–100k pairs; dev eval.
* D5: Asymmetric heads; quick compare.
* D6: Optional hybrid loss.
* D7: Optional reranker; measure latency/gains.

**W2**

* D8–D9: Scale best 1–2 configs to 100–200k pairs; early stop.
* D10: Ablations (paraphrase on/off, mined negatives, head dim).
* D11: Internal + TalentCLEF EN–EN eval (no re-train).
* D12: Robustness, seeds, `reproduce.sh`.
* D13–D14: Writeup (methods, results, limitations); finalize artifacts.

---

## 19) Acceptance Criteria

* **Baseline**: Linear-T run produces valid R@k/MAP@10 and diagnostics.
* **Contrastive**: Improves over linear on dev MAP@10.
* **Indexing**: Deterministic, version-locked ESCO index.
* **CLI**: Single entry point `src/cli.py` handles train/index/eval/rerank via YAML.
* **Repro**: `reproduce.sh` yields same metrics (±small variance).
* **Report**: Markdown table comparing strategies (R@1/5/10, MAP@10, MRR, latency).

---

## 20) Pseudocode Snippets

**Linear-T fit**

```python
A = enc(job_texts)   # N x d
B = enc(esco_texts)  # N x d
T, *_ = np.linalg.lstsq(A, B, rcond=None)
# Diagnostics
mse = np.mean(np.sum((A @ T - B)**2, axis=1))
fro = np.linalg.norm(T - np.eye(T.shape[1]), 'fro')
```

**InfoNCE step (bi-encoder)**

```python
z_job  = normalize(job_head(enc(job_batch)))
z_esco = normalize(esco_head(enc(esco_batch)))
sim = (z_job @ z_esco.T) / tau              # B x B
loss = cross_entropy(sim, target=arange(B)) # positives on diagonal
```

**R@k / MAP@10**

```python
idx, scores = faiss.search(query_vecs, k=10)
# compute R@k, AP@10 per query, then average
```

---

## 21) Prompt Cheat-Sheet (for LLM pair-programming)

* “Implement `train_contrastive.py` with SentenceTransformers and MultipleNegativesRankingLoss; support optional asymmetric heads and multitask batches (job↔ESCO, job↔job).”
* “Add `faiss_index.py::build_index(embs, ids)` that L2-normalizes, builds IndexFlatIP, and saves index + id map.”
* “Write `tests/test_metrics.py` with a toy case where gold ESCO is exact; assert R@1==1.0 and MAP@10==1.0.”
* “Review `evaluate.py` for normalization mismatches and ensure encoder/heads used for ESCO index match those used for queries.”
