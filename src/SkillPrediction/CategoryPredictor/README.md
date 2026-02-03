# Skill Mapping v2 - Modular Pipeline

A modular, experiment-friendly skill mapping pipeline for mapping job titles/descriptions to ESCO skills.

## Overview

| Script | Purpose |
|--------|---------|
| `skill_indexer.py` | Build FAISS vector index from ESCO skills |
| `similarity_scorer.py` | Map jobs to skills via semantic similarity |
| `category_trainer.py` | Train category predictor with Optuna HPO |
| `category_inference.py` | Predict skill categories for new jobs |
| `isco_trainer.py` | Train ISCO group classifier (single-label) |
| `isco_inference.py` | Predict ISCO codes for jobs |
| `bayesian_fuser.py` | Re-rank skills by fusing IR scores with category priors |

---

## 1. `skill_indexer.py` - The Vector Store

Creates a static FAISS index of the entire ESCO skill library for fast similarity search.

### Usage

```bash
python -m skill_mapping.v2.skill_indexer \
    --skills_csv data/processed/augmentation/augmented_esco_skills.csv \
    --model_name pj-mathematician/JobSkillBGE-large-en-v1.5 \
    --output_dir outputs/skill_index \
    --text_column job_brief \
    --batch_size 64 \
    --device cuda
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--skills_csv` | str | **required** | Path to augmented ESCO skills CSV |
| `--model_name` | str | `pj-mathematician/JobSkillBGE-large-en-v1.5` | HuggingFace model for encoding |
| `--output_dir` | str | **required** | Directory to save index and metadata |
| `--text_column` | str | `job_brief` | Column name for skill text to encode |
| `--batch_size` | int | `64` | Batch size for encoding |
| `--device` | str | `cuda` | Device for encoding (cuda/cpu) |

### Output Files

```
outputs/skill_index/
├── skill.index           # FAISS IndexFlatIP
├── skill_embeddings.npy  # Pre-computed vectors (N, dim)
└── skill_metadata.json   # Metadata + skill info mapping
```

### Metadata Format

```json
{
  "model_name": "pj-mathematician/JobSkillBGE-large-en-v1.5",
  "text_column": "job_brief",
  "num_skills": 3000,
  "embedding_dim": 1024,
  "skills": [
    {"conceptUri": "http://...", "preferredLabel": "skill name"},
    ...
  ]
}
```

---

## 2. `similarity_scorer.py` - Semantic Retrieval

High-speed semantic search to map job titles to ESCO skills using cosine similarity.

> **Note:** For very large datasets (e.g., millions of jobs), use `similarity_scorer_opt.py`. It is functionally identical but optimized for lower memory usage and batch processing on large-scale data (like the full Karrierewege dataset).

### Usage

```bash
python -m skill_mapping.v2.similarity_scorer \
    --jobs_csv data/processed/job_titles.csv \
    --index_dir outputs/skill_index \
    --model_path pj-mathematician/JobSkillBGE-large-en-v1.5 \
    --output_path outputs/similarity_scores.json \
    --text_column processed_text \
    --id_column job_id \
    --top_k 100 \
    --batch_size 64 \
    --device cuda \
    --save_embeddings outputs/job_embeddings
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--jobs_csv` | str | **required** | Path to CSV with job texts |
| `--index_dir` | str | **required** | Directory containing `skill.index` and `skill_metadata.json` |
| `--model_path` | str | `pj-mathematician/JobSkillBGE-large-en-v1.5` | SentenceTransformer model path |
| `--output_path` | str | **required** | Path to save output JSON |
| `--text_column` | str | `processed_text` | Column name for job text |
| `--id_column` | str | `None` | Column name for job ID (defaults to row index) |
| `--top_k` | int | `100` | Number of top candidates per job |
| `--batch_size` | int | `64` | Batch size for encoding |
| `--device` | str | `cuda` | Device for encoding (cuda/cpu) |
| `--save_embeddings` | str | `None` | Optional: Directory to save query embeddings |

### Input CSV Format

```csv
job_id,processed_text
0,"Software Engineer: Develops web applications using modern frameworks..."
1,"Data Analyst: Analyzes business data and creates reports..."
```

### Output Format

```json
{
  "0": [
    {"skill_uri": "http://data.europa.eu/esco/skill/...", "score": 0.985432, "rank": 1},
    {"skill_uri": "http://data.europa.eu/esco/skill/...", "score": 0.972156, "rank": 2},
    ...
  ],
  "1": [...]
}
```

### Optional: Saved Embeddings

When `--save_embeddings` is provided:

```
outputs/job_embeddings/
├── job_embeddings.npy       # (N, dim) float32 vectors
└── job_embeddings_meta.json # job_id to index mapping
```

---

## 3. `category_trainer.py` - Category Predictor

Trains a multi-label classifier to predict skill categories from job text, with Optuna hyperparameter optimization.

Supports two labeling modes:
- **Hard Labels (default)**: Multi-hot encoding where each category is binary (present/absent)
- **Soft Labels**: Label Distribution Learning where targets are category densities (# skills in category / total skills)

### Usage

**Standard (hard labels):**

```bash
python -m skill_mapping.v2.category_trainer \
    --esco_path data/processed/master_datasets_2/master_complete_hierarchy_w_occ.csv \
    --val_path data/validation.csv \
    --test_path data/test.csv \
    --encoder_ckpt pj-mathematician/JobSkillBGE-large-en-v1.5 \
    --target_level 1 \
    --output_dir outputs/category_model \
    --cache_name my_experiment \
    --n_trials 20 \
    --max_epochs 50 \
    --final_epochs 100 \
    --device cuda
```

**With soft labels (Label Distribution Learning):**

```bash
python -m skill_mapping.v2.category_trainer \
    --esco_path data/processed/master_datasets_2/master_complete_hierarchy_w_occ.csv \
    --val_path data/validation.csv \
    --test_path data/test.csv \
    --encoder_ckpt pj-mathematician/JobSkillBGE-large-en-v1.5 \
    --target_level 1 \
    --output_dir outputs/category_model_soft \
    --soft_labels \
    --n_trials 20 \
    --device cuda
```

**With augmented data (multiple text variants per occupation):**

```bash
python -m skill_mapping.v2.category_trainer \
    --esco_path data/processed/master_datasets_2/master_complete_hierarchy_w_occ.csv \
    --augmented_path data/processed/augmentation/augmented_esco_occupations.csv \
    --val_path data/validation.csv \
    --encoder_ckpt pj-mathematician/JobSkillBGE-large-en-v1.5 \
    --target_level 1 \
    --output_dir outputs/category_model_aug \
    --soft_labels \
    --n_trials 20
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--esco_path` | str | **required** | Path to ESCO master CSV |
| `--val_path` | str | **required** | Path to validation CSV |
| `--test_path` | str | `None` | Path to test CSV (optional) |
| `--encoder_ckpt` | str | `pj-mathematician/JobSkillBGE-large-en-v1.5` | Encoder model |
| `--target_level` | int | `1` | Hierarchy level (0=pillar, 1-3=levels) |
| `--n_trials` | int | `20` | Number of Optuna trials |
| `--max_epochs` | int | `50` | Max epochs per trial |
| `--final_epochs` | int | `100` | Max epochs for final training |
| `--output_dir` | str | **required** | Directory to save model |
| `--cache_name` | str | `embeddings` | Base name for embedding cache files |
| `--device` | str | `cuda` | Device (cuda/cpu) |
| `--soft_labels` | flag | `False` | Enable Label Distribution Learning mode |
| `--samples_per_occupation` | int | `None` | Enable Stochastic Dynamic Resampling with k samples per occupation per epoch |
| `--augmented_path` | str | `None` | Path to augmented occupation CSV with multiple text variants |

### Data Augmentation (Multiple Text Variants)

When `--augmented_path` is provided, the script supports **multiple `skill_brief` texts per occupation** (`conceptUri`). This creates multiple training samples per occupation, each with:
- Different input text (the augmented `skill_brief` variant)
- Same target labels (category distribution is shared)

**Augmented CSV Format:**

```csv
conceptUri,skill_brief
http://data.europa.eu/esco/occupation/abc123,"Software developer building web applications..."
http://data.europa.eu/esco/occupation/abc123,"Web developer creating responsive interfaces..."
http://data.europa.eu/esco/occupation/abc123,"Full-stack engineer developing modern apps..."
```

This effectively increases training data by creating multiple perspectives of the same occupation.

### Stochastic Dynamic Resampling

When using augmented data, different occupations may have varying numbers of text variants, causing semantic imbalance. The `--samples_per_occupation` option enables **Stochastic Dynamic Resampling** to ensure balanced representation:

```bash
python -m skill_mapping.v2.category_trainer \
    --esco_path data/esco.csv \
    --augmented_path data/augmented_occupations.csv \
    --val_path data/val.csv \
    --output_dir outputs/balanced_model \
    --samples_per_occupation 10 \
    --soft_labels
```

**How it works:**
- Each epoch samples exactly `k` text variants per occupation
- If occupation has > k samples: randomly sample k (different selection each epoch)
- If occupation has < k samples: oversample with replacement to reach k
- Full semantic coverage across epochs while maintaining per-epoch balance

**Example:**
| Occupation | Total Variants | Per Epoch (k=10) |
|------------|----------------|------------------|
| Software Engineer | 25 | 10 (random subset) |
| Data Analyst | 15 | 10 (random subset) |
| Librarian | 3 | 10 (oversampled) |

This ensures each occupation contributes equally regardless of augmentation count, preventing dominant occupations from biasing the model.

### Soft Labels (Label Distribution Learning)

When `--soft_labels` is enabled, the training targets change from binary multi-hot vectors to probability distributions:

**Hard Labels Example** (occupation with 5 skills):
- 3 skills in "Communication", 2 skills in "IT" → `[1, 1, 0, 0, ...]` (binary)

**Soft Labels Example** (same occupation):
- 3 skills in "Communication", 2 skills in "IT" → `[0.6, 0.4, 0, 0, ...]` (density)

This approach:
- Captures the **relative importance** of categories for each occupation
- Uses **KL divergence loss** instead of BCE loss
- Outputs a **probability distribution** over categories (softmax)
- Provides richer supervision signal than binary labels
- Applies **epsilon smoothing** (1e-6) to all categories for numerical stability in KL divergence

#### Soft Label Metrics

| Metric | Description |
|--------|-------------|
| `top1_overlap` | Fraction of top-1 predictions matching top-1 targets |
| `top3_overlap` | Fraction of top-3 predictions in top-3 targets |
| `top5_overlap` | Fraction of top-5 predictions in top-5 targets |
| `mse` | Mean squared error between predicted and target distributions |
| `cosine_similarity` | Cosine similarity between predicted and target distributions |
| `f1_micro` / `f1_macro` | Standard F1 (after thresholding for comparison) |

### Hierarchy Levels

| Level | Column | Description |
|-------|--------|-------------|
| 0 | `pillar_label` | Top-level category (knowledge/skills) |
| 1 | `level1_label` | Broad category |
| 2 | `level2_label` | Intermediate category |
| 3 | `level3_label` | Fine-grained category |

### Validation/Test CSV Format

```csv
job_id,job_text,skillUri
0,"Software Engineer developing web applications",http://data.europa.eu/esco/skill/abc123
0,"Software Engineer developing web applications",http://data.europa.eu/esco/skill/def456
1,"Data Analyst with Python experience",http://data.europa.eu/esco/skill/ghi789
```

### Hyperparameters Tuned

- `hidden_dim`: [256, 512, 1024]
- `dropout`: 0.0 - 0.5
- `lr`: 1e-5 - 1e-2 (log scale)
- `batch_size`: [32, 64, 128]
- `weight_decay`: 1e-6 - 1e-3 (log scale)

### Output Files

```
outputs/category_model/
├── category_classifier.pt  # Model weights
├── label_encoder.json      # Category label mapping
├── results.json            # Best params + metrics
└── cache/
    ├── embeddings_train.npy
    ├── embeddings_val.npy
    └── embeddings_test.npy
```

### Results Format

**Hard Labels:**

```json
{
  "best_params": {
    "hidden_dim": 512,
    "dropout": 0.2,
    "lr": 0.001,
    "batch_size": 64,
    "weight_decay": 1e-5
  },
  "best_val_loss": 0.1234,
  "final_val_metrics": {
    "loss": 0.1234,
    "f1_micro": 0.85,
    "f1_macro": 0.72,
    "precision_micro": 0.88,
    "recall_micro": 0.82
  },
  "test_metrics": {...},
  "config": {
    "encoder_ckpt": "pj-mathematician/JobSkillBGE-large-en-v1.5",
    "target_level": 1,
    "cat_col": "level1_label",
    "num_classes": 25,
    "input_dim": 1024,
    "soft_labels": false
  }
}
```

**Soft Labels:**

```json
{
  "best_params": {...},
  "best_val_loss": 0.0567,
  "final_val_metrics": {
    "loss": 0.0567,
    "top1_overlap": 0.78,
    "top3_overlap": 0.85,
    "top5_overlap": 0.91,
    "mse": 0.012,
    "cosine_similarity": 0.92,
    "f1_micro": 0.80,
    "f1_macro": 0.68
  },
  "config": {
    ...
    "soft_labels": true
  }
}
```

---

## 4. `category_inference.py` - Category Prediction

Use a trained category classifier to predict the skill category distribution for new jobs.

### Usage

```bash
python -m skill_mapping.v2.category_inference \
    --jobs_csv data/processed/job_titles.csv \
    --model_dir outputs/category_model \
    --output_path outputs/category_scores.json \
    --text_col processed_text \
    --id_col job_id \
    --embeddings_path outputs/job_embeddings \
    --device cuda
```

### Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--jobs_csv` | str | **required** | CSV containing jobs to predict |
| `--model_dir` | str | **required** | Directory containing trained model (`results.json`, `.pt`, `label_encoder.json`) |
| `--output_path` | str | **required** | Output JSON filename |
| `--text_col` | str | `processed_text` | Column containing job description text |
| `--id_col` | str | `None` | Job ID column (optional) |
| `--embeddings_path` | str | `None` | Pre-computed embeddings file/dir. If not provided, encoding runs on-the-fly. |
| `--compute_metrics` | flag | `False` | Run evaluation if ground truth is present |

### Output Format (JSON)

```json
{
  "job_id_1": [
    {"category": "Communication, collaboration and creativity", "score": 0.45, "logit": -0.1},
    {"category": "Information skills", "score": 0.22, "logit": -0.8},
    ...
  ],
  ...
}
```

---

## 5. `isco_trainer.py` & `isco_inference.py` - ISCO Classifier

Train a single-label classifier to predict ISCO codes (Level 1-4) from occupation text.

### Trainer Usage

```bash
python -m skill_mapping.v2.isco_trainer \
    --train_path data/occupations_en.csv \
    --output_dir outputs/isco_model \
    --isco_level 4 \
    --text_col preferredLabel \
    --desc_col description \
    --include_description \
    --n_trials 20
```

### Inference Usage

```bash
python -m skill_mapping.v2.isco_inference \
    --jobs_csv data/new_jobs.csv \
    --model_dir outputs/isco_model \
    --output_path outputs/isco_predictions \
    --topk 5
```

---

## 6. `bayesian_fuser.py` - Re-ranking (Bayesian Fusion)

Combine IR similarity scores with category priors to re-rank skill candidates.

$$ Score_{final} = Similarity \times (P(Category|Job))^w $$

### Usage

```bash
python -m skill_mapping.v2.bayesian_fuser \
    --similarity_scores_json outputs/similarity_scores.json \
    --category_scores_json outputs/category_scores.json \
    --skill_hierarchy_csv data/processed/master_skill_complete_hierarchy.csv \
    --output_dir outputs/fused_results \
    --fusion_mode bayesian \
    --weights 0.5,1.0,2.0 \
    --temperatures 1.0 \
    --thresholds 0.1
```

### Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `--similarity_scores_json` | str | Output from `similarity_scorer.py` |
| `--category_scores_json` | str | Output from `category_inference.py` |
| `--skill_hierarchy_csv` | str | ESCO skill hierarchy (maps skill -> category) |
| `--fusion_mode` | `bayesian`, `linear` | Fusion formula to use |
| `--weights` / `--alphas` | list | Comma-separated weights (w) or alphas (for linear) |
| `--temperatures` | list | Temperature scaling factors for category probs |
| `--thresholds` | list | Minimum probability thresholds to apply weighting |

### Output

Detailed metrics for each hyperparameter combination (MRR@K, R@K, P@K) are logged to console and saved in the output directory.

---

## Pipeline Example

### Step 1: Build Skill Index
```bash
python -m skill_mapping.v2.skill_indexer \
    --skills_csv data/processed/augmentation/augmented_esco_skills.csv \
    --output_dir outputs/skill_index
```

### Step 2: Calculate Raw Similarity
```bash
python -m skill_mapping.v2.similarity_scorer \
    --jobs_csv data/processed/job_titles.csv \
    --index_dir outputs/skill_index \
    --output_path outputs/similarity_scores.json
```

### Step 3: Train & Predict Categories
```bash
# Train
python -m skill_mapping.v2.category_trainer \
    --esco_path data/processed/master_datasets_2/master_complete_hierarchy_w_occ.csv \
    --val_path data/validation.csv \
    --output_dir outputs/category_model \
    --soft_labels

# Inference
python -m skill_mapping.v2.category_inference \
    --model_dir outputs/category_model \
    --jobs_csv data/processed/job_titles.csv \
    --output_path outputs/category_scores.json
```

### Step 4: Fuse Scores
```bash
python -m skill_mapping.v2.bayesian_fuser \
    --similarity_scores_json outputs/similarity_scores.json \
    --category_scores_json outputs/category_scores.json \
    --skill_hierarchy_csv data/processed/master_skill_complete_hierarchy.csv \
    --output_dir outputs/final_results \
    --fusion_mode bayesian \
    --weights 1.0
```

---

## Technical Notes

- **Cosine Similarity**: All vectors are L2-normalized; FAISS `IndexFlatIP` computes inner product = cosine similarity
- **Multi-Label Classification**: Uses `BCEWithLogitsLoss` with Sigmoid activation for independent category probabilities
- **Embedding Caching**: Embeddings are saved as `.npy` files for reuse across experiments
- **Class Imbalance**: Positive weights are computed automatically for BCEWithLogitsLoss
