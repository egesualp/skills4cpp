# CPP (Career Path Prediction) Scripts Documentation

This module contains the core scripts for the **Career Path Prediction (CPP)** project. It implements multi-modal learning approaches that combine text embeddings, skill information, and structured features to predict users' next career steps.

---

## 📁 Directory Structure

```
src/cpp/
├── 📄 Core Training Scripts
│   ├── train_cpp_enhanced_v2.py          # Main multi-modal training (recommended)
│   ├── train_cpp_enhanced_v2_fixed.py    # Fixed modality weights + InfoNCE loss option
│   ├── train_cpp_enhanced_v2_gated.py    # Gated attention fusion variant
│   ├── train_cpp_enhanced_v2_tuned_weights.py  # Learnable pooling weights variant
│   ├── train_cpp_enhanced_v3.py          # Last-job skill document approach
│   └── finetune_last_job_skills.py       # SentenceTransformer fine-tuning for skills
│
├── 📄 Data & Dataset Classes
│   ├── data_classes.py                   # Data class for loading datasets
│   ├── data_loaders.py                   # Skill/vocabulary loading utilities
│   ├── data_loaders_tuned.py             # Data loaders for tuned weight variants
│   ├── cpp_dataset.py                    # PyTorch Dataset (on-the-fly embeddings)
│   ├── cpp_dataset_tuned.py              # Dataset for tuned weight variants
│   ├── skill_dataset.py                  # Skill-specific dataset utilities
│   └── utils.py                          # Dataset loading functions
│
├── 📄 Skill Processing
│   └── skill_pooling.py                  # Skill pooling strategies & IDF calculation
│
├── 📄 Evaluation & Analysis
│   ├── evaluation.py                     # Evaluation metrics utilities
│   ├── validate_encoder.py               # Encoder validation utilities
│   ├── analyze_skill_scores.py           # Skill score analysis
│   ├── profile_training.py               # Training profiling utilities
│   └── create_vocab.py                   # Vocabulary creation script
│
├── 📂 decorte/                           # Decorte dataset scoring & fusion
│   ├── skill_overlap_scoring_v2.py       # Skill overlap scoring (basic)
│   ├── skill_overlap_scoring_v3.py       # Skill overlap scoring (advanced)
│   ├── occupation_score_fuser.py         # Score fusion (text + skill overlap)
│   ├── commands.sh                       # Example command references
│   └── README_SCORING_AND_FUSION.md      # Scoring pipeline documentation
│
├── 📂 scripts/                           # Utility scripts
│   ├── create_job_skills_mapping.py      # Generate job-skill mappings
│   ├── create_job_id_mapping.py          # Generate job ID mappings
│   └── README_skill_mapping.md           # Skill mapping documentation
│
└── 📂 helpers/                           # Helper & diagnostic scripts
    ├── diagnose_skill_embeddings.py      # Embedding diagnostics
    ├── example_training_with_dataset.py  # Training example
    ├── profile_idf_stats.py              # IDF statistics profiling
    ├── profile_skill_stats.py            # Skill statistics profiling
    ├── test_bottlenecks.py               # Performance bottleneck testing
    └── test_dataset.py                   # Dataset testing
```

---

## 🛠️ Pre-configured Experiment Scripts

For ready-to-run experiments, see the bash scripts in:
```
experiments/configs/run_cpp/
```

| Script | Dataset | Description |
|--------|---------|-------------|
| `run_cpp_decorte_static.sh` | Decorte | Comprehensive ablation study on Decorte |
| `run_cpp_decorte_optuna.sh` | Decorte | Hyperparameter optimization with Optuna |
| `run_cpp_kw_cp_static.sh` | Karrierewege_CP | Static experiments on Karrierewege |
| `run_cpp_kw_cp_static_v3.sh` | Karrierewege_CP | V3 (last-job skills) on Karrierewege |
| `run_cpp_decorte_static_v3.sh` | Decorte | V3 experiments on Decorte |
| `run_cpp_baseline_kw_100k.sh` | Karrierewege 100k | Baseline on larger dataset |
| `run_cpp_kw_100k_skills_battle.sh` | Karrierewege 100k | Skills comparison experiments |
| `run_cpp_static_ablation_kw_100k.sh` | Karrierewege 100k | Full ablation on 100k |

**Example: View one of these scripts for configuration reference:**
```bash
cat experiments/configs/run_cpp/run_cpp_decorte_static.sh
```

---

## 🚀 Training Scripts Overview

### Main Training Script: `train_cpp_enhanced_v2.py` ⭐

The primary training script for multi-modal Career Path Prediction.

**Features:**
- **Multi-Modal Learning**: Fuses text history, skill embeddings, and structured features
- **Optuna Optimization**: Automated hyperparameter search
- **GPU Optimization**: Supports GPU-pinned embeddings and mixed precision (FP16)
- **Flexible Pooling**: Multiple skill pooling strategies (mean, weighted_mean, weighted_idf)

**Architecture Modes:**
| Flag | Description |
|------|-------------|
| Default | Simple concatenation (early fusion) |
| `--use_advanced` | Multi-modal architecture (late fusion with modality encoders) |

### Training Script Variants

| Script | Description | Key Feature |
|--------|-------------|-------------|
| `train_cpp_enhanced_v2.py` | Standard multi-modal training | Baseline with CosineEmbeddingLoss |
| `train_cpp_enhanced_v2_fixed.py` | Fixed modality weights | Adds `--loss_type infonce` + `--temperature` |
| `train_cpp_enhanced_v2_gated.py` | Gated attention fusion | Learnable attention across modalities |
| `train_cpp_enhanced_v2_tuned_weights.py` | Learnable pooling | Trains α, β, γ during backprop |
| `train_cpp_enhanced_v3.py` | Last-job skill documents | Single encoder for skills from last job only |

---

## 📋 Complete Command-Line Arguments

### Data & Skill Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_type` | `decorte` | Dataset (`decorte`, `decorte_esco`, `karrierewege`, `karrierewege_100k`, `kw_cp`) |
| `--skill_scores_file` | **Required** | Path to JSON/JSONL with predicted skills per job_id (e.g., `fused_predictions.json`) |
| `--skill_embeddings_dir` | None | Directory with pre-computed skill embeddings (`skill_embeddings.npy` + `skill_metadata.json`) |
| `--esco_skills_file` | `data/esco_datasets/skills_en.csv` | ESCO skills metadata |
| `--esco_taxonomy_file` | `...occupationSkillRelations_en.csv` | ESCO taxonomy for IDF calculation |
| `--vocab_dir` | `data/processed/master_datasets_2/` | Vocabulary files for structured features |
| `--skill_properties_file` | `...skill_properties_map.json` | Skill type/hierarchy metadata |

### Encoder Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--encoder_text` | `ElenaSenger/career-path-representation-mpnet-decorte` | Encoder for text history |
| `--encoder_skill` | (same as encoder_text) | Encoder for skills (ignored if `--skill_embeddings_dir` is set) |

### Feature & Modality Selection

| Argument | Description |
|----------|-------------|
| `--use_text_history` | Enable job history text embedding |
| `--use_text_description` | Include job descriptions in text (not just titles) |
| `--use_skill_text` | Enable skill text embeddings |
| `--use_skill_description` | Include skill descriptions (not just names) |
| `--use_structured` | Enable structured meta-features (skill types, hierarchy) |
| `--use_advanced` | Use MultiModalCPPModel (late fusion) instead of SimpleConcatModel |
| `--use_modality_weights` | Add learnable modality weights (α_text, α_skill, α_struct) |

### Skill Pooling Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--pooling_strategy` | `weighted_idf` | Pooling method: `mean`, `weighted_mean`, `weighted_idf` |
| `--alpha` | 1.0 | Exponent for skill confidence scores |
| `--beta` | 1.0 | Exponent for IDF weights |
| `--use_skill_path_log_pooling` | False | Use logarithmic position weighting across jobs in career path |
| `--skill_path_alpha_decay` | 0.5 | Decay rate for log pooling (0 = mean over jobs) |
| `--skill_confidence_threshold` | None | Filter skills below this prediction score |
| `--min_max_normalize_skills` | False | Normalize skill scores to [0, 1] per job |

### Training Hyperparameters (Static Mode)

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr` | 2e-5 | Learning rate |
| `--hidden_dim` | 512 | MLP hidden dimension |
| `--n_layers` | 1 | Number of MLP layers |
| `--dropout` | 0.1 | Dropout rate |
| `--max_epochs` | 10 | Maximum training epochs |
| `--batch_size` | 512 | Training batch size |
| `--eval_batch_size` | 4092 | Evaluation batch size |
| `--patience` | 2 | Early stopping patience |
| `--seed` | 42 | Random seed |

### Optimizer Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--optimizer` | `adam` | Optimizer: `adam` or `sgd` |
| `--weight_decay` | 0.0 | Weight decay (L2 regularization) |
| `--momentum` | 0.9 | Momentum for SGD |
| `--nesterov` | False | Use Nesterov momentum for SGD |

### Optuna Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--optuna` | False | Enable Optuna hyperparameter optimization |
| `--n_trials` | 50 | Number of Optuna trials |
| `--optuna_patience` | 3 | Early stopping patience per trial |
| `--val_sample_ratio` | 0.1 | Fraction of validation set for Optuna trials |
| `--train_sample_ratio` | 1.0 | Fraction of training set for Optuna trials |
| `--min_delta` | 0.001 | Minimum MRR improvement to reset patience |

### GPU & Performance

| Argument | Default | Description |
|----------|---------|-------------|
| `--device` | `cuda` | Device: `cuda` or `cpu` |
| `--num_workers` | auto | DataLoader workers (auto-detects from SLURM) |
| `--mixed_precision` | False | Enable FP16 mixed precision training |
| `--gradient_accumulation_steps` | 1 | Gradient accumulation for larger effective batch size |
| `--pin_embeddings_to_gpu` | False | Pin pre-computed embeddings to GPU memory |
| `--embeddings_cache_dir` | `...embeddings/` | Cache directory for pre-computed embeddings |
| `--force_recompute` | False | Force recomputation of cached embeddings |

**Note on Caching:** The caching mechanism automatically computes a unique MD5 hash of the input data (including dataset size and content) and includes it in the cache filename (e.g., `history_emb_{encoder}_{hash}.npy`). This ensures that if the underlying data changes, the system will automatically recompute embeddings instead of loading stale data, even if the cache directory remains the same.

### Output & Logging

| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | `results/cpp` | Output directory |
| `--run_name` | `cpp_enhanced` | Run identifier |
| `--save_model` | False | Save final model checkpoint |
| `--save_scores` | False | Save prediction scores for fusion with skill overlap |
| `--scores_output_dir` | `{output_dir}/scores` | Directory for saved scores |
| `--save_study` | False | Save Optuna study to pickle |
| `--results_csv_path` | `results/cpp/experiment_results.csv` | CSV for experiment results |

### W&B Logging

| Argument | Default | Description |
|----------|---------|-------------|
| `--use_wandb` | False | Enable Weights & Biases logging |
| `--wandb_project` | `cpp-enhanced` | W&B project name |
| `--wandb_entity` | None | W&B entity/team name |

### Data Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--no_subspans` | False | Disable subspan generation in data loading |
| `--eval_clean_test` | False | Also evaluate on clean test set |
| `--filter_repetitive` | False | Remove samples where history ends with same occupation as target |

### Study Alignment / Reproducibility

| Argument | Default | Description |
|----------|---------|-------------|
| `--normalize_input` | False | L2-normalize embeddings before MLP (matches vector_transformation.py) |
| `--early_stop_metric` | `mrr` | Metric for early stopping: `mrr` or `loss` |

---

## 📊 Data Pipeline

### Data Loading Flow

```
1. Data Class (data_classes.py)
   ├── load_prepare_decorte()      → Decorte dataset (free-text)
   ├── load_prepare_decorte_esco() → Decorte with ESCO titles
   └── load_prepare_karrierewege() → Karrierewege dataset

2. Skill Mapping (data_loaders.py)
   └── load_job_skill_data_by_id() → Job ID → Skills mapping (from fused_predictions.json)

3. Dataset Creation (cpp_dataset.py)
   └── CareerPathDataset           → PyTorch Dataset with on-the-fly embeddings

4. DataLoader
   └── collate_career_path_batch() → Custom collation for batched training
```

### Supported Datasets

| Dataset | Description | Data Type |
|---------|-------------|-----------|
| `decorte` | Free-text job titles | Raw titles + descriptions |
| `decorte_esco` | ESCO-normalized titles | Standard ESCO occupations |
| `karrierewege` | German career paths | ESCO occupations |
| `karrierewege_100k` | Larger karrierewege split | ESCO occupations |
| `kw_cp` | Karrierewege Career Path | With free-text descriptions |

---

## 🔧 Skill Processing

### Skill Pooling Strategies (`skill_pooling.py`)

| Strategy | Formula | Description |
|----------|---------|-------------|
| `mean` | `E = (1/n) Σ E_s` | Simple average |
| `weighted_mean` | `E = Σ(score_s · E_s) / Σ(score_s)` | Weighted by prediction confidence |
| `weighted_idf` | `E = Σ(score^α · IDF^β · E_s) / Σ(w_s)` | IDF + confidence weighting |

**IDF Calculation:**
```
IDF(s) = log((N_jobs + 1) / (n_s + 1))
```
Where `N_jobs` is total jobs in train+val and `n_s` is jobs containing skill `s`.

---

## 📈 Scoring & Fusion Pipeline

### Skill Overlap Scoring

Compute skill-based similarity scores for ranking:

```
S_overlap(history, target) = |S_career ∩ S(target)| / |S(target)|
```

```bash
python -m src.cpp.decorte.skill_overlap_scoring_v3 \
    --data_type decorte \
    --skill_scores_file path/to/fused_predictions.json \
    --output_dir results/skill_overlap \
    --top_k_skills 10 \
    --save_scores
```

### Score Fusion

Combine text-based and skill-based scores:

**Linear Fusion:**
```
S_hybrid = α · S_text + (1 - α) · S_overlap
```

```bash
python -m src.cpp.decorte.occupation_score_fuser \
    --text_scores_dir results/cpp/scores \
    --overlap_scores_dir results/skill_overlap \
    --output_dir results/fused \
    --fusion_mode linear \
    --alphas "0.0,0.3,0.5,0.7,1.0" \
    --splits val --eval_splits test
```

See `decorte/README_SCORING_AND_FUSION.md` for detailed documentation.

---

## 🧪 Ablation Studies

The scripts support systematic modality ablation experiments.

### Modality Control Flags

```bash
--use_text_history   # Job history text embedding
--use_skill_text     # Skill text embedding  
--use_structured     # Structured meta-features
--use_text_description  # Include descriptions (for both history and targets)
```

### Example: Run All Ablations

```bash
# 1. Job Path Only (titles + descriptions)
python -m src.cpp.train_cpp_enhanced_v2 \
    --use_text_history --use_text_description \
    --skill_scores_file $SKILL_FILE ...

# 2. Skills Only
python -m src.cpp.train_cpp_enhanced_v2 \
    --use_skill_text --use_text_description \
    --skill_scores_file $SKILL_FILE ...

# 3. Multimodal (text + skills)
python -m src.cpp.train_cpp_enhanced_v2 \
    --use_text_history --use_skill_text --use_text_description \
    --use_advanced \
    --skill_scores_file $SKILL_FILE ...
```

---

## 🏃 Quick Start Examples

### 1. Static Training with Pre-computed Skills

```bash
python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type decorte \
    --skill_scores_file path/to/fused_predictions.json \
    --skill_embeddings_dir path/to/skill_index \
    --encoder_text "ElenaSenger/career-path-representation-mpnet-decorte" \
    --use_text_history --use_skill_text --use_text_description \
    --use_advanced \
    --pooling_strategy weighted_idf \
    --lr 2e-5 \
    --hidden_dim 512 \
    --n_layers 1 \
    --dropout 0.1 \
    --max_epochs 50 \
    --batch_size 16 \
    --patience 2 \
    --normalize_input \
    --early_stop_metric loss \
    --save_scores \
    --eval_clean_test \
    --device cuda \
    --output_dir results/cpp/my_experiment
```

### 2. Optuna Hyperparameter Search

```bash
python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type decorte \
    --skill_scores_file path/to/fused_predictions.json \
    --use_text_history --use_skill_text \
    --optuna --n_trials 50 \
    --max_epochs 10 \
    --device cuda \
    --output_dir results/cpp/optuna_search
```

### 3. Learnable Pooling Weights (Tuned Weights Variant)

```bash
python -m src.cpp.train_cpp_enhanced_v2_tuned_weights \
    --data_type decorte \
    --skill_scores_file path/to/fused_predictions.json \
    --skill_embeddings_dir path/to/skill_index \
    --use_text_history --use_skill_text --use_text_description \
    --use_advanced \
    --use_learnable_pooling \
    --pooling_lr_multiplier 0.1 \
    --initial_alpha 1.0 \
    --initial_beta 1.0 \
    --initial_gamma 0.0 \
    --max_skills_per_path 400 \
    --max_epochs 50 \
    --device cuda
```

### 4. Log Pooling (Position-Weighted Skills)

```bash
python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type decorte \
    --skill_scores_file path/to/fused_predictions.json \
    --use_text_history --use_skill_text --use_text_description \
    --use_advanced \
    --pooling_strategy weighted_idf \
    --use_skill_path_log_pooling \
    --skill_path_alpha_decay 0.5 \
    --device cuda
```

---

## 📋 Evaluation Metrics

The scripts compute standard information retrieval metrics:

| Metric | Description |
|--------|-------------|
| **MRR** | Mean Reciprocal Rank |
| **R@1** | Recall at 1 (exact match) |
| **R@5** | Recall at 5 |
| **R@10** | Recall at 10 |
| **R@20** | Recall at 20 |

**Example Output:**
```
FINAL TEST SET RESULTS
================================================================================
MRR: 0.5678
R@1: 0.3456
R@5: 0.6789
R@10: 0.7890
R@20: 0.8567
================================================================================
```

---

## 🔍 Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of Memory | Reduce `--batch_size` (try 16 or 8) or use `--gradient_accumulation_steps` |
| DataLoader crashes | Set `--num_workers 0` for debugging |
| Slow training | Increase `--num_workers` (4-8), use `--pin_embeddings_to_gpu`, enable `--mixed_precision` |
| Low MRR scores | Try `--use_advanced`, increase `--max_epochs`, check skill file matches dataset |
| Skill file mismatch | Ensure `skill_scores_file` contains job_ids from the same dataset |
| Missing skills | Check job_id format in skill file; use `--min_max_normalize_skills` for score normalization |

---

## 📚 Related Documentation

| Document | Description |
|----------|-------------|
| [decorte/README_SCORING_AND_FUSION.md](decorte/README_SCORING_AND_FUSION.md) | Scoring & fusion pipeline |
| [scripts/README_skill_mapping.md](scripts/README_skill_mapping.md) | Skill mapping generation |

---

*Last Updated: February 2026*