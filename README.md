# Skills4CPP

**Leveraging Skills for Career Path Prediction**

This project implements a multi-stage pipeline for Career Path Prediction (CPP), combining skill extraction, skill-based scoring, and multi-modal neural learning to predict users' next career steps.

---

## 📁 Project Structure

```
skills4cpp/
├── src/                          # Source code modules
│   ├── cpp/                      # Career Path Prediction (main component)
│   ├── SkillPrediction/          # Skill prediction pipeline
│   ├── JobToESCO/                # Job title to ESCO occupation mapping
│   ├── ErrorAnalysis/            # Analysis and visualization tools
│   ├── LLMDataAugmentation/      # LLM-based data augmentation
│   ├── seq_transformer/          # Sequence transformer experiments
│   └── helpers/                  # Utility scripts and evaluators
│
├── experiments/configs/          # Pre-configured experiment scripts
│   ├── run_cpp/                  # CPP training bash scripts
│   ├── run_fused_scorer.sh       # Skill scorer configuration
│   ├── job_to_esco/              # Job-to-ESCO mapping scripts
│   └── ...
│
├── data/                         # Datasets and ESCO taxonomy files
├── docs/                         # Documentation
├── notebooks/                    # Jupyter notebooks for exploration
├── reports/                      # Generated figures and analysis
├── results/                      # Experiment outputs
├── models/                       # Saved model checkpoints
├── tests/                        # Unit tests
└── outputs/                      # Pipeline intermediate outputs
```

---

## 🧩 Source Modules (`src/`)

### `cpp/` — Career Path Prediction ⭐
The core module for predicting next career steps using multi-modal learning.

**Features:**
- Multi-modal fusion (text history + skill embeddings + structured features)
- Multiple training script variants (v2, v2_fixed, v2_gated, v2_tuned_weights, v3)
- Optuna hyperparameter optimization
- Skill pooling strategies (mean, weighted_mean, weighted_idf)

📖 See [`src/cpp/README.md`](src/cpp/README.md) for detailed documentation.

---

### `SkillPrediction/` — Skill Extraction Pipeline
Predicts skills associated with job experiences.

| Submodule | Description |
|-----------|-------------|
| `CategoryPredictor/` | Predicts ESCO skill categories for jobs |
| `FusedScorer/` | Fuses multiple skill prediction signals |
| `LLMReranking/` | LLM-based skill reranking |
| `helpers/` | Utility functions |

---

### `JobToESCO/` — Job Title Mapping
Maps raw job titles to standardized ESCO occupation codes.

**Key files:**
- `model.py` — Bi-encoder model for title-to-occupation matching
- `evaluate.py` — Evaluation and ranking metrics
- `utils.py` — Data loading and preprocessing

---

### `ErrorAnalysis/` — Analysis Tools
Tools for analyzing model behavior and generating visualizations.

**Key scripts:**
- `pooling_method_analysis.py` — Compare skill pooling strategies
- `analyze_skill_confidence.py` — Analyze prediction confidence scores

📖 See [`src/ErrorAnalysis/README_POOLING_ANALYSIS.md`](src/ErrorAnalysis/README_POOLING_ANALYSIS.md)

---

### `LLMDataAugmentation/`
LLM-based data augmentation for training data enrichment.

---

### `seq_transformer/`
Experimental sequence transformer models for career path modeling.

---

### `helpers/`
Shared utility scripts including:
- `SkillRetrievalEvaluator.py` — Evaluate skill retrieval performance
- `vanilla_ir_eval.py` — Information retrieval evaluation

---

## ⚙️ Experiment Configurations

Pre-configured bash scripts for running experiments are located in:

```
experiments/configs/
```

### CPP Training Scripts
```
experiments/configs/run_cpp/
├── run_cpp_decorte_static.sh      # Decorte static ablation
├── run_cpp_decorte_optuna.sh      # Decorte with Optuna
├── run_cpp_kw_cp_static.sh        # Karrierewege static experiments
├── run_cpp_kw_cp_static_v3.sh     # V3 on Karrierewege
├── run_cpp_decorte_static_v3.sh   # V3 on Decorte
└── ...
```

### Other Experiment Scripts
```
experiments/configs/
├── run_fused_scorer.sh            # Skill scorer training
├── job_to_esco/                   # Job-to-ESCO mapping
└── category_classifier/           # Category prediction
```

---

## 📚 Documentation

Detailed documentation is available in the `docs/` folder:

| Document | Description |
|----------|-------------|
| `esco_dataset_map.md` | ESCO dataset structure and relationships |
| `master_datasets_guide.md` | Master datasets created from ESCO |
| `train_cpp_enhanced_v2.md` | CPP v2 training script documentation |
| `train_cpp_enhanced_v3.md` | CPP v3 (last-job skills) documentation |
| `finetune_last_job_skills.md` | Skill encoder fine-tuning guide |
| `skill_overlap_scoring_v2.md` | Skill overlap scoring documentation |
| `occupation_score_fuser.md` | Score fusion documentation |
| `build_hierarchy_guide.md` | ESCO hierarchy processing guide |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
pip install -r requirements_dev.txt
```

### 2. Run CPP Training
```bash
# Example: Static training on Decorte
python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type decorte \
    --skill_scores_file path/to/fused_predictions.json \
    --use_text_history --use_skill_text \
    --output_dir results/cpp/my_experiment
```

### 3. Use Pre-configured Scripts
```bash
# View available experiment configs
ls experiments/configs/run_cpp/

# Run a pre-configured experiment
sbatch experiments/configs/run_cpp/run_cpp_decorte_static.sh
```

---

## 📊 Supported Datasets

| Dataset | Description | Type |
|---------|-------------|------|
| `decorte` | Anonymous working histories | Free-text |
| `decorte_esco` | Decorte with ESCO mappings | ESCO titles |
| `karrierewege` | German career paths | ESCO titles |
| `karrierewege_100k` | 100k subset | ESCO titles |
| `kw_cp` | Karrierewege Career Path | With descriptions |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## 📝 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Last Updated: February 2026*
