# FusedScorer

Multi-modal score fusion module for job-to-skill mapping. This module combines skill similarity scores with occupational classification signals (ISCO codes or skill categories) to improve skill prediction accuracy.

## Overview

The FusedScorer module provides two main approaches for fusing skill similarity scores with auxiliary occupation-based signals:

| Script | Signal Source | Description |
|--------|---------------|-------------|
| `isco_fuser.py` | ISCO predictions | Fuses IR similarity scores with ISCO group probabilities via a skill-ISCO affinity matrix |
| `category_fuser.py` | Skill categories | Bayesian re-ranking using predicted skill category probabilities |
| `isco_fuser_chunked.py` | ISCO predictions | Memory-efficient version for large-scale datasets with Parquet support |

## Fusion Strategies

Both fusers support two fusion modes:

- **Multiplicative (Bayesian)**: `score_final = similarity_ir × (P_signal)^α`
- **Linear**: `score_final = (1 - α) × similarity_ir + α × P_signal`

### Normalization Methods

Scores can be normalized before fusion using:
- `minmax`: Scales values to [0, 1]
- `zscore`: Z-score normalization with sigmoid transformation
- `rank`: Percentile-based normalization

## Key Features

- **Grid Search Optimization**: Automatically searches over hyperparameter combinations (α, γ, normalization method)
- **ISCO Level Support**: Works with ISCO hierarchy levels 1-4
- **Parallel Processing**: Multi-threaded evaluation using joblib
- **Chunked Processing**: For large datasets, use `isco_fuser_chunked.py` with Parquet format

## Usage

### ISCO-based Fusion

```bash
python -m SkillPrediction.FusedScorer.isco_fuser \
    --esco_dir data/esco_datasets \
    --label_encoder path/to/label_encoder.json \
    --task_a path/to/occupation_predictions.jsonl \
    --task_b path/to/similarity_scores.json \
    --isco_preds path/to/isco_predictions.json \
    --decorte_map path/to/job_mapping.csv \
    --output_dir path/to/output \
    --fusion_strategy multiplicative \
    --isco_level 2 \
    --n_jobs 4
```

### Category-based Fusion

```bash
python -m SkillPrediction.FusedScorer.category_fuser \
    --similarity_scores_json path/to/similarity_scores.json \
    --category_scores_json path/to/category_scores.json \
    --skill_hierarchy path/to/skill_hierarchy.csv \
    --output_dir path/to/output \
    --fusion_mode bayesian \
    --top_k 20
```

### Large-scale Processing (Chunked)

For datasets that don't fit in memory, first convert to Parquet:

```bash
# Convert similarity scores to Parquet
python -m SkillPrediction.FusedScorer.isco_fuser_chunked --convert-to-parquet \
    --task_b path/to/similarity_scores.json \
    --parquet_output path/to/similarity_scores.parquet

# Run fusion with chunked processing
python -m SkillPrediction.FusedScorer.isco_fuser_chunked \
    --task_b_parquet path/to/similarity_scores.parquet \
    --chunk_size 5000 \
    ...
```

## Required Inputs

| Argument | Description |
|----------|-------------|
| `--task_a` | Occupation predictions (JSONL) from occupation classifier |
| `--task_b` | Skill similarity scores (JSON) from IR/embedding model |
| `--isco_preds` | ISCO classification probabilities (JSON) |
| `--esco_dir` | Path to ESCO taxonomy data |
| `--label_encoder` | Label encoder from trained ISCO classifier |
| `--decorte_map` | Job ID to ESCO occupation mapping (CSV) |

## Output

The module produces:
- `fused_scores.json`: Re-ranked skill predictions per job
- `grid_search_results.csv`: Hyperparameter search results with metrics
- `best_config.json`: Optimal hyperparameter configuration

## Example Scripts

See `command.sh` and `run_experiments.sh` for complete usage examples.
