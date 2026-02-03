# Skill-Based Sentence Transformer Finetuning

This module provides a skill-based approach to training sentence transformers for career path prediction.

## Overview

The training pipeline:

1. **Data Loading**: Loads career paths from `karrierewege_100k` dataset
2. **Skill Extraction**: Maps job titles to their associated skills
3. **IDF Calculation**: Computes IDF scores for skill weighting
4. **Skill Encoding**: Encodes skills as `"skill: {name} \n description: {description}"`
5. **IDF-Weighted Pooling**: Pools skills per job using IDF weights
6. **Logarithmic Position Pooling**: Pools jobs in career path with log decay
7. **Contrastive Learning**: Uses MultipleNegativesRankingLoss with ISCO group filtering

## Key Features

- **IDF-weighted skill pooling**: Skills are weighted by their inverse document frequency
- **Logarithmic position weighting**: Recent jobs have higher weight (w_i = log(1 + α × i))
- **ISCO group-aware batching**: Prevents trivial negatives in contrastive learning
- **Full integration**: Supports wandb logging, early stopping, and model checkpointing

## Files

- `src/cpp/skill_dataset.py`: Custom dataset and batch sampler
- `src/cpp/train_cpp_skills.py`: Main training script
- `scripts/test_train_cpp_skills.sh`: Test script

## Usage

### Basic Training

```bash
python -m src.cpp.train_cpp_skills \
    --data_type karrierewege_100k \
    --job_title_skills_csv results/karrierewege_esco_100k_esco_ground_truth/job_title_skills_master.csv \
    --skills_csv data/esco_datasets/skills_en.csv \
    --occupations_csv data/esco_datasets/occupations_en.csv \
    --model_name ElenaSenger/career-path-representation-mpnet-karrierewege \
    --alpha_decay 0.5 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_epochs 10 \
    --patience 3 \
    --use_skill_description \
    --output_dir results/cpp_skills \
    --save_model \
    --run_name "skill_training_exp1"
```

### With Weights & Biases Logging

```bash
python -m src.cpp.train_cpp_skills \
    --data_type karrierewege_100k \
    --job_title_skills_csv results/karrierewege_esco_100k_esco_ground_truth/job_title_skills_master.csv \
    --skills_csv data/esco_datasets/skills_en.csv \
    --occupations_csv data/esco_datasets/occupations_en.csv \
    --model_name ElenaSenger/career-path-representation-mpnet-karrierewege \
    --alpha_decay 0.5 \
    --batch_size 32 \
    --learning_rate 2e-5 \
    --num_epochs 10 \
    --use_skill_description \
    --output_dir results/cpp_skills \
    --save_model \
    --use_wandb \
    --wandb_project cpp-skills \
    --run_name "skill_training_with_logging"
```

### Quick Test (CPU, small epochs)

```bash
bash scripts/test_train_cpp_skills.sh
```

## Key Parameters

### Alpha Decay (`--alpha_decay`)
- Controls logarithmic position weighting for jobs
- Default: 0.5
- Range: [0.1, 1.0]
- Higher values give more weight to recent jobs

### Skill Description (`--use_skill_description`)
- If set, encodes skills with descriptions
- Format: `"skill: {name} \n description: {description}"`
- Otherwise: `"skill: {name}"`

### Batch Size (`--batch_size`)
- Number of samples per batch
- Note: ISCO group filtering may affect effective batch composition
- Default: 32

### Learning Rate (`--learning_rate`)
- Learning rate for AdamW optimizer
- Default: 2e-5

## Architecture

### IDF-Weighted Skill Pooling

For each job, skills are pooled using IDF weights:

```
IDF(skill) = log(total_occupations / occupation_count)
job_vector = Σ(IDF_i × skill_embedding_i) / Σ(IDF_i)
```

### Logarithmic Position Pooling

Jobs in a career path are pooled with logarithmic weights:

```
w_i = log(1 + α × i)  where i = job position (0-indexed)
career_vector = Σ(w_i × job_vector_i) / Σ(w_i)
```

### Loss Function

MultipleNegativesRankingLoss:
- Anchor: Career path skill vector
- Positive: Next ESCO occupation
- Negatives: Other occupations in batch (excluding same ISCO group)

## Data Requirements

1. **Career Path Data**: Karrierewege dataset via `Data` class
2. **Skill Mappings**: `job_title_skills_master.csv` with columns:
   - `job_title`: Job title
   - `skill`: Skill name
   - `skillUri`: ESCO skill URI
   - `score`: Skill confidence score
3. **Skill Descriptions**: `skills_en.csv` (ESCO skills)
4. **Occupation Data**: `occupations_en.csv` with ISCO groups

## Output

Training produces:
- `best_model/`: Saved model checkpoint (if `--save_model`)
- `experiment_results.csv`: Training metrics
- `train_cpp_skills.log`: Detailed logs

## Evaluation Metrics

- **MRR**: Mean Reciprocal Rank
- **R@1**: Recall at 1
- **R@5**: Recall at 5
- **R@10**: Recall at 10
- **R@20**: Recall at 20

## Notes

- ISCO group-aware sampling ensures diverse batches
- Career paths without skills get zero vectors
- Training uses AdamW optimizer with early stopping
- Model checkpoints save best validation MRR

