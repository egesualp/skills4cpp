# Skill-Based Sentence Transformer Training - Implementation Summary

## Overview

Successfully implemented a complete skill-based training pipeline for sentence transformers that uses skill representations with IDF weighting and logarithmic position pooling for career path prediction.

## Files Created

### 1. Core Implementation Files

#### `src/cpp/skill_dataset.py`
- **SkillBasedCareerPathDataset**: PyTorch Dataset for skill-based career paths
  - Extracts job titles from career history
  - Maps titles to skills using job_skill_map
  - Stores target occupation information with ISCO groups
  
- **ISCOGroupBatchSampler**: Custom batch sampler
  - Ensures no duplicate ISCO groups within a batch
  - Prevents trivial negatives in contrastive learning
  - Implements smart grouping and shuffling logic

- **collate_skill_batch**: Custom collate function for batching

#### `src/cpp/train_cpp_skills.py`
Main training script with the following components:

**Data Loading Functions:**
- `load_skill_mappings()`: Loads job-to-skills mapping from CSV
- `load_skill_descriptions()`: Loads ESCO skill descriptions
- `load_occupation_isco_groups()`: Loads ISCO group information
- `calculate_idf_scores()`: Computes IDF weights for skills
- `create_target_occupation_map()`: Creates target occupation metadata

**Pooling Functions:**
- `encode_skills()`: Encodes skills with "skill: ... \n description: ..." format
- `pool_skills_with_idf()`: IDF-weighted pooling for skills per job
  ```
  job_vector = Σ(IDF_i × skill_embedding_i) / Σ(IDF_i)
  ```
- `pool_jobs_with_log_decay()`: Logarithmic position weighting for jobs
  ```
  w_i = log(1 + α × i)  where i is job position
  career_vector = Σ(w_i × job_vector_i) / Σ(w_i)
  ```

**Training Functions:**
- `manual_train_loop()`: Complete training loop with AdamW optimizer
  - Implements MultipleNegativesRankingLoss manually
  - Proper backpropagation through model parameters
  - Early stopping based on validation MRR
  - Model checkpointing

**Evaluation Functions:**
- `evaluate_ranking()`: Comprehensive ranking metrics
  - MRR (Mean Reciprocal Rank)
  - Recall@1, Recall@5, Recall@10, Recall@20
  - Cosine similarity-based ranking

**Main Function:**
- Complete end-to-end pipeline
- Wandb integration for experiment tracking
- Results saving to CSV
- Test set evaluation with final metrics

### 2. Supporting Files

#### `scripts/test_train_cpp_skills.sh`
Bash script for quick testing:
- Runs training with minimal configuration
- CPU mode for local testing
- 2 epochs for quick validation

#### `src/cpp/test_skill_imports.py`
Validation script that tests:
- Module imports
- Basic pooling functionality
- Dataset creation with minimal data
- Component integration

#### `README_skills_training.md`
Comprehensive documentation:
- Overview of the approach
- Usage examples
- Parameter descriptions
- Architecture details
- Data requirements
- Output format

#### `IMPLEMENTATION_SUMMARY.md` (this file)
Complete implementation summary and reference

## Key Features Implemented

### 1. IDF-Weighted Skill Pooling ✓
- Skills are weighted by inverse document frequency
- Formula: `IDF = log(total_occupations / occupation_count)`
- Normalized weighted sum for each job

### 2. Logarithmic Position Pooling ✓
- Jobs weighted by position in career path
- Last job (most recent) has highest weight
- Formula: `w_i = log(1 + α × i)`
- Configurable decay parameter α (default: 0.5)

### 3. ISCO Group-Aware Batch Sampling ✓
- Custom sampler ensures unique ISCO groups per batch
- Prevents trivial negatives in contrastive learning
- Smart shuffling and iteration logic

### 4. MultipleNegativesRankingLoss ✓
- Manual implementation of contrastive loss
- Cosine similarity matrix computation
- Cross-entropy on similarity scores
- Batch negatives as hard negatives

### 5. Training Infrastructure ✓
- AdamW optimizer with configurable learning rate
- Early stopping with patience parameter
- Model checkpointing (saves best validation model)
- Comprehensive logging with loguru

### 6. Integration Features ✓
- **Wandb logging**: Track experiments, metrics, hyperparameters
- **Argument parsing**: Full CLI with all configurable options
- **Results saving**: CSV output for experiment tracking
- **Device support**: CUDA/CPU compatibility

### 7. Data Loading ✓
- Loads karrierewege_100k via Data class
- Skill mappings from CSV
- ESCO skill descriptions
- ISCO group information
- IDF calculation from training data

## Usage

### Basic Command
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
    --run_name "skill_training_v1"
```

### With Wandb Logging
```bash
python -m src.cpp.train_cpp_skills \
    --data_type karrierewege_100k \
    --job_title_skills_csv results/karrierewege_esco_100k_esco_ground_truth/job_title_skills_master.csv \
    --skills_csv data/esco_datasets/skills_en.csv \
    --occupations_csv data/esco_datasets/occupations_en.csv \
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

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--alpha_decay` | 0.5 | Logarithmic decay for job position weighting |
| `--batch_size` | 32 | Training batch size |
| `--learning_rate` | 2e-5 | Learning rate for AdamW |
| `--num_epochs` | 10 | Maximum training epochs |
| `--patience` | 3 | Early stopping patience |
| `--use_skill_description` | False | Include skill descriptions in encoding |

## Architecture Flow

```
1. Load career paths from karrierewege_100k
   ↓
2. Extract job titles from each career path
   ↓
3. Map job titles → skills (from job_title_skills_master.csv)
   ↓
4. Calculate IDF scores for each skill
   ↓
5. For each job in career path:
   a. Encode skills: "skill: {name} \n description: {desc}"
   b. Pool skills with IDF weights → job_vector
   ↓
6. Pool job vectors with logarithmic position weights → career_vector
   ↓
7. Encode target: "role: {title} \n description: {desc}"
   ↓
8. Compute MultipleNegativesRankingLoss
   - Anchor: career_vector
   - Positive: target occupation
   - Negatives: other targets in batch (different ISCO groups)
   ↓
9. Backpropagate and update model parameters
   ↓
10. Evaluate on validation set (MRR, Recall@K)
    ↓
11. Early stopping / save best model
```

## Output Files

When running training, the following files are generated:

- `{output_dir}/best_model/`: Saved model checkpoint (if --save_model)
- `{output_dir}/experiment_results.csv`: Training metrics CSV
- `logs/train_cpp_skills.log`: Detailed training logs

## Testing

Run validation tests:
```bash
python src/cpp/test_skill_imports.py
```

Quick integration test:
```bash
bash scripts/test_train_cpp_skills.sh
```

## Evaluation Metrics

The model is evaluated using:
- **MRR**: Mean Reciprocal Rank (primary metric)
- **R@1**: Recall at 1 (exact match)
- **R@5**: Recall at 5
- **R@10**: Recall at 10
- **R@20**: Recall at 20

## Dependencies

All dependencies are standard for the project:
- torch
- sentence-transformers
- pandas
- numpy
- sklearn
- tqdm
- loguru
- wandb (optional)

## Implementation Notes

### IDF Calculation
- IDF is calculated only from training+validation jobs (no test leakage)
- Counts number of distinct occupations each skill appears in
- Formula: `log(total_occupations / occupation_count)`

### ISCO Group Filtering
- Prevents same ISCO group in batch negatives
- Uses `iscoGroup` column from occupations_en.csv
- Falls back to "unknown" for unmapped occupations

### Edge Cases Handled
- Career paths without skills → zero vector
- Jobs without skill mappings → skipped in pooling
- Missing ISCO groups → labeled as "unknown"
- Empty skill lists → zero embeddings

### Training Strategy
- Manual training loop with PyTorch optimizer
- Model parameters updated via backpropagation
- SentenceTransformer encode() used for target encoding
- Career embeddings computed via skill pooling (not trainable)

## Future Enhancements

Possible improvements:
1. **Optuna integration**: Add hyperparameter tuning for α, learning rate, batch size
2. **Advanced pooling**: Attention-based pooling instead of weighted average
3. **Skill embeddings**: Make skill embeddings trainable
4. **Hard negative mining**: More sophisticated negative sampling beyond ISCO groups
5. **Multi-task learning**: Joint training on multiple objectives

## Completion Status

All planned features implemented:
- ✅ Data loading (skills, occupations, IDF)
- ✅ Custom dataset with ISCO-aware sampling
- ✅ IDF-weighted pooling
- ✅ Logarithmic position pooling
- ✅ Training loop with MultipleNegativesRankingLoss
- ✅ Evaluation metrics (MRR, Recall@K)
- ✅ Wandb integration
- ✅ Command-line interface
- ✅ Model checkpointing
- ✅ Results saving
- ✅ Documentation
- ✅ Test scripts

## Contact & Support

For questions or issues:
1. Check `README_skills_training.md` for detailed usage
2. Review logs in `logs/train_cpp_skills.log`
3. Validate imports with `python src/cpp/test_skill_imports.py`
4. Run quick test with `bash scripts/test_train_cpp_skills.sh`

