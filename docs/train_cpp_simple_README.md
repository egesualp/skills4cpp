# Simple CPP Training Script

A simplified training script for Career Path Prediction that reproduces MLP baseline metrics without the complexity of Optuna hyperparameter optimization.

## Overview

This script (`src/cpp/train_cpp_simple.py`) is similar in simplicity to `vector_transformation.py` but designed for the career path prediction task. It provides:

- ✅ Simple concatenation MLP architecture
- ✅ Fixed hyperparameters (no Optuna)
- ✅ Early stopping with patience
- ✅ WandB integration
- ✅ Ranking metrics (MRR, Recall@K)
- ✅ Uses same `data_classes.py` logic as main pipeline
- ✅ Pre-computed embeddings with caching

## Quick Start

### 1. Using config file (recommended)

```bash
python src/cpp/train_cpp_simple.py --config configs/cpp_simple_baseline.yaml
```

Or with the helper script:

```bash
bash scripts/run_cpp_simple.sh
```

### 2. Using command-line arguments

```bash
python src/cpp/train_cpp_simple.py \
    --data_type decorte \
    --encoder ElenaSenger/career-path-representation-mpnet-decorte \
    --hidden_dim 512 \
    --n_layers 2 \
    --dropout 0.1 \
    --max_epochs 50 \
    --patience 5 \
    --lr 0.0001 \
    --batch_size 512 \
    --use_wandb \
    --run_name my_experiment
```

### 3. Override config with command-line args

```bash
bash scripts/run_cpp_simple.sh --max_epochs 100 --patience 10 --run_name longer_training
```

## Key Arguments

### Data
- `--data_type`: Dataset to use (decorte, karrierewege, etc.)
- `--use_text_description`: Include job descriptions (default: titles only)
- `--use_skill_description`: Include skill descriptions

### Model
- `--hidden_dim`: Hidden layer size (default: 512)
- `--n_layers`: Number of hidden layers (default: 2)
- `--dropout`: Dropout rate (default: 0.1)

### Training
- `--max_epochs`: Maximum training epochs (default: 50)
- `--patience`: Early stopping patience (default: 5)
- `--min_delta`: Minimum improvement threshold (default: 0.001)
- `--lr`: Learning rate (default: 0.0001)
- `--weight_decay`: L2 regularization (default: 0.0001)
- `--batch_size`: Training batch size (default: 512)
- `--eval_batch_size`: Evaluation batch size (default: 2048)

### WandB
- `--use_wandb`: Enable WandB logging
- `--wandb_project`: WandB project name
- `--wandb_entity`: WandB entity/username
- `--run_name`: Experiment name

### Output
- `--output_dir`: Directory to save results (default: results/cpp_simple)
- `--save_model`: Save trained model checkpoint

## Architecture

The model uses a simple concatenation architecture:

```
Input: [Text Embeddings | Skill Embeddings | Structured Features]
  ↓
Linear(input_dim → hidden_dim) + ReLU + Dropout
  ↓
Linear(hidden_dim → hidden_dim) + ReLU + Dropout  (repeated n_layers times)
  ↓
Linear(hidden_dim → output_dim)
  ↓
Output: Predicted career path embedding
```

Loss: `CosineEmbeddingLoss` (similarity-based)

## Features

### Input Features (all enabled by default)
1. **Text History**: Job title/description embeddings (from encoder)
2. **Skill Text**: Aggregated skill embeddings for jobs
3. **Structured Meta-features**: One-hot encoded skill properties

### Evaluation Metrics
- **MRR**: Mean Reciprocal Rank
- **Recall@K**: K ∈ {1, 5, 10, 20}

## Example Experiments

### Experiment 1: Baseline with default settings
```bash
python src/cpp/train_cpp_simple.py \
    --config configs/cpp_simple_baseline.yaml \
    --run_name baseline_v1
```

### Experiment 2: Deeper network
```bash
python src/cpp/train_cpp_simple.py \
    --config configs/cpp_simple_baseline.yaml \
    --hidden_dim 768 \
    --n_layers 3 \
    --run_name deeper_network
```

### Experiment 3: With job descriptions
```bash
python src/cpp/train_cpp_simple.py \
    --config configs/cpp_simple_baseline.yaml \
    --use_text_description \
    --run_name with_descriptions
```

### Experiment 4: Longer training
```bash
python src/cpp/train_cpp_simple.py \
    --config configs/cpp_simple_baseline.yaml \
    --max_epochs 100 \
    --patience 10 \
    --run_name longer_training
```

## Output

The script produces:

1. **Console logs**: Training progress and final metrics
2. **WandB dashboard**: Live training curves (if enabled)
3. **Model checkpoint**: `output_dir/model.pt` (if `--save_model`)
4. **Results CSV**: `output_dir/results.csv` with all metrics

### Example output:
```
================================================================================
FINAL TEST RESULTS
================================================================================
MRR: 0.4521
R@1: 0.3245
R@5: 0.5632
R@10: 0.6789
R@20: 0.7654
================================================================================
```

## Differences from `train_cpp_enhanced_v2.py`

| Feature | train_cpp_simple.py | train_cpp_enhanced_v2.py |
|---------|---------------------|--------------------------|
| Lines of code | ~650 | ~1834 |
| Optuna HPO | ❌ | ✅ |
| Modality ablation | ❌ | ✅ |
| Multi-modal architecture | ❌ | ✅ |
| Mixed precision | ❌ | ✅ |
| Gradient accumulation | ❌ | ✅ |
| Score saving (fusion) | ❌ | ✅ |
| Clean test set | ❌ | ✅ |
| Simple to understand | ✅ | ❌ |

## Notes

- **Training strategy**: Combines train+val sets for final training (no validation split during training)
- **Early stopping**: Uses test set performance for early stopping (since train+val are combined)
- **Embeddings caching**: Automatically caches computed embeddings for faster re-runs
- **GPU memory**: Uses ~4-6GB GPU memory with default batch size

## Troubleshooting

### Out of memory error
Reduce batch size:
```bash
python src/cpp/train_cpp_simple.py --config configs/cpp_simple_baseline.yaml --batch_size 256
```

### Slow data loading
Increase workers:
```bash
python src/cpp/train_cpp_simple.py --config configs/cpp_simple_baseline.yaml --num_workers 8
```

### WandB not available
Install WandB:
```bash
pip install wandb
wandb login
```

## Citation

If you use this script, please cite the original work and this implementation.





