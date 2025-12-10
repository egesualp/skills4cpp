# Enhanced Training Script - Complete Guide

## Overview

`train_cpp_enhanced.py` combines the best features from both approaches:

### ✅ From Original Script (`train_cpp.py`)
- On-the-fly dataset (storage efficient)
- Multi-modal architecture
- Support for different encoders

### ✅ From Gemini Script (`train_cpp_gemini.py`)
- Optuna hyperparameter optimization
- CosineEmbeddingLoss (better for embeddings)
- Proper ranking metrics (MRR, Recall@K)
- Early stopping
- Pruning bad trials
- Final training on train+val
- Test set evaluation

## Key Features

### 🎯 Two Architecture Modes

#### Mode 1: Simple Concatenation (Gemini's Approach)
```bash
python -m src.cpp.train_cpp_enhanced
# Default mode, no --use_advanced flag
```

- Early fusion: concatenates all features immediately
- Single MLP processes everything
- Smaller search space (faster optimization)
- Good baseline approach

#### Mode 2: Multi-Modal with Shared Hyperparameters (Option 2)
```bash
python -m src.cpp.train_cpp_enhanced --use_advanced
```

- Late fusion: each modality processed separately
- Shared encoder hyperparameters (efficient search)
- Simple linear fusion head
- Better interpretability

### 🔧 Flexible Encoder Configuration

#### Same Encoder for Everything
```bash
python -m src.cpp.train_cpp_enhanced \
    --encoder_text "ElenaSenger/career-path-representation-mpnet-decorte"
# encoder_skill will default to same as encoder_text
```

#### Different Encoders for Text vs Skills
```bash
python -m src.cpp.train_cpp_enhanced \
    --encoder_text "ElenaSenger/career-path-representation-mpnet-decorte" \
    --encoder_skill "sentence-transformers/all-MiniLM-L6-v2" \
    --use_advanced
```

**Why use different encoders?**
- Domain-specific model for skills (e.g., SkillBERT)
- Smaller/faster model for skills
- Research question: does specialized encoding help?

### 📊 Proper Evaluation Metrics

The script calculates:
- **MRR (Mean Reciprocal Rank)**: Primary optimization metric
- **Recall@1**: Exact match accuracy
- **Recall@5**: Top-5 accuracy
- **Recall@10**: Top-10 accuracy
- **Recall@20**: Top-20 accuracy

These are the standard metrics for ranking tasks!

### 🔍 Optuna Hyperparameter Search

#### Search Space (Simple Mode)
```python
hidden_dim: [256, 512, 768, 1024]
n_layers: [1, 2, 3, 4]
dropout: [0.1 - 0.5]
lr: [1e-5 - 1e-3] (log scale)
```

#### Search Space (Advanced/Multi-Modal Mode)
```python
hidden_dim: [256, 512, 768]
n_layers: [1, 2, 3]
dropout: [0.1 - 0.5]
use_modality_weights: [True, False]
lr: [1e-5 - 1e-3] (log scale)
```

Only 5-6 hyperparameters = efficient optimization!

## Usage Examples

### Example 1: Quick Test (Simple Mode)
```bash
python -m src.cpp.train_cpp_enhanced \
    --n_trials 10 \
    --max_epochs 10 \
    --batch_size 32 \
    --num_workers 4
```

### Example 2: Full Training (Multi-Modal, IDF-weighted)
```bash
python -m src.cpp.train_cpp_enhanced \
    --use_advanced \
    --pooling_strategy weighted_idf \
    --alpha 1.0 \
    --beta 1.0 \
    --n_trials 50 \
    --max_epochs 30 \
    --patience 5 \
    --batch_size 32 \
    --num_workers 4 \
    --device cuda
```

### Example 3: Different Encoders + Skill Descriptions
```bash
python -m src.cpp.train_cpp_enhanced \
    --use_advanced \
    --encoder_text "ElenaSenger/career-path-representation-mpnet-decorte" \
    --encoder_skill "sentence-transformers/all-MiniLM-L6-v2" \
    --use_skill_description \
    --pooling_strategy weighted_idf \
    --n_trials 50 \
    --max_epochs 30
```

### Example 4: Last Job Only (Simple Task)
```bash
python -m src.cpp.train_cpp_enhanced \
    --last_job_only \
    --pooling_strategy mean \
    --n_trials 30 \
    --max_epochs 20
```

## Command-Line Arguments

### Data Configuration
```
--data_type              Dataset type (default: decorte)
--master_skill_file      Path to job-skill mapping CSV
--esco_skills_file       Path to ESCO skills CSV
--vocab_dir              Directory with vocabulary files
--skill_properties_file  Path to skill properties JSON
```

### Encoder Configuration
```
--encoder_text           Encoder for text history (default: mpnet-decorte)
--encoder_skill          Encoder for skills (default: same as encoder_text)
```

### Feature Configuration
```
--use_skill_description  Include skill descriptions in text
--last_job_only          Use only last job pairs
--pooling_strategy       mean | weighted_mean | weighted_idf
--alpha                  Exponent for confidence score (default: 1.0)
--beta                   Exponent for IDF score (default: 1.0)
```

### Architecture
```
--use_advanced           Use multi-modal architecture (Option 2)
                        If not set, uses simple concatenation
```

### Optuna Configuration
```
--n_trials              Number of optimization trials (default: 50)
--max_epochs            Max epochs per trial (default: 30)
--patience              Early stopping patience (default: 5)
```

### Training Configuration
```
--batch_size            Batch size (default: 32)
--num_workers           DataLoader workers (default: 4)
--device                cuda | cpu (default: auto-detect)
```

### Output
```
--output_dir            Output directory (default: results/cpp_training_enhanced)
--study_name            Optuna study name (default: cpp_optuna_study)
```

## Understanding the Output

### During Optimization
```
[I 2025-11-16 14:32:15] Trial 0 finished with value: 0.5234
[I 2025-11-16 14:35:42] Trial 1 finished with value: 0.5512
[I 2025-11-16 14:36:20] Trial 2 pruned.
...
```

- Each trial tests different hyperparameters
- Shows validation MRR
- Pruned trials are stopped early (bad performance)

### Final Results
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

**Interpreting metrics:**
- **MRR = 0.5678**: On average, correct answer is in top ~2 positions
- **R@1 = 0.3456**: 34.56% exact match accuracy
- **R@5 = 0.6789**: Correct answer in top-5 for 67.89% of samples
- **R@10 = 0.7890**: Correct answer in top-10 for 78.90% of samples

## Output Files

### `final_model.pt`
Contains:
```python
{
    'model_state_dict': ...,      # Trained model weights
    'best_params': {...},          # Best hyperparameters found
    'test_metrics': {...},         # Test set performance
    'args': {...}                  # Training configuration
}
```

### `optuna_study.pkl`
Complete Optuna study object with all trial history

### Loading a Trained Model
```python
import torch
from src.cpp.train_cpp_enhanced import MultiModalCPPModel, SimpleConcatModel

checkpoint = torch.load('results/cpp_training_enhanced/final_model.pt')

# Reconstruct model
if checkpoint['args']['use_advanced']:
    model = MultiModalCPPModel(
        text_dim=768,
        skill_text_dim=768,
        structured_dim=2000,
        hidden_dim=checkpoint['best_params']['hidden_dim'],
        n_layers=checkpoint['best_params']['n_layers'],
        dropout=checkpoint['best_params']['dropout'],
        output_dim=768,
        use_modality_weights=checkpoint['best_params']['use_modality_weights']
    )
else:
    model = SimpleConcatModel(...)

model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Use for inference
# predictions = model(batch)
```

## Performance Tips

### 1. Choose Appropriate Number of Trials
- **Quick test**: 10 trials
- **Standard**: 50 trials
- **Thorough**: 100+ trials

More trials = better hyperparameters, but longer time

### 2. Adjust num_workers
- **CPU**: Try 2-4 workers
- **Multi-core**: Try 4-8 workers
- **If issues**: Set to 0 for debugging

### 3. Use GPU if Available
```bash
python -m src.cpp.train_cpp_enhanced --device cuda
```

Much faster training!

### 4. Start Simple, Then Advanced
1. First run without `--use_advanced` (baseline)
2. Then try `--use_advanced` (multi-modal)
3. Compare results

### 5. Monitor Memory Usage
If out of memory:
- Reduce `--batch_size` (try 16 or 8)
- Reduce `--num_workers` (try 2 or 0)
- Use smaller `--max_epochs` (try 20)

## Comparison: Simple vs Advanced Mode

| Feature | Simple Mode | Advanced Mode |
|---------|-------------|---------------|
| Architecture | Early fusion (concat) | Late fusion (multi-modal) |
| Search space | Slightly larger | Smaller (5-6 params) |
| Speed | Slightly faster | Slightly slower |
| Interpretability | Lower | Higher (can analyze modalities) |
| Flexibility | Single concatenated input | Separate modality processing |
| Best for | Quick baseline | Final model |

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution:**
```bash
--batch_size 16 --num_workers 2
```

### Issue: "DataLoader worker died"
**Solution:**
```bash
--num_workers 0  # Debug mode
```

### Issue: All trials being pruned
**Solution:**
- Increase `--max_epochs` (give trials more time)
- Adjust search space in code
- Check data quality

### Issue: Low MRR scores
**Possible causes:**
- Dataset too difficult
- Wrong pooling strategy (try different ones)
- Need more trials
- Encoder not suitable

### Issue: Training is very slow
**Solution:**
- Reduce `--n_trials` for testing
- Increase `--batch_size` if memory allows
- Use `--device cuda` if available
- Increase `--num_workers`

## Best Practices

1. ✅ **Start with defaults** to establish baseline
2. ✅ **Compare modes** (simple vs advanced)
3. ✅ **Try different pooling strategies** (mean vs weighted_idf)
4. ✅ **Use validation metrics** to select best model
5. ✅ **Only trust test set** for final evaluation
6. ✅ **Save your config** for reproducibility
7. ✅ **Monitor training** (watch for overfitting)

## Research Questions You Can Answer

1. **Does multi-modal help?**
   ```bash
   # Run both and compare test MRR
   python -m src.cpp.train_cpp_enhanced
   python -m src.cpp.train_cpp_enhanced --use_advanced
   ```

2. **Do skill-specific encoders help?**
   ```bash
   # Same encoder
   python -m src.cpp.train_cpp_enhanced --use_advanced
   
   # Different encoder
   python -m src.cpp.train_cpp_enhanced --use_advanced \
       --encoder_skill "sentence-transformers/all-MiniLM-L6-v2"
   ```

3. **Which pooling strategy is best?**
   ```bash
   for strategy in mean weighted_mean weighted_idf; do
       python -m src.cpp.train_cpp_enhanced \
           --pooling_strategy $strategy \
           --output_dir "results/pooling_${strategy}"
   done
   ```

4. **Do skill descriptions help?**
   ```bash
   # Without descriptions
   python -m src.cpp.train_cpp_enhanced
   
   # With descriptions
   python -m src.cpp.train_cpp_enhanced --use_skill_description
   ```

## Next Steps

1. Run a quick test to verify everything works
2. Compare simple vs advanced modes
3. Experiment with different encoders
4. Try different feature configurations
5. Analyze results and iterate

---

**Need help?** Check the main documentation or the example scripts!



