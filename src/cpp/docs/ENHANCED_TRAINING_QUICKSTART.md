# Enhanced Training Script - Quick Start

## TL;DR

I've created **`train_cpp_enhanced.py`** that combines the best of both worlds:

✅ On-the-fly embeddings (storage efficient)  
✅ Optuna optimization (automatic hyperparameter tuning)  
✅ Proper metrics (MRR, Recall@K)  
✅ Two architecture modes (simple concatenation vs multi-modal)  
✅ Support for different encoders (text vs skills)  
✅ All best practices (early stopping, train+val, test evaluation)

## Run It Now

### Quick Test (10 minutes)
```bash
cd /dss/dsshome1/02/ra95kix2/thesis/skills4cpp

python -m src.cpp.train_cpp_enhanced \
    --n_trials 10 \
    --max_epochs 10 \
    --batch_size 32 \
    --num_workers 4
```

### Full Training with Multi-Modal Architecture (2-3 hours)
```bash
python -m src.cpp.train_cpp_enhanced \
    --use_advanced \
    --pooling_strategy weighted_idf \
    --alpha 1.0 \
    --beta 1.0 \
    --n_trials 50 \
    --max_epochs 30 \
    --batch_size 32 \
    --num_workers 4 \
    --device cuda
```

### With Different Encoders
```bash
python -m src.cpp.train_cpp_enhanced \
    --use_advanced \
    --encoder_text "ElenaSenger/career-path-representation-mpnet-decorte" \
    --encoder_skill "sentence-transformers/all-MiniLM-L6-v2" \
    --n_trials 50 \
    --max_epochs 30
```

## What You Get

### Output Files
- `results/cpp_training_enhanced/final_model.pt` - Trained model
- `results/cpp_training_enhanced/optuna_study.pkl` - Optimization history

### Evaluation Metrics
```
FINAL TEST SET RESULTS
================================================================================
MRR: 0.5678        # Mean Reciprocal Rank (primary metric)
R@1: 0.3456        # Top-1 accuracy
R@5: 0.6789        # Top-5 accuracy
R@10: 0.7890       # Top-10 accuracy
R@20: 0.8567       # Top-20 accuracy
================================================================================
```

## Key Flags

| Flag | Effect |
|------|--------|
| `--use_advanced` | Use multi-modal architecture (Option 2) instead of simple concatenation |
| `--encoder_skill` | Use different encoder for skills (default: same as encoder_text) |
| `--pooling_strategy` | How to pool skills: `mean`, `weighted_mean`, `weighted_idf` |
| `--use_skill_description` | Include skill descriptions in text |
| `--last_job_only` | Use only last job pairs (simpler task) |
| `--n_trials` | Number of Optuna trials (more = better hyperparameters) |
| `--max_epochs` | Max epochs per trial |
| `--patience` | Early stopping patience (default: 5) |

## Architecture Modes

### Mode 1: Simple Concatenation (Default)
```
Input: [text | skill_text | structured]
  ↓
MLP (optimized by Optuna)
  ↓
Output
```

**When to use:** Quick baseline, smaller search space

### Mode 2: Multi-Modal (`--use_advanced`)
```
text → Encoder → hidden
skill_text → Encoder → hidden  (shared hyperparameters)
structured → Encoder → hidden
  ↓
Concatenate → Linear → Output
```

**When to use:** Better interpretability, can analyze modality contributions

## What Changed from Original Scripts

### vs `train_cpp.py`
- ✅ Added Optuna hyperparameter optimization
- ✅ Added CosineEmbeddingLoss (better for embeddings)
- ✅ Added proper ranking metrics (MRR, Recall@K)
- ✅ Added early stopping
- ✅ Added test set evaluation
- ✅ Added final training on train+val
- ✅ Kept on-the-fly generation (storage efficient)

### vs `train_cpp_gemini.py`
- ✅ Kept Optuna optimization
- ✅ Kept proper metrics
- ✅ Changed to on-the-fly generation (no pre-computation needed!)
- ✅ Added multi-modal architecture option
- ✅ Added support for different encoders
- ✅ More flexible configuration

## Files Modified/Created

### New Files
1. **`src/cpp/train_cpp_enhanced.py`** ⭐ - Main training script
2. **`src/cpp/README_ENHANCED_TRAINING.md`** - Detailed documentation
3. **`TRAINING_SCRIPTS_COMPARISON.md`** - Comparison of all scripts
4. **`ENHANCED_TRAINING_QUICKSTART.md`** - This file

### Modified Files
1. **`src/cpp/cpp_dataset.py`** - Added support for separate skill encoder

## Common Use Cases

### Research Question 1: Does Multi-Modal Help?
```bash
# Simple (baseline)
python -m src.cpp.train_cpp_enhanced \
    --output_dir results/simple

# Multi-modal
python -m src.cpp.train_cpp_enhanced \
    --use_advanced \
    --output_dir results/multimodal

# Compare test MRR
```

### Research Question 2: Do Domain-Specific Encoders Help?
```bash
# Same encoder
python -m src.cpp.train_cpp_enhanced --use_advanced

# Different encoder for skills
python -m src.cpp.train_cpp_enhanced \
    --use_advanced \
    --encoder_skill "different-model"
```

### Research Question 3: Which Pooling Strategy is Best?
```bash
for strategy in mean weighted_mean weighted_idf; do
    python -m src.cpp.train_cpp_enhanced \
        --use_advanced \
        --pooling_strategy $strategy \
        --output_dir "results/pooling_${strategy}"
done
```

## Troubleshooting

### Out of Memory?
```bash
--batch_size 16 --num_workers 2
```

### DataLoader Issues?
```bash
--num_workers 0  # Single process for debugging
```

### Want Faster Testing?
```bash
--n_trials 10 --max_epochs 10
```

## Next Steps

1. ✅ Run quick test to verify everything works
2. ✅ Compare simple vs advanced modes
3. ✅ Experiment with different configurations
4. ✅ Analyze results and select best model

## Need More Help?

- **Detailed guide:** `src/cpp/README_ENHANCED_TRAINING.md`
- **Script comparison:** `TRAINING_SCRIPTS_COMPARISON.md`
- **Dataset docs:** `src/cpp/README_DATASET.md`
- **Quick reference:** `QUICK_START.md`

---

**Ready to train? Just run the commands above!** 🚀

