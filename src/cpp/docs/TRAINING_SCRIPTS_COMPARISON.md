# Training Scripts Comparison

## Overview of Available Scripts

You now have **3 training scripts** in `src/cpp/`:

1. **`train_cpp.py`** - Original simple training script
2. **`train_cpp_gemini.py`** - Gemini's Optuna-optimized script (pre-computed features)
3. **`train_cpp_enhanced.py`** ⭐ **RECOMMENDED** - Best of both worlds

## Quick Comparison Table

| Feature | train_cpp.py | train_cpp_gemini.py | train_cpp_enhanced.py ⭐ |
|---------|--------------|---------------------|------------------------|
| **Data Loading** | On-the-fly | Pre-computed .npy | On-the-fly |
| **Storage Needed** | Minimal | **GBs** | Minimal |
| **Hyperparameter Tuning** | ❌ Manual | ✅ Optuna | ✅ Optuna |
| **Loss Function** | MSE | CosineEmbedding | CosineEmbedding |
| **Metrics** | MSE only | MRR | MRR + Recall@K |
| **Early Stopping** | ❌ No | ✅ Yes | ✅ Yes |
| **Test Evaluation** | ❌ No | ✅ Yes | ✅ Yes |
| **Multi-Modal** | ✅ Yes | ❌ No | ✅ Yes (optional) |
| **Different Encoders** | Limited | ❌ No | ✅ Yes |
| **Architecture Options** | 1 (multi-modal) | 1 (concat) | 2 (both) |
| **Final Train+Val** | ❌ No | ✅ Yes | ✅ Yes |
| **Flexibility** | ✅ High | ❌ Low | ✅ Very High |

## When to Use Each

### Use `train_cpp.py` If:
- ❌ **Generally not recommended anymore**
- Use `train_cpp_enhanced.py` instead with fixed hyperparameters

### Use `train_cpp_gemini.py` If:
- ✅ You've already pre-computed all features
- ✅ You want fastest possible training
- ✅ You have plenty of storage space
- ❌ But storage inefficient and less flexible

### Use `train_cpp_enhanced.py` ⭐ If:
- ✅ **Recommended for most use cases**
- ✅ You want optimal hyperparameters (Optuna)
- ✅ You want storage efficiency (on-the-fly)
- ✅ You want proper metrics (MRR, Recall@K)
- ✅ You want flexibility (different encoders, architectures)
- ✅ You want to compare simple vs multi-modal

## Migration Guide

### From `train_cpp.py` → `train_cpp_enhanced.py`

**Before:**
```bash
python -m src.cpp.train_cpp \
    --pooling_strategy weighted_idf \
    --batch_size 32 \
    --num_epochs 10 \
    --hidden_dim 512
```

**After:**
```bash
python -m src.cpp.train_cpp_enhanced \
    --use_advanced \
    --pooling_strategy weighted_idf \
    --batch_size 32 \
    --n_trials 50 \
    --max_epochs 30
    # hidden_dim will be optimized automatically!
```

### From `train_cpp_gemini.py` → `train_cpp_enhanced.py`

**Before:**
```bash
# Step 1: Pre-compute features (takes time + storage)
python -m src.cpp.generate_embeddings \
    --pooling_strategy weighted_idf \
    --output_dir precomputed_features/

# Step 2: Train
python -m src.cpp.train_cpp_gemini \
    --features_dir precomputed_features/ \
    --n_trials 50 \
    --use_text --use_skill_text --use_structured
```

**After:**
```bash
# Single command, no pre-computation needed!
python -m src.cpp.train_cpp_enhanced \
    --pooling_strategy weighted_idf \
    --n_trials 50 \
    --use_advanced
```

## Detailed Feature Comparison

### 1. Data Loading Strategy

#### train_cpp.py & train_cpp_enhanced.py
```python
# Generate embeddings on-the-fly during training
for batch in train_loader:
    h_text = encoder.encode(history_doc)  # Dynamic
    h_skill = pool(encoder.encode(skills))  # Dynamic
    # ...
```
**Pros:** Minimal storage, flexible  
**Cons:** Slightly slower training

#### train_cpp_gemini.py
```python
# Load pre-computed embeddings
X_train = np.load("train_text.npy")  # Pre-computed
X_skill = np.load("train_skill_text.npy")  # Pre-computed
# ...
```
**Pros:** Faster training  
**Cons:** Requires GBs of storage, inflexible

---

### 2. Architecture Comparison

#### train_cpp.py: Multi-Modal Only
```python
class MultiModalCPPModel:
    text_proj = MLP(text → hidden)
    skill_proj = MLP(skill → hidden)
    struct_proj = MLP(struct → hidden)
    fusion = Linear(3*hidden → output)
```

#### train_cpp_gemini.py: Concatenation Only
```python
class MLP:
    x = concat([text, skill, struct])  # Immediate fusion
    output = MLP(x)
```

#### train_cpp_enhanced.py: BOTH!
```python
# Option 1: Concatenation (--use_advanced NOT set)
class SimpleConcatModel:
    x = concat([text, skill, struct])
    output = MLP(x)

# Option 2: Multi-Modal (--use_advanced)
class MultiModalCPPModel:
    # Separate projections + linear fusion
    # (same as train_cpp.py but with Optuna)
```

---

### 3. Hyperparameter Optimization

#### train_cpp.py: Manual
```python
# Fixed in command line
hidden_dim = args.hidden_dim  # You must guess
lr = args.lr  # You must guess
```
**Problem:** Requires manual tuning (time-consuming)

#### train_cpp_gemini.py & train_cpp_enhanced.py: Optuna
```python
# Automatic search
hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 768])
lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
# Optuna finds best values automatically!
```
**Benefit:** Finds optimal hyperparameters automatically

---

### 4. Loss Functions

#### train_cpp.py: MSE Loss
```python
loss = nn.MSELoss()(pred, target)
```
**Problem:** Optimizes Euclidean distance, not ideal for embeddings

#### train_cpp_gemini.py & train_cpp_enhanced.py: Cosine Loss
```python
loss = nn.CosineEmbeddingLoss()(pred, target, label=1)
```
**Benefit:** Optimizes cosine similarity (better for embeddings)

---

### 5. Evaluation Metrics

#### train_cpp.py: Loss Only
```python
val_loss = mse(predictions, targets)
print(f"Val Loss: {val_loss}")
```
**Problem:** MSE doesn't reflect ranking quality

#### train_cpp_gemini.py & train_cpp_enhanced.py: Ranking Metrics
```python
mrr = calculate_mrr(predictions, targets, all_targets)
recall_at_5 = calculate_recall_at_k(predictions, targets, all_targets, k=5)
print(f"MRR: {mrr}, R@5: {recall_at_5}")
```
**Benefit:** Proper metrics for ranking tasks

---

## Performance Comparison (Expected)

### Speed (Training Time)

**Fastest to Slowest:**
1. `train_cpp_gemini.py` (pre-computed) - 100% baseline
2. `train_cpp.py` (on-the-fly, fixed params) - ~110%
3. `train_cpp_enhanced.py` (on-the-fly + Optuna) - ~120% per trial

**Note:** train_cpp_enhanced.py runs multiple trials, but finds better models!

### Storage Requirements

1. `train_cpp.py` & `train_cpp_enhanced.py`: **~1 GB** (raw data only)
2. `train_cpp_gemini.py`: **~10-50 GB** (pre-computed embeddings)

### Model Quality (Expected)

**Best to Worst:**
1. `train_cpp_enhanced.py` with `--use_advanced` ⭐
2. `train_cpp_enhanced.py` without `--use_advanced`
3. `train_cpp_gemini.py`
4. `train_cpp.py` (unless you're very lucky with hyperparameters)

---

## Recommended Workflow

### Phase 1: Quick Baseline (5 minutes)
```bash
# Simple concatenation, few trials
python -m src.cpp.train_cpp_enhanced \
    --n_trials 10 \
    --max_epochs 10 \
    --batch_size 32
```

### Phase 2: Multi-Modal (30 minutes)
```bash
# Multi-modal architecture
python -m src.cpp.train_cpp_enhanced \
    --use_advanced \
    --n_trials 30 \
    --max_epochs 20
```

### Phase 3: Full Optimization (2-4 hours)
```bash
# Best hyperparameters + IDF weighting
python -m src.cpp.train_cpp_enhanced \
    --use_advanced \
    --pooling_strategy weighted_idf \
    --n_trials 50 \
    --max_epochs 30 \
    --device cuda
```

### Phase 4: Experiment (varies)
```bash
# Try different encoders, skill descriptions, etc.
python -m src.cpp.train_cpp_enhanced \
    --use_advanced \
    --encoder_skill "sentence-transformers/all-MiniLM-L6-v2" \
    --use_skill_description \
    --n_trials 50
```

---

## Summary Recommendation

### 🥇 **Best Choice: `train_cpp_enhanced.py`**

**Why?**
- ✅ All features from Gemini's script (Optuna, metrics, etc.)
- ✅ Storage efficient (on-the-fly generation)
- ✅ Flexible (multiple architectures, encoders)
- ✅ Best practices (early stopping, train+val, test evaluation)
- ✅ Research-friendly (easy to compare configurations)

**Use this for:**
- Production training
- Research experiments
- Comparing different approaches
- Finding optimal hyperparameters

### When NOT to Use Enhanced Script

❌ If you've already pre-computed features and just want quick iteration:
→ Use `train_cpp_gemini.py` (but consider deleting pre-computed files after!)

---

## Command Cheat Sheet

```bash
# Quick test (simple mode)
python -m src.cpp.train_cpp_enhanced --n_trials 10

# Full training (multi-modal)
python -m src.cpp.train_cpp_enhanced --use_advanced --n_trials 50

# Different encoders
python -m src.cpp.train_cpp_enhanced \
    --use_advanced \
    --encoder_text "model1" \
    --encoder_skill "model2"

# With skill descriptions
python -m src.cpp.train_cpp_enhanced \
    --use_advanced \
    --use_skill_description \
    --pooling_strategy weighted_idf

# Last job only
python -m src.cpp.train_cpp_enhanced --last_job_only

# Compare architectures
python -m src.cpp.train_cpp_enhanced  # Simple
python -m src.cpp.train_cpp_enhanced --use_advanced  # Multi-modal
```

---

**Bottom Line:** Use `train_cpp_enhanced.py` for everything! 🚀

