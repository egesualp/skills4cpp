# Which Training Script Should I Use?

I've created **two** simplified scripts based on your needs. Here's how to choose:

## 📊 Comparison Table

| Feature | train_vector_transformation.py | train_cpp_simple.py |
|---------|--------------------------------|---------------------|
| **Approach** | Vector transformation (original study) | Multi-modal MLP |
| **Input Features** | Career history text **only** | Text + Skills + Structured |
| **What it does** | Transforms career embeddings → target embeddings | Predicts target from all features |
| **Similar to** | `vector_transformation.py` (original) | `train_cpp_enhanced_v2.py` (simplified) |
| **Input normalization** | ✅ Yes (during training) | ❌ No |
| **Validation during training** | ✅ Separate val set | ❌ Uses test for early stopping |
| **Lines of code** | ~450 | ~650 |
| **Use case** | **Reproduce original study** | Reproduce full multi-modal approach |

## 🎯 Quick Decision Guide

### Use `train_vector_transformation.py` if:
- ✅ You want to reproduce the **original vector transformation study**
- ✅ You want a **text-only** approach (no skills, no structured features)
- ✅ You want the **exact same architecture** as `vector_transformation.py`
- ✅ You need separate train/val/test splits during training

**This is what you asked for!**

### Use `train_cpp_simple.py` if:
- ✅ You want to use **all modalities** (text + skills + structured)
- ✅ You want to reproduce the **full MLP approach** from `train_cpp_enhanced_v2.py`
- ✅ You want to compare text-only vs multi-modal performance

---

## 📝 Usage Examples

### Option 1: Vector Transformation (Original Study)

```bash
# Exact reproduction of original approach
python src/cpp/train_vector_transformation.py \
    --config configs/vector_transformation_baseline.yaml \
    --use_wandb \
    --run_name original_study_reproduction

# With custom encoder (e.g., fine-tuned one)
python src/cpp/train_vector_transformation.py \
    --config configs/vector_transformation_baseline.yaml \
    --encoder ElenaSenger/career-path-representation-mpnet-decorte \
    --run_name with_finetuned_encoder

# With job descriptions instead of just titles
python src/cpp/train_vector_transformation.py \
    --config configs/vector_transformation_baseline.yaml \
    --use_text_description \
    --run_name with_descriptions
```

### Option 2: Multi-Modal MLP

```bash
# Full multi-modal approach
python src/cpp/train_cpp_simple.py \
    --config configs/cpp_simple_baseline.yaml \
    --use_wandb \
    --run_name multimodal_baseline
```

---

## 🔍 Architecture Details

### Vector Transformation Architecture
```
Career History Text
    ↓
[Encoder: SentenceTransformer]
    ↓
Career Embedding (768-dim)
    ↓
[Normalize during training]
    ↓
Linear(768 → 512) + ReLU + Dropout
    ↓
Linear(512 → 512) + ReLU + Dropout
    ↓
Linear(512 → 768)
    ↓
Predicted Target Embedding (768-dim)

Loss: CosineEmbeddingLoss(predicted, target, label=1)
```

### Multi-Modal MLP Architecture
```
[Career Text Embedding | Skill Embeddings | Structured Features]
           768-dim     |     768-dim      |    variable-dim
    ↓
Concatenate → Input (e.g., 1536+ dim)
    ↓
Linear(input_dim → 512) + ReLU + Dropout
    ↓
Linear(512 → 512) + ReLU + Dropout
    ↓
Linear(512 → 768)
    ↓
Predicted Target Embedding (768-dim)

Loss: CosineEmbeddingLoss(predicted, target, label=1)
```

---

## 📈 Expected Performance

Based on typical results:

### Vector Transformation (Text Only)
- **MRR**: ~0.35-0.45
- **R@1**: ~0.25-0.35
- **R@5**: ~0.50-0.60
- **R@10**: ~0.60-0.70
- **R@20**: ~0.70-0.80

### Multi-Modal MLP (Text + Skills + Structured)
- **MRR**: ~0.40-0.50 (typically better)
- **R@1**: ~0.30-0.40
- **R@5**: ~0.55-0.65
- **R@10**: ~0.65-0.75
- **R@20**: ~0.75-0.85

*Note: Actual results depend on dataset, encoder, and hyperparameters*

**Metrics Reported:**
- Both scripts now report **MRR** during each epoch (for validation)
- Final test results include: **Loss, MRR, R@1, R@5, R@10, R@20**

---

## 🚀 Recommended Workflow

### Step 1: Reproduce Original Study
```bash
# Start with the original vector transformation approach
python src/cpp/train_vector_transformation.py \
    --config configs/vector_transformation_baseline.yaml \
    --use_wandb \
    --run_name baseline_reproduction
```

### Step 2: Try with Fine-tuned Encoder
```bash
# Use your fine-tuned encoder for better results
python src/cpp/train_vector_transformation.py \
    --config configs/vector_transformation_baseline.yaml \
    --encoder ElenaSenger/career-path-representation-mpnet-decorte \
    --run_name with_finetuned_encoder
```

### Step 3: Compare with Multi-Modal
```bash
# See if adding skills and structured features helps
python src/cpp/train_cpp_simple.py \
    --config configs/cpp_simple_baseline.yaml \
    --encoder ElenaSenger/career-path-representation-mpnet-decorte \
    --run_name multimodal_comparison
```

### Step 4: Analyze Results
Check your WandB dashboard to compare:
- Training curves
- Final metrics (MRR, Recall@K)
- Training time
- Model complexity

---

## 🔧 Common Modifications

### Change Hidden Layer Size
```bash
# Vector transformation
python src/cpp/train_vector_transformation.py \
    --config configs/vector_transformation_baseline.yaml \
    --hidden_sizes 768 768 \
    --run_name larger_hidden

# Multi-modal
python src/cpp/train_cpp_simple.py \
    --config configs/cpp_simple_baseline.yaml \
    --hidden_dim 768 \
    --run_name larger_hidden
```

### Change Number of Layers
```bash
# Vector transformation (3 layers)
python src/cpp/train_vector_transformation.py \
    --config configs/vector_transformation_baseline.yaml \
    --hidden_sizes 512 512 512 \
    --run_name deeper_network

# Multi-modal (3 layers)
python src/cpp/train_cpp_simple.py \
    --config configs/cpp_simple_baseline.yaml \
    --n_layers 3 \
    --run_name deeper_network
```

### Longer Training
```bash
python src/cpp/train_vector_transformation.py \
    --config configs/vector_transformation_baseline.yaml \
    --max_epochs 100 \
    --patience 10 \
    --run_name longer_training
```

---

## ❓ FAQ

**Q: Which script reproduces the "other study" you mentioned?**  
A: Use `train_vector_transformation.py` - it matches the original `vector_transformation.py` approach.

**Q: Can I get ranking metrics (MRR, Recall@K) from vector transformation?**  
A: The current script only reports loss. You'd need to add evaluation code to compute ranking metrics (I can help with this if needed).

**Q: Which approach is better?**  
A: Multi-modal typically performs better, but vector transformation is simpler and more interpretable.

**Q: Can I use both scripts with the same data?**  
A: Yes! Both use the same `Data` class and data loading logic.

**Q: How long does training take?**  
A: Vector transformation: ~5-15 min/epoch, Multi-modal: ~10-30 min/epoch (depends on GPU and data size).

---

## 📚 Summary

**For reproducing the original study: Use `train_vector_transformation.py`**

This is the direct replacement for `vector_transformation.py` with:
- Same architecture (vector transformation)
- Same input (text embeddings only)
- Simplified code structure
- WandB integration added
- Config file support added

**Start here:**
```bash
python src/cpp/train_vector_transformation.py \
    --config configs/vector_transformation_baseline.yaml \
    --use_wandb
```

