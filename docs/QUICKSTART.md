# 🚀 Quick Start Guide

## TL;DR - Run This Now

### To reproduce the original vector transformation study:
```bash
python src/cpp/train_vector_transformation.py \
    --config configs/vector_transformation_baseline.yaml \
    --use_wandb
```

### To run multi-modal MLP baseline:
```bash
python src/cpp/train_cpp_simple.py \
    --config configs/cpp_simple_baseline.yaml \
    --use_wandb
```

---

## 📂 What Files Were Created

### Training Scripts (Pick ONE)

1. **`src/cpp/train_vector_transformation.py`** ⭐ **USE THIS FOR ORIGINAL STUDY**
   - Reproduces `vector_transformation.py` approach
   - Text embeddings only
   - ~450 lines, very simple
   
2. **`src/cpp/train_cpp_simple.py`**
   - Multi-modal MLP (text + skills + structured)
   - Simplified version of `train_cpp_enhanced_v2.py`
   - ~650 lines

### Config Files

- **`configs/vector_transformation_baseline.yaml`** - For vector transformation
- **`configs/cpp_simple_baseline.yaml`** - For multi-modal MLP

### Documentation

- **`docs/WHICH_SCRIPT_TO_USE.md`** - Detailed comparison guide ⭐ **READ THIS**
- **`docs/train_cpp_simple_README.md`** - Full documentation for multi-modal script
- **`examples/train_cpp_simple_example.sh`** - Usage examples

---

## 🎯 Your Goal: Reproduce Original Study

**Use `train_vector_transformation.py`** because:
- ✅ Same architecture as original `vector_transformation.py`
- ✅ Text embeddings only (no skills, no structured features)
- ✅ Input normalization during training
- ✅ Separate validation set
- ✅ Simple and clean code

### Step-by-Step

#### 1. Basic run (uses default encoder)
```bash
python src/cpp/train_vector_transformation.py \
    --config configs/vector_transformation_baseline.yaml \
    --use_wandb \
    --run_name baseline_run
```

#### 2. With your fine-tuned encoder
```bash
python src/cpp/train_vector_transformation.py \
    --config configs/vector_transformation_baseline.yaml \
    --encoder ElenaSenger/career-path-representation-mpnet-decorte \
    --use_wandb \
    --run_name finetuned_encoder
```

#### 3. Different dataset
```bash
python src/cpp/train_vector_transformation.py \
    --config configs/vector_transformation_baseline.yaml \
    --data_type karrierewege_100k \
    --use_wandb \
    --run_name karrierewege_experiment
```

#### 4. Custom hyperparameters
```bash
python src/cpp/train_vector_transformation.py \
    --data_type decorte \
    --encoder sentence-transformers/all-mpnet-base-v2 \
    --hidden_sizes 768 768 \
    --dropout \
    --dropout_rate 0.2 \
    --max_epochs 100 \
    --patience 10 \
    --lr 0.001 \
    --batch_size 256 \
    --use_wandb \
    --run_name custom_hparams \
    --save_model
```

---

## 📊 What to Expect

### Training Output
```
================================================================================
Vector Transformation Training
================================================================================
Run: baseline_run
Data: decorte
Encoder: sentence-transformers/all-mpnet-base-v2
Hidden sizes: [512, 512]
Dropout: True (rate: 0.1)
Max epochs: 50, Patience: 5
Learning rate: 0.001, Batch size: 256
================================================================================

[1/4] Loading encoder...
  ✓ Encoder loaded

[2/4] Loading data...
  ✓ Train: 12543, Val: 1568, Test: 1568

[3/4] Creating data loaders...
Processing train: encoding 12543 pairs...
  Encoding career histories...
  Encoding targets...
  ✓ train loader created: 12543 samples

...

================================================================================
TRAINING
================================================================================

Epoch   1/50 | Train Loss: 0.4523 | Val Loss: 0.4812 | Val MRR: 0.3245 | Time: 45.3s
  → New best model! Val Loss: 0.4812, MRR: 0.3245
Epoch   2/50 | Train Loss: 0.3987 | Val Loss: 0.4234 | Val MRR: 0.3687 | Time: 44.8s
  → New best model! Val Loss: 0.4234, MRR: 0.3687
Epoch   3/50 | Train Loss: 0.3654 | Val Loss: 0.4098 | Val MRR: 0.3923 | Time: 45.1s
  → New best model! Val Loss: 0.4098, MRR: 0.3923
...

Early stopping after 15 epochs (patience: 5)

Training completed in 11.23 minutes
Loaded best model from epoch 10 (Val Loss: 0.3845)

================================================================================
FINAL TEST RESULTS
================================================================================
Test Loss: 0.3923
Test MRR: 0.4156
Test R@1: 0.2987
Test R@5: 0.5234
Test R@10: 0.6345
Test R@20: 0.7456
================================================================================

Model saved to: results/vector_transformation/model.pth
Results saved to: results/vector_transformation/results.csv

✓ Done!
```

### Output Files
- `results/vector_transformation/model.pth` - Trained model
- `results/vector_transformation/results.csv` - Metrics log
- WandB dashboard with training curves

---

## 🔧 Key Differences from Original `vector_transformation.py`

| Feature | Original | New Script |
|---------|----------|------------|
| Architecture | ✅ Same | ✅ Same |
| Input normalization | ✅ Yes | ✅ Yes |
| Validation during training | ✅ Yes | ✅ Yes |
| Config file support | ❌ No | ✅ Yes |
| WandB integration | ❌ No | ✅ Yes |
| Command-line args | ❌ Limited | ✅ Full |
| Code structure | ❌ Complex | ✅ Simple |
| Lines of code | ~353 | ~450 |

**Bottom line**: Same functionality, cleaner code, better UX.

---

## ⚙️ Configuration Options

### All Available Arguments

```bash
python src/cpp/train_vector_transformation.py --help

# Key arguments:
--config              # Path to YAML config file
--data_type           # Dataset: decorte, karrierewege, etc.
--use_text_description  # Include job descriptions (default: titles only)
--encoder             # SentenceTransformer model name
--hidden_sizes        # List of hidden layer sizes (e.g., 512 512)
--dropout             # Enable dropout
--dropout_rate        # Dropout rate (default: 0.1)
--max_epochs          # Maximum training epochs (default: 50)
--patience            # Early stopping patience (default: 5)
--batch_size          # Batch size (default: 256)
--lr                  # Learning rate (default: 0.001)
--device              # cuda or cpu
--output_dir          # Output directory
--run_name            # Experiment name
--save_model          # Save trained model
--use_wandb           # Enable WandB logging
--wandb_project       # WandB project name
--wandb_entity        # WandB entity/username
```

---

## 🆘 Troubleshooting

### Import Error: No module named 'src.cpp'
```bash
# Make sure you're in the project root
cd /dss/dsshome1/02/ra95kix2/thesis/skills4cpp
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### CUDA Out of Memory (during training)
```bash
# Reduce batch size
python src/cpp/train_vector_transformation.py \
    --config configs/vector_transformation_baseline.yaml \
    --batch_size 128
```

### CUDA Out of Memory (during metric computation on large datasets)
```bash
# The script auto-handles this for most datasets
# But if you still get OOM with very large datasets (>200k samples):

# Option 1: Use smaller chunk size
python src/cpp/train_vector_transformation.py \
    --config configs/vector_transformation_baseline.yaml \
    --ranking_chunk_size 500

# Option 2: See docs/OOM_FIX_LARGE_DATASETS.md for details
```

### Slow Data Loading
```bash
# Already optimized with batched encoding
# But you can try CPU if GPU is busy
python src/cpp/train_vector_transformation.py \
    --config configs/vector_transformation_baseline.yaml \
    --device cpu
```

---

## 📚 Next Steps

1. **Run the baseline**: Get a reference result
2. **Experiment with encoders**: Try different pre-trained models
3. **Tune hyperparameters**: Adjust hidden sizes, dropout, learning rate
4. **Compare approaches**: Try multi-modal MLP with `train_cpp_simple.py`
5. **Analyze results**: Use WandB to compare runs

---

## 💡 Pro Tips

1. **Always use WandB** for experiment tracking
2. **Use meaningful run names** (e.g., `encoder_name_hidden512_lr0.001`)
3. **Save your best models** with `--save_model`
4. **Start with default config** and override specific params
5. **Check the docs** at `docs/WHICH_SCRIPT_TO_USE.md` for detailed comparison

---

## 📞 Need Help?

Check these files:
- `docs/WHICH_SCRIPT_TO_USE.md` - Detailed comparison
- `docs/train_cpp_simple_README.md` - Multi-modal documentation
- `examples/train_cpp_simple_example.sh` - More examples

Happy experimenting! 🎉

