# Performance Optimization Guide

This guide helps you identify and fix bottlenecks in the training pipeline.

## Quick Start: Profiling Your Training

### 1. Run Quick Bottleneck Test
```bash
python src/cpp/test_bottlenecks.py --quick
```

This will test all components with a subset of data (~5 minutes).

### 2. Run Full Profiling
```bash
python src/cpp/test_bottlenecks.py --full
```

### 3. Test Specific Components
```bash
# Test only data loading
python src/cpp/test_bottlenecks.py --component data_loading

# Test only DataLoader configuration
python src/cpp/test_bottlenecks.py --component dataloader

# Test only model training speed
python src/cpp/test_bottlenecks.py --component model

# Test full epoch breakdown
python src/cpp/test_bottlenecks.py --component epoch
```

## Common Bottlenecks and Solutions

### 1. Data Loading Bottleneck

**Symptoms:**
- Pre-computing embeddings takes a long time
- High CPU usage, low GPU usage during training
- Data loading time > 30% of batch time

**Solutions:**

#### A. Optimize Encoder Batch Size
The embedding computation can be batched more efficiently:

```python
# In data_loaders.py, modify batch size for encoder.encode()
embeddings = encoder.encode(
    texts, 
    batch_size=128,  # Try 64, 128, 256
    show_progress_bar=True,
    convert_to_numpy=False
)
```

#### B. Cache Pre-computed Embeddings
Add caching to avoid recomputing embeddings:

```python
import pickle
import hashlib

def get_cache_key(texts, encoder_name):
    """Generate cache key from texts and encoder."""
    text_hash = hashlib.md5("".join(texts).encode()).hexdigest()
    return f"cache/{encoder_name}_{text_hash}.pkl"

def precompute_with_cache(encoder, texts, cache_dir="cache/embeddings"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_key = get_cache_key(texts, encoder_name)
    cache_path = os.path.join(cache_dir, cache_key)
    
    if os.path.exists(cache_path):
        logger.info(f"Loading from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    # Compute embeddings
    embeddings = encoder.encode(texts, batch_size=128)
    
    # Save to cache
    with open(cache_path, 'wb') as f:
        pickle.dump(embeddings, f)
    
    return embeddings
```

#### C. Use Faster Storage
- Move data to SSD if on HDD
- Use `/tmp` or fast local storage on cluster nodes
- Pre-load data to RAM disk for very fast access

### 2. DataLoader Bottleneck

**Symptoms:**
- GPU utilization < 80%
- `num_workers=0` is slow
- Data loading time dominates training

**Solutions:**

#### A. Tune num_workers
```bash
# Test different values
python src/cpp/test_bottlenecks.py --component dataloader
```

**Rule of thumb:**
- Start with `num_workers = 4 * num_gpus`
- On SLURM: Use `--num_workers=$SLURM_CPUS_PER_TASK`
- Too many workers can slow down due to overhead

#### B. Use pin_memory
Already enabled in your code, but make sure:
```python
DataLoader(..., pin_memory=True)  # For CUDA
```

#### C. Increase Batch Size
Larger batches mean fewer DataLoader calls:
```bash
# Try doubling batch size
python src/cpp/train_cpp_enhanced.py --batch_size 64  # from 32
```

**Warning:** Monitor GPU memory with `nvidia-smi`

### 3. Model Training Bottleneck

**Symptoms:**
- GPU utilization is high (90%+)
- Training step time dominates
- Model is large

**Solutions:**

#### A. Use Mixed Precision Training
Add to your training script:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

def train_epoch_amp(model, dataloader, optimizer, criterion, device):
    """Train with automatic mixed precision."""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with autocast():
            y_pred = model(batch)
            target = torch.ones(y_pred.size(0)).to(device)
            loss = criterion(y_pred, batch['y'], target)
        
        # Scaled backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)
```

**Expected speedup:** 1.5-2x faster, 30-40% less memory

#### B. Gradient Accumulation
If memory-limited, accumulate gradients over multiple batches:

```python
accumulation_steps = 4  # Effective batch_size = 32 * 4 = 128

for i, batch in enumerate(dataloader):
    y_pred = model(batch)
    loss = criterion(y_pred, batch['y'], target) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

#### C. Model Architecture Optimization
```python
# Use smaller hidden dimensions
--hidden_dim 256  # instead of 512

# Use fewer layers
--n_layers 1  # instead of 2-3

# Disable modality weights if not needed
# (removes extra parameters)
```

### 4. Optuna Optimization Bottleneck

**Symptoms:**
- 50 trials take too long
- Most trials are pruned early
- Hyperparameter search is slow

**Solutions:**

#### A. Reduce Number of Trials
```bash
# Quick optimization
--n_trials 20  # instead of 50

# Use better pruner
# Already using MedianPruner in your code
```

#### B. Reduce max_epochs per Trial
```bash
# Reduce from 10 to 5
--max_epochs 5 --patience 2
```

#### C. Parallel Optuna Trials
Run multiple trials in parallel (requires setup):

```python
# Create a shared study with database
import optuna

# On first run, create study
study = optuna.create_study(
    study_name="cpp_study",
    storage="sqlite:///optuna.db",  # Shared database
    direction="maximize",
    load_if_exists=True
)

# Submit multiple SLURM jobs, each runs:
study.optimize(objective, n_trials=10)  # Each job does 10 trials
```

**SLURM array job:**
```bash
#SBATCH --array=1-5  # 5 parallel workers

python train_cpp_enhanced.py --n_trials 10  # Total: 5*10=50 trials
```

#### D. Skip Optuna for Ablations
When doing ablation studies with similar configs:

```bash
# Use static hyperparameters (no Optuna)
python src/cpp/train_cpp_enhanced.py \
    --use_text_history \
    --use_skill_text \
    --use_structured \
    --lr 2e-5 \
    --hidden_dim 512 \
    --n_layers 1 \
    --dropout 0.1 \
    --max_epochs 10 \
    --patience 2
```

This skips Optuna and trains directly with specified hyperparameters.

### 5. I/O Bottleneck

**Symptoms:**
- Loading vocabs/mappings is slow
- File I/O dominates startup time

**Solutions:**

#### A. Cache Vocabs and Mappings
```python
def load_all_vocabs_cached(vocab_dir, cache_file="cache/vocabs.pkl"):
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    vocabs = load_all_vocabs(vocab_dir)
    
    with open(cache_file, 'wb') as f:
        pickle.dump(vocabs, f)
    
    return vocabs
```

#### B. Use Faster File Formats
- Convert CSV → Parquet (much faster for pandas)
- Use HDF5 for large arrays
- Use memory-mapped files for huge datasets

## Advanced Profiling

### PyTorch Profiler
For detailed GPU/CPU analysis:

```python
from src.cpp.profile_training import pytorch_profiler

with pytorch_profiler(output_dir="profiler_output"):
    # Run training
    for epoch in range(3):
        train_epoch(model, loader, optimizer, criterion, device)
        prof.step()  # Step profiler

# View results
# tensorboard --logdir=profiler_output
```

### Line Profiler
For line-by-line Python profiling:

```bash
pip install line_profiler

# Add @profile decorator to functions you want to profile
kernprof -l -v src/cpp/train_cpp_enhanced.py
```

### GPU Monitoring
Monitor GPU usage in real-time:

```bash
# Terminal 1: Run training
python src/cpp/train_cpp_enhanced.py ...

# Terminal 2: Monitor GPU
watch -n 0.5 nvidia-smi

# Or use nvtop (better UI)
nvtop
```

## Performance Checklist

Before starting large experiments:

- [ ] Run `test_bottlenecks.py --quick` to identify bottlenecks
- [ ] Optimize num_workers for DataLoader
- [ ] Consider caching pre-computed embeddings
- [ ] Test larger batch sizes (if GPU memory allows)
- [ ] Use mixed precision training for 1.5-2x speedup
- [ ] Monitor GPU utilization (should be >80%)
- [ ] For Optuna: reduce trials or max_epochs for faster iteration
- [ ] For ablations: use static hyperparameters (skip Optuna)

## Expected Performance

Based on your setup, approximate times:

**Data Loading (first time):**
- Load encoders: ~10-30s
- Pre-compute embeddings: ~2-5 min (depends on dataset size)

**Training:**
- Batch time: ~50-200ms (depends on model size, batch size)
- Epoch time: ~2-10 min (depends on dataset size)
- Optuna (50 trials × 10 epochs): ~4-20 hours

**Optimized:**
With mixed precision + caching + optimal num_workers:
- 1.5-2x faster training
- Instant data loading (cached)
- ~2-10 hours for full Optuna optimization

## Debugging Slow Training

If training is unexpectedly slow:

1. **Check GPU utilization:**
   ```bash
   nvidia-smi dmon -s u
   ```
   Should show >80% GPU utilization during training.

2. **Profile one epoch:**
   ```bash
   python src/cpp/test_bottlenecks.py --component epoch --quick
   ```

3. **Check data loading:**
   If "Data loading overhead" > 30%, it's a bottleneck.

4. **Check batch size:**
   Larger batches = better GPU utilization (until memory limit).

5. **Check for CPU bottlenecks:**
   ```bash
   htop  # Monitor CPU usage
   ```
   If CPU is maxed out, increase num_workers or use caching.

## Questions?

Check profiling results and look for:
- Which component takes the most time?
- Is GPU utilization high or low?
- Is data loading slow or fast?

Then apply solutions from the relevant section above!














