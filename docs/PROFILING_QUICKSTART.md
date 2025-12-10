# Profiling Quick Start Guide

## 🚀 Quick Commands

### 1. Find bottlenecks (recommended first step)
```bash
python src/cpp/test_bottlenecks.py --quick
```
**Time:** ~5 minutes  
**Output:** Identifies which parts are slow (data loading, DataLoader, model, etc.)

### 2. Run your training with timing
```bash
python src/cpp/train_with_profiling.py --quick-profile \
    --use_text_history --use_skill_text --use_structured \
    --n_trials 5 --max_epochs 5
```
**Output:** Shows timing for each trial and epoch

### 3. Use convenience script
```bash
# Make it executable first
chmod +x scripts/profile_pipeline.sh

# Then use it
./scripts/profile_pipeline.sh quick      # Quick test
./scripts/profile_pipeline.sh epoch      # Test epoch speed
./scripts/profile_pipeline.sh help       # See all options
```

## 📊 Understanding Results

After running `test_bottlenecks.py`, you'll see output like:

```
TEST 4: FULL EPOCH BOTTLENECK
================================
Avg data loading:   45.32ms (30.5% of batch time)
Avg training step:  103.21ms (69.5% of batch time)
Avg batch total:    148.53ms
Throughput:         215.4 samples/s

Time Estimation:
  Time per epoch: 8.45 minutes
  Total time (50 trials × 10 epochs): 70.42 hours
```

### What to look for:

1. **Data loading > 30%?** → Data loading bottleneck
   - Solution: Increase `num_workers`, use caching, optimize embeddings

2. **Training step > 70%?** → Compute-bound (good!)
   - Solution: Use mixed precision, increase batch size

3. **GPU utilization < 80%?** → Not using GPU efficiently
   - Solution: Increase batch size, check DataLoader settings

## 🔧 Quick Fixes

### If data loading is slow:
```bash
# Test different num_workers
python src/cpp/test_bottlenecks.py --component dataloader

# Use caching (add to your code)
# See docs/PERFORMANCE_OPTIMIZATION.md for details
```

### If training is slow:
```bash
# Try larger batch size
python src/cpp/train_cpp_enhanced.py --batch_size 64  # instead of 32

# Use smaller model for testing
python src/cpp/train_cpp_enhanced.py --hidden_dim 256 --n_layers 1
```

### If Optuna is too slow:
```bash
# Reduce trials and epochs
python src/cpp/train_cpp_enhanced.py --n_trials 20 --max_epochs 5 --patience 2

# Or skip Optuna for ablations
python src/cpp/train_cpp_enhanced.py \
    --use_text_history --use_skill_text \
    --lr 2e-5 --hidden_dim 512  # Use static hyperparameters
```

## 📈 Optimization Workflow

1. **Profile** → Identify bottleneck
   ```bash
   python src/cpp/test_bottlenecks.py --quick
   ```

2. **Fix** → Apply solution (see PERFORMANCE_OPTIMIZATION.md)

3. **Verify** → Test again to confirm improvement
   ```bash
   python src/cpp/test_bottlenecks.py --component epoch --quick
   ```

4. **Iterate** → Repeat until satisfied

## 💡 Pro Tips

### Monitor GPU while training
```bash
# Terminal 1: Run training
python src/cpp/train_cpp_enhanced.py ...

# Terminal 2: Watch GPU
watch -n 0.5 nvidia-smi
```

### Estimate time before full run
```bash
# Test with small dataset first
python src/cpp/test_bottlenecks.py --quick
# Look at "Time Estimation" section
```

### Compare configurations
```bash
# Test baseline
python src/cpp/test_bottlenecks.py --component epoch --batch_size 32 > baseline.log

# Test optimized
python src/cpp/test_bottlenecks.py --component epoch --batch_size 64 > optimized.log

# Compare
diff baseline.log optimized.log
```

## 📚 Full Documentation

- **Detailed guide:** `docs/PERFORMANCE_OPTIMIZATION.md`
- **Profiling tools:** `src/cpp/profile_training.py`
- **Test script:** `src/cpp/test_bottlenecks.py`
- **Wrapper script:** `src/cpp/train_with_profiling.py`

## 🐛 Troubleshooting

### "CUDA out of memory"
- Reduce `--batch_size`
- Reduce `--hidden_dim`
- Use gradient accumulation (see PERFORMANCE_OPTIMIZATION.md)

### "DataLoader workers slow"
- Try `--num_workers 0` (no multiprocessing)
- Or try `--num_workers 4` (sweet spot for most cases)

### "Pre-computing embeddings takes forever"
- Add caching (see PERFORMANCE_OPTIMIZATION.md)
- Use faster encoder batch size
- Use SSD instead of HDD

## 🎯 Recommended First Steps

1. Run quick profile:
   ```bash
   python src/cpp/test_bottlenecks.py --quick
   ```

2. Look at the bottleneck (data loading? training? Optuna?)

3. Apply relevant solution from `docs/PERFORMANCE_OPTIMIZATION.md`

4. Test again to verify improvement

5. Run full training with optimized settings!

## Questions?

Check `docs/PERFORMANCE_OPTIMIZATION.md` for detailed solutions to specific bottlenecks.









