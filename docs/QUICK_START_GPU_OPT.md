# Quick Start: GPU Optimization

## TL;DR - Just Use This

For **maximum GPU utilization** with cached embeddings:

```bash
sbatch experiments/configs/run_cpp_baseline_kw_100k_v3.sh
```

This script now includes:
- ✅ `--num_workers 0` (no multi-process overhead)
- ✅ `--pin_embeddings_to_gpu` (embeddings stay on GPU)
- ✅ `--batch_size 2048` (larger batches)
- ✅ `--mixed_precision` (FP16 for A100)

**Expected Result**: ~70-80% GPU utilization (vs 1-3% before)

---

## What Changed?

### Before (SLOW)
```bash
--num_workers 6              # Multi-process serialization overhead
--batch_size 1024            # Smaller batches
# No GPU pinning              # CPU→GPU transfers every batch
```

**Result**: 
- Data loading: 73s (93%)
- GPU compute: 5s (7%)
- GPU idle most of the time

### After (FAST)
```bash
--num_workers 0              # Single process (embeddings in memory)
--batch_size 2048            # Larger batches
--pin_embeddings_to_gpu      # Embeddings live on GPU
```

**Result**:
- Data loading: 2s (15%)
- GPU compute: 11s (85%)
- GPU working hard!

---

## When to Use Each Solution

| Scenario | Configuration | GPU Util | Speed |
|----------|--------------|----------|-------|
| **A100 40GB (Recommended)** | `--pin_embeddings_to_gpu --num_workers 0` | **70-80%** | **Fastest** |
| **Smaller GPU (<16GB)** | `--num_workers 0` (no pinning) | 40-50% | Fast |
| **CPU Training** | `--num_workers 2` (shared memory) | N/A | Baseline |

---

## Verify It's Working

### 1. Check Logs
Look for this message:
```
📌 Pinning embeddings to GPU (cuda:0)...
  ✓ Text embeddings on GPU: torch.Size([1089621, 768]) (3.12 GB)
  ✓ Skill embeddings on GPU: torch.Size([1089621, 768]) (3.12 GB)
```

### 2. Monitor GPU
```bash
watch -n 1 nvidia-smi
```
You should see **GPU Util: 70-90%** during training.

### 3. Check Timing
Logs will show:
```
⏱️  Epoch timing: Data loading: 2.15s (15%), GPU compute: 11.91s (85%)
```
Good: **GPU compute > 70%**

---

## Troubleshooting

### "CUDA out of memory"
**Solution**: Reduce batch size
```bash
--batch_size 1024 \
--gradient_accumulation_steps 4  # Keep effective batch size = 4096
```

Or disable GPU pinning:
```bash
# Remove --pin_embeddings_to_gpu flag
--num_workers 0  # Still much better than workers>0
```

### Still seeing low GPU utilization?
1. Check `nvidia-smi` - is GPU memory full?
2. Verify `--num_workers 0` is set
3. Look for error messages about embedding transfer
4. Try without `--pin_embeddings_to_gpu` first (shared memory fallback)

---

## Advanced: Compare Different Configurations

### Test Shared Memory (Fallback)
```bash
python -m src.cpp.train_cpp_enhanced_v2 \
    --num_workers 0 \
    --batch_size 2048 \
    # No --pin_embeddings_to_gpu
```

### Test GPU Pinning (Maximum Performance)
```bash
python -m src.cpp.train_cpp_enhanced_v2 \
    --num_workers 0 \
    --batch_size 2048 \
    --pin_embeddings_to_gpu
```

### Profile Data Loading
Add this flag to see detailed timing:
```bash
--profile_data_loading
```

---

## Questions?

See [GPU_OPTIMIZATION_GUIDE.md](GPU_OPTIMIZATION_GUIDE.md) for:
- Technical implementation details
- Performance benchmarks
- Code architecture
- Troubleshooting guide






