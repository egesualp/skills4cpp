# Changelog - OOM Fix for Large Datasets

## Version: 2025-12-23

### 🐛 Bug Fix: CUDA Out of Memory on Large Datasets

**Issue**: Training failed with OOM error when computing ranking metrics on large datasets (e.g., karrierewege_100k with ~138k validation samples).

**Root Cause**: Creating full similarity matrix `[n_samples × n_samples]` exceeded GPU memory:
- karrierewege_100k: 138k × 138k × 4 bytes = ~76.5 GB
- Available GPU memory: 39 GB

**Solution**: Implemented batched similarity computation

### Changes Made

#### 1. Updated `evaluate()` function in `train_vector_transformation.py`

**Key improvements:**
- ✅ Move embeddings to CPU immediately after computation
- ✅ Process similarities in chunks (default: 1000 samples)
- ✅ Adaptive chunk sizing based on dataset size
- ✅ New optional argument: `--ranking_chunk_size`

**Memory reduction:**
- Before: O(n²) - Full matrix
- After: O(n × chunk_size) - One chunk at a time
- For n=138k: ~76 GB → ~0.5 GB per chunk

#### 2. Added command-line argument

```bash
--ranking_chunk_size INT    # Manual chunk size control (optional)
```

Default behavior (auto-detect):
- Small datasets (<5k): Compute all at once
- Medium datasets (5k-10k): Chunk size 2000
- Large datasets (>10k): Chunk size 1000

#### 3. Performance impact

**Pros:**
- ✅ No more OOM errors on large datasets
- ✅ Works for any dataset size
- ✅ Automatic optimization (no config changes needed)

**Cons:**
- ⚠️ Slightly slower (~30-60s extra per epoch for 138k samples)
- ⚠️ Uses CPU for similarity computation (GPU still used for model inference)

### Usage

**No changes needed!** Existing commands work automatically:

```bash
# Works now without OOM
python src/cpp/train_vector_transformation.py \
    --config configs/vector_transformation_baseline.yaml \
    --data_type karrierewege_100k \
    --use_wandb
```

**Optional: Manual control**

```bash
# Smaller chunks (more memory-efficient)
python src/cpp/train_vector_transformation.py \
    --config configs/vector_transformation_baseline.yaml \
    --data_type karrierewege_100k \
    --ranking_chunk_size 500

# Larger chunks (faster if you have memory)
python src/cpp/train_vector_transformation.py \
    --config configs/vector_transformation_baseline.yaml \
    --data_type karrierewege_100k \
    --ranking_chunk_size 2000
```

### Testing

**Verified on:**
- ✅ decorte (~1.5k samples) - Works, no performance impact
- ✅ karrierewege (~5k samples) - Works, minimal impact
- ✅ karrierewege_100k (~138k samples) - **Now works!** Previously OOM

### Output Example

```
Epoch   1/10 | Train Loss: 0.4523 | Val Loss: 0.4812 | Val MRR: 0.3245 | Time: 45.3s
  Computing ranking metrics for 138275 samples (batched processing)...
  Computing similarities: 100%|██████████| 139/139 [00:45<00:00,  3.07it/s]
  Calculating final metrics...
  → New best model! Val Loss: 0.4812, MRR: 0.3245
```

### Documentation

New files:
- `docs/OOM_FIX_LARGE_DATASETS.md` - Detailed explanation
- `CHANGELOG_OOM_FIX.md` - This file

Updated files:
- `src/cpp/train_vector_transformation.py` - Core fix
- `QUICKSTART.md` - Mentions large dataset support

### Migration Guide

**If you were avoiding large datasets due to OOM:**
- ✅ You can now train on them directly
- ✅ No configuration changes required
- ✅ Script auto-detects optimal chunk size

**If you had custom workarounds:**
- ✅ You can remove them
- ✅ Use `--ranking_chunk_size` if you want manual control

### Technical Details

**Before:**
```python
# Single matrix multiplication (OOM for large n)
similarity_matrix = torch.mm(all_outputs, all_targets.t())  # [n × n] on GPU
sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
```

**After:**
```python
# Batched computation (memory-safe)
for i in range(0, n_samples, chunk_size):
    chunk_outputs = all_outputs[i:i + chunk_size]
    similarity_chunk = torch.mm(chunk_outputs, all_targets.t())  # [chunk × n] on CPU
    sorted_indices_chunk = torch.argsort(similarity_chunk, dim=1, descending=True)
    # Process this chunk...
```

### Backward Compatibility

✅ **Fully backward compatible**
- Old commands work without modification
- No breaking changes to API
- Results are numerically identical

### Future Improvements

Potential optimizations (not implemented yet):
- [ ] GPU batched computation (more complex but faster)
- [ ] Approximate nearest neighbors (FAISS) for very large datasets (>500k)
- [ ] Async similarity computation during training
- [ ] Option to skip validation ranking metrics (only compute for test)

### Credits

- **Issue reported**: karrierewege_100k training OOM
- **Fixed**: Batched similarity computation with CPU fallback
- **Date**: 2025-12-23

---

## Summary

**What changed**: Ranking metric computation now uses batched processing  
**Why**: Avoid OOM on large datasets  
**Impact**: Works for all dataset sizes, slightly slower for large datasets  
**Action needed**: None - works automatically! 🎉



