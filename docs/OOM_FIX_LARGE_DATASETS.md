# OOM Fix for Large Datasets

## Problem

When computing ranking metrics (MRR, R@k) on large datasets like **karrierewege_100k** (~138k validation samples), the original implementation created a huge similarity matrix that exceeded GPU memory:

```python
similarity_matrix = torch.mm(all_outputs, all_targets.t())  # [138k x 138k]
# Size: 138,275 × 138,275 × 4 bytes = ~76.5 GB
# Error: CUDA out of memory (GPU has only 39 GB)
```

## Solution: Batched Similarity Computation

The fix processes similarities in **chunks** instead of computing the full matrix at once:

### Key Changes

1. **Move embeddings to CPU immediately** after computing them
2. **Process similarities in batches** (default: 1000 samples at a time)
3. **Adaptive chunk sizing** based on dataset size

### Implementation

```python
# Old (OOM for large datasets):
similarity_matrix = torch.mm(all_outputs, all_targets.t())  # [n_samples x n_samples] on GPU
sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)

# New (memory-efficient):
for i in range(0, n_samples, chunk_size):
    chunk_outputs = all_outputs[i:i + chunk_size]  # Process 1000 at a time
    similarity_chunk = torch.mm(chunk_outputs, all_targets.t())  # [1000 x n_samples] on CPU
    sorted_indices_chunk = torch.argsort(similarity_chunk, dim=1, descending=True)
```

## Performance Impact

### Memory Usage
- **Before**: O(n²) - Full similarity matrix
- **After**: O(n × chunk_size) - Only one chunk at a time
- **Reduction**: For n=138k, from ~76 GB to ~0.5 GB per chunk

### Speed
- **CPU computation** is slower than GPU but acceptable
- **Typical overhead**: +30-60 seconds for 138k samples
- **Trade-off**: Slightly slower but won't OOM

### Chunk Size Settings

The script **auto-detects** optimal chunk size:

| Dataset Size | Default Chunk Size | Approx. Memory per Chunk |
|--------------|-------------------|--------------------------|
| < 5,000 | All at once | Minimal |
| 5,000 - 10,000 | 2,000 | ~60 MB |
| > 10,000 | 1,000 | ~30 MB |

## Usage

### Default (Auto-detect)
```bash
python src/cpp/train_vector_transformation.py \
    --config configs/vector_transformation_baseline.yaml \
    --data_type karrierewege_100k
```

The script automatically uses:
- Chunk size 1000 for large datasets (>10k samples)
- Chunk size 2000 for medium datasets (5k-10k)
- Full matrix for small datasets (<5k)

### Manual Chunk Size

If you experience OOM or want to optimize:

```bash
# Smaller chunks (more memory-efficient, slower)
python src/cpp/train_vector_transformation.py \
    --config configs/vector_transformation_baseline.yaml \
    --data_type karrierewege_100k \
    --ranking_chunk_size 500

# Larger chunks (faster, more memory)
python src/cpp/train_vector_transformation.py \
    --config configs/vector_transformation_baseline.yaml \
    --data_type karrierewege_100k \
    --ranking_chunk_size 2000
```

## Output Example

```
[2/4] Loading data...
  ✓ Train: 1089621, Val: 138275, Test: 137530

...

Epoch   1/10 | Train Loss: 0.4523 | Val Loss: 0.4812 | Val MRR: 0.3245 | Time: 45.3s
  Computing ranking metrics for 138275 samples (batched processing)...
  Computing similarities: 100%|██████████| 139/139 [00:45<00:00,  3.07it/s]
  Calculating final metrics...
  → New best model! Val Loss: 0.4812, MRR: 0.3245
```

## Technical Details

### Memory Breakdown

For **karrierewege_100k** validation set (138,275 samples):

**Old approach (OOM):**
```
all_outputs: [138275, 768] × 4 bytes = 425 MB ✓
all_targets: [138275, 768] × 4 bytes = 425 MB ✓
similarity_matrix: [138275, 138275] × 4 bytes = 76,524 MB ✗ OOM!
```

**New approach (Success):**
```
all_outputs: [138275, 768] × 4 bytes = 425 MB on CPU ✓
all_targets: [138275, 768] × 4 bytes = 425 MB on CPU ✓

Per chunk:
  chunk_outputs: [1000, 768] × 4 bytes = 3 MB ✓
  similarity_chunk: [1000, 138275] × 4 bytes = 553 MB ✓
  Total per iteration: ~560 MB ✓
```

### Why CPU Instead of GPU?

1. **Embedding computation** still happens on GPU (fast)
2. **Similarity computation** happens on CPU (slower but memory-safe)
3. **Trade-off**: Acceptable speed loss to avoid OOM

You could keep it on GPU with very small chunks, but CPU is simpler and works universally.

## Dataset Size Recommendations

| Dataset | Val/Test Size | Works OOT? | Notes |
|---------|--------------|-----------|-------|
| decorte | ~1,500 | ✅ Yes | Original implementation works |
| karrierewege | ~5,000 | ✅ Yes | Auto-chunks to 2000 |
| karrierewege_100k | ~138,000 | ✅ Yes (now) | Auto-chunks to 1000 |
| Custom > 200k | Very large | ⚠️ Maybe | Try chunk_size=500 |

## Troubleshooting

### Still Getting OOM?

1. **Reduce chunk size**:
   ```bash
   --ranking_chunk_size 500
   ```

2. **Skip validation ranking metrics** (only compute for test):
   - Modify code to set `compute_ranking_metrics=False` during training
   - Keep it `True` only for final test evaluation

3. **Subsample validation set**:
   ```bash
   # Evaluate on subset during training
   --val_sample_ratio 0.1  # Use 10% of val set
   ```

### Too Slow?

1. **Increase chunk size** (if you have more memory):
   ```bash
   --ranking_chunk_size 2000
   ```

2. **Reduce validation frequency**:
   - Only compute ranking metrics every N epochs
   - Requires code modification

## Verification

After the fix, you should see:

✅ No OOM errors during validation  
✅ Ranking metrics computed successfully  
✅ Progress bar showing chunk processing  
✅ Slightly slower per epoch (~30-60s extra for large datasets)  

## Related Files

- `src/cpp/train_vector_transformation.py` - Main training script (updated)
- `src/cpp/evaluation.py` - Metric computation functions (unchanged)
- `configs/vector_transformation_baseline.yaml` - Config file

## Summary

**Problem**: OOM when computing ranking metrics on large datasets  
**Solution**: Batched similarity computation on CPU  
**Impact**: +30-60s per epoch, no OOM  
**Usage**: Works automatically, no config changes needed  

The fix is **transparent** - existing commands work without modification! 🎉





