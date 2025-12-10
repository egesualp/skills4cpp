# Skill Encoding Optimization Plan

## Problem

Currently, the system encodes skills for every sample (13,484 training samples), which means:
- The same skill gets encoded multiple times if it appears in multiple jobs
- O(n_samples × avg_skills_per_job) encoding operations
- Very slow: 8.99 it/s with 23+ minutes remaining for just one split

## Current Flow

```python
# In precompute_input_embeddings():
for idx, (h, _) in enumerate(tqdm(filtered_pairs)):
    infos = _extract_skill_infos(h, job_skill_map)  # Extract skills for this job
    skill_vecs.append(_pooled_skill_vec(
        infos, encoder_skill, esco_skill_text_map, ...  # Encode skills HERE
    ))
```

Inside `_pooled_skill_vec()`:
```python
for info in infos:
    uri = info['skillUri']
    text = format_skill_text(uri)  # Get text
    strings.append(text)

embs = encoder_skill.encode(strings)  # ENCODE EVERY TIME
```

## Proposed Optimization

### Step 1: Extract Unique Skills from Dataset

```python
def extract_unique_skills_from_dataset(data_pairs, job_skill_map):
    """Extract all unique skills used in the dataset."""
    unique_skills = set()
    for history_doc, _ in data_pairs:
        infos = _extract_skill_infos(history_doc, job_skill_map)
        for info in infos:
            unique_skills.add(info['skillUri'])
    return unique_skills
```

### Step 2: Pre-encode All Unique Skills Once

```python
def precompute_skill_embeddings(unique_skill_uris, encoder_skill, esco_skill_text_map, use_skill_description):
    """Pre-compute embeddings for all unique skills in the dataset."""
    skill_texts = []
    skill_uris_ordered = []
    
    for uri in unique_skill_uris:
        if uri in esco_skill_text_map:
            st = esco_skill_text_map[uri]
            text = f"role: {st['name']} \n description: {st['desc']}" if use_skill_description else st['name']
            skill_texts.append(text)
            skill_uris_ordered.append(uri)
    
    # Encode all skills at once (batch encoding is efficient)
    skill_embeddings = encoder_skill.encode(skill_texts, convert_to_numpy=True, show_progress_bar=True)
    
    # Create lookup dictionary
    skill_embedding_map = dict(zip(skill_uris_ordered, skill_embeddings))
    return skill_embedding_map
```

### Step 3: Use Pre-computed Embeddings

```python
def _pooled_skill_vec_optimized(infos, skill_embedding_map, pooling_strategy, alpha, beta, embed_dim):
    """Compute pooled skill vector using pre-computed embeddings."""
    embeddings = []
    weights = []
    
    for info in infos:
        uri = info['skillUri']
        if uri in skill_embedding_map:
            embeddings.append(skill_embedding_map[uri])
            
            if pooling_strategy == "mean":
                weights.append(1.0)
            elif pooling_strategy == "weighted_mean":
                weights.append(info['score'])
            else:  # weighted_idf
                c = info['score']
                idf = info.get('idf', 0)
                weights.append((c ** alpha) * (idf ** beta))
    
    if not embeddings:
        return np.zeros(embed_dim, dtype=np.float32)
    
    embs = np.array(embeddings)
    w = np.array(weights, dtype=np.float32)
    
    vec = embs.mean(axis=0) if pooling_strategy == "mean" or w.sum() == 0 else np.average(embs, axis=0, weights=w)
    return vec.astype(np.float32)
```

### Step 4: Modified precompute_input_embeddings

```python
def precompute_input_embeddings_optimized(...):
    # Filter pairs
    filtered_pairs = [(h, t) for (h, t) in data_pairs if t in Y_target_dict]
    
    # *** NEW: Extract unique skills first ***
    if use_skill_text:
        unique_skills = extract_unique_skills_from_dataset(filtered_pairs, job_skill_map)
        print(f"  > Found {len(unique_skills)} unique skills in dataset")
        print(f"  > (vs {sum(len(skills) for skills in job_skill_map.values())} total in taxonomy)")
        
        # Pre-encode all unique skills once
        skill_embedding_map = precompute_skill_embeddings(
            unique_skills, encoder_skill, esco_skill_text_map, use_skill_description
        )
        
        # Now process samples using pre-computed embeddings
        skill_vecs = []
        for h, _ in tqdm(filtered_pairs, desc="  > Aggregating skill vectors"):
            infos = _extract_skill_infos(h, job_skill_map)
            skill_vecs.append(_pooled_skill_vec_optimized(
                infos, skill_embedding_map, pooling_strategy, alpha, beta, embed_dim
            ))
        h_skill = np.stack(skill_vecs, axis=0)
```

## Expected Improvements

### Before (Current):
- Encodes: n_samples × avg_skills_per_job skill texts
- Example: 13,484 samples × 30 skills/job = ~404,520 encoding operations
- Speed: ~9 samples/second = ~25 minutes

### After (Optimized):
- Encodes: unique_skills (typically 500-2000)
- Example: 1,500 unique skills (one-time batch encoding)
- Then: 13,484 lookups + weighted averaging
- Expected speed: ~1000+ samples/second = ~13 seconds

**Speedup: ~100-200x faster!**

## Implementation Priority

1. ✅ **HIGH**: Implement this optimization in `data_loaders.py` - **COMPLETED**
2. ✅ **MEDIUM**: Add logging to show unique vs total skills - **COMPLETED**
3. ⏸️ **LOW**: Add caching option to save/load pre-computed skill embeddings - **FUTURE WORK**

## Additional Benefit

This also addresses your concern: **We're only encoding skills that actually appear in the dataset, not the entire ESCO taxonomy!**

## Implementation Status

### ✅ COMPLETED (2025-11-23)

The optimization has been successfully implemented in `data_loaders.py`:

1. **New Functions Added:**
   - `extract_unique_skills_from_dataset()` - Extracts all unique skill URIs from dataset
   - `precompute_skill_embeddings()` - Batch encodes all unique skills once
   - `_pooled_skill_vec_optimized()` - Uses pre-computed embeddings for pooling

2. **Modified Functions:**
   - `precompute_input_embeddings()` - Now uses the optimized 3-step approach:
     - Step 1: Extract unique skills
     - Step 2: Pre-encode all unique skills (batch)
     - Step 3: Aggregate skills per sample using lookups

3. **Logging Improvements:**
   - Shows number of unique skills vs total skill instances
   - Displays efficiency gain (e.g., "~100x speedup")
   - Progress bars for each step

4. **Backward Compatibility:**
   - Old `_pooled_skill_vec()` function kept but marked as deprecated
   - All existing code continues to work

### Expected Performance Impact

For a typical dataset with:
- 13,484 samples
- ~30 skills per job on average
- ~1,500 unique skills

**Before:** 
- Encoding operations: 13,484 × 30 = ~404,520
- Time: ~25 minutes

**After:**
- Encoding operations: 1,500 (one-time batch)
- Time: ~13 seconds for encoding + ~10 seconds for aggregation
- **Speedup: ~100-200x faster!**


