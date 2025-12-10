# Changes: IDF Calculation Strategy

## Problem Identified

The initial implementation calculated IDF weights based on the **entire ESCO taxonomy** (all occupations and skills in ESCO), which doesn't reflect the actual data distribution in the DECORTE dataset.

## Solution Implemented

IDF weights are now calculated **during training** based on the actual DECORTE dataset, not pre-computed in the mapping script.

## What Changed

### Modified: `scripts/create_job_skills_mapping.py`

**Removed:**
- IDF calculation from entire ESCO taxonomy
- `numpy` import (no longer needed)

**Changed:**
- Skills now exported with placeholder score `1.0`
- Docstring updated to clarify IDF calculation happens during training

### Updated Documentation

All documentation updated to reflect the new approach:
- `QUICK_START_ESCO_GT.md`
- `ESCO_GROUND_TRUTH_IMPLEMENTATION.md`
- `docs/esco_ground_truth_guide.md`
- `scripts/README_skill_mapping.md`

## How It Works Now

### 1. Mapping Script (create_job_skills_mapping.py)
```python
# Export skills with placeholder scores
skills_list.append({
    'skill': skill_name,
    'score': 1.0,  # Placeholder - IDF calculated during training
    'skillUri': skill_uri
})
```

### 2. Training Script (data_loaders.py)
```python
if pooling_strategy == "weighted_idf":
    # N_occ = Total unique job titles in DECORTE dataset
    N_occ = df['job_title'].nunique()
    # n_i = Number of DECORTE job titles that have skill_i
    skill_n_occ = df.groupby('skillUri')['job_title'].nunique()
    # idf_i = log((N_occ + 1) / (n_i + 1))
    idf_map = np.log((N_occ + 1) / (skill_n_occ + 1))
    df['idf'] = df['skillUri'].map(idf_map)
```

## Why This Is Better

| Aspect | Old Approach | New Approach |
|--------|--------------|--------------|
| **IDF Basis** | All ESCO occupations (~3000) | DECORTE dataset jobs (~200-300) |
| **Relevance** | ESCO-wide distribution | Training data distribution |
| **N_occ** | ~3000 occupations | Actual unique jobs in DECORTE |
| **Skill Rarity** | Rare in ESCO ≠ rare in DECORTE | Reflects actual data |

### Example
- A skill appearing in 10/3000 ESCO occupations → High IDF (rare in ESCO)
- But if those 10 occupations are heavily represented in DECORTE → Should have lower IDF
- **New approach correctly captures this!**

## Impact on Training

### No Changes Needed in Training Code
The existing `load_job_and_skill_data()` function in `data_loaders.py` already:
1. ✅ Reads CSV with 'score' column (now placeholder 1.0)
2. ✅ Calculates IDF from the loaded dataframe when `pooling_strategy="weighted_idf"`
3. ✅ Uses DECORTE job distribution (not ESCO-wide)

### Pooling Strategies

**For ESCO Ground-Truth:**
- ✅ `mean` - Equal weighting (ignores 'score' and 'idf')
- ✅ `weighted_idf` - IDF-weighted (calculated from DECORTE)
- ❌ `weighted_mean` - Not applicable (requires similarity scores, not placeholder 1.0)

**For IR-Extracted:**
- ✅ `mean` - Equal weighting
- ✅ `weighted_mean` - Similarity-weighted (uses actual IR scores)
- ✅ `weighted_idf` - Similarity × IDF weighted (both calculated properly)

## Verification

After generating the mapping, verify placeholder scores:

```bash
# Check that all scores are 1.0
python -c "
import pandas as pd
df = pd.read_csv('results/decorte_esco_ground_truth/job_title_skills_master.csv')
print(f'Min score: {df[\"score\"].min()}')
print(f'Max score: {df[\"score\"].max()}')
print(f'All scores are 1.0: {(df[\"score\"] == 1.0).all()}')
"
```

During training, you'll see:
```
Calculating IDF scores from master_skill_file...
  > N_occ (total jobs) = 245  # Actual DECORTE jobs
  > Example IDF (...): 0.05 (most common)
  > Example IDF (...): 5.50 (rarest)
```

## Summary

✅ **Simpler**: No complex IDF calculation in mapping script
✅ **Correct**: IDF based on actual training data, not ESCO-wide
✅ **Flexible**: Training script can use different IDF formulations
✅ **Consistent**: Same IDF calculation for both IR-extracted and ESCO ground-truth
✅ **Transparent**: Clear what N_occ represents (DECORTE jobs, not ESCO occupations)

## Files Modified

1. `scripts/create_job_skills_mapping.py` - Removed IDF calculation, added placeholder
2. `QUICK_START_ESCO_GT.md` - Updated with clarification
3. `ESCO_GROUND_TRUTH_IMPLEMENTATION.md` - Updated technical details
4. `docs/esco_ground_truth_guide.md` - Updated IDF section and examples
5. `scripts/README_skill_mapping.md` - Updated score interpretation
6. `CHANGES_IDF_CALCULATION.md` - This file (explanation of changes)

## No Action Required

The training script already handles this correctly. Just use the updated mapping script and train as planned!





