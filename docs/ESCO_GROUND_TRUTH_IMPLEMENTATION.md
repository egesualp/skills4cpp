# ESCO Ground-Truth Skills Implementation Summary

## Problem Statement

Previously, the career path prediction model used skills extracted from raw job titles via information retrieval (IR). This approach has potential noise and depends on the IR model's predictions. 

**Goal**: Enable training with ESCO's official occupation-skill relations as ground truth.

## Solution Design

Instead of modifying the training pipeline, we extended the `create_job_skills_mapping.py` script to generate ESCO ground-truth mappings in the **same format** as IR-extracted skills. This minimizes changes to the training code.

## Implementation

### 1. Modified Script: `scripts/create_job_skills_mapping.py`

**New Function**: `create_esco_ground_truth_mapping(base_dir)`

**What it does:**
1. Loads `decorte_esco` dataset (with ESCO job titles)
2. Extracts all unique ESCO titles from train/val/test splits
3. Maps ESCO titles → occupation URIs (from `occupations_en.csv`)
4. Retrieves related skills from ESCO taxonomy (`occupationSkillRelations_en.csv`)
5. Exports with placeholder scores (IDF calculated during training)
6. Exports in same format as IR-extracted skills (JSON + CSV)

**Key Features:**
- Title normalization matching `utils.py` (replacements for consistency)
- IDF-only weighting (no similarity scores since ground-truth)
- Same output format → no training code changes needed
- Progress tracking with detailed statistics

### 2. Output Files

**Location**: `results/decorte_esco_ground_truth/`

**Files:**
- `job_title_skills_master.json` - Nested format (for reference)
- `job_title_skills_master.csv` - Flattened format (used by training)

**CSV Schema:**
```
original_row_index, job_title, skill, score, skillUri
```
- `score` = Placeholder (1.0) - IDF calculated during training on DECORTE data

### 3. Usage

**Step 1: Generate ESCO Ground-Truth Mapping**
```bash
cd /dss/dsshome1/02/ra95kix2/thesis/skills4cpp

python scripts/create_job_skills_mapping.py \
    --mode esco_ground_truth \
    --base_dir .
```

**Step 2: Train with ESCO Ground-Truth**
```bash
python src/cpp/train_cpp_enhanced.py \
    --data_type decorte_esco \
    --master_skill_file results/decorte_esco_ground_truth/job_title_skills_master.csv \
    --use_text_history \
    --use_skill_text \
    --pooling_strategy weighted_idf \
    --batch_size 32 \
    --max_epochs 10
```

## Key Differences: IR-Extracted vs. ESCO Ground-Truth

| Aspect | IR-Extracted | ESCO Ground-Truth |
|--------|--------------|-------------------|
| **Data Source** | JobBERT predictions | ESCO taxonomy |
| **Dataset** | `decorte` (raw titles) | `decorte_esco` (ESCO titles) |
| **Skill Weights** | Similarity + IDF | IDF only |
| **Pooling Options** | `mean`, `weighted_mean`, `weighted_idf` | `mean`, `weighted_idf` |
| **Noise Level** | Depends on IR model | Curated ground-truth |
| **Reproducibility** | Depends on IR model version | Fixed (ESCO version) |

## Technical Details

### IDF Calculation
**Important**: IDF is NOT pre-calculated in the mapping script. Instead, it's calculated during training based on the actual DECORTE dataset:

```python
# In data_loaders.py (during training):
N_occ = total unique job titles in DECORTE dataset
n_i = number of DECORTE job titles that have skill_i
idf_i = log((N_occ + 1) / (n_i + 1))
```

This ensures IDF reflects the actual training data distribution, not the entire ESCO taxonomy.

### Job Title Matching
- Decorte dataset provides ESCO URIs and titles
- Title replacements applied (e.g., "ICT security engineer" → "cyber incident responder")
- Exact matching ensures consistent skill retrieval

### Handling Missing Mappings
- Some job titles may not have ESCO skills (reported in output)
- Training will handle empty skill sets gracefully (returns zero vector)

## Integration with Existing Pipeline

### No Changes Needed in:
- ✅ `train_cpp_enhanced.py` - Works with any CSV in correct format
- ✅ `data_loaders.py` - `load_job_and_skill_data()` already handles IDF calculation
- ✅ `cpp_dataset.py` - Dataset class works with any skill mapping
- ✅ Training configs - Just change `--data_type` and `--master_skill_file`

### Only Changes Made:
- ✅ `scripts/create_job_skills_mapping.py` - Added new mode
- ✅ `scripts/README_skill_mapping.md` - Documentation
- ✅ `docs/esco_ground_truth_guide.md` - Usage guide

## Validation Steps

### 1. Verify Mapping Generation
```bash
# Generate mapping
python scripts/create_job_skills_mapping.py --mode esco_ground_truth --base_dir .

# Check output
head results/decorte_esco_ground_truth/job_title_skills_master.csv
```

### 2. Verify Skill Statistics
```bash
python -c "
import pandas as pd
df = pd.read_csv('results/decorte_esco_ground_truth/job_title_skills_master.csv')
print(f'Total job-skill pairs: {len(df)}')
print(f'Unique jobs: {df[\"job_title\"].nunique()}')
print(f'Unique skills: {df[\"skillUri\"].nunique()}')
print(f'Avg skills per job: {len(df) / df[\"job_title\"].nunique():.1f}')
print(f'IDF score range: [{df[\"score\"].min():.3f}, {df[\"score\"].max():.3f}]')
"
```

### 3. Test Training (Quick Run)
```bash
python src/cpp/train_cpp_enhanced.py \
    --data_type decorte_esco \
    --master_skill_file results/decorte_esco_ground_truth/job_title_skills_master.csv \
    --use_skill_text \
    --pooling_strategy weighted_idf \
    --batch_size 8 \
    --max_epochs 1 \
    --output_dir test_esco_gt
```

## Expected Behavior

### During Mapping Generation:
- Progress bars for dataset processing
- Statistics on unique ESCO titles found
- Report of titles without mappings (if any)
- Confirmation of output files created

### During Training:
- Should load correctly with no errors
- Skill embeddings should be computed via IDF-weighted pooling
- Model should train normally

## Advantages of This Approach

1. **✅ Minimal Code Changes**: Only modified one script
2. **✅ Backward Compatible**: IR-extracted mode still works
3. **✅ Reuses Pipeline**: All training logic unchanged
4. **✅ Easy Comparison**: Same experiments, different skill source
5. **✅ Reproducible**: Ground-truth fixed by ESCO version
6. **✅ Authoritative**: Official ESCO relations

## Future Enhancements

- [ ] Filter by `relationType` (essential vs. optional skills)
- [ ] Add skill type filtering (knowledge vs. skill/competence)
- [ ] Support alternative IDF formulations
- [ ] Cache ESCO data for faster repeated runs
- [ ] Add skill overlap analysis (IR vs. ground-truth)

## Files Modified/Created

### Modified:
- `scripts/create_job_skills_mapping.py` (+100 lines)

### Created:
- `docs/esco_ground_truth_guide.md`
- `scripts/README_skill_mapping.md`
- `ESCO_GROUND_TRUTH_IMPLEMENTATION.md` (this file)

### Will be Created (when run):
- `results/decorte_esco_ground_truth/job_title_skills_master.json`
- `results/decorte_esco_ground_truth/job_title_skills_master.csv`

## Testing Checklist

- [ ] Run mapping generation script
- [ ] Verify output files exist and have correct format
- [ ] Check statistics (job count, skill count, IDF range)
- [ ] Run quick training test (1 epoch)
- [ ] Compare results with IR-extracted baseline
- [ ] Full training run with optimal hyperparameters

## Contact & Support

For issues or questions:
1. Check the documentation in `docs/esco_ground_truth_guide.md`
2. Verify ESCO files are present in `data/esco_datasets/`
3. Check that decorte_esco dataset loads correctly
4. Review mapping generation output for errors

## References

- ESCO Taxonomy: https://esco.ec.europa.eu/
- Decorte Dataset: `jensjorisdecorte/anonymous-working-histories`
- Training Script: `src/cpp/train_cpp_enhanced.py`
- Data Loaders: `src/cpp/data_loaders.py`

