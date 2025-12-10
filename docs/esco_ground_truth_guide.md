# ESCO Ground-Truth Skills Integration Guide

This guide explains how to use ESCO ground-truth occupation-skill relations instead of IR-extracted skills for career path prediction.

## Overview

**Problem**: Previously, skills were extracted from raw job titles using information retrieval, which may introduce noise.

**Solution**: Use ESCO's ground-truth occupation-skill relations directly, weighted only by IDF scores (no similarity scores needed).

## Data Flow

```
decorte_esco dataset (ESCO titles)
         ↓
ESCO occupations → occupation URIs
         ↓
ESCO occupation-skill relations → skill URIs + IDF weights
         ↓
job_title_skills_master.json (same format as IR-extracted)
         ↓
train_cpp_enhanced.py (no changes needed!)
```

## Step 1: Generate ESCO Ground-Truth Skill Mapping

Run the mapping script with the new mode:

```bash
cd /dss/dsshome1/02/ra95kix2/thesis/skills4cpp

python scripts/create_job_skills_mapping.py \
    --mode esco_ground_truth \
    --base_dir .
```

**Output Files:**
- `results/decorte_esco_ground_truth/job_title_skills_master.json` - Main mapping file
- `results/decorte_esco_ground_truth/job_title_skills_master.csv` - CSV version for inspection

**Output Format** (same as IR-extracted):
```json
[
  {
    "job_title": "software developer",
    "skills": [
      {
        "skill": "Python",
        "score": 1.0,
        "skillUri": "http://data.europa.eu/esco/skill/..."
      },
      ...
    ]
  }
]
```

**Note**: `score` is a placeholder (1.0). IDF weights are calculated during training based on DECORTE dataset.

## Step 2: Train with ESCO Ground-Truth Skills

Use the generated mapping file in your training:

```bash
python src/cpp/train_cpp_enhanced.py \
    --data_type decorte_esco \
    --master_skill_file results/decorte_esco_ground_truth/job_title_skills_master.json \
    --use_text_history \
    --use_skill_text \
    --pooling_strategy weighted_idf \
    --batch_size 32 \
    --max_epochs 10 \
    --patience 2
```

**Key Parameters:**
- `--data_type decorte_esco`: Uses ESCO job titles from decorte dataset
- `--master_skill_file`: Points to ESCO ground-truth mapping
- `--pooling_strategy weighted_idf`: Use IDF-weighted skill aggregation
  - Alternative: `mean` for equal weighting
  - **Note**: `weighted_mean` not applicable (no similarity scores)

## Differences from IR-Extracted Skills

| Aspect | IR-Extracted | ESCO Ground-Truth |
|--------|--------------|-------------------|
| Skill Source | Information retrieval from raw titles | ESCO taxonomy relations |
| Similarity Scores | ✓ (from IR matching) | ✗ (placeholder 1.0 in CSV) |
| IDF Weights | ✓ (calculated in training) | ✓ (calculated in training) |
| IDF Calculation | Based on DECORTE jobs | Based on DECORTE jobs |
| Weighting Options | `mean`, `weighted_mean`, `weighted_idf` | `mean`, `weighted_idf` |
| alpha/beta params | Used in `weighted_mean` | Not applicable |

## Key Implementation Details

### IDF Calculation
**Important**: IDF weights are NOT pre-calculated in the mapping script. They are calculated during training based on the actual DECORTE dataset:

```python
# Calculated in data_loaders.py during training
N_occ = total unique job titles in DECORTE training data
n_i = number of job titles in DECORTE that have skill_i  
idf_i = log((N_occ + 1) / (n_i + 1))
```

This ensures IDF reflects the actual data distribution in DECORTE, not the entire ESCO taxonomy. Skills are exported with placeholder scores (1.0) in the CSV.

### Job Title Matching
- ESCO titles are normalized and replacements applied (same as in `utils.py`)
- Titles from decorte_esco dataset are matched to ESCO occupation URIs
- Skills are retrieved via `occupationSkillRelations_en.csv`

### Handling Missing Mappings
- Some job titles may not have ESCO mappings (reported in output)
- Dataset instances with unmapped titles will have empty skill sets

## Verification Steps

1. **Check the generated mapping:**
```bash
head -20 results/decorte_esco_ground_truth/job_title_skills_master.csv
```

2. **Verify skill counts:**
```bash
python -c "
import json
with open('results/decorte_esco_ground_truth/job_title_skills_master.json') as f:
    data = json.load(f)
print(f'Jobs mapped: {len(data)}')
print(f'Avg skills per job: {sum(len(d[\"skills\"]) for d in data) / len(data):.1f}')
print(f'All scores are placeholder: {all(s[\"score\"] == 1.0 for d in data for s in d[\"skills\"])}')
"
```

3. **Compare with IR-extracted:**
```bash
# Compare coverage
python -c "
import json
with open('results/decorte_jobbert_v2_baseline/job_title_skills_master.json') as f:
    ir_data = json.load(f)
with open('results/decorte_esco_ground_truth/job_title_skills_master.json') as f:
    esco_data = json.load(f)
print(f'IR-extracted: {len(ir_data)} jobs')
print(f'ESCO ground-truth: {len(esco_data)} jobs')
"
```

## Example Usage in Experiments

Create a new experiment config (e.g., `experiments/configs/run_cpp_esco_ground_truth.sh`):

```bash
#!/bin/bash
#SBATCH --job-name=cpp_esco_gt
#SBATCH --output=logs/cpp_esco_gt_%j.out
#SBATCH --error=logs/cpp_esco_gt_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

python src/cpp/train_cpp_enhanced.py \
    --data_type decorte_esco \
    --master_skill_file results/decorte_esco_ground_truth/job_title_skills_master.json \
    --encoder_text ElenaSenger/career-path-representation-mpnet-decorte \
    --use_text_history \
    --use_skill_text \
    --use_structured \
    --pooling_strategy weighted_idf \
    --batch_size 32 \
    --n_trials 50 \
    --max_epochs 10 \
    --patience 2 \
    --output_dir results/cpp_esco_ground_truth \
    --run_name cpp_esco_gt_all_modalities
```

## Troubleshooting

**Issue**: Job titles not found in ESCO
- Check the output for "titles without mapping"
- Verify title normalization/replacements match `utils.py`
- Some raw titles in decorte may not have exact ESCO matches

**Issue**: Different number of skills vs IR-extracted
- Expected! ESCO ground-truth is curated, IR-extracted is top-K predictions
- ESCO typically has fewer, more relevant skills per occupation

**Issue**: Training performance different
- Expected! Different skill sets will affect model performance
- This is the point of the comparison - ground-truth vs. extracted skills

## Benefits of ESCO Ground-Truth

1. ✓ **Cleaner data**: No IR extraction noise
2. ✓ **Authoritative**: Official ESCO taxonomy relations
3. ✓ **Reproducible**: Fixed skill sets (no dependency on IR model)
4. ✓ **Comparable**: Standard benchmark for future work
5. ✓ **Essential/Optional**: Can filter by relation type if needed

## Next Steps

- Run experiments comparing IR-extracted vs. ESCO ground-truth
- Analyze performance differences
- Consider filtering by `relationType` (essential vs. optional skills)
- Experiment with different pooling strategies

