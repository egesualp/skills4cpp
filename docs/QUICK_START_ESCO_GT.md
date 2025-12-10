# Quick Start: ESCO Ground-Truth Skills

## TL;DR

Run career path prediction with ESCO's official occupation-skill relations instead of IR-extracted skills.

## Step 1: Generate Mapping (One-time)

```bash
cd /dss/dsshome1/02/ra95kix2/thesis/skills4cpp

python scripts/create_job_skills_mapping.py \
    --mode esco_ground_truth \
    --base_dir .
```

**Output**: `results/decorte_esco_ground_truth/job_title_skills_master.csv`

**Note**: Skills are exported with placeholder scores (1.0). IDF weights will be calculated during training based on the actual DECORTE dataset distribution.

## Step 2: Train Model

```bash
python src/cpp/train_cpp_enhanced.py \
    --data_type decorte_esco \
    --master_skill_file results/decorte_esco_ground_truth/job_title_skills_master.csv \
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
    --run_name esco_gt_experiment
```

## Key Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| `--data_type` | `decorte_esco` | Uses ESCO titles |
| `--master_skill_file` | Path to ESCO GT CSV | Ground-truth skills |
| `--pooling_strategy` | `weighted_idf` or `mean` | IDF-weighted or equal |

⚠️ **Note**: Don't use `weighted_mean` - it requires similarity scores (not available in ground-truth)

## Verify It Works

```bash
# Check mapping was created
wc -l results/decorte_esco_ground_truth/job_title_skills_master.csv

# Check statistics
python -c "
import pandas as pd
df = pd.read_csv('results/decorte_esco_ground_truth/job_title_skills_master.csv')
print(f'Jobs: {df[\"job_title\"].nunique()}')
print(f'Skills: {df[\"skillUri\"].nunique()}')
print(f'Avg skills/job: {len(df) / df[\"job_title\"].nunique():.1f}')
"
```

## Compare with IR-Extracted

**IR-Extracted (baseline)**:
```bash
python src/cpp/train_cpp_enhanced.py \
    --data_type decorte \
    --master_skill_file results/decorte_jobbert_v2_baseline/job_title_skills_master.csv \
    --pooling_strategy weighted_mean \
    ...
```

**ESCO Ground-Truth (new)**:
```bash
python src/cpp/train_cpp_enhanced.py \
    --data_type decorte_esco \
    --master_skill_file results/decorte_esco_ground_truth/job_title_skills_master.csv \
    --pooling_strategy weighted_idf \
    ...
```

## Documentation

- **Full Implementation**: `ESCO_GROUND_TRUTH_IMPLEMENTATION.md`
- **Detailed Guide**: `docs/esco_ground_truth_guide.md`
- **Script README**: `scripts/README_skill_mapping.md`

## Troubleshooting

**Error: "File not found: occupationSkillRelations_en.csv"**
- Download ESCO datasets from https://esco.ec.europa.eu/en/use-esco/download
- Place in `data/esco_datasets/`

**Few job titles mapped**
- Check title normalization in script matches your data
- Some titles may not have exact ESCO matches

**Different performance than IR-extracted**
- Expected! Ground-truth skills are different from IR predictions
- This is the comparison you want to make

