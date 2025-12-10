# Job-Skills Mapping Generation

This directory contains scripts for generating job-to-skills mappings used in career path prediction.

## Overview

The `create_job_skills_mapping.py` script supports two modes:

### 1. IR-Extracted Skills (`ir_extracted`)
- Uses information retrieval predictions from JobBERT model
- Skills are extracted from raw job titles
- Includes similarity scores from IR matching + IDF weights

### 2. ESCO Ground-Truth Skills (`esco_ground_truth`) 
- Uses official ESCO occupation-skill relations
- Skills come directly from ESCO taxonomy
- Exports placeholder scores (1.0) - IDF calculated during training on DECORTE data

## Usage

### Generate IR-Extracted Mapping

```bash
python scripts/create_job_skills_mapping.py \
    --mode ir_extracted \
    --base_dir .
```

**Output:**
- `results/decorte_jobbert_v2_baseline/job_title_skills_master.json`
- `results/decorte_jobbert_v2_baseline/job_title_skills_master.csv`
- `results/decorte_jobbert_v2_baseline/job_title_skills_train.json`
- `results/decorte_jobbert_v2_baseline/job_title_skills_val.json`
- `results/decorte_jobbert_v2_baseline/job_title_skills_test.json`

### Generate ESCO Ground-Truth Mapping

```bash
python scripts/create_job_skills_mapping.py \
    --mode esco_ground_truth \
    --base_dir .
```

**Output:**
- `results/decorte_esco_ground_truth/job_title_skills_master.json`
- `results/decorte_esco_ground_truth/job_title_skills_master.csv`

## Output Format

### JSON Format
```json
[
  {
    "job_title": "software developer",
    "skills": [
      {
        "skill": "Python",
        "score": 0.845,
        "skillUri": "http://data.europa.eu/esco/skill/..."
      }
    ]
  }
]
```

**Note**: For ESCO ground-truth, `score` will be 1.0 (placeholder). For IR-extracted, `score` is the similarity from JobBERT.

### CSV Format (Flattened)
```csv
original_row_index,job_title,skill,score,skillUri
0,software developer,Python,0.845,http://data.europa.eu/esco/skill/...
0,software developer,Java,0.762,http://data.europa.eu/esco/skill/...
1,data scientist,Python,0.845,http://data.europa.eu/esco/skill/...
```

## Score Interpretation

- **IR-Extracted**: Similarity score from JobBERT matching (0-1 range)
- **ESCO Ground-Truth**: Placeholder value (1.0) in CSV
  - IDF weights are calculated during training based on DECORTE dataset
  - Formula: `idf_i = log((N_occ + 1) / (n_i + 1))` where N_occ = job titles in DECORTE
  - Calculated by `data_loaders.py` when `pooling_strategy="weighted_idf"`

## Requirements

- `pandas`
- `numpy`
- `datasets` (HuggingFace)
- `tqdm`

## Data Sources

### IR-Extracted Mode
- Predictions: `results/decorte_jobbert_v2_baseline/eval_baseline_JobBERT_decorte_{split}_predictions.json`
- Job data: `data/title_pairs_desc/decorte_{split}_pairs.csv`
- ESCO skills: `data/esco_datasets/skills_en.csv`

### ESCO Ground-Truth Mode
- Dataset: `jensjorisdecorte/anonymous-working-histories` (HuggingFace)
- ESCO occupations: `data/esco_datasets/occupations_en.csv`
- ESCO skills: `data/esco_datasets/skills_en.csv`
- Occupation-skill relations: `data/esco_datasets/occupationSkillRelations_en.csv`

## Use in Training

Point to the generated CSV file:

```bash
python src/cpp/train_cpp_enhanced.py \
    --data_type decorte \
    --master_skill_file results/decorte_jobbert_v2_baseline/job_title_skills_master.csv \
    ...
```

Or for ESCO ground-truth:

```bash
python src/cpp/train_cpp_enhanced.py \
    --data_type decorte_esco \
    --master_skill_file results/decorte_esco_ground_truth/job_title_skills_master.csv \
    ...
```

## Troubleshooting

**Missing prediction files (IR-extracted mode)**
- Run JobBERT evaluation first to generate predictions

**Missing ESCO files**
- Download ESCO datasets from https://esco.ec.europa.eu/en/use-esco/download
- Place in `data/esco_datasets/`

**Job titles not found (ESCO mode)**
- Check that ESCO title normalization matches `src/cpp/utils.py`
- Some raw titles may not have exact ESCO matches

