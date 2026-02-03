# Pooling Method Analysis for Career Path Prediction (CPP)

This analysis script examines how different skill pooling methods affect Career Path Prediction performance. It investigates the relationship between skill characteristics (frequency, quality, specificity) and prediction accuracy.

## Table of Contents
- [Overview](#overview)
- [Quick Start](#quick-start)
- [Input Data](#input-data)
- [Key Concepts](#key-concepts)
- [Output Files](#output-files)
- [Statistical Report Explanation](#statistical-report-explanation)
- [Visualization Guide](#visualization-guide)

---

## Overview

### What This Script Does

1. **Loads CPP predictions** from three model variants:
   - **TEXT**: Uses only job title/description text embeddings
   - **SKILL**: Uses only skill text embeddings  
   - **HYBRID**: Combines text and skill embeddings

2. **Computes skill-level statistics** for each career path:
   - Skill frequency (how many ESCO occupations use each skill)
   - IDF scores (inverse document frequency - rarity measure)
   - Skill prediction confidence scores
   - Genericness scores (normalized inverse of IDF)

3. **Simulates pooling strategy weights**:
   - Mean pooling (equal weights)
   - Weighted mean (weight = skill score)
   - Weighted IDF (weight = score^α × idf^β)

4. **Generates visualizations and statistical reports** to understand what makes predictions succeed or fail.

---

## Quick Start

```bash
# Activate the thesis conda environment
source /dss/dsshome1/02/ra95kix2/miniconda3/etc/profile.d/conda.sh
conda activate thesis

# Run the analysis
bash src/error_analysis/run_pooling_analysis.sh

# Or run directly with custom paths:
python src/error_analysis/pooling_method_analysis.py \
    --text_scores <path_to_text_scores.pkl> \
    --skill_scores <path_to_skill_scores.pkl> \
    --hybrid_scores <path_to_hybrid_scores.pkl> \
    --master_skill_file <path_to_fused_predictions.json> \
    --output_dir <output_directory>
```

---

## Input Data

### Required Files

| File | Description | Format |
|------|-------------|--------|
| `test_scores_text.pkl` | CPP prediction scores for text-only model | Pickle with `scores`, `target_labels`, `job_ids`, `true_target_indices` |
| `test_scores_skill.pkl` | CPP prediction scores for skill-only model | Same format |
| `test_scores_hybrid.pkl` | CPP prediction scores for hybrid model | Same format |
| `fused_predictions.json` | Skill predictions per job | `{job_id: [[skill_uri, score], ...]}` |
| `occupationSkillRelations_en.csv` | ESCO taxonomy | Columns: `occupationUri`, `skillUri` |

### Prediction Score File Structure

```python
{
    'scores': np.ndarray,           # Shape: [n_samples, n_targets] - similarity scores
    'target_labels': List[str],     # All possible target occupation labels
    'true_target_indices': List[int], # Index of correct target for each sample
    'job_ids': List[List[str]],     # Job IDs in each career path
    'split': str                    # 'train', 'val', or 'test'
}
```

---

## Key Concepts

### Skill Frequency and IDF

The script computes IDF (Inverse Document Frequency) from **two independent sources**:

#### 1. ESCO Taxonomy IDF (Static)
- **Source**: Official ESCO occupation-skill relations database
- **Frequency** = Number of ESCO occupations that use a skill
- **IDF** = log((N_occupations + 1) / (n_skill + 1))

**Interpretation**:
- High IDF → Rare skill in ESCO taxonomy
- Low IDF → Common skill across ESCO occupations

#### 2. Decorte Dataset IDF (Dynamic)
- **Source**: Actual job postings from the Decorte dataset (fused_predictions.json)
- **Frequency** = Number of unique job_ids that have this skill predicted
- **IDF** = log((N_jobs + 1) / (n_skill + 1))

**Interpretation**:
- Reflects real-world skill usage in job postings
- May differ from ESCO taxonomy due to industry bias in the dataset

#### Why Compare Both Sources?

| Aspect | ESCO Taxonomy | Decorte Dataset |
|--------|---------------|-----------------|
| **Coverage** | All ESCO skills (~13K) | Only predicted skills |
| **Scope** | All occupations | Sample of job postings |
| **Updates** | Static (official) | Dynamic (from model) |
| **Bias** | Occupation-centric | Industry sample bias |

**Genericness** = Normalized inverse of IDF, scaled to [0, 1]
- High genericness (→1) = Common/generic skill
- Low genericness (→0) = Rare/specific skill

### Pooling Strategies

When a career path has multiple jobs, and each job has multiple skills, we need to **pool** (aggregate) skill embeddings into a single vector. Three strategies are compared:

| Strategy | Weight Formula | Effect |
|----------|----------------|--------|
| **Mean** | w_i = 1/N | Equal contribution from all skills |
| **Weighted Mean** | w_i = score_i / Σ(scores) | High-confidence skills contribute more |
| **Weighted IDF** | w_i = score_i^α × idf_i^β / Σ(...) | Rare, high-confidence skills contribute most |

### Effective Number of Skills

Measures **weight concentration** using entropy:
```
Entropy = -Σ(w_i × log(w_i))
Effective Skills = exp(Entropy)
```

- **High effective skills** → Weights are dispersed (many skills contribute)
- **Low effective skills** → Weights are concentrated (few skills dominate)

---

## Output Files

### Data Files

| File | Description |
|------|-------------|
| `analysis_dataframe.csv` | Per-sample metrics (confidence, correctness, ranks, skill stats) |
| `path_statistics.csv` | Detailed per-career-path skill statistics (ESCO + Decorte) |
| `statistics_report.txt` | Human-readable statistical summary |

### Visualizations

| File | Description |
|------|-------------|
| `pooling_analysis.png/pdf` | Main 1×2 plot: Genericness vs confidence, correctness box plots |
| `pooling_strategy_analysis.png` | 2×2 plot: Pooling strategy comparisons |
| `frequency_analysis.png` | 2×2 plot: Frequency-based analysis |
| `skill_confidence_analysis.png` | Score distribution and skill count analysis |
| **`idf_source_comparison.png`** | **2×2 plot: ESCO vs Decorte IDF comparison** |

---

## Statistical Report Explanation

### Section: OVERALL METRICS

```
TEXT    : Acc@1 = 18.59%, MRR = 0.2657, Median Rank = 24
SKILL   : Acc@1 = 14.93%, MRR = 0.2222, Median Rank = 38
HYBRID  : Acc@1 = 17.98%, MRR = 0.2603, Median Rank = 20
```

| Metric | Description |
|--------|-------------|
| **Acc@1** | % of samples where the top-1 prediction is correct |
| **MRR** | Mean Reciprocal Rank = average of 1/rank |
| **Median Rank** | 50th percentile of true target ranks |

### Section: SKILL GENERICNESS STATISTICS

```
Mean genericness: 0.4796 (std: 0.0508)
Mean skill score: 0.5986 (std: 0.0669)
Mean num skills: 295.77 (std: 208.34)
```

| Metric | Description |
|--------|-------------|
| **Mean genericness** | Average skill commonality across all paths (0=rare, 1=common) |
| **Mean skill score** | Average skill prediction confidence |
| **Mean num skills** | Average total skill occurrences per career path |

### Section: DETAILED FREQUENCY STATISTICS

```
min_freq    : mean=1.31, std=0.58, median=1.00
median_freq : mean=23.77, std=9.51, median=21.50
max_freq    : mean=335.31, std=33.01, median=347.00
avg_freq    : mean=45.96, std=11.29, median=45.20
```

For each career path, we compute the **min/median/max/avg frequency** of its skills. Then we report statistics across all paths.

**Interpretation**: Most paths contain at least one rare skill (min_freq ≈ 1) and at least one very common skill (max_freq ≈ 335).

### Section: POOLING STRATEGY ANALYSIS

```
MEAN           : Effective skills: mean=295.77
WEIGHTED_MEAN  : Effective skills: mean=283.33
WEIGHTED_IDF   : Effective skills: mean=277.46
```

| Strategy | Interpretation |
|----------|----------------|
| **Mean** | All 295.77 skills contribute equally |
| **Weighted Mean** | ~283 skills effectively contribute (some have higher weights) |
| **Weighted IDF** | ~277 skills effectively contribute (most concentrated) |

### Section: CORRELATION ANALYSIS

```
TEXT confidence vs genericness: r = -0.1601 (p = 8.0922e-12) ***
```

**Interpretation**: Negative correlation means career paths with **more generic skills** have **lower prediction confidence**. The *** indicates p < 0.001.

### Section: GENERICNESS BY PREDICTION CORRECTNESS

```
TEXT    : Correct mean = 0.4644, Incorrect mean = 0.4831
          t = -6.131, p = 1.0732e-09 ***
```

**Interpretation**: Correctly predicted paths have **lower genericness** (more specific skills) than incorrectly predicted paths. This difference is statistically significant.

### Section: MEDIAN FREQUENCY BY PREDICTION CORRECTNESS

```
TEXT    : Correct mean = 21.19, Incorrect mean = 24.36
          t = -5.554, p = 3.2132e-08 ***
```

**Interpretation**: Correctly predicted paths have skills with **lower median frequency** (rarer skills). Rare skills are more discriminative.

### Section: SKILL SCORE QUALITY IMPACT (Quartiles)

The data is split into 4 quartiles based on mean skill prediction score:

| Quartile | Description | Example: HYBRID |
|----------|-------------|-----------------|
| **Q1 (Low)** | Lowest 25% skill scores | Acc@1 = 12.64% |
| **Q2** | 25-50th percentile | Acc@1 = 15.33% |
| **Q3** | 50-75th percentile | Acc@1 = 19.33% |
| **Q4 (High)** | Top 25% skill scores | Acc@1 = 24.61% |

**Interpretation**: Higher skill prediction quality leads to dramatically better CPP performance.

### Section: POOLING WEIGHT CONCENTRATION IMPACT

The data is split by **effective number of skills** (weight concentration):

| Quartile | Description | Interpretation |
|----------|-------------|----------------|
| **Q1 (Concentrated)** | Fewest effective skills | Few skills dominate embedding |
| **Q2-Q3** | Moderate concentration | Balanced contribution |
| **Q4 (Dispersed)** | Most effective skills | Many skills contribute equally |

**Finding**: Q3 (moderate concentration) performs best, not extremes.

---

## Visualization Guide

### 1. pooling_analysis.png

**Left Panel: Scatter Plot**
- X-axis: Skill genericness (0=specific, 1=generic)
- Y-axis: CPP prediction confidence for true target
- Colors: Text (blue), Skill (orange), Hybrid (green)
- Dashed lines: Linear regression trends with r-value
- **Expected**: Negative slope (generic skills → lower confidence)

**Right Panel: Box Plot**
- Grouped by model (Text, Skill, Hybrid) and correctness (✓/✗)
- Y-axis: Mean skill genericness
- **Expected**: "Correct" boxes should be lower (more specific skills)

### 2. pooling_strategy_analysis.png

**Top-Left: Effective Skills by Strategy**
- Box plots comparing mean/weighted_mean/weighted_idf
- Shows how much weight concentration differs

**Top-Right: Weight Distribution**
- Histogram of maximum weights per sample
- Shows how often a single skill dominates

**Bottom-Left: Weight Concentration vs Accuracy**
- Histogram split by correct/incorrect predictions
- Shows if concentrated or dispersed weights help

**Bottom-Right: Skill Frequency vs Rank**
- Scatter plot: median frequency → prediction rank
- Color: average skill score
- **Expected**: Low frequency (specific skills) → better rank

### 3. frequency_analysis.png

**Top-Left: Skill Frequency by Correctness**
- Box plots of median frequency per model/correctness
- **Expected**: Correct predictions have lower median frequency

**Top-Right: Skill Frequency Range**
- Scatter: min_freq vs max_freq
- Shows the spread of skill specificity in each path

**Bottom-Left: Frequency Variance**
- Histogram of frequency standard deviation
- Shows if uniform or mixed skill profiles help

**Bottom-Right: Skill Diversity vs Repetition**
- Scatter: unique skills vs total occurrences
- Points on diagonal = no skill repetition
- Points above = skill repetition in path

### 4. idf_source_comparison.png

This visualization compares IDF scores from ESCO taxonomy vs Decorte dataset:

**Top-Left: IDF Scatter (ESCO vs Decorte)**
- Each point is a skill present in both sources
- X-axis: ESCO IDF, Y-axis: Decorte IDF
- Red dashed line: y=x (perfect agreement)
- Green dashed line: Regression with correlation coefficient
- **Expected**: Strong positive correlation if sources agree

**Top-Right: IDF Distribution**
- Overlaid histograms showing IDF distribution from each source
- Shows if the range and shape of IDF differs

**Bottom-Left: Per-Path Genericness Comparison**
- For each career path, plots average genericness from ESCO vs Decorte
- Shows if path-level genericness is consistent across sources

**Bottom-Right: Median Frequency Comparison**
- Log-scale scatter of median skill frequency per path
- X-axis: ESCO frequency, Y-axis: Decorte frequency
- Note: ESCO counts occupations, Decorte counts job postings

---

## Path Statistics CSV Columns

The `path_statistics.csv` file contains one row per career path with these columns:

### Identifier
| Column | Description |
|--------|-------------|
| `path_idx` | Index of the career path (matches `sample_idx` in main DataFrame) |

### Frequency Statistics (from ESCO taxonomy)
| Column | Description |
|--------|-------------|
| `min_freq` | Minimum skill frequency (rarest skill in path) |
| `median_freq` | Median skill frequency |
| `max_freq` | Maximum skill frequency (most common skill in path) |
| `avg_freq` | Average skill frequency |
| `std_freq` | Standard deviation of frequencies |
| `q25_freq` | 25th percentile of frequencies |
| `q75_freq` | 75th percentile of frequencies |

### IDF Statistics
| Column | Description |
|--------|-------------|
| `min_idf` | Minimum IDF (most common skill) |
| `median_idf` | Median IDF |
| `max_idf` | Maximum IDF (rarest skill) |
| `avg_idf` | Average IDF |
| `std_idf` | Standard deviation of IDF |

### Skill Score Statistics (prediction confidence)
| Column | Description |
|--------|-------------|
| `min_score` | Minimum skill prediction score |
| `median_score` | Median skill prediction score |
| `max_score` | Maximum skill prediction score |
| `avg_score` | Average skill prediction score |
| `std_score` | Standard deviation of scores |

### Genericness Statistics (normalized inverse IDF)
| Column | Description |
|--------|-------------|
| `min_genericness` | Minimum genericness (most specific skill) |
| `median_genericness` | Median genericness |
| `max_genericness` | Maximum genericness (most common skill) |
| `avg_genericness` | Average genericness |
| `std_genericness` | Standard deviation of genericness |

### Count Statistics
| Column | Description |
|--------|-------------|
| `num_unique_skills` | Number of distinct skills in the path |
| `total_skills` | Total skill occurrences (with repetition) |
| `num_jobs` | Number of jobs in the career path |

### Pooling Weight Statistics
| Column | Description |
|--------|-------------|
| `mean_weight_entropy` | Entropy of mean pooling weights |
| `weighted_mean_weight_entropy` | Entropy of weighted mean weights |
| `weighted_idf_weight_entropy` | Entropy of weighted IDF weights |
| `mean_max_weight` | Maximum weight for mean pooling (= 1/N) |
| `weighted_mean_max_weight` | Maximum weight for weighted mean |
| `weighted_idf_max_weight` | Maximum weight for weighted IDF |
| `mean_effective_skills` | Effective skills for mean pooling (= N) |
| `weighted_mean_effective_skills` | Effective skills for weighted mean |
| `weighted_idf_effective_skills` | Effective skills for weighted IDF |

### Decorte Dataset Statistics (from job postings)
| Column | Description |
|--------|-------------|
| `decorte_min_freq` | Minimum skill frequency from Decorte dataset |
| `decorte_median_freq` | Median skill frequency |
| `decorte_max_freq` | Maximum skill frequency |
| `decorte_avg_freq` | Average skill frequency |
| `decorte_std_freq` | Standard deviation of frequencies |
| `decorte_min_idf` | Minimum IDF (from Decorte) |
| `decorte_median_idf` | Median IDF |
| `decorte_max_idf` | Maximum IDF |
| `decorte_avg_idf` | Average IDF |
| `decorte_std_idf` | Standard deviation of IDF |
| `decorte_min_genericness` | Minimum genericness (from Decorte) |
| `decorte_median_genericness` | Median genericness |
| `decorte_max_genericness` | Maximum genericness |
| `decorte_avg_genericness` | Average genericness |
| `decorte_std_genericness` | Standard deviation of genericness |

---

## Command-Line Arguments

```
python pooling_method_analysis.py [OPTIONS]

Required:
  --text_scores PATH       Path to text-only CPP scores pickle
  --skill_scores PATH      Path to skill-only CPP scores pickle
  --hybrid_scores PATH     Path to hybrid CPP scores pickle
  --master_skill_file PATH Path to fused_predictions.json
  --output_dir PATH        Output directory for results

Optional:
  --master_job_data PATH   Path to decorte_master_3.csv (for metadata)
  --esco_taxonomy_file PATH Path to ESCO occupation-skill relations CSV
  --alpha FLOAT            Exponent for score in weighted_idf (default: 1.0)
  --beta FLOAT             Exponent for IDF in weighted_idf (default: 1.0)
```
