# Skill Scoring and Fusion Pipeline

This directory contains scripts for computing skill-based overlap scores and fusing them with text-based predictions for career path prediction.

## 1. Skill Overlap Scoring (`skill_overlap_scoring_v2.py` / `skill_overlap_scoring_v3.py`)

These scripts compute a skill overlap score between a user's career history and potential next occupations.

**Formula:**
$$ S_{skills}(history, target) = \frac{|S_{career} \cap S(target)|}{|S(target)|} $$

Where:
*   $S_{career}$: Union of all skills associated with the user's past jobs.
*   $S(target)$: Set of skills required by the target occupation (from ESCO).

### Modes

The scripts automatically detect the mode based on the `--skill_scores_file` argument:

1.  **Job ID Mode (Default for `decorte` raw titles)**
    *   Career skills are looked up via `job_id` from a pre-computed skill prediction file (e.g., `best_fused_scores.json`).
    *   Usage: Provide `--skill_scores_file`.

2.  **ESCO Mode (For `decorte_esco`)**
    *   Career skills are looked up directly from the ESCO taxonomy using the job titles in the history.
    *   Usage: Do **not** provide `--skill_scores_file`.

### Differences: v2 vs v3

| Feature | `skill_overlap_scoring_v2.py` | `skill_overlap_scoring_v3.py` |
| :--- | :--- | :--- |
| **Skill Filtering** | Filters by Top-K only. | Adds **Confidence Threshold** (`--min_skill_confidence`) to filter low-confidence predictions. |
| **Target Matching** | Uses raw strings. | **Normalizes labels** (strip, lowercase) for robust de-duplication and matching. |
| **Evaluation** | **Penalizes** unmapped targets (MRR = 0). | **Excludes** unmapped targets from metrics (reports performance only on valid data). |
| **Reporting** | Basic metrics. | Detailed statistics on filtered skills and unmapped targets. |
| **Recall@K** | Only computes R@1, 5, 10. | Same (typically 1, 5, 10, 20). |

### Usage Example

**Running v3 (Recommended):**

```bash
python -m cpp.decorte.skill_overlap_scoring_v3 \
    --data_type decorte \
    --skill_scores_file ../../../results/skill_prediction/best_scores.json \
    --output_dir results/cpp/decorte/skill_overlap_scores \
    --top_k_skills 10 \
    --min_skill_confidence 0.5 \
    --splits train val test \
    --save_scores
```

**Arguments:**
*   `--data_type`: Dataset name (e.g., `decorte`, `karrierewege`).
*   `--skill_scores_file`: JSON/JSONL file mapping `job_id` -> predicted skills (required for Job ID mode).
*   `--top_k_skills`: (Optional) Use only top K predicted skills per job.
*   `--min_skill_confidence`: (v3 only) Minimum score threshold for a skill to be included.
*   `--eval_clean_test`: If set, also processes the `test_clean` split.

---

## 2. Occupation Score Fuser (`occupation_score_fuser.py`)

This script combines the **text-based scores** (from the main MLP model) with the **skill overlap scores** computed above to improve prediction accuracy.

It performs a **Grid Search** on the validation set to find optimal fusion parameters, then evaluates the best configuration on the test set.

### Fusion Modes

1.  **Linear**:
    $$ S_{hybrid} = \alpha \cdot S_{text} + (1 - \alpha) \cdot S_{overlap} $$
    *   Tunable parameter: `alpha` (0.0 to 1.0)

2.  **Bayesian**:
    $$ S_{hybrid} = S_{text} \cdot (S_{overlap} + \epsilon)^w $$
    *   Tunable parameter: `weight` (exponent) and `epsilon` (smoothing).

### Usage Example

```bash
python -m cpp.decorte.occupation_score_fuser \
    --text_scores_dir results/cpp/decorte_static/scores \
    --overlap_scores_dir results/cpp/decorte/skill_overlap_scores \
    --output_dir results/cpp/decorte/fused_scores \
    --fusion_mode linear \
    --alphas 0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
    --splits val \
    --eval_splits test clean_test \
    --normalize_text \
    --save_scores
```

**Key Arguments:**
*   `--text_scores_dir`: Directory containing `*_scores_text.pkl` files (output from `train_cpp_enhanced_v2.py`).
*   `--overlap_scores_dir`: Directory containing `*_scores_skill_overlap.pkl` files.
*   `--fusion_mode`: `linear` or `bayesian`.
*   `--alphas`: Comma-separated list of alphas to test (linear mode).
*   `--weights`: Comma-separated list of weights to test (bayesian mode).
*   `--splits`: Splits to use for **tuning** (usually `val`).
*   `--eval_splits`: Splits to **evaluate** the best tuned config on (e.g., `test`, `clean_test`).
*   `--num_workers`: Number of parallel workers for grid search (-1 for all cores).
*   `--grid_normalize`: If set, adds normalization (True/False) to the grid search.

### Output
*   `grid_search_metrics.json`: Detailed metrics for every configuration tested.
*   `*_scores_fused.pkl`: Final fused scores for the best configuration (if `--save_scores` is used).
