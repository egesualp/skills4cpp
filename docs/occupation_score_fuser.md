# Occupation Score Fuser Documentation

**Script**: `src/cpp/decorte/occupation_score_fuser.py`

## Overview
The `occupation_score_fuser.py` script combines predictions from two distinct sources to improve next-occupation prediction accuracy. It fuses:
1.  **Text-based Scores ($S_{text}$)**: Output from an MLP model (from `train_cpp_enhanced_v2.py`).
2.  **Skill Overlap Scores ($S_{overlap}$)**: Output based on skill similarities (from `skill_overlap_scoring_v2.py`).

The script performs a grid search over fusion hyperparameters to find the optimal combination that maximizes ranking metrics (MRR, Recall@K).

---

## Execution Flow

The script follows a linear pipeline:

1.  **Initialization**:
    -   Parses CLI arguments (paths, fusion mode, hyperparameter ranges).
    -   Sets up logging.

2.  **Data Loading & Alignment**:
    -   Loads *Text* score matrices (`*_scores_text.pkl`).
    -   Loads *Overlap* score matrices (`*_scores_skill_overlap.pkl`).
    -   **Alignment**: Reorders the columns of the Overlap matrix so that they match the target label ordering of the Text matrix. This ensures the $j$-th column in both matrices corresponds to the same occupation.

3.  **Grid Search (Tuning)**:
    -   Iterates over specified *splits* (usually `val`).
    -   Generates a grid of hyperparameter configurations (alphas, weights, normalization flags).
    -   **Parallel Processing**: Distributes configurations across worker processes.
    -   For each configuration:
        -   Fuses scores using the specified mathematical formula.
        -   Calculates ranking metrics (MRR, Recall@K).
    -   Identifies the best configuration based on a target metric (default: MRR).

4.  **Evaluation**:
    -   Applies the best configuration found during tuning to the *evaluation splits* (e.g., `test`, `clean_test`).
    -   Computes final performance metrics.

5.  **Output**:
    -   Logs a summary table of the grid search.
    -   Saves the best fused scores (`*_scores_fused.pkl`).
    -   Saves comprehensive metrics to `grid_search_metrics.json`.

---

## Mathematical Equations

### 1. Score Fusion

The script supports two fusion modes:

**A. Linear Fusion**
A weighted convex combination of the scores.
$$S_{hybrid} = \alpha \cdot S_{text} + (1 - \alpha) \cdot S_{overlap}$$
*   $\alpha \in [0, 1]$: Controls the importance of the text model.

**B. Bayesian (Multiplicative) Fusion**
Treats overlap scores as a prior or boosting factor.
$$S_{hybrid} = S_{text} \cdot (S_{overlap} + \epsilon)^w$$
*   $w$: Weight/exponent applied to the overlap scores.
*   $\epsilon$: A small constant (e.g., $10^{-6}$) to prevent zeroing out scores.

### 2. Normalization
Before fusion, scores can be optionally normalized per sample (row-wise).

**Min-Max Normalization**:
$$x' = \frac{x - \min(x)}{\max(x) - \min(x)}$$

**Z-Score Normalization**:
$$x' = \frac{x - \mu}{\sigma}$$

### 3. Evaluation Metrics
Metrics are calculated by comparing the predicted ranking against the true ground-truth indices.

**Mean Reciprocal Rank (MRR)**:
$$MRR = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{rank_i}$$
Where $rank_i$ is the rank position of the true occupation for sample $i$.

**Recall@K (R@K)**:
$$R@K = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \mathbb{I}(rank_i \le K)$$
Fraction of samples where the true occupation appears in the top $K$ predictions.

---

## Preprocessing Steps

### 1. Label Extraction
Raw labels often contain metadata (e.g., `esco role: Chef \n description: ...`). The script normalizes these to simple occupation names to ensure robust matching between datasets.

The `extract_occupation_name` function handles:
-   Prefixes like `esco role:` or `role:`.
-   Suffixes starting with `\n`.
-   Whitespace and casing.

**Example**:
`"esco role: Software Engineer \n description: writes code"` $\rightarrow$ `"software engineer"`

### 2. Score Alignment
Since the two input sources might produce score matrices with different column orderings (target labels), the script aligns them dynamically.
1.  Extracts normalized names for both `text_labels` and `overlap_labels`.
2.  Maps each `text_label` to its corresponding index in `overlap_labels`.
3.  Reorders the `overlap_scores` matrix columns:
    `aligned_overlap = overlap_scores[:, reorder_indices]`

---

## Usage

**Example Command (Linear Fusion)**:
```bash
python -m src.cpp.decorte.occupation_score_fuser \
    --text_scores_dir results/cpp/decorte_static/scores \
    --overlap_scores_dir results/cpp/decorte/skill_overlap \
    --output_dir results/cpp/decorte/fused \
    --fusion_mode linear \
    --alphas "0.0,0.5,1.0" \
    --splits val \
    --eval_splits test
```

**Key Arguments**:
-   `--fusion_mode`: `linear` or `bayesian`.
-   `--alphas`: Comma-separated list of alpha values (Linear only).
-   `--weights`: Comma-separated weights (Bayesian only).
-   `--grid_normalize`: If set, adds normalization options (True/False for both sources) to the grid search.
