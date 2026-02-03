# Documentation: `finetune_last_job_skills.py`

This document provides a comprehensive overview of the `finetune_last_job_skills.py` script, which is designed to fine-tune SentenceTransformer models on career path data. The goal is to learn embeddings that map the **skills of a user's last job** to the **description of their next occupation**.

## 1. Overview

The script performs the following high-level operations:
1.  Loads career path datasets (either with standard ESCO titles or free-text titles).
2.  Maps jobs to their underlying skills (using either a taxonomy or predicted skills).
3.  Constructs training pairs: `(Skills of Last Job, Next Job Description)`.
4.  Fine-tunes a `SentenceTransformer` model using **Multiple Negatives Ranking Loss (MNRL)**.
5.  Evaluates the model using Information Retrieval metrics (Recall@k, MRR).

## 2. Data Processing

The data processing pipeline transforms raw career paths into pairs suitable for contrastive learning.

### 2.1 Inputs
The script accepts several data types via the `--data_type` argument:
*   **ESCO-based (`karrierewege`, `decorte_esco`, etc.)**: Jobs are standard ESCO titles. Skills are looked up from a static `Job Title -> Skills` CSV mapping.
*   **Free-text / Predicted (`decorte`, `karrierewege_occ`, etc.)**: Jobs are free-text. Skills are retrieved from a "Fused Predictions" JSON file (`fused_predictions.json`), which contains predicted skills and confidence scores for each specific `job_id`.

### 2.2 Skill Representation
For the "Anchor" document (the user's current state), the script constructs a text representation of their skills:

1.  **Selection**: The script selects the top-$k$ most relevant skills for the job.
    *   **Strategies**: Top-k (highest scored) or Stratified sampling.
    *   **Scoring**: Based on IDF (rarity), Prediction Confidence (from classifier), or a Weighted combination.
2.  **Formatting**: The skills are concatenated into a single string.
    *   *With Descriptions*: `skill: <name> \n description: <text> [SEP] skill: <name> ...`
    *   *Without Descriptions*: `name [SEP] name ...`

### 2.3 Pair Construction
The core learning signal comes from transitions between jobs $J_t$ and $J_{t+1}$.

*   **Anchor ($doc_1$)**: Validated skills of job $J_t$ (Last Job).
*   **Positive ($doc_2$)**: The *description* and *title* of job $J_{t+1}$ (Target Occupation).
    *   Format: `esco role: <title> \n description: <text>`

> **Note**: The script handles data splitting (train/val/test) and filters out transitions where skill data is missing.

## 3. Main Flow

The execution flow involves the following stages:

1.  **Initialization**:
    *   Parses arguments.
    *   Sets up logging (File & Weights & Biases).
    *   Loads resources (Skill descriptions, Mappings).

2.  **Data Loading & Preparation**:
    *   Loads the dataset split (Train/Val/Test).
    *   **Scoring**: Calculates IDF or Weighted scores for all skills in the dataset.
    *   **Pair Construction**: Iterates through career paths to build `(doc1, doc2)` pairs.
        *   *Debug*: Optionally prints detailed samples of constructed pairs (`--print_sample`).
    *   **Analysis**: Checks token lengths to warn about potential truncation by the model.

3.  **Baseline Evaluation (Optional)**:
    *   If `--test_base_model` is set, computes retrieval metrics on the test set using the pre-trained (un-finetuned) model.

4.  **Training (Fine-tuning)**:
    *   Instantiates the `SentenceTransformer`.
    *   Uses `MultipleNegativesRankingLoss` (MNRL).
    *   Runs for $N$ epochs with the configured batch size and learning rate.
    *   Evaluates on the validation set during training.
    *   Optionally runs retrieval evaluation on the test set at the end of every epoch (`--test_strategy epoch`).

5.  **Finalization**:
    *   Saves the best model foundation.
    *   Runs final retrieval evaluation on the test set.
    *   Saves metrics to `metrics.json` and logs to W&B.

## 4. Experimentation Options

The script offers extensive command-line arguments for experimentation.

| Category | Argument | Description |
| :--- | :--- | :--- |
| **Data & Resources** | `--data_type` | Dataset variant (e.g., `karrierewege_100k`, `decorte`). |
| | `--job_title_skills_csv` | CSV for ESCO taxonomy skills (Title -> Skills). |
| | `--skill_scores_json` | JSON for predicted skills (Job ID -> Skills + Scores). |
| | `--skills_csv` | CSV containing skill descriptions. |
| **Model** | `--model_name` | Base Hugging Face model (default: `all-mpnet-base-v2`). |
| **Skill Selection** | `--top_k_skills` | Number of skills to include in the Anchor document. |
| | `--skill_selection_strategy` | `top_k` (best) or `stratified` (random sampling from tiers). |
| | `--scoring_mode` | `idf_only` (rarity), `scores_only` (confidence), or `weighted`. |
| | `--importance_weight` | Weight $\alpha$ for mixing scores in `weighted` mode ($0 \le \alpha \le 1$). |
| | `--no_skill_descriptions` | If set, omits skill descriptions from the Anchor document. |
| **Training** | `--epochs`, `--batch_size` | Standard training hyperparameters. |
| | `--learning_rate` | Learning rate (default: `2e-5`). |
| | `--scheduler` | LR Schedule (e.g., `linear`). |
| **Evaluation** | `--test_strategy` | When to run full retrieval eval (`final`, `epoch`, `none`). |
| | `--test_base_model` | simple flag to benchmark the starting model. |

## 5. Mathematical Representations

### 5.1 Multiple Negatives Ranking Loss (MNRL)
The model is trained to maximize the similarity between the skills of job $i$ ($a_i$) and the description of the *actual* next job $i$ ($p_i$), while minimizing similarity to next jobs of *other* samples in the batch ($p_j$).

For a batch of size $N$, the loss is the Cross Entropy of the softmax-normalized similarity scores:

$$
\mathcal{L} = - \frac{1}{N} \sum_{i=1}^N \log \frac{e^{\text{sim}(a_i, p_i) / \tau}}{\sum_{j=1}^N e^{\text{sim}(a_i, p_j) / \tau}}
$$

*   $\text{sim}(u, v)$: Cosine similarity $\frac{u \cdot v}{\|u\| \|v\|}$.
*   $\tau$: Temperature parameter (scaled by 20.0 in code $\implies \tau = 0.05$).

### 5.2 Weighted Scoring
When using `--scoring_mode weighted`, skills are ranked by a linear combination of their **Prediction Confidence** and **IDF (Inverse Document Frequency)**.

$$
\text{Weighted Score} = \alpha \cdot \text{Score}_{\text{pred}} + (1 - \alpha) \cdot \text{Score}_{\text{IDF}}
$$

*   $\alpha$: Controlled by `--importance_weight`.
*   $\text{Score}_{\text{pred}}$: Output from the skill classifier (0-1).
*   $\text{Score}_{\text{IDF}}$: Normalized IDF score representing skill specificity.

### 5.3 Retrieval Metrics
The model's performance is measured by its ability to retrieve the correct next job description ($doc_2$) given the previous skills ($doc_1$) from a set of all possible target jobs in the test set.

*   **Recall@k (R@k)**: The probability that the correct target job is within the top $k$ retrieved results.
    $$ R@k = \frac{1}{Q} \sum_{i=1}^Q \mathbb{I}(\text{rank}_i \le k) $$

*   **Mean Reciprocal Rank (MRR)**: The average of the multiplicative inverse of the rank of the first correct answer.
    $$ \text{MRR} = \frac{1}{Q} \sum_{i=1}^Q \frac{1}{\text{rank}_i} $$
