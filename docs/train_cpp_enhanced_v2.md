# Documentation: `train_cpp_enhanced_v2.py`

This document provides a comprehensive overview of the `train_cpp_enhanced_v2.py` script. This script is designed for **enhanced Career Path Prediction (CPP)** modelling, incorporating multi-modal data fusion, Optuna hyperparameter optimization, and efficient training on GPU.

## 1. Overview

The script performs the following primary functions:
1.  **Multi-Modal Learning**: Fuses information from three sources:
    *   **Text History**: Embeddings of previous job titles/descriptions.
    *   **Skills**: Aggregated embeddings of skills associated with jobs (via pre-computed maps).
    *   **Structured Data**: One-hot encoded features derived from skill properties.
2.  **Optimized Training**:
    *   **Optuna**:  Automated hyperparameter search (learning rate, layers, dimension, dropout, optimizer).
    *   **GPU Optimization**: Supports GPU-pinned embeddings to minimize CPU-GPU transfer overhead.
    *   **Mixed Precision**: Supports FP16 training.
3.  **Model Architectures**:
    *   **Simple Concatenation (Early Fusion)**: Concatenates all input features before passing them to an MLP.
    *   **Multi-Modal (Late Fusion)**: Uses separate encoders for each modality, optionally with learnable scalar weights, before fusion.
4.  **Evaluation**: Computes standard Information Retrieval metrics (MRR, Recall@k) using cosine similarity against all possible target occupations.

## 2. Data Processing

The data pipeline is designed for scalability and flexibility.

### 2.1 Input Data
*   **Career Paths**: Loaded via the `Data` class (supports `decorte` and other datasets).
    *   Splits: Train, Validation, Test (and optionally "Clean Test" without subspans).
*   **Skill Mapping**:
    *   **Predicted Skills**: Uses a JSON file (`--skill_scores_file`) mapping `job_id` → list of skills with confidence scores.
    *   **ESCO Taxonomy**: Uses static CSV mappings for some datasets.
*   **Precomputed Embeddings**: Can load cached skill embeddings (`--skill_embeddings_dir`) to speed up training.

### 2.2 Feature Engineering
The model can consume three types of inputs, controlled by flags:

1.  **Text History (`--use_text_history`)**:
    *   Encodes the sequence of past jobs using a SentenceTransformer.
    *   Options: Titles only or full descriptions (`--use_text_description`).
2.  **Skill Text (`--use_skill_text`)**:
    *   Aggregates embeddings of all skills in the history.
    *   **Pooling Strategies** (`--pooling_strategy`):
        *   `mean`: Simple average.
        *   `weighted_mean`: Weighted by skill confidence scores.
        *   `weighted_idf`: Weighted by IDF (specificity) and confidence.
    *   **Logarithmic Decay**: Optionally weights recent jobs higher (`--use_skill_path_log_pooling`).
3.  **Structured Features (`--use_structured`)**:
    *   One-hot vectors representing skill properties (e.g., skill type, reuse level).

### 2.3 Labeling
*   **Target ($y$)**: The embedding of the **next job title/description**.
*   **Target Vocabulary**: The set of *all* possible target embeddings ($Y_{all}$) is precomputed for efficient ranking evaluation.

## 3. Main Flow

The execution consists of two main phases: **Optimization (Optuna)** and **Final Training**.

1.  **Setup**:
    *   Configures logging (File & W&B).
    *   Detects hardware (GPU/CPU) and optimization settings (pin_memory).
    *   Loads encoders (Text & Skill).

2.  **Data Preparation**:
    *   Loads career path pairs.
    *   Filters data (e.g., removes repetitive transitions).
    *   **Embedding Precomputation**:
        *   Computes/Loads embeddings for Targets.
        *   Computes/Loads inputs for Train/Val/Test splits (cached to disk).
    *   **Dataset Creation**: Initializes `CareerPathDataset` and `DataLoader` for each split.

3.  **Optuna Optimization (Optional)**:
    *   If `--optuna` is set:
        *   Runs $N$ trials.
        *   Samples hyperparameters (Hidden Dim, Layers, Dropout, LR, Optimizer).
        *   Trains on a *subset* of data for speed.
        *   Selects the best hyperparameters based on Validation MRR.

4.  **Final Training**:
    *   Instantiates the final model with the best (or static) hyperparameters.
    *   Trains on the **full** training set (concatenated Train + Val).
    *   Uses **CosineEmbeddingLoss** with hard negatives (implicit in batch) or explicit targets.
    *   **Early Stopping**: Monitors Validation MRR/Loss.

5.  **Evaluation**:
    *   Computes Final Metrics (MRR, R@1, R@5, R@10, R@20) on Test (and Clean Test) sets.
    *   Saves the model to disk.
    *   Optionally saves prediction scores (`.pkl`) for ensemble fusion.

## 4. Experimentation Options

Key command-line arguments:

| Category | Argument | Description |
| :--- | :--- | :--- |
| **Data** | `--data_type` | Dataset used (e.g., `decorte`). |
| | `--skill_scores_file` | JSON mapping Job IDs to Skills. |
| | `--embeddings_cache_dir` | Directory to cache intermediate embeddings. |
| | `--pin_embeddings_to_gpu` | Optimization: keep embeddings in VRAM. |
| **Features** | `--use_text_history` | Enable text modality. |
| | `--use_skill_text` | Enable skill text modality. |
| | `--use_structured` | Enable structured dictionary features. |
| | `--pooling_strategy` | How to aggregate skills (`mean`, `weighted_idf`). |
| **Model** | `--use_advanced` | Use the Multi-Modal architecture (vs. Simple Concat). |
| | `--encoder_text` | Hugging Face model for text encoding. |
| **Training** | `--optuna` | Enable hyperparameter search. |
| | `--optimizer` | `adam` or `sgd`. |
| | `--mixed_precision` | Enable FP16 training. |
| **Output** | `--save_scores` | Save raw prediction scores for later fusion. |

## 5. Mathematical Representations

### 5.1 Cosine Embedding Loss
The model is trained to minimize the distance between the predicted career vector $v_{pred}$ and the true next job vector $v_{target}$.

$$
\mathcal{L}(x, y) = \begin{cases}
1 - \cos(x, y), & \text{if } y = 1 \text{ (similar)} \\
\max(0, \cos(x, y) - \text{margin}), & \text{if } y = -1 \text{ (dissimilar)}
\end{cases}
$$

In this script, the target label is always $1$ (maximize similarity to the ground truth). The batch construction effectively provides "in-batch negatives" for the ranking metrics, though the loss itself here focuses on alignment.

### 5.2 Ranking Metrics
Evaluation is performed by ranking *all* unique targets $t_j \in Y_{all}$ against the prediction $v_i$.

*   **Ranking**: Compute cosine similarity $S(v_i, t_j)$ for all $j$. Sort $j$ by score.
*   **Reciprocal Rank**: $\frac{1}{\text{rank}(t_{true})}$.
*   **MRR**: $\frac{1}{N} \sum \frac{1}{\text{rank}_i}$.

### 5.3 Skill Pooling (Weighted IDF)
For a set of skills $S$, the aggregate embedding $E_{skills}$ is:

$$
E_{skills} = \frac{\sum_{s \in S} w_s \cdot E(s)}{\sum_{s \in S} w_s}
$$

Where weights $w_s$ depend on the strategy:
*   **Mean**: $w_s = 1$
*   **Weighted Mean**: $w_s = \text{score}(s)$
*   **Weighted IDF**: $w_s = \text{score}(s)^\alpha \cdot \text{IDF}(s)^\beta$

### 5.4 Multi-Modal Fusion (Advanced Model)
$$
\text{Output} = \text{Linear}\left( [\alpha_{text} E_{text} ; \alpha_{skill} E_{skills} ; \alpha_{struct} E_{struct}] \right)
$$
*   $\alpha$: Learnable scalar weights (if `--use_modality_weights` is enabled).
