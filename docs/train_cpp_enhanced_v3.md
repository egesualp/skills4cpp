# Documentation: `train_cpp_enhanced_v3.py`

This document provides a comprehensive overview of the `train_cpp_enhanced_v3.py` script. This script is a specialized variant of `train_cpp_enhanced_v2.py` designed to leverage **Last-Job Skill Documents**, particularly those encoded by models fine-tuned via `finetune_last_job_skills.py`.

## 1. Overview

While v2 allows for aggregations of skills across the entire career path (e.g., via weighted averaging or logarithmic decay), **v3 enforces a specific "Skill Document" approach**:

1.  **Last-Job Focus**: It extracts skills *only* from the last job in the input history.
2.  **Document Formatting**: It constructs a single text document for these skills, following the exact format used during fine-tuning:
    ```text
    skill: <name> \n description: <desc> <SEP> skill: <name> ...
    ```
3.  **Advanced Filtering**: It applies **IDF-based scoring** and **Top-K capping** (with lexicographic tie-breaking) to select the most relevant skills for that job.
4.  **Dedicated Encoding**: It allows encoding this skill document with a separate, fine-tuned SentenceTransformer (`--encoder_skill`), distinct from the career history encoder.

The rest of the pipeline (Optuna optimization, multi-modal fusion logic, evaluation metrics) remains identical to v2.

## 2. Key Differences from v2

| Feature | `train_cpp_enhanced_v2.py` | `train_cpp_enhanced_v3.py` |
| :--- | :--- | :--- |
| **Skill Scope** | Aggregates skills from *all* jobs in history (optional decay). | **Last Job Only**. |
| **Skill Representation** | Weighted average of individual skill embeddings. | **Single encoded vector** of a concatenated "Skill Document". |
| **Skill Selection** | Uses all mapped skills (weighted by confidence). | **Top-K** (e.g., 10) selected by **IDF** score. |
| **Encoders** | Typically shares one encoder for text & skills (feature-level fusion). | Designed for **Dual Encoders**: General MPNet for history, Fine-tuned MPNet for skills. |
| **Hardcoded Params** | Configurable pooling (`mean`, `weighted_idf`, etc.). | **Enforced**: `weighted_idf` logic for selection, flattened document for encoding. |

## 3. Data Processing

### 3.1 Input Data & Skill Mapping
*   **Career Paths**: Same as v2 (history $\to$ target).
*   **Skill Mapping**:
    *   For **Decorte/Standard Datasets**: Uses `dataset_name` to load standard ESCO mappings.
    *   **Predicted Skills**: Supports `fused_predictions.json` via `--skill_scores_json` for datasets without ground-truth skills.
    *   **Taxonomy**: Can load raw ESCO files (`--raw_esco_dir`) or pre-processed masters (`--master_skill_file`).

### 3.2 Last-Job Skill Document Generation
For every sample, the script performs:
1.  **Extraction**: Identifies the last job title in the history.
2.  **Lookup & Scoring**:
    *   Retrieves potential skills for that job.
    *   Calculates/Retrieves **IDF scores** for each skill (measuring specificity).
    *   **Scoring Mode** (`--scoring_mode`): Can use `idf_only` (default), `scores_only` (confidence), or `weighted` (combination).
3.  **Selection**: Selects the top $K$ skills (`--top_k_skills`, default 10).
    *   *Tie-breaking*: IDF $\to$ Description Length $\to$ Lexicographical Name.
4.  **Formatting**:
    *   Constructs the text document.
    *   Separator: `[SEP]` (or tokenizer equivalent).
    *   Format: `skill: {name} \n description: {desc}`.

### 3.3 Text History
*   Processed normally using `--encoder_text`.
*   Can include descriptions (`--use_text_description`) or titles only.

## 4. Experimentation Options

Key command-line arguments unique to or critical for v3:

| Category | Argument | Description |
| :--- | :--- | :--- |
| **Skill Encoder** | `--encoder_skill` | **Critical**: Path to the fine-tuned skill model (e.g., from `finetune_last_job_skills.py`). |
| **Skill Selection** | `--top_k_skills` | Max skills to include for the last job (default: `10`). |
| | `--skill_selection_strategy` | `top_k` (default) or `stratified` (sampling). |
| | `--scoring_mode` | `idf_only` (default) is recommended for stable canonical skill representations. |
| **Skill Data** | `--master_skill_file` | Path to master CSV mapping jobs to skills. |
| | `--raw_esco_dir` | Path to raw ESCO CSVs (alternative to master file). |
| | `--skill_scores_json` | Path to predicted scores (for non-standard datasets). |
| **General** | `--force_recompute` | Force re-creation of skill document embeddings (bypasses cache). |

**Note**: The following v2 arguments are **ignored/overridden** in v3:
*   `--pooling_strategy` (forced to internal logic)
*   `--use_skill_path_log_pooling` (forced to `False`)
*   `--alpha`, `--beta` (forced to `1.0`)

## 5. Mathematical Representations

### 5.1 Skill Embedding ($E_{skills}$)
Unlike v2's weighted average, v3 encodes the document $D_{skills}$:

$$ D_{skills} = \text{Join}_{SEP}(\{ \text{fmt}(s) \mid s \in \text{TopK}_{IDF}(S_{last\_job}) \}) $$

$$ E_{skills} = \text{Encoder}_{skill}(D_{skills}) $$

Where $\text{fmt}(s)$ is `"skill: name \n description: desc"`.

### 5.2 Multi-Modal Fusion
The final model input is the concatenation (late fusion) of the history embedding and this specific skill document embedding:

$$ \text{Input} = [\alpha_{text} E_{history}(H) ; \alpha_{skill} E_{skills}(D_{skills}) ; \alpha_{struct} E_{struct}] $$

## 6. Usage Example

```bash
python src/cpp/train_cpp_enhanced_v3.py \
    --data_type decorte \
    --encoder_text "sentence-transformers/all-mpnet-base-v2" \
    --encoder_skill "path/to/fine_tuned_skill_encoder_v1" \
    --top_k_skills 10 \
    --use_text_history --use_skill_text --use_advanced \
    --output_dir results/cpp_v3_run1
```
