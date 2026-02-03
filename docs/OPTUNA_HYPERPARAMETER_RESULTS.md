# Optuna Hyperparameter Optimization Results

## Overview

This document summarizes the best hyperparameters found by Optuna for different experimental setups on the `cpp_karrierewege_100k` dataset. All experiments used MRR (Mean Reciprocal Rank) as the optimization metric.

## Experimental Setups

The experiments varied across several key configuration parameters:
- **Skill encoder**: Different pre-trained models (`best-model`, `with_skill_description/best-model`, `no_skill_descriptions/best-model`)
- **Skill descriptions**: Whether to include skill descriptions (`use_skill_description`)
- **Top-k skills**: Number of skills per job (15 or 20)
- **Structured features**: Whether to include structured features (`use_structured`)
- **Pooling strategy**: Embedding pooling method (typically `weighted_idf`)

---

## Results Summary

### Setup 1: With Skill Descriptions (top_k=15, encoder=with_skill_description)

**Log File**: `cpp_karrierewege_100k_profiling_5412590.log`

**Configuration**:
- `use_skill_description`: True
- `top_k_skills`: 15
- `encoder_skill`: `/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/with_skill_description/best-model`
- `use_structured`: False
- `pooling_strategy`: `weighted_idf`

**Best Results**:
- **Best Trial**: 8
- **Best Validation MRR**: **0.4820**
- **Optimal Training Epochs**: 12

**Best Hyperparameters**:
- `hidden_dim`: 1024
- `n_layers`: 2
- `dropout`: 0.0538
- `use_modality_weights`: True
- `lr`: 1.678e-05
- `weight_decay`: 1.078e-05

---

### Setup 2: Without Skill Descriptions (top_k=20, encoder=no_skill_descriptions)

**Log File**: `cpp_karrierewege_100k_profiling_5412591.log`

**Configuration**:
- `use_skill_description`: False
- `top_k_skills`: 20
- `encoder_skill`: `/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/no_skill_descriptions/best-model`
- `use_structured`: False
- `pooling_strategy`: `weighted_idf`

**Best Results**:
- **Best Trial**: 3
- **Best Validation MRR**: **0.4795**
- **Optimal Training Epochs**: 7

**Best Hyperparameters**:
- `hidden_dim`: 1024
- `n_layers`: 3
- `dropout`: 0.0623
- `use_modality_weights`: True
- `lr`: 3.437e-05
- `weight_decay`: 2.599e-06

---

### Setup 3: With Skill Descriptions (top_k=15, encoder=best-model) - Best Overall

**Log File**: `cpp_karrierewege_100k_profiling_5410871.log`

**Configuration**:
- `use_skill_description`: True
- `top_k_skills`: 15
- `encoder_skill`: `/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/best-model`
- `use_structured`: False
- `pooling_strategy`: `weighted_idf`

**Best Results**:
- **Best Trial**: 1
- **Best Validation MRR**: **0.4857** ⭐ (Highest MRR)
- **Optimal Training Epochs**: 9

**Best Hyperparameters**:
- `hidden_dim`: 1024
- `n_layers`: 1
- `dropout`: 0.0875
- `use_modality_weights`: False
- `lr`: 2.285e-04
- `weight_decay`: 3.533e-06

---

### Setup 4: With Skill Descriptions (top_k=15, encoder=best-model, variant)

**Log File**: `cpp_karrierewege_100k_profiling_5410805.log`

**Configuration**:
- `use_skill_description`: True
- `top_k_skills`: 15 (implicit, not explicitly set)
- `encoder_skill`: `/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/best-model`
- `use_structured`: False
- `pooling_strategy`: `weighted_idf`

**Best Results**:
- **Best Trial**: 4
- **Best Validation MRR**: **0.4844**
- **Optimal Training Epochs**: 9

**Best Hyperparameters**:
- `hidden_dim`: 1024
- `n_layers`: 1
- `dropout`: 0.2537
- `use_modality_weights`: True
- `lr`: 1.020e-04
- `weight_decay`: 5.608e-05

---

### Setup 5: Without Skill Descriptions (encoder=no_skill_descriptions)

**Log File**: `cpp_karrierewege_100k_profiling_5410772.log`

**Configuration**:
- `use_skill_description`: False
- `encoder_skill`: `/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/no_skill_descriptions/best-model`
- `use_structured`: False
- `pooling_strategy`: `weighted_idf`

**Best Results**:
- **Best Trial**: 1
- **Best Validation MRR**: **0.4807**
- **Optimal Training Epochs**: 9

**Best Hyperparameters**:
- `hidden_dim`: 768
- `n_layers`: 1
- `dropout`: 0.4037
- `use_modality_weights`: True
- `lr`: 2.235e-04
- `weight_decay`: 1.394e-06

---

### Setup 6: With Skill Descriptions (top_k=15, encoder=best-model, batch_size=1024)

**Log File**: `cpp_karrierewege_100k_profiling_5410213.log`

**Configuration**:
- `use_skill_description`: True
- `top_k_skills`: 15
- `encoder_skill`: `/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/best-model`
- `use_structured`: True
- `batch_size`: 1024 (smaller than other experiments)

**Best Results**:
- **Best Trial**: 4
- **Best Validation MRR**: **0.4790**
- **Optimal Training Epochs**: 12

**Best Hyperparameters**:
- `hidden_dim`: 512
- `n_layers`: 3
- `dropout`: 0.0370
- `use_modality_weights`: True
- `lr`: 5.555e-05
- `weight_decay`: 4.457e-05

---

### Setup 7: With Skill Descriptions (top_k=15, encoder=best-model, older run)

**Log File**: `cpp_karrierewege_100k_profiling_5402197.log`

**Configuration**:
- `use_skill_description`: True
- `top_k_skills`: 15
- `encoder_skill`: `/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/best-model`
- `use_structured`: False

**Best Results**:
- **Best Trial**: 9
- **Best Validation MRR**: **0.4817**
- **Optimal Training Epochs**: 12

**Best Hyperparameters**:
- `hidden_dim`: 1024
- `n_layers`: 1
- `dropout`: 0.4223
- `use_modality_weights`: True
- `lr`: 3.970e-05
- `weight_decay`: 4.568e-03

---

### Setup 8: With Skill Descriptions (top_k=15, encoder=best-model, variant 2)

**Log File**: `cpp_karrierewege_100k_profiling_5402198.log`

**Configuration**:
- `use_skill_description`: True
- `top_k_skills`: 15
- `encoder_skill`: `/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/best-model`
- `use_structured`: False

**Best Results**:
- **Best Trial**: 6
- **Best Validation MRR**: **0.4830**
- **Optimal Training Epochs**: 6

**Best Hyperparameters**:
- `hidden_dim`: 1024
- `n_layers`: 1
- `dropout`: 0.4027
- `use_modality_weights`: False
- `lr`: 7.216e-04
- `weight_decay`: 6.042e-06

---

### Setup 9: With Skill Descriptions (top_k=15, encoder=best-model, older run 2)

**Log File**: `cpp_karrierewege_100k_profiling_5401953.log`

**Configuration**:
- `use_skill_description`: True
- `top_k_skills`: 15
- `encoder_skill`: `/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/best-model`
- `use_structured`: False

**Best Results**:
- **Best Trial**: 0
- **Best Validation MRR**: **0.4841**
- **Optimal Training Epochs**: 28

**Best Hyperparameters**:
- `hidden_dim`: 1024
- `n_layers`: 3
- `dropout`: 0.2899
- `use_modality_weights`: True
- `lr`: 4.426e-05
- `weight_decay`: 8.237e-06

---

### Setup 10: With Skill Descriptions (top_k=15, encoder=best-model, older run 3)

**Log File**: `cpp_karrierewege_100k_profiling_5401876.log`

**Configuration**:
- `use_skill_description`: True
- `top_k_skills`: 15
- `encoder_skill`: `/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/best-model`
- `use_structured`: False

**Best Results**:
- **Best Trial**: 3
- **Best Validation MRR**: **0.4718**
- **Optimal Training Epochs**: 28

**Best Hyperparameters**:
- `hidden_dim`: 1024
- `n_layers`: 2
- `dropout`: 0.2534
- `use_modality_weights`: Not specified (likely False)
- `lr`: 1.284e-05
- `weight_decay`: 4.159e-04

---

### Setup 11: With Skill Descriptions (top_k=15, encoder=best-model, older run 4)

**Log File**: `cpp_karrierewege_100k_profiling_5401875.log`

**Configuration**:
- `use_skill_description`: True
- `top_k_skills`: 15
- `encoder_skill`: `/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/best-model`
- `use_structured`: False

**Best Results**:
- **Best Trial**: 1
- **Best Validation MRR**: **0.4779**
- **Optimal Training Epochs**: 7

**Best Hyperparameters**:
- `hidden_dim`: 1024
- `n_layers`: 3
- `dropout`: 0.1404
- `use_modality_weights`: Not specified (likely False)
- `lr`: 3.964e-04
- `weight_decay`: 1.111e-06

---

### Setup 12: With Skill Descriptions (top_k=15, encoder=best-model, older run 5)

**Log File**: `cpp_karrierewege_100k_profiling_5401362.log`

**Configuration**:
- `use_skill_description`: True
- `top_k_skills`: 15
- `encoder_skill`: `/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/best-model`
- `use_structured`: False

**Best Results**:
- **Best Trial**: 1
- **Best Validation MRR**: **0.4640**
- **Optimal Training Epochs**: 7

**Best Hyperparameters**:
- `hidden_dim`: 512
- `n_layers`: 3
- `dropout`: 0.1589
- `use_modality_weights`: Not specified (likely False)
- `lr`: 7.157e-04
- `weight_decay`: 1.249e-04

---

### Setup 13: With Skill Descriptions (top_k=15, encoder=best-model, older run 6)

**Log File**: `cpp_karrierewege_100k_5400594.log`

**Configuration**:
- `use_skill_description`: True
- `top_k_skills`: 15
- `encoder_skill`: `/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/best-model`
- `use_structured`: False

**Best Results**:
- **Best Trial**: 4
- **Best Validation MRR**: **0.4757**
- **Optimal Training Epochs**: 22

**Best Hyperparameters**:
- `hidden_dim`: 512
- `n_layers`: 3
- `dropout`: 0.1168
- `use_modality_weights`: Not specified (likely False)
- `lr`: 2.493e-04
- `weight_decay`: 2.121e-06

---

## Key Findings

### Best Overall Performance
- **Highest MRR**: **0.4857** (Setup 3)
- **Configuration**: With skill descriptions, top_k=15, encoder=best-model
- **Key Hyperparameters**: 
  - `hidden_dim`: 1024
  - `n_layers`: 1
  - `dropout`: 0.0875
  - `use_modality_weights`: False
  - `lr`: 2.285e-04

### Common Patterns

1. **Hidden Dimension**: Most best-performing setups use `hidden_dim=1024`, though some use 512 or 768
2. **Number of Layers**: Typically 1-3 layers, with 1 layer being common in top performers
3. **Dropout**: Varies widely (0.037-0.4037), suggesting the optimal value is highly dependent on other hyperparameters
4. **Learning Rate**: Generally in the range 1e-05 to 7e-04, with most around 2e-05 to 4e-05
5. **Weight Decay**: Very small values (1e-06 to 1e-04), indicating minimal regularization needed
6. **Modality Weights**: Mixed results - some best setups use it (True), others don't (False)

### Configuration Impact

- **Skill Descriptions**: Including skill descriptions generally improves performance (best MRR: 0.4857 vs 0.4807 without)
- **Top-k Skills**: Using 15 skills appears optimal (vs 20)
- **Encoder Model**: The `best-model` encoder performs best overall
- **Structured Features**: Including structured features (Setup 6) shows slightly lower performance (0.4790)

## Recommendations

For future experiments, consider:
1. **Primary Configuration**: Use skill descriptions with top_k=15 and the `best-model` encoder
2. **Hyperparameter Ranges**:
   - `hidden_dim`: [512, 768, 1024] - prioritize 1024
   - `n_layers`: [1, 2, 3] - start with 1
   - `dropout`: [0.05, 0.15] - explore around 0.09
   - `lr`: [1e-05, 5e-04] - focus on 2e-04 to 3e-04
   - `weight_decay`: [1e-06, 1e-04] - keep small
   - `use_modality_weights`: Try both True and False

## Notes

- All experiments used 10 Optuna trials
- Validation was performed on a 10% sample of the validation set
- Early stopping was used with patience=3
- Mixed precision training was enabled
- Batch size was typically 4096 (except Setup 6 with 1024)

---

*Last Updated: Based on log files from experiments/logs/cpp_karrierewege_100k_*.log*
