#!/bin/bash

# Experiment 1: Multiplicative Fusion
# Uses grid search over alpha (exponent) and gamma (ISCO exponent)
python src/skill_mapping/v5/fused_scorer.py \
    --esco_dir data/esco_datasets \
    --label_encoder /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h2_soft_deep_larger_val_mpnet/label_encoder.json \
    --task_a /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/experiments/results/infer_decorte_all_jobbert_final_2/predictions.jsonl \
    --task_b /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/decorte_w_desc_2/similarity_scores.json \
    --isco_preds /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h2_soft_deep_larger_val_mpnet/inference_results_h2_decorte_master_2/isco_predictions.json \
    --decorte_map /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/decorte_master_2.csv \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative \
    --fusion_strategy multiplicative \
    --n_jobs 4

# Experiment 2: Linear Fusion
# Uses grid search over alpha (interpolation weight) and normalization methods
python src/skill_mapping/v5/fused_scorer.py \
    --esco_dir data/esco_datasets \
    --label_encoder /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h2_soft_deep_larger_val_mpnet/label_encoder.json \
    --task_a /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/experiments/results/infer_decorte_all_jobbert_final_2/predictions.jsonl \
    --task_b /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/decorte_w_desc_2/similarity_scores.json \
    --isco_preds /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h2_soft_deep_larger_val_mpnet/inference_results_h2_decorte_master_2/isco_predictions.json \
    --decorte_map /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/decorte_master_2.csv \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/linear \
    --fusion_strategy linear










