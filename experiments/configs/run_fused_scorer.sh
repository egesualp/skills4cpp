#!/bin/bash
#SBATCH --job-name=fused_scorer        # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/skill_mapping/kw_cp_fused_scorer_%j.log    # Log output (%j expands to job ID)
#SBATCH --error=experiments/logs/skill_mapping/kw_cp_fused_scorer_%j.err     # Error output
#SBATCH --partition=lrz-cpu  # Specify the partition to use
#SBATCH --qos=cpu                     # Number of GPUs to use
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=8                 # Number of CPUs per task
#SBATCH --mem=128G                       # Memory per task
#SBATCH --time=03:00:00                   # Time limit

source /dss/dsshome1/02/ra95kix2/miniconda3/bin/activate thesis

python src/skill_mapping/v5/fused_scorer_chunked.py \
    --esco_dir data/esco_datasets \
    --label_encoder /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/kw_cp/isco_model_h2_soft_deep_larger_val_mpnet/label_encoder.json \
    --task_a /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/experiments/results/infer_kw_cp_desc_all_jobbert_final_2/predictions.jsonl \
    --task_b_parquet /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/kw_cp/kw_cp_jobbert/similarity_scores_chunked.parquet \
    --isco_preds /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/kw_cp/isco_model_h2_soft_deep_larger_val_mpnet/inference_results_soft_labels_deep_larger_val_mpnet_kw_cp/isco_predictions.json \
    --decorte_map /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/karrierewege_plus_cp_master_3.csv \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/kw_cp/v5_fused_scorer/multiplicative_h2_kw_cp \
    --fusion_strategy multiplicative \
    --isco_level 2 \
    --chunk_size 5000 