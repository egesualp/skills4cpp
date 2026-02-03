#!/bin/bash
#SBATCH --job-name=skill_isco        # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/category_classifier/skill_isco_%j.log    # Log output (%j expands to job ID)
#SBATCH --error=experiments/logs/category_classifier/skill_isco_%j.err     # Error output
#SBATCH --partition=mcml-dgx-a100-40x8  # Specify the partition to use
#SBATCH --gres=gpu:1                     # Number of GPUs to use
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=8                 # Number of CPUs per task
#SBATCH --time=1:00:00                   # Time limit (increased for multiple runs)

#
# Static Hyperparameter Ablation Study for Career Path Prediction
# Data: Karrierewege 100k
#
# Runs:
# 1. Job Titles Only
# 2. Job Titles + Descriptions
# 3. Skills Only (Names, IDF)
# 4. Skills Only (Names + Desc, IDF)
# 5. Multimodal: Job Titles + Skills (Names, IDF)
# 6. Multimodal: Job Titles + Skills (Names, Log Pooling)
# 7. V3: Last Job Skills Only (Names, IDF, Top-10)
# 8. V3: Last Job Skills Only (Names + Desc, IDF, Top-10)
#

source /dss/dsshome1/02/ra95kix2/miniconda3/bin/activate thesis

# Common Configuration
DATA_TYPE="decorte"
MASTER_SKILL_FILE="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/linear_fusion_sum/best_fused_scores.json"
ENCODER_TEXT="ElenaSenger/career-path-representation-mpnet-decorte"
ENCODER_SKILL="ElenaSenger/career-path-representation-mpnet-decorte"
OUTPUT_BASE="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte_static"
WANDB_PROJECT="cpp-decorte-static"

python -m skill_mapping.v2.isco_trainer \
    --esco_path ./data/esco_datasets/occupations_en.csv \
    --train_path /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/karrierewege_plus_cp_master_2.csv \
    --test_path /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/category_test_split_isco.csv \
    --model_path ElenaSenger/career-path-representation-mpnet-karrierewege-cp \
    --isco_level 2 \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h2_soft_deep_larger_val_mpnet_kw_cp \
    --cache_name soft_labels_deep_larger_val_mpnet_kw_cp \
    --n_trials 30 \
    --max_epochs 50 \
    --final_epochs 100 \
    --device cuda

python -m skill_mapping.v2.isco_trainer \
    --esco_path ./data/esco_datasets/occupations_en.csv \
    --train_path /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/karrierewege_plus_cp_master_2.csv \
    --test_path /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/category_test_split_isco.csv \
    --model_path ElenaSenger/career-path-representation-mpnet-decorte \
    --isco_level 2 \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h2_soft_deep_larger_val_mpnet_kw_cp_decorte \
    --cache_name soft_labels_deep_larger_val_mpnet_kw_cp_decorte \
    --n_trials 30 \
    --max_epochs 50 \
    --final_epochs 100 \
    --device cuda

python -m skill_mapping.v2.isco_trainer \
    --esco_path ./data/esco_datasets/occupations_en.csv \
    --test_path /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/category_test_split_isco.csv \
    --model_path pj-mathematician/JobSkillBGE-large-en-v1.5 \
    --isco_level 1 \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h1_soft_deep_larger_val \
    --cache_name soft_labels_deep_larger_val \
    --n_trials 30 \
    --max_epochs 50 \
    --final_epochs 100 \
    --device cuda

python -m skill_mapping.v2.isco_trainer \
    --esco_path ./data/esco_datasets/occupations_en.csv \
    --test_path /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/category_test_split_isco.csv \
    --model_path pj-mathematician/JobSkillBGE-large-en-v1.5 \
    --isco_level 3 \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h3_soft_deep_larger_val \
    --cache_name soft_labels_deep_larger_val \
    --n_trials 30 \
    --max_epochs 50 \
    --final_epochs 100 \
    --device cuda

python -m skill_mapping.v2.isco_trainer \
    --esco_path ./data/esco_datasets/occupations_en.csv \
    --test_path /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/category_test_split_isco.csv \
    --model_path pj-mathematician/JobSkillBGE-large-en-v1.5 \
    --isco_level 4 \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h3_soft_deep_larger_val \
    --cache_name soft_labels_deep_larger_val \
    --n_trials 30 \
    --max_epochs 50 \
    --final_epochs 100 \
    --device cuda