#!/bin/bash
#SBATCH --job-name=cpp_decorte_esco_skills_only        # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/cpp_decorte_esco_skills_only_%j.log    # Log output (%j expands to job ID)
#SBATCH --error=experiments/logs/cpp_decorte_esco_skills_only_%j.err     # Error output
#SBATCH --partition=mcml-dgx-a100-40x8  # Specify the partition to use
#SBATCH --gres=gpu:1                     # Number of GPUs to use
#SBATCH --ntasks=8                       # Number of tasks
#SBATCH --time=03:00:00                   # Time limit



#
# This script runs the full 4-step evaluation for the hybrid skill model.
# It runs:
#   1. Baseline with text history
#   2. Baseline with text history and skills
#   3. Baseline with skills only
#   4. Baseline with text history and skills (structured)
#
# It will exit immediately if any command fails.
#
source /dss/dsshome1/02/ra95kix2/miniconda3/bin/activate thesis
# --- 1. Define Variables ---
# Path to your python module

# Run the python module
echo "Running baseline with skills only"
python -m src.cpp.train_cpp_enhanced \
    --data_type decorte_esco \
    --master_skill_file results/decorte_esco_ground_truth/job_title_skills_master.csv \
    --encoder_text "ElenaSenger/career-path-representation-mpnet-decorte-esco" \
    --max_epochs 100 \
    --batch_size 4096 \
    --output_dir results/cpp/decorte_esco_skills_only \
    --use_skill_text \
    --use_skill_description \
    --run_name test_skills_only \
    --use_wandb \
    --wandb_project "cpp-decorte-esco-skills" \
    --optuna \
    --n_trials 20 \
    --patience 3 \
    --num_workers 0 \
    --pooling_strategy weighted_idf

python -m src.cpp.train_cpp_enhanced \
    --data_type decorte_esco \
    --master_skill_file results/decorte_esco_ground_truth/job_title_skills_master_essential.csv \
    --encoder_text "ElenaSenger/career-path-representation-mpnet-decorte-esco" \
    --max_epochs 100 \
    --batch_size 4096 \
    --output_dir results/cpp/decorte_esco_skills_only_essential \
    --use_skill_text \
    --use_skill_description \
    --run_name test_skills_only_essential \
    --use_wandb \
    --wandb_project "cpp-decorte-esco-skills" \
    --optuna \
    --n_trials 20 \
    --patience 3 \
    --num_workers 0 \
    --pooling_strategy weighted_idf

python -m src.cpp.train_cpp_enhanced \
    --data_type decorte_esco \
    --master_skill_file results/decorte_esco_ground_truth/job_title_skills_master.csv \
    --encoder_text "ElenaSenger/career-path-representation-mpnet-decorte-esco" \
    --max_epochs 100 \
    --batch_size 4096 \
    --output_dir results/cpp/decorte_esco_skills_only_mean \
    --use_skill_text \
    --use_skill_description \
    --run_name test_skills_only_mean \
    --use_wandb \
    --wandb_project "cpp-decorte-esco-skills" \
    --optuna \
    --n_trials 20 \
    --patience 3 \
    --num_workers 0 \
    --pooling_strategy mean

python -m src.cpp.train_cpp_enhanced \
    --data_type decorte_esco \
    --master_skill_file results/decorte_esco_ground_truth/job_title_skills_master_essential.csv \
    --encoder_text "ElenaSenger/career-path-representation-mpnet-decorte-esco" \
    --max_epochs 100 \
    --batch_size 4096 \
    --output_dir results/cpp/decorte_esco_skills_only_mean_essential \
    --use_skill_text \
    --use_skill_description \
    --run_name test_skills_only_mean_essential \
    --use_wandb \
    --wandb_project "cpp-decorte-esco-skills" \
    --optuna \
    --n_trials 20 \
    --patience 3 \
    --num_workers 0 \
    --pooling_strategy mean

python -m src.cpp.train_cpp_enhanced \
    --data_type decorte_esco \
    --master_skill_file results/decorte_esco_ground_truth/job_title_skills_master.csv \
    --encoder_text "ElenaSenger/career-path-representation-mpnet-decorte-esco" \
    --encoder_skill "sentence-transformers/all-mpnet-base-v2" \
    --max_epochs 100 \
    --batch_size 4096 \
    --output_dir results/cpp/decorte_esco_skills_only_mpnet \
    --use_skill_text \
    --use_skill_description \
    --run_name test_skills_only_mpnet \
    --use_wandb \
    --wandb_project "cpp-decorte-esco-skills" \
    --optuna \
    --n_trials 20 \
    --patience 3 \
    --num_workers 0 \
    --pooling_strategy weighted_idf

python -m src.cpp.train_cpp_enhanced \
    --data_type decorte_esco \
    --master_skill_file results/decorte_esco_ground_truth/job_title_skills_master_essential.csv \
    --encoder_text "ElenaSenger/career-path-representation-mpnet-decorte-esco" \
    --encoder_skill "sentence-transformers/all-mpnet-base-v2" \
    --max_epochs 100 \
    --batch_size 4096 \
    --output_dir results/cpp/decorte_esco_skills_only_mpnet_essential \
    --use_skill_text \
    --use_skill_description \
    --run_name test_skills_only_mpnet_essential \
    --use_wandb \
    --wandb_project "cpp-decorte-esco-skills" \
    --optuna \
    --n_trials 20 \
    --patience 3 \
    --num_workers 0 \
    --pooling_strategy weighted_idf