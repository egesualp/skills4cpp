#!/bin/bash
#SBATCH --job-name=cpp_decorte_esco        # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/cpp_decorte_esco_%j.log    # Log output (%j expands to job ID)
#SBATCH --error=experiments/logs/cpp_decorte_esco_%j.err     # Error output
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
# Run the python module
echo "Running baseline with text history"
python -m src.cpp.train_cpp_enhanced \
    --data_type decorte_esco \
    --master_skill_file results/decorte_esco_ground_truth/job_title_skills_master.csv \
    --encoder_text "ElenaSenger/career-path-representation-mpnet-decorte-esco" \
    --use_text_history \
    --max_epochs 100 \
    --batch_size 4096 \
    --output_dir results/cpp/decorte_esco_baseline \
    --use_text_description \
    --run_name test_baseline_2 \
    --use_wandb \
    --wandb_project "cpp-decorte-esco" \
    --optuna \
    --n_trials 30 \
    --patience 3 \
    --num_workers 0 \
    --pooling_strategy weighted_idf \
    --optimizer adam

echo "Running baseline with text history and skills"
python -m src.cpp.train_cpp_enhanced \
    --data_type decorte_esco \
    --master_skill_file results/decorte_esco_ground_truth/job_title_skills_master.csv \
    --encoder_text "ElenaSenger/career-path-representation-mpnet-decorte" \
    --use_text_history \
    --max_epochs 100 \
    --batch_size 4096 \
    --output_dir results/cpp/decorte_esco_baseline_w_skills \
    --use_text_description \
    --use_skill_text \
    --use_skill_description \
    --run_name test_baseline_w_skills_2 \
    --use_wandb \
    --wandb_project "cpp-decorte-esco" \
    --optuna \
    --n_trials 30 \
    --patience 3 \
    --num_workers 0 \
    --pooling_strategy weighted_idf \
    --use_advanced \
    --optimizer sgd

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
    --run_name test_skills_only_2 \
    --use_wandb \
    --wandb_project "cpp-decorte-esco" \
    --optuna \
    --n_trials 30 \
    --patience 3 \
    --num_workers 0 \
    --pooling_strategy weighted_idf \
    --optimizer sgd

echo "Running baseline with text history and skills"
python -m src.cpp.train_cpp_enhanced \
    --data_type decorte_esco \
    --master_skill_file results/decorte_esco_ground_truth/job_title_skills_master.csv \
    --encoder_text "ElenaSenger/career-path-representation-mpnet-decorte-esco" \
    --use_text_history \
    --max_epochs 100 \
    --batch_size 4096 \
    --output_dir results/cpp/decorte_esco_baseline_w_skills \
    --use_text_description \
    --use_skill_text \
    --use_skill_description \
    --run_name test_baseline_w_skills_advanced \
    --use_wandb \
    --wandb_project "cpp-decorte-esco" \
    --optuna \
    --n_trials 30 \
    --patience 3 \
    --num_workers 0 \
    --pooling_strategy weighted_idf \
    --optimizer sgd \
    --use_advanced


python -m src.cpp.train_cpp_enhanced \
    --data_type decorte_esco \
    --master_skill_file results/decorte_esco_ground_truth/job_title_skills_master.csv \
    --encoder_text "ElenaSenger/career-path-representation-mpnet-decorte-esco" \
    --use_text_history \
    --max_epochs 100 \
    --batch_size 4096 \
    --output_dir results/cpp/decorte_esco_baseline_w_skills \
    --use_text_description \
    --use_skill_text \
    --use_skill_description \
    --run_name test_baseline_w_skills_static \
    --use_wandb \
    --wandb_project "cpp-decorte-esco" \
    --hidden_dim 512 \
    --n_layers 1 \
    --dropout 0.35 \
    --lr 0.000668890 \
    --num_workers 0 \
    --pooling_strategy weighted_idf \
    --optimizer adam \
    --use_advanced \
    --patience 3