#!/bin/bash
#SBATCH --job-name=cpp_baselines        # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/cpp_baselines_%j.log    # Log output (%j expands to job ID)
#SBATCH --error=experiments/logs/cpp_baselines_%j.err     # Error output
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
    --max_epochs 15 \
    --batch_size 4096 \
    --output_dir results/test_skills_only \
    --use_skill_text \
    --use_skill_description \
    --run_name test_skills_only \
    --use_wandb \
    --wandb_project "cpp-baseline" \
    --optuna \
    --n_trials 10 \
    --patience 2 \
    --num_workers 0

echo "Running baseline with text history and skills (structured)"
python -m src.cpp.train_cpp_enhanced \
    --use_text_history \
    --max_epochs 15 \
    --batch_size 4096 \
    --output_dir results/test_baseline_w_skills_structured \
    --use_text_description \
    --use_skill_text \
    --use_skill_description \
    --use_structured \
    --run_name test_baseline_w_skills_structured \
    --use_wandb \
    --wandb_project "cpp-baseline" \
    --optuna \
    --n_trials 10 \
    --patience 2 \
    --num_workers 0