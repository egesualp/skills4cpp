#!/bin/bash
#SBATCH --job-name=map_job_to_esco        # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/taskA_mapping/map_job_to_esco_%j.log    # Log output (%j expands to job ID)
#SBATCH --error=experiments/logs/taskA_mapping/map_job_to_esco_%j.err     # Error output
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
# 1. Map job titles to ESCO occupations
#

source /dss/dsshome1/02/ra95kix2/miniconda3/bin/activate thesis

python skills4cpp/src/evaluate.py --cfg experiments/configs/final_dance/decorte_all_jobbert_expanded.yaml