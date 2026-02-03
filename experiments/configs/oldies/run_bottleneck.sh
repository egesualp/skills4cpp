#!/bin/bash
#SBATCH --job-name=bottleneck        # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/bottleneck_%j.log    # Log output (%j expands to job ID)
#SBATCH --error=experiments/logs/bottleneck_%j.err     # Error output
#SBATCH --partition=mcml-dgx-a100-40x8  # Specify the partition to use
#SBATCH --gres=gpu:1                     # Number of GPUs to use
#SBATCH --ntasks=8                       # Number of tasks
#SBATCH --time=01:00:00                   # Time limit



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
echo "Running quick bottleneck test..."
python -m src.cpp.test_bottlenecks --quick