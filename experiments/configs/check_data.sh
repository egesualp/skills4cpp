#!/bin/bash
#SBATCH --job-name=data_augmentation        # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/data_augmentation_%j.log    # Log output (%j expands to job ID)
#SBATCH --error=experiments/logs/data_augmentation_%j.err     # Error output
#SBATCH --partition=lrz-cpu  # Specify the partition to use
#SBATCH --qos=cpu                     # Number of GPUs to use
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=8                 # Number of CPUs per task
#SBATCH --mem=16GB                       # Memory per task
#SBATCH --time=04:00:00                   # Time limit

source /dss/dsshome1/02/ra95kix2/miniconda3/bin/activate thesis
python -m src.llm_augmentation.augment_data