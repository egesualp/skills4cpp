#!/bin/bash
#SBATCH --job-name=embedding_finetuning        # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/embedding_finetuning_%j.log    # Log output (%j expands to job ID)
#SBATCH --error=experiments/logs/embedding_finetuning_%j.err     # Error output
#SBATCH --partition=mcml-dgx-a100-40x8  # Specify the partition to use
#SBATCH --gres=gpu:1                     # Number of GPUs to use
#SBATCH --ntasks=8                       # Number of tasks
#SBATCH --time=03:00:00                   # Time limit

# Set paths
source /dss/dsshome1/02/ra95kix2/miniconda3/bin/activate thesis

python -m src.cpp.finetune_last_job_skills \
  --data_type karrierewege_100k \
  --job_title_skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/results/karrierewege_esco_100k_esco_ground_truth/job_title_skills_master.csv \
  --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
  --model_name ElenaSenger/career-path-representation-mpnet-karrierewege \
  --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models \
  --top_k_skills 10 \
  --epochs 1 \
  --batch_size 32 \
  --learning_rate 2e-5 \
  --epoch_eval_frac 0.1