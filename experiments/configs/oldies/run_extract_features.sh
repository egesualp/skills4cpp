#!/bin/bash
#SBATCH --job-name=cpp_karrierewege_100k_profiling        # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/cpp_karrierewege_100k_profiling_%j.log    # Log output (%j expands to job ID)
#SBATCH --error=experiments/logs/cpp_karrierewege_100k_profiling_%j.err     # Error output
#SBATCH --partition=mcml-dgx-a100-40x8  # Specify the partition to use
#SBATCH --gres=gpu:1                     # Number of GPUs to use
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=8                 # Number of CPUs per task
#SBATCH --time=04:00:00                   # Time limit


source /dss/dsshome1/02/ra95kix2/miniconda3/bin/activate thesis

python -m src.seq_transformer.extract_features \
    --model_name "pj-mathematician/JobSkillBGE-large-en-v1.5" \
    --checkpoint_subfolder "checkpoint-4480" \
    --template "{{ skill_brief }}" \
    --input_path data/processed/augmentation/augmented_esco_occupations_2.csv \
    --id_column "conceptUri" \
    --output_path /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/embeddings/occupations_skill_brief_bge_v1.pt \
