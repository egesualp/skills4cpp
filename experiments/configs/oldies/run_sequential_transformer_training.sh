#!/bin/bash
#SBATCH --job-name=cpp_sequential_training_kw_100k        # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/sequential_transformer/cpp_sequential_training_kw_100k_%j.log    # Log output (%j expands to job ID)
#SBATCH --error=experiments/logs/sequential_transformer/cpp_sequential_training_kw_100k_%j.err     # Error output
#SBATCH --partition=mcml-dgx-a100-40x8  # Specify the partition to use
#SBATCH --gres=gpu:1                     # Number of GPUs to use
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=8                 # Number of CPUs per task
#SBATCH --time=04:00:00                   # Time limit


source /dss/dsshome1/02/ra95kix2/miniconda3/bin/activate thesis

python -m src.seq_transformer.train \
    --job_embeddings_path "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/embeddings/occupations_skill_brief_bge_v1.pt" \
    --data_type "karrierewege_100k" \
    --occupations_path "data/esco_datasets/occupations_en.csv" \
    --epochs 30 \
    --batch_size 512 \
    --output_dir "results/seq_transformer/run_02" \
    --combine_method "concat" \
    --use_wandb \
    --wandb_project "career-path-transformer" \
    --run_name "transformer_job_only_run_v2_final" \
    --use_all_subspans \
    --skip_hpo \
    --d_model 512 \
    --n_layers 3 \
    --n_heads 8 \
    --lr 7.031085421259402e-05 \
    --dropout 0.13797567636647634