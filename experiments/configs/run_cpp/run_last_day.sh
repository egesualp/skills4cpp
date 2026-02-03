#!/bin/bash
#SBATCH --job-name=embedding_finetuning        # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/embedding_finetuning/embedding_finetuning_%j.log    # Log output (%j expands to job ID)
#SBATCH --error=experiments/logs/embedding_finetuning/embedding_finetuning_%j.err     # Error output
#SBATCH --partition=mcml-dgx-a100-40x8  # Specify the partition to use
#SBATCH --gres=gpu:1                     # Number of GPUs to use
#SBATCH --ntasks=8                       # Number of tasks
#SBATCH --time=02:00:00                   # Time limit



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

MASTER_OUTPUT_DIR="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/karrierewege_cp_embedding_final"
WANDB_PROJECT="karrierewege_cp_skills_embedding_mpnet" 
MODEL_NAME="ElenaSenger/career-path-representation-mpnet-karrierewege-cp"
MODEL_NAME_GTE="Alibaba-NLP/gte-base-en-v1.5"
WANDB_PROJECT_GTE="karrierewege_cp_skills_embedding_gte" 
MASTER_SKILL_FILE="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/kw_cp/v5_fused_scorer/linear_h2_kw_cp/fused_predictions.jsonl"
HF_HOME="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/.cache/huggingface"

#PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#python -m src.cpp.finetune_last_job_skills \
#  --data_type karrierewege_cp \
#  --skill_scores_json ${MASTER_SKILL_FILE} \
#  --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
#  --model_name ${MODEL_NAME} \
#  --output_dir ${MASTER_OUTPUT_DIR}/kw_cp_weighted_0_8_topk_10skills_es_mpnet \
#  --scoring_mode weighted \
#  --skill_selection_strategy top_k \
#  --top_k_skills 10 \
#  --epochs 3 \
#  --batch_size 32 \
#  --learning_rate 3e-5 \
#  --epoch_eval_frac 0.1 \
#  --use_wandb \
#  --wandb_project ${WANDB_PROJECT} \
#  --run_name "kw_cp_weighted_0_8_topk_10skills_es_mpnet" \
#  --scheduler "cosine" \
#  --test_strategy final \
#  --dataloader_num_workers 0 \
#  --gradient_accumulation_steps 2 \
#  --print_sample \
#  --test_base_model \
#  --importance_weight 0.8 \
#  --save_model

#PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#python -m src.cpp.finetune_last_job_skills \
#  --data_type karrierewege_cp \
#  --skill_scores_json ${MASTER_SKILL_FILE} \
#  --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
#  --model_name ${MODEL_NAME} \
#  --output_dir ${MASTER_OUTPUT_DIR}/kw_cp_weighted_0_8_topk_10skills_es_mpnet \
#  --scoring_mode weighted \
#  --skill_selection_strategy top_k \
#  --top_k_skills 10 \
#  --epochs 3 \
#  --batch_size 32 \
#  --learning_rate 3e-5 \
#  --epoch_eval_frac 0.1 \
#  --use_wandb \
#  --wandb_project ${WANDB_PROJECT} \
#  --run_name "kw_cp_weighted_0_8_topk_10skills_es_mpnet" \
#  --scheduler "cosine" \
#  --test_strategy final \
#  --dataloader_num_workers 0 \
#  --gradient_accumulation_steps 2 \
#  --print_sample \
#  --test_base_model \
#  --importance_weight 0.8 \
#  --save_model

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.cpp.finetune_last_job_skills \
  --data_type karrierewege_cp \
  --skill_scores_json ${MASTER_SKILL_FILE} \
  --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
  --model_name ${MODEL_NAME_GTE} \
  --output_dir ${MASTER_OUTPUT_DIR}/kw_cp_weighted_0_8_topk_60skills_gte_base \
  --scoring_mode weighted \
  --skill_selection_strategy top_k \
  --top_k_skills 60 \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 3e-5 \
  --epoch_eval_frac 0.1 \
  --use_wandb \
  --wandb_project ${WANDB_PROJECT} \
  --run_name "kw_cp_weighted_0_8_topk_60skills_gte_base" \
  --scheduler "cosine" \
  --test_strategy final \
  --dataloader_num_workers 0 \
  --gradient_accumulation_steps 4 \
  --print_sample \
  --test_base_model \
  --importance_weight 0.8 \
  --save_model