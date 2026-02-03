#!/bin/bash
#SBATCH --job-name=embedding_finetuning        # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/embedding_finetuning_%j.log    # Log output (%j expands to job ID)
#SBATCH --error=experiments/logs/embedding_finetuning_%j.err     # Error output
#SBATCH --partition=mcml-dgx-a100-40x8  # Specify the partition to use
#SBATCH --gres=gpu:1                     # Number of GPUs to use
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=8                 # Number of CPUs per task
#SBATCH --time=05:00:00                   # Time limit

# Set paths
source /dss/dsshome1/02/ra95kix2/miniconda3/bin/activate thesis

# ==============================================================================
# DECORTE EXPERIMENTS - Job ID based skill loading
# ==============================================================================

# --- Experiment 1: scores_only mode with top-k selection ---
# Uses prediction scores from fused_predictions.json to select top-k skills
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.cpp.finetune_last_job_skills \
  --data_type decorte \
  --skill_scores_json /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_3/fused_predictions.json \
  --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
  --model_name Alibaba-NLP/gte-base-en-v1.5 \
  --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/final_models/decorte_scores_only_topk_20skills \
  --scoring_mode scores_only \
  --skill_selection_strategy top_k \
  --top_k_skills 20 \
  --epochs 3 \
  --batch_size 32 \
  --learning_rate 3e-5 \
  --epoch_eval_frac 0.1 \
  --use_wandb \
  --wandb_project "decorte_skills_embedding" \
  --run_name "decorte_scores_only_topk_20skills" \
  --scheduler "cosine" \
  --test_strategy final \
  --dataloader_num_workers 0 \
  --gradient_accumulation_steps 2 \
  --print_sample

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.cpp.finetune_last_job_skills \
  --data_type decorte \
  --skill_scores_json /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_3/fused_predictions.json \
  --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
  --model_name ElenaSenger/career-path-representation-mpnet-decorte \
  --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/decorte_scores_only_topk_20skills_es_mpnet \
  --scoring_mode scores_only \
  --skill_selection_strategy top_k \
  --top_k_skills 20 \
  --epochs 3 \
  --batch_size 32 \
  --learning_rate 3e-5 \
  --epoch_eval_frac 0.1 \
  --use_wandb \
  --wandb_project "decorte_skills_embedding" \
  --run_name "decorte_scores_only_topk_20skills_es_mpnet" \
  --scheduler "cosine" \
  --test_strategy final \
  --dataloader_num_workers 0 \
  --gradient_accumulation_steps 2 \
  --print_sample

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.cpp.finetune_last_job_skills \
  --data_type decorte \
  --skill_scores_json /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_3/fused_predictions.json \
  --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
  --model_name Alibaba-NLP/gte-base-en-v1.5 \
  --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/decorte_scores_only_topk_40skills \
  --scoring_mode scores_only \
  --skill_selection_strategy top_k \
  --top_k_skills 40 \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 3e-5 \
  --epoch_eval_frac 0.1 \
  --use_wandb \
  --wandb_project "decorte_skills_embedding" \
  --run_name "decorte_scores_only_topk_40skills" \
  --scheduler "cosine" \
  --test_strategy final \
  --dataloader_num_workers 0 \
  --gradient_accumulation_steps 4

# --- Experiment 2: scores_only mode with stratified selection ---
 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
 python -m src.cpp.finetune_last_job_skills \
   --data_type decorte \
   --skill_scores_json /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_3/fused_predictions.json \
   --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
   --model_name ElenaSenger/career-path-representation-mpnet-decorte \
   --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/decorte_scores_only_stratified_es_mpnet \
   --scoring_mode scores_only \
   --skill_selection_strategy stratified \
   --top_k_skills 20 \
   --epochs 3 \
   --batch_size 32 \
   --learning_rate 3e-5 \
   --epoch_eval_frac 0.1 \
   --use_wandb \
   --wandb_project "decorte_skills_embedding" \
   --run_name "decorte_scores_only_stratified_es_mpnet" \
   --scheduler "cosine" \
   --test_strategy final \
   --dataloader_num_workers 0 \
   --gradient_accumulation_steps 2

  # --- Experiment 2.1: weighted mode with stratified selection ---
 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
 python -m src.cpp.finetune_last_job_skills \
   --data_type decorte \
   --skill_scores_json /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_3/fused_predictions.json \
   --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
   --model_name ElenaSenger/career-path-representation-mpnet-decorte \
   --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/decorte_weighted_stratified_es_mpnet \
   --scoring_mode weighted \
   --skill_selection_strategy stratified \
   --top_k_skills 20 \
   --epochs 3 \
   --batch_size 32 \
   --learning_rate 3e-5 \
   --epoch_eval_frac 0.1 \
   --use_wandb \
   --wandb_project "decorte_skills_embedding" \
   --run_name "decorte_weighted_stratified_es_mpnet" \
   --scheduler "cosine" \
   --test_strategy final \
   --dataloader_num_workers 0 \
   --gradient_accumulation_steps 2

 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
 python -m src.cpp.finetune_last_job_skills \
   --data_type decorte \
   --skill_scores_json /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_3/fused_predictions.json \
   --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
   --model_name Alibaba-NLP/gte-base-en-v1.5 \
   --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/decorte_weighted_stratified_gte_base \
   --scoring_mode weighted \
   --skill_selection_strategy stratified \
   --top_k_skills 20 \
   --epochs 3 \
   --batch_size 32 \
   --learning_rate 3e-5 \
   --epoch_eval_frac 0.1 \
   --use_wandb \
   --wandb_project "decorte_skills_embedding" \
   --run_name "decorte_weighted_stratified_gte_base" \
   --scheduler "cosine" \
   --test_strategy final \
   --dataloader_num_workers 0 \
   --gradient_accumulation_steps 2

# --- Experiment 3: weighted mode (IDF + scores) with top-k ---
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
 python -m src.cpp.finetune_last_job_skills \
   --data_type decorte \
   --skill_scores_json /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_3/fused_predictions.json \
   --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
   --model_name ElenaSenger/career-path-representation-mpnet-decorte \
   --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/decorte_weighted_topk_40skills_es_mpnet_no_desc \
   --scoring_mode weighted \
   --importance_weight 0.5 \
   --skill_selection_strategy top_k \
   --top_k_skills 40 \
   --epochs 3 \
   --batch_size 32 \
   --learning_rate 3e-5 \
   --epoch_eval_frac 0.1 \
   --use_wandb \
   --wandb_project "decorte_skills_embedding" \
   --run_name "decorte_weighted_0.5_topk_40skills_es_mpnet_no_desc" \
   --scheduler "cosine" \
   --test_strategy final \
   --dataloader_num_workers 0 \
   --gradient_accumulation_steps 2 \
   --print_sample \
   --no_skill_descriptions

   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
 python -m src.cpp.finetune_last_job_skills \
   --data_type decorte \
   --skill_scores_json /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_3/fused_predictions.json \
   --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
   --model_name Alibaba-NLP/gte-base-en-v1.5 \
   --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/decorte_weighted_topk_100skills_gte_base_no_desc \
   --scoring_mode weighted \
   --importance_weight 0.5 \
   --skill_selection_strategy top_k \
   --top_k_skills 100 \
   --epochs 3 \
   --batch_size 32 \
   --learning_rate 3e-5 \
   --epoch_eval_frac 0.1 \
   --use_wandb \
   --wandb_project "decorte_skills_embedding" \
   --run_name "decorte_weighted_0.5_topk_100skills_gte_base_no_desc" \
   --scheduler "cosine" \
   --test_strategy final \
   --dataloader_num_workers 0 \
   --gradient_accumulation_steps 2 \
   --print_sample \
   --no_skill_descriptions

# --- Experiment 4: idf_only mode (baseline) ---
 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
 python -m src.cpp.finetune_last_job_skills \
   --data_type decorte \
   --skill_scores_json /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_3/fused_predictions.json \
   --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
   --model_name ElenaSenger/career-path-representation-mpnet-decorte \
   --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/decorte_idf_only_topk_es_mpnet \
   --scoring_mode idf_only \
   --skill_selection_strategy top_k \
   --top_k_skills 20 \
   --epochs 3 \
   --batch_size 32 \
   --learning_rate 3e-5 \
   --epoch_eval_frac 0.1 \
   --use_wandb \
   --wandb_project "decorte_skills_embedding" \
   --run_name "decorte_idf_only_topk_es_mpnet" \
   --scheduler "cosine" \
   --test_strategy final \
   --dataloader_num_workers 0 \
   --gradient_accumulation_steps 2 \
   --print_sample

# ==============================================================================
# KARRIEREWEGE EXPERIMENTS - Original job title based skill loading (reference)
# ==============================================================================

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# python -m src.cpp.finetune_last_job_skills \
#   --data_type karrierewege_100k \
#   --job_title_skills_csv /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/karrierewege_esco_100k_esco_ground_truth/job_title_skills_master.csv \
#   --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
#   --model_name Alibaba-NLP/gte-base-en-v1.5 \
#   --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/stratified_with_descriptions \
#   --scoring_mode idf_only \
#   --top_k_skills 12 \
#   --epochs 3 \
#   --batch_size 32 \
#   --learning_rate 3e-5 \
#   --epoch_eval_frac 0.1 \
#   --use_wandb \
#   --wandb_project "decorte_skills_embedding" \
#   --run_name "karrierewege_idf_stratified_12skills" \
#   --scheduler "cosine" \
#   --test_strategy final \
#   --dataloader_num_workers 0 \
#   --gradient_accumulation_steps 2 \
#   --skill_selection_strategy stratified

