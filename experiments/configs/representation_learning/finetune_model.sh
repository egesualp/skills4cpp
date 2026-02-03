#!/bin/bash
#SBATCH --job-name=embedding_finetuning        # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/embedding_finetuning/decorte_embedding_final_%j.log    # Log output (%j expands to job ID)
#SBATCH --error=experiments/logs/embedding_finetuning/decorte_embedding_final_%j.err     # Error output
#SBATCH --partition=mcml-dgx-a100-40x8  # Specify the partition to use
#SBATCH --gres=gpu:1                     # Number of GPUs to use
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=8                 # Number of CPUs per task
#SBATCH --time=05:00:00                   # Time limit

# Set paths
source /dss/dsshome1/02/ra95kix2/miniconda3/bin/activate thesis

SKILL_SCORES_JSON="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_h2_pjmath/fused_predictions.json"
MASTER_OUTPUT_DIR="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/decorte_embedding_final"
WANDB_PROJECT="decorte_skills_embedding_mpnet" 
MODEL_NAME="ElenaSenger/career-path-representation-mpnet-decorte"
MODEL_NAME_GTE="Alibaba-NLP/gte-base-en-v1.5"
WANDB_PROJECT_GTE="decorte_skills_embedding_gte" 

# ==============================================================================
# DECORTE EXPERIMENTS - Job ID based skill loading
# ==============================================================================

# --- Round 1: decide on scoring method ---
# Round 1: Scoring mode (3 runs)
# Fix: GTE, k=20, desc=yes, selection=top-k

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.cpp.finetune_last_job_skills \
  --data_type decorte \
  --skill_scores_json ${SKILL_SCORES_JSON} \
  --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
  --model_name ${MODEL_NAME} \
  --output_dir ${MASTER_OUTPUT_DIR}/decorte_scores_only_topk_20skills_gte_base \
  --scoring_mode scores_only \
  --skill_selection_strategy top_k \
  --top_k_skills 20 \
  --epochs 3 \
  --batch_size 32 \
  --learning_rate 3e-5 \
  --epoch_eval_frac 0.1 \
  --use_wandb \
  --wandb_project ${WANDB_PROJECT} \
  --run_name "decorte_scores_only_topk_20skills_gte_base" \
  --scheduler "cosine" \
  --test_strategy final \
  --dataloader_num_workers 0 \
  --gradient_accumulation_steps 2 \
  --print_sample \
  --test_base_model

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.cpp.finetune_last_job_skills \
  --data_type decorte \
  --skill_scores_json ${SKILL_SCORES_JSON} \
  --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
  --model_name ${MODEL_NAME} \
  --output_dir ${MASTER_OUTPUT_DIR}/decorte_idf_only_topk_20skills_gte_base \
  --scoring_mode idf_only \
  --skill_selection_strategy top_k \
  --top_k_skills 20 \
  --epochs 3 \
  --batch_size 32 \
  --learning_rate 3e-5 \
  --epoch_eval_frac 0.1 \
  --use_wandb \
  --wandb_project ${WANDB_PROJECT} \
  --run_name "decorte_idf_only_topk_20skills_gte_base" \
  --scheduler "cosine" \
  --test_strategy final \
  --dataloader_num_workers 0 \
  --gradient_accumulation_steps 2 \
  --print_sample \
  --test_base_model

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.cpp.finetune_last_job_skills \
  --data_type decorte \
  --skill_scores_json ${SKILL_SCORES_JSON} \
  --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
  --model_name ${MODEL_NAME} \
  --output_dir ${MASTER_OUTPUT_DIR}/decorte_weighted_topk_20skills_gte_base \
  --scoring_mode weighted \
  --skill_selection_strategy top_k \
  --top_k_skills 20 \
  --epochs 3 \
  --batch_size 32 \
  --learning_rate 3e-5 \
  --epoch_eval_frac 0.1 \
  --use_wandb \
  --wandb_project ${WANDB_PROJECT} \
  --run_name "decorte_weighted_topk_20skills_gte_base" \
  --scheduler "cosine" \
  --test_strategy final \
  --dataloader_num_workers 0 \
  --gradient_accumulation_steps 2 \
  --print_sample \
  --test_base_model

  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.cpp.finetune_last_job_skills \
  --data_type decorte \
  --skill_scores_json ${SKILL_SCORES_JSON} \
  --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
  --model_name ${MODEL_NAME} \
  --output_dir ${MASTER_OUTPUT_DIR}/decorte_weighted_0_8_topk_20skills_gte_base \
  --scoring_mode weighted \
  --skill_selection_strategy top_k \
  --top_k_skills 20 \
  --epochs 3 \
  --batch_size 32 \
  --learning_rate 3e-5 \
  --epoch_eval_frac 0.1 \
  --use_wandb \
  --wandb_project ${WANDB_PROJECT} \
  --run_name "decorte_weighted_0_8_topk_20skills_gte_base" \
  --scheduler "cosine" \
  --test_strategy final \
  --dataloader_num_workers 0 \
  --gradient_accumulation_steps 2 \
  --print_sample \
  --test_base_model \
  --importance_weight 0.8

# --- Round 2: decide on selection method ---
# Round 2: Selection method (2 runs)
# Fix: GTE, k=20, desc=yes, selection=top-k

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.cpp.finetune_last_job_skills \
  --data_type decorte \
  --skill_scores_json ${SKILL_SCORES_JSON} \
  --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
  --model_name ${MODEL_NAME} \
  --output_dir ${MASTER_OUTPUT_DIR}/decorte_scores_only_stratified_20skills_gte_base \
  --scoring_mode scores_only \
  --skill_selection_strategy stratified \
  --top_k_skills 20 \
  --epochs 3 \
  --batch_size 32 \
  --learning_rate 3e-5 \
  --epoch_eval_frac 0.1 \
  --use_wandb \
  --wandb_project ${WANDB_PROJECT} \
  --run_name "decorte_scores_only_stratified_20skills_gte_base" \
  --scheduler "cosine" \
  --test_strategy final \
  --dataloader_num_workers 0 \
  --gradient_accumulation_steps 2 \
  --print_sample \
  --test_base_model

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.cpp.finetune_last_job_skills \
  --data_type decorte \
  --skill_scores_json ${SKILL_SCORES_JSON} \
  --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
  --model_name ${MODEL_NAME} \
  --output_dir ${MASTER_OUTPUT_DIR}/decorte_weighted_stratified_20skills_gte_base \
  --scoring_mode weighted \
  --skill_selection_strategy stratified \
  --top_k_skills 20 \
  --epochs 3 \
  --batch_size 32 \
  --learning_rate 3e-5 \
  --epoch_eval_frac 0.1 \
  --use_wandb \
  --wandb_project ${WANDB_PROJECT} \
  --run_name "decorte_weighted_stratified_20skills_gte_base" \
  --scheduler "cosine" \
  --test_strategy final \
  --dataloader_num_workers 0 \
  --gradient_accumulation_steps 2 \
  --print_sample \
  --test_base_model

# --- Round 3: decide on top-k skills ---
# Round 3: Top-k skills (2 runs)
# Fix: GTE, k=20, desc=yes, selection=top-k
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.cpp.finetune_last_job_skills \
  --data_type decorte \
  --skill_scores_json ${SKILL_SCORES_JSON} \
  --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
  --model_name ${MODEL_NAME} \
  --output_dir ${MASTER_OUTPUT_DIR}/decorte_weighted_0_8_topk_10skills_gte_base \
  --scoring_mode weighted \
  --skill_selection_strategy top_k \
  --top_k_skills 10 \
  --epochs 3 \
  --batch_size 32 \
  --learning_rate 3e-5 \
  --epoch_eval_frac 0.1 \
  --use_wandb \
  --wandb_project ${WANDB_PROJECT} \
  --run_name "decorte_weighted_0_8_topk_10skills_gte_base" \
  --scheduler "cosine" \
  --test_strategy final \
  --dataloader_num_workers 0 \
  --gradient_accumulation_steps 2 \
  --print_sample \
  --test_base_model \
  --importance_weight 0.8

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.cpp.finetune_last_job_skills \
  --data_type decorte \
  --skill_scores_json ${SKILL_SCORES_JSON} \
  --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
  --model_name ${MODEL_NAME} \
  --output_dir ${MASTER_OUTPUT_DIR}/decorte_weighted_0_8_topk_40skills_gte_base \
  --scoring_mode weighted \
  --skill_selection_strategy top_k \
  --top_k_skills 40 \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 3e-5 \
  --epoch_eval_frac 0.1 \
  --use_wandb \
  --wandb_project ${WANDB_PROJECT} \
  --run_name "decorte_weighted_0_8_topk_40skills_gte_base" \
  --scheduler "cosine" \
  --test_strategy final \
  --dataloader_num_workers 0 \
  --gradient_accumulation_steps 4 \
  --print_sample \
  --test_base_model \
  --importance_weight 0.8

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.cpp.finetune_last_job_skills \
  --data_type decorte \
  --skill_scores_json ${SKILL_SCORES_JSON} \
  --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
  --model_name ${MODEL_NAME} \
  --output_dir ${MASTER_OUTPUT_DIR}/decorte_weighted_0_8_topk_80skills_gte_base \
  --scoring_mode weighted \
  --skill_selection_strategy top_k \
  --top_k_skills 80 \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 3e-5 \
  --epoch_eval_frac 0.1 \
  --use_wandb \
  --wandb_project ${WANDB_PROJECT} \
  --run_name "decorte_weighted_0_8_topk_80skills_gte_base" \
  --scheduler "cosine" \
  --test_strategy final \
  --dataloader_num_workers 0 \
  --gradient_accumulation_steps 4 \
  --print_sample \
  --test_base_model \
  --importance_weight 0.8 \
  --save_strategy best \
  --save_model

# --- Round 4: decide on embedding model ---
# Round 4: Embedding model (2 runs)
# Fix: GTE, k=20, desc=yes, selection=top-k
MODEL_NAME = "ElenaSenger/career-path-representation-mpnet-decorte"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.cpp.finetune_last_job_skills \
  --data_type decorte \
  --skill_scores_json ${SKILL_SCORES_JSON} \
  --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
  --model_name ${MODEL_NAME} \
  --output_dir ${MASTER_OUTPUT_DIR}/decorte_weighted_0_8_topk_20skills_es_mpnet \
  --scoring_mode weighted \
  --skill_selection_strategy top_k \
  --top_k_skills 20 \
  --epochs 3 \
  --batch_size 32 \
  --learning_rate 3e-5 \
  --epoch_eval_frac 0.1 \
  --use_wandb \
  --wandb_project ${WANDB_PROJECT} \
  --run_name "decorte_weighted_0_8_topk_20skills_es_mpnet" \
  --scheduler "cosine" \
  --test_strategy final \
  --dataloader_num_workers 0 \
  --gradient_accumulation_steps 2 \
  --print_sample \
  --test_base_model \
  --importance_weight 0.8

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.cpp.finetune_last_job_skills \
  --data_type decorte \
  --skill_scores_json ${SKILL_SCORES_JSON} \
  --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
  --model_name ${MODEL_NAME} \
  --output_dir ${MASTER_OUTPUT_DIR}/decorte_weighted_0_8_topk_10skills_es_mpnet \
  --scoring_mode weighted \
  --skill_selection_strategy top_k \
  --top_k_skills 10 \
  --epochs 3 \
  --batch_size 32 \
  --learning_rate 3e-5 \
  --epoch_eval_frac 0.1 \
  --use_wandb \
  --wandb_project ${WANDB_PROJECT} \
  --run_name "decorte_weighted_0_8_topk_10skills_es_mpnet" \
  --scheduler "cosine" \
  --test_strategy final \
  --dataloader_num_workers 0 \
  --gradient_accumulation_steps 2 \
  --print_sample \
  --test_base_model \
  --importance_weight 0.8

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.cpp.finetune_last_job_skills \
  --data_type karrierewege_cp \
  --skill_scores_json ${SKILL_SCORES_JSON} \
  --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
  --model_name ${MODEL_NAME} \
  --output_dir ${MASTER_OUTPUT_DIR}/kw_cp_weighted_0_8_topk_10skills_es_mpnet \
  --scoring_mode weighted \
  --skill_selection_strategy top_k \
  --top_k_skills 10 \
  --epochs 3 \
  --batch_size 32 \
  --learning_rate 3e-5 \
  --epoch_eval_frac 0.1 \
  --use_wandb \
  --wandb_project ${WANDB_PROJECT} \
  --run_name "kw_cp_weighted_0_8_topk_10skills_es_mpnet" \
  --scheduler "cosine" \
  --test_strategy final \
  --dataloader_num_workers 0 \
  --gradient_accumulation_steps 2 \
  --print_sample \
  --test_base_model \
  --importance_weight 0.8

# --- Round 5 Final: decide on description ---
# Round 5: Description (2 runs)
# Fix: GTE, k=20, desc=yes, selection=top-k

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.cpp.finetune_last_job_skills \
  --data_type decorte \
  --skill_scores_json ${SKILL_SCORES_JSON} \
  --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
  --model_name ${MODEL_NAME} \
  --output_dir ${MASTER_OUTPUT_DIR}/decorte_weighted_0_8_topk_20skills_no_desc_es_mpnet \
  --scoring_mode weighted \
  --skill_selection_strategy top_k \
  --top_k_skills 20 \
  --epochs 3 \
  --batch_size 32 \
  --learning_rate 3e-5 \
  --epoch_eval_frac 0.1 \
  --use_wandb \
  --wandb_project ${WANDB_PROJECT} \
  --run_name "decorte_weighted_0_8_topk_20skills_no_desc_es_mpnet" \
  --scheduler "cosine" \
  --test_strategy final \
  --dataloader_num_workers 0 \
  --gradient_accumulation_steps 2 \
  --print_sample \
  --test_base_model \
  --importance_weight 0.8 \
  --no_skill_descriptions


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.cpp.finetune_last_job_skills \
  --data_type decorte \
  --skill_scores_json ${SKILL_SCORES_JSON} \
  --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
  --model_name ${MODEL_NAME} \
  --output_dir ${MASTER_OUTPUT_DIR}/decorte_weighted_topk_40skills_no_desc_es_mpnet \
  --scoring_mode weighted \
  --skill_selection_strategy top_k \
  --top_k_skills 40 \
  --epochs 3 \
  --batch_size 32 \
  --learning_rate 3e-5 \
  --epoch_eval_frac 0.1 \
  --use_wandb \
  --wandb_project ${WANDB_PROJECT} \
  --run_name "decorte_weighted_topk_40skills_no_desc_es_mpnet" \
  --scheduler "cosine" \
  --test_strategy final \
  --dataloader_num_workers 0 \
  --gradient_accumulation_steps 2 \
  --print_sample \
  --test_base_model \
  --no_skill_descriptions

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.cpp.finetune_last_job_skills \
  --data_type decorte \
  --skill_scores_json ${SKILL_SCORES_JSON} \
  --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
  --model_name ${MODEL_NAME} \
  --output_dir ${MASTER_OUTPUT_DIR}/decorte_weighted_0_8_topk_80skills_no_desc_es_mpnet \
  --scoring_mode weighted \
  --skill_selection_strategy top_k \
  --top_k_skills 80 \
  --epochs 3 \
  --batch_size 32 \
  --learning_rate 3e-5 \
  --epoch_eval_frac 0.1 \
  --use_wandb \
  --wandb_project ${WANDB_PROJECT} \
  --run_name "decorte_weighted_0_8_topk_80skills_no_desc_es_mpnet" \
  --scheduler "cosine" \
  --test_strategy final \
  --dataloader_num_workers 0 \
  --gradient_accumulation_steps 2 \
  --print_sample \
  --test_base_model \
  --importance_weight 0.8 \
  --no_skill_descriptions

# --- Round 5 Final: decide on description ---
# Round 5: Description (2 runs) GTE MODELS
# Fix: GTE, k=20, desc=yes, selection=top-k

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.cpp.finetune_last_job_skills \
  --data_type decorte \
  --skill_scores_json ${SKILL_SCORES_JSON} \
  --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
  --model_name ${MODEL_NAME_GTE} \
  --output_dir ${MASTER_OUTPUT_DIR}/decorte_weighted_0_8_topk_10skills_no_desc_gte_base \
  --scoring_mode weighted \
  --skill_selection_strategy top_k \
  --top_k_skills 10 \
  --epochs 3 \
  --batch_size 32 \
  --learning_rate 3e-5 \
  --epoch_eval_frac 0.1 \
  --use_wandb \
  --wandb_project ${WANDB_PROJECT_GTE} \
  --run_name "decorte_weighted_0_8_topk_10skills_no_desc_gte_base" \
  --scheduler "cosine" \
  --test_strategy final \
  --dataloader_num_workers 0 \
  --gradient_accumulation_steps 2 \
  --print_sample \
  --test_base_model \
  --importance_weight 0.8 \
  --no_skill_descriptions

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.cpp.finetune_last_job_skills \
  --data_type decorte \
  --skill_scores_json ${SKILL_SCORES_JSON} \
  --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
  --model_name ${MODEL_NAME_GTE} \
  --output_dir ${MASTER_OUTPUT_DIR}/decorte_weighted_topk_40skills_no_desc_gte_base \
  --scoring_mode weighted \
  --skill_selection_strategy top_k \
  --top_k_skills 40 \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 3e-5 \
  --epoch_eval_frac 0.1 \
  --use_wandb \
  --wandb_project ${WANDB_PROJECT_GTE} \
  --run_name "decorte_weighted_topk_40skills_no_desc_gte_base" \
  --scheduler "cosine" \
  --test_strategy final \
  --dataloader_num_workers 0 \
  --gradient_accumulation_steps 4 \
  --print_sample \
  --test_base_model \
  --importance_weight 0.8 \
  --no_skill_descriptions

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.cpp.finetune_last_job_skills \
  --data_type decorte \
  --skill_scores_json ${SKILL_SCORES_JSON} \
  --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
  --model_name ${MODEL_NAME_GTE} \
  --output_dir ${MASTER_OUTPUT_DIR}/decorte_weighted_topk_80skills_no_desc_gte_base \
  --scoring_mode weighted \
  --skill_selection_strategy top_k \
  --top_k_skills 80 \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 3e-5 \
  --epoch_eval_frac 0.1 \
  --use_wandb \
  --wandb_project ${WANDB_PROJECT_GTE} \
  --run_name "decorte_weighted_topk_80skills_no_desc_gte_base" \
  --scheduler "cosine" \
  --test_strategy final \
  --dataloader_num_workers 0 \
  --gradient_accumulation_steps 4 \
  --print_sample \
  --test_base_model \
  --importance_weight 0.8 \
  --no_skill_descriptions

### REAL FINAL

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.cpp.finetune_last_job_skills \
  --data_type decorte \
  --skill_scores_json ${SKILL_SCORES_JSON} \
  --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
  --model_name ${MODEL_NAME_GTE} \
  --output_dir ${MASTER_OUTPUT_DIR}/z_decorte_weighted_topk_60skills_subspans_gte_base \
  --scoring_mode weighted \
  --skill_selection_strategy top_k \
  --top_k_skills 60 \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 3e-5 \
  --epoch_eval_frac 0.1 \
  --use_wandb \
  --wandb_project ${WANDB_PROJECT_GTE} \
  --run_name "z_decorte_weighted_topk_60skills_subspans_gte_base" \
  --scheduler "cosine" \
  --test_strategy final \
  --dataloader_num_workers 0 \
  --gradient_accumulation_steps 4 \
  --print_sample \
  --test_base_model \
  --importance_weight 0.8 \
  --consider_subspans


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.cpp.finetune_last_job_skills \
  --data_type decorte \
  --skill_scores_json ${SKILL_SCORES_JSON} \
  --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
  --model_name ${MODEL_NAME_GTE} \
  --output_dir ${MASTER_OUTPUT_DIR}/z_decorte_weighted_topk_20skills_subspans_gte_base \
  --scoring_mode weighted \
  --skill_selection_strategy top_k \
  --top_k_skills 20 \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 3e-5 \
  --epoch_eval_frac 0.1 \
  --use_wandb \
  --wandb_project ${WANDB_PROJECT_GTE} \
  --run_name "z_decorte_weighted_topk_20skills_subspans_gte_base" \
  --scheduler "cosine" \
  --test_strategy final \
  --dataloader_num_workers 0 \
  --gradient_accumulation_steps 4 \
  --print_sample \
  --test_base_model \
  --importance_weight 0.8 \
  --consider_subspans


PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.cpp.finetune_last_job_skills \
  --data_type decorte \
  --skill_scores_json ${SKILL_SCORES_JSON} \
  --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
  --model_name ${MODEL_NAME_GTE} \
  --output_dir ${MASTER_OUTPUT_DIR}/z_decorte_weighted_topk_20skills_subspans_gte_base \
  --scoring_mode weighted \
  --skill_selection_strategy top_k \
  --top_k_skills 20 \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 3e-5 \
  --epoch_eval_frac 0.1 \
  --use_wandb \
  --wandb_project ${WANDB_PROJECT_GTE} \
  --run_name "z_decorte_weighted_topk_20skills_subspans_gte_base" \
  --scheduler "cosine" \
  --test_strategy final \
  --dataloader_num_workers 0 \
  --gradient_accumulation_steps 4 \
  --print_sample \
  --test_base_model \
  --importance_weight 0.8 \
  --consider_subspans
  

### STRATIFIED
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.cpp.finetune_last_job_skills \
  --data_type decorte \
  --skill_scores_json ${SKILL_SCORES_JSON} \
  --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
  --model_name ${MODEL_NAME} \
  --output_dir ${MASTER_OUTPUT_DIR}/decorte_stratified_idf_scores_only_20skills_es_mpnet \
  --skill_selection_strategy stratified \
  --stratified_sampling_basis idf_only \
  --top_k_skills 20 \
  --epochs 3 \
  --batch_size 32 \
  --learning_rate 3e-5 \
  --epoch_eval_frac 0.1 \
  --use_wandb \
  --wandb_project ${WANDB_PROJECT} \
  --run_name "decorte_stratified_idf_scores_only_20skills_es_mpnet" \
  --scheduler "cosine" \
  --test_strategy final \
  --dataloader_num_workers 0 \
  --gradient_accumulation_steps 2 \
  --print_sample \
  --test_base_model \
  --scoring_mode scores_only

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.cpp.finetune_last_job_skills \
  --data_type decorte \
  --skill_scores_json ${SKILL_SCORES_JSON} \
  --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
  --model_name ${MODEL_NAME_GTE} \
  --output_dir ${MASTER_OUTPUT_DIR}/decorte_stratified_idf_scores_only_20skills_gte_base \
  --skill_selection_strategy stratified \
  --stratified_sampling_basis idf_only \
  --top_k_skills 20 \
  --epochs 3 \
  --batch_size 32 \
  --learning_rate 3e-5 \
  --epoch_eval_frac 0.1 \
  --use_wandb \
  --wandb_project ${WANDB_PROJECT_GTE} \
  --run_name "decorte_stratified_idf_scores_only_20skills_gte_base" \
  --scheduler "cosine" \
  --test_strategy final \
  --dataloader_num_workers 0 \
  --gradient_accumulation_steps 2 \
  --print_sample \
  --test_base_model \
  --scoring_mode scores_only




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

