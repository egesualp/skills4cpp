#!/bin/bash
#SBATCH --job-name=cpp_pjmath_skills        # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/cpp_decorte/cpp_pjmath_skills_%j.log    # Log output (%j expands to job ID)
#SBATCH --error=experiments/logs/cpp_decorte/cpp_pjmath_skills_%j.err     # Error output
#SBATCH --partition=lrz-cpu  # Specify the partition to use
#SBATCH --qos=cpu                     # Number of GPUs to use
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=8                 # Number of CPUs per task
#SBATCH --mem=16GB                       # Memory per task
#SBATCH --time=02:00:00                   # Time limit

#
# Static Hyperparameter Ablation Study for Career Path Prediction
# Data: Karrierewege 100k
#
# Runs:
# 1. Job Titles Only
# 2. Job Titles + Descriptions
# 3. Skills Only (Names, IDF)
# 4. Skills Only (Names + Desc, IDF)
# 5. Multimodal: Job Titles + Skills (Names, IDF)
# 6. Multimodal: Job Titles + Skills (Names, Log Pooling)
# 7. V3: Last Job Skills Only (Names, IDF, Top-10)
# 8. V3: Last Job Skills Only (Names + Desc, IDF, Top-10)
#

source /dss/dsshome1/02/ra95kix2/miniconda3/bin/activate thesis

# Common Configuration
DATA_TYPE="decorte"
MASTER_SKILL_FILE="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_h2_pjmath/fused_predictions.json"
ENCODER_TEXT="ElenaSenger/career-path-representation-mpnet-decorte"
ENCODER_SKILL="ElenaSenger/career-path-representation-mpnet-decorte"
OUTPUT_BASE="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte"
WANDB_PROJECT="cpp-decorte-optuna-final"

#MASTER_SKILL_FILE="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_3/fused_predictions.json"


# Static Hyperparameters
BATCH_SIZE=16
EVAL_BATCH_SIZE=32
MAX_EPOCHS=50
PATIENCE=2
LR=2e-5
HIDDEN_DIM=512
N_LAYERS=1
DROPOUT=0.1
USE_MODALITY_WEIGHTS=true

echo "Starting Ablation Study..."
echo "Data: $DATA_TYPE"
echo "Project: $WANDB_PROJECT"


# -----------------------------------------------------------------------------
# 1. Job Titles with desc
# -----------------------------------------------------------------------------
echo -e "\n\n=== Running Experiment 1: Job Titles Only ==="
python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --skill_embeddings_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index \
    --skill_scores_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --output_dir "${OUTPUT_BASE}/job_title_only" \
    --run_name "job_title_only" \
    --use_text_history \
    --batch_size 16 \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --num_workers 4 \
    --eval_clean_test \
    --save_scores \
    --normalize_input \
    --early_stop_metric loss \
    --seed 42 \
    --optuna


python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --skill_embeddings_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index \
    --skill_scores_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --output_dir "${OUTPUT_BASE}/job_title_desc" \
    --run_name "job_title_desc" \
    --use_text_history \
    --use_text_description \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --num_workers 4 \
    --eval_clean_test \
    --save_scores \
    --normalize_input \
    --early_stop_metric loss \
    --seed 42 \
    --optuna

# -----------------------------------------------------------------------------
# 3. Job Titles with Pooled Skills
# -----------------------------------------------------------------------------

python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --skill_embeddings_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index \
    --skill_scores_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --output_dir "${OUTPUT_BASE}/job_titles_skills_pjmath_desc_weighted_idf_logpool_adv" \
    --run_name "job_titles_skills_pjmath_desc_weighted_idf_logpool_adv" \
    --use_text_history \
    --use_text_description \
    --use_skill_text \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --num_workers 4 \
    --eval_clean_test \
    --save_scores \
    --normalize_input \
    --early_stop_metric mrr \
    --seed 42 \
    --optuna \
    --pooling_strategy weighted_idf \
    --use_skill_path_log_pooling \
    --max_epochs 50 \
    --n_trials 30 \
    --use_advanced


python -m src.cpp.train_cpp_enhanced_v2_fixed \
    --data_type $DATA_TYPE \
    --skill_embeddings_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index \
    --skill_scores_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --output_dir "${OUTPUT_BASE}/job_titles_skills_pjmath_desc_weighted_idf_logpool_adv_infonce" \
    --run_name "job_titles_skills_pjmath_desc_weighted_idf_logpool_adv_infonce" \
    --use_text_history \
    --use_text_description \
    --use_skill_text \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --num_workers 4 \
    --eval_clean_test \
    --save_scores \
    --normalize_input \
    --early_stop_metric mrr \
    --seed 42 \
    --optuna \
    --pooling_strategy weighted_idf \
    --use_skill_path_log_pooling \
    --max_epochs 50 \
    --n_trials 30 \
    --use_advanced \
    --loss_type infonce \
    --temperature 0.07

python -m src.cpp.train_cpp_enhanced_v3 \
    --data_type $DATA_TYPE \
    --encoder_text $ENCODER_TEXT \
    --output_dir "${OUTPUT_BASE}/job_titles_skills_mpnet_desc_2" \
    --run_name "job_titles_skills_mpnet_desc_2" \
    --use_text_history \
    --use_skill_text \
    --use_text_description \
    --use_skill_description \
    --max_epochs $MAX_EPOCHS \
    --patience $PATIENCE \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM \
    --n_layers $N_LAYERS \
    --dropout $DROPOUT \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --patience $PATIENCE \
    --num_workers 4 \
    --eval_clean_test \
    --normalize_input \
    --early_stop_metric loss \
    --use_advanced \
    --seed 42 \
    --top_k_skills 10 \
    --skill_selection_strategy top_k \
    --scoring_mode weighted \
    --importance_weight 0.8 \
    --skill_scores_json $MASTER_SKILL_FILE \
    --encoder_skill $ENCODER_SKILL_MPNET \
    --save_scores


python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --skill_embeddings_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index \
    --skill_scores_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --output_dir "${OUTPUT_BASE}/skills_only_pjmath_desc_weighted_idf_logpool_adv" \
    --run_name "skills_only_pjmath_desc_weighted_idf_logpool_adv" \
    --use_skill_text \
    --use_text_description \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --num_workers 4 \
    --eval_clean_test \
    --save_scores \
    --normalize_input \
    --early_stop_metric mrr \
    --seed 42 \
    --optuna \
    --pooling_strategy weighted_idf \
    --use_skill_path_log_pooling \
    --max_epochs 50 \
    --n_trials 30

python -m src.cpp.train_cpp_enhanced_v2_tuned_weights \
    --data_type $DATA_TYPE \
    --skill_embeddings_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index \
    --skill_scores_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --output_dir "${OUTPUT_BASE}/job_titles_skills_pjmath_desc_weighted_idf_logpool_adv_weights_tuned" \
    --run_name "job_titles_skills_pjmath_desc_weighted_idf_logpool_adv_weights_tuned" \
    --use_text_history \
    --use_skill_text \
    --use_text_description \
    --patience $PATIENCE \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --num_workers 4 \
    --eval_clean_test \
    --normalize_input \
    --early_stop_metric loss \
    --seed 42 \
    --use_advanced \
    --pooling_strategy weighted_idf \
    --use_skill_path_log_pooling \
    --use_learnable_pooling \
    --pooling_lr_multiplier 0.1 \
    --initial_alpha 1.0 \
    --initial_beta 1.0 \
    --initial_gamma 0.0 \
    --max_skills_per_path 400 \
    --device cuda \
    --optuna \
    --n_trials 30 \
    --max_epochs 50



## not started
python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --skill_embeddings_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index \
    --skill_scores_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --output_dir "${OUTPUT_BASE}/skills_only_pjmath_desc_weighted_idf_jobbert" \
    --run_name "skills_only_pjmath_desc_weighted_idf_jobbert" \
    --use_skill_text \
    --use_text_description \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --num_workers 4 \
    --eval_clean_test \
    --save_scores \
    --normalize_input \
    --early_stop_metric mrr \
    --seed 42 \
    --optuna \
    --pooling_strategy weighted_idf \
    --max_epochs 50 \
    --n_trials 30

    ## ongoing
    python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --skill_embeddings_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index \
    --skill_scores_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --output_dir "${OUTPUT_BASE}/skills_only_pjmath_desc_weighted_idf_logpool_jobbert" \
    --run_name "skills_only_pjmath_desc_weighted_idf_logpool_jobbert" \
    --use_skill_text \
    --use_text_description \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --num_workers 4 \
    --eval_clean_test \
    --save_scores \
    --normalize_input \
    --early_stop_metric mrr \
    --seed 42 \
    --optuna \
    --pooling_strategy weighted_idf \
    --max_epochs 50 \
    --n_trials 30


