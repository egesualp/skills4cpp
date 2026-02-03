#!/bin/bash
#SBATCH --job-name=cpp_kw_cp_static_v3        # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/cpp_kw/cpp_kw_cp_static_v3_%j.log    # Log output (%j expands to job ID)
#SBATCH --error=experiments/logs/cpp_kw/cpp_kw_cp_static_v3_%j.err     # Error output
#SBATCH --qos=mcml
#SBATCH --partition=mcml-dgx-a100-40x8  # Specify the partition to use
#SBATCH --gres=gpu:1                     # Number of GPUs to use
#SBATCH --ntasks=6                       # Number of tasks
#SBATCH --time=04:00:00 
#SBATCH --mem=64G

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
DATA_TYPE="karrierewege_cp"

MASTER_SKILL_FILE="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/kw_cp/v5_fused_scorer/linear_h2_kw_cp/fused_predictions.jsonl"
ENCODER_TEXT="ElenaSenger/career-path-representation-mpnet-karrierewege-cp"
OUTPUT_BASE="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/karrierewege_cp_static_v3"
WANDB_PROJECT="cpp-karrierewege_cp-static-v3-final"

ENCODER_SKILL_MPNET=/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/karrierewege_cp_embedding_final/kw_cp_weighted_0_8_topk_10skills_es_mpnet/best-model-kw-mpnet

HF_HOME="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/.cache/huggingface"


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

python -m src.cpp.train_cpp_enhanced_v3 \
    --data_type $DATA_TYPE \
    --encoder_text $ENCODER_TEXT \
    --output_dir "${OUTPUT_BASE}/job_titles_skills_mpnet_desc" \
    --run_name "job_titles_skills_mpnet_desc" \
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

#python -m src.cpp.train_cpp_enhanced_v3 \
#    --data_type $DATA_TYPE \
#    --encoder_text $ENCODER_TEXT \
#    --output_dir "${OUTPUT_BASE}/job_titles_skills_gte_desc" \
#    --run_name "job_titles_skills_gte_desc" \
#    --use_text_history \
#    --use_skill_text \
#    --use_text_description \
#    --use_skill_description \
#    --max_epochs $MAX_EPOCHS \
#    --patience $PATIENCE \
#    --use_wandb \
#    --wandb_project $WANDB_PROJECT \
#    --lr $LR \
#    --hidden_dim $HIDDEN_DIM \
#    --n_layers $N_LAYERS \
#    --dropout $DROPOUT \
#    --max_epochs $MAX_EPOCHS \
#    --batch_size $BATCH_SIZE \
#    --eval_batch_size $EVAL_BATCH_SIZE \
#    --patience $PATIENCE \
#    --num_workers 4 \
#    --eval_clean_test \
#    --normalize_input \
#    --early_stop_metric loss \
#    --use_advanced \
#    --seed 42 \
#    --top_k_skills 80 \
#    --skill_selection_strategy top_k \
#    --scoring_mode weighted \
#    --importance_weight 0.8 \
#    --skill_scores_json $MASTER_SKILL_FILE \
#    --encoder_skill $ENCODER_SKILL



#python -m src.cpp.train_cpp_enhanced_v3 \
#    --data_type $DATA_TYPE \
#    --encoder_text $ENCODER_TEXT \
#    --output_dir "${OUTPUT_BASE}/job_titles_skills_mpnet_desc_optuna" \
#    --run_name "job_titles_skills_mpnet_desc_optuna" \
#    --use_text_history \
#    --use_skill_text \
#    --use_text_description \
#    --use_skill_description \
#    --max_epochs $MAX_EPOCHS \
#    --patience $PATIENCE \
#    --use_wandb \
#    --wandb_project $WANDB_PROJECT \
#    --num_workers 4 \
#    --eval_clean_test \
#    --normalize_input \
#    --early_stop_metric loss \
#    --use_advanced \
#    --seed 42 \
#    --top_k_skills 10 \
#    --skill_selection_strategy top_k \
#    --scoring_mode weighted \
#    --importance_weight 0.8 \
#    --skill_scores_json $MASTER_SKILL_FILE \
#    --encoder_skill $ENCODER_SKILL_MPNET \
#    --optuna \
#    --save_scores
#
## Skills Only (Names + Desc, IDF)
#python -m src.cpp.train_cpp_enhanced_v3 \
#    --data_type $DATA_TYPE \
#    --encoder_text $ENCODER_TEXT \
#    --output_dir "${OUTPUT_BASE}/skills_only_mpnet_desc_optuna_2" \
#    --run_name "skills_only_mpnet_desc_optuna_2" \
#    --use_skill_text \
#    --use_text_description \
#    --use_skill_description \
#    --max_epochs $MAX_EPOCHS \
#    --patience $PATIENCE \
#    --use_wandb \
#    --wandb_project $WANDB_PROJECT \
#    --num_workers 24 \
#    --eval_clean_test \
#    --normalize_input \
#    --early_stop_metric loss \
#    --seed 42 \
#    --top_k_skills 10 \
#    --skill_selection_strategy top_k \
#    --scoring_mode weighted \
#    --importance_weight 0.8 \
#    --skill_scores_json $MASTER_SKILL_FILE \
#    --encoder_skill $ENCODER_SKILL_MPNET \
#    --optuna \
#    --save_scores \
#    --n_trials 20


