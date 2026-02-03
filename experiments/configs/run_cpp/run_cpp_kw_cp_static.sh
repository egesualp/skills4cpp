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
DATA_TYPE="karrierewege_cp"
MASTER_SKILL_FILE="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/kw_cp/v5_fused_scorer/linear_h2_kw_cp/fused_predictions.jsonl"
ENCODER_TEXT="ElenaSenger/career-path-representation-mpnet-karrierewege-cp"
ENCODER_SKILL="ElenaSenger/career-path-representation-mpnet-karrierewege-cp"
OUTPUT_BASE="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/karrierewege_cp_static"
WANDB_PROJECT="cpp-karrierewege-cp-static-final"

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
    --output_dir "${OUTPUT_BASE}/job_titles" \
    --run_name "job_titles" \
    --use_text_history \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM \
    --n_layers $N_LAYERS \
    --dropout $DROPOUT \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --patience $PATIENCE \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --num_workers 4 \
    --eval_clean_test \
    --save_scores \
    --normalize_input \
    --early_stop_metric loss \
    --seed 42 \
    --save_scores

python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --skill_embeddings_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index \
    --skill_scores_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --output_dir "${OUTPUT_BASE}/job_titles_desc" \
    --run_name "job_titles_desc" \
    --use_text_history \
    --use_text_description \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM \
    --n_layers $N_LAYERS \
    --dropout $DROPOUT \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --patience $PATIENCE \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --num_workers 4 \
    --eval_clean_test \
    --save_scores \
    --normalize_input \
    --early_stop_metric loss \
    --seed 42 \
    --save_scores 



python -m src.cpp.train_cpp_enhanced_v2_fixed \
    --data_type $DATA_TYPE \
    --skill_embeddings_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index \
    --skill_scores_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --output_dir "${OUTPUT_BASE}/job_titles_desc_infonce" \
    --run_name "job_titles_desc_infonce" \
    --use_text_history \
    --use_text_description \
    --loss_type infonce \
    --temperature 0.07 \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM \
    --n_layers $N_LAYERS \
    --dropout $DROPOUT \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --patience $PATIENCE \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --num_workers 4 \
    --eval_clean_test \
    --save_scores \
    --normalize_input \
    --early_stop_metric loss \
    --seed 42 \
    --save_scores

# -----------------------------------------------------------------------------
# 3. Job Titles with Pooled Skills
# -----------------------------------------------------------------------------
echo -e "\n\n=== Running Experiment 1: Job Titles Only ==="
python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --skill_embeddings_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index \
    --skill_scores_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --output_dir "${OUTPUT_BASE}/job_titles_skills_pjmath_desc_adv" \
    --run_name "job_titles_skills_pjmath_desc_adv" \
    --use_text_history \
    --use_skill_text \
    --use_text_description \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM \
    --n_layers $N_LAYERS \
    --dropout $DROPOUT \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --patience $PATIENCE \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --num_workers 4 \
    --eval_clean_test \
    --normalize_input \
    --early_stop_metric loss \
    --use_advanced \
    --seed 42 \
    --pooling_strategy mean



python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --skill_embeddings_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index \
    --skill_scores_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --output_dir "${OUTPUT_BASE}/job_titles_skills_pjmath_adv_desc_weighted_idf" \
    --run_name "job_titles_skills_pjmath_adv_desc_weighted_idf" \
    --use_text_history \
    --use_skill_text \
    --use_text_description \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM \
    --n_layers $N_LAYERS \
    --dropout $DROPOUT \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --patience $PATIENCE \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --num_workers 4 \
    --eval_clean_test \
    --normalize_input \
    --early_stop_metric loss \
    --seed 42 \
    --use_advanced \
    --use_modality_weights \
    --pooling_strategy weighted_idf


python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --skill_embeddings_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index \
    --skill_scores_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --output_dir "${OUTPUT_BASE}/job_titles_skills_pjmath_adv_desc_weighted_mean" \
    --run_name "job_titles_skills_pjmath_adv_desc_weighted_mean" \
    --use_text_history \
    --use_skill_text \
    --use_text_description \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM \
    --n_layers $N_LAYERS \
    --dropout $DROPOUT \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --patience $PATIENCE \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --num_workers 4 \
    --eval_clean_test \
    --normalize_input \
    --early_stop_metric loss \
    --seed 42 \
    --use_advanced \
    --use_modality_weights \
    --pooling_strategy weighted_mean

# -----------------------------------------------------------------------------
# 3. Job Titles with Pooled Skills
# -----------------------------------------------------------------------------
echo -e "\n\n=== Running Experiment 1: Job Titles Only ==="
python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --skill_embeddings_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index \
    --skill_scores_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --output_dir "${OUTPUT_BASE}/job_titles_skills_pjmath_adv_desc_weighted_idf_logpool" \
    --run_name "job_titles_skills_pjmath_adv_desc_weighted_idf_logpool" \
    --use_text_history \
    --use_skill_text \
    --use_text_description \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM \
    --n_layers $N_LAYERS \
    --dropout $DROPOUT \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --patience $PATIENCE \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --num_workers 4 \
    --eval_clean_test \
    --normalize_input \
    --early_stop_metric loss \
    --seed 42 \
    --use_advanced \
    --use_modality_weights \
    --pooling_strategy weighted_idf \
    --use_skill_path_log_pooling

python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --skill_embeddings_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index \
    --skill_scores_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --output_dir "${OUTPUT_BASE}/job_titles_skills_pjmath_desc_weighted_idf_logpool_adv_2" \
    --run_name "job_titles_skills_pjmath_desc_weighted_idf_logpool_adv_2" \
    --use_text_history \
    --use_skill_text \
    --use_text_description \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM \
    --n_layers $N_LAYERS \
    --dropout $DROPOUT \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --patience $PATIENCE \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --num_workers 4 \
    --eval_clean_test \
    --normalize_input \
    --early_stop_metric loss \
    --seed 42 \
    --use_advanced \
    --use_modality_weights \
    --pooling_strategy weighted_idf \
    --use_skill_path_log_pooling

python -m src.cpp.train_cpp_enhanced_v2_tuned_weights \
    --data_type $DATA_TYPE \
    --skill_embeddings_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index \
    --skill_scores_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --output_dir "${OUTPUT_BASE}/job_titles_skills_pjmath_desc_adv_tuned" \
    --run_name "job_titles_skills_pjmath_desc_adv_tuned" \
    --use_text_history \
    --use_skill_text \
    --use_text_description \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM \
    --n_layers $N_LAYERS \
    --dropout $DROPOUT \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
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
    --device cuda


MASTER_OUTPUT_DIR="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/karrierewege_cp_embedding_final"
WANDB_PROJECT="karrierewege_cp_skills_embedding_mpnet" 
MODEL_NAME="ElenaSenger/career-path-representation-mpnet-karrierewege-cp"
MODEL_NAME_GTE="Alibaba-NLP/gte-base-en-v1.5"
WANDB_PROJECT_GTE="karrierewege_cp_skills_embedding_gte" 

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m src.cpp.finetune_last_job_skills \
  --data_type karrierewege_cp \
  --skill_scores_json ${MASTER_SKILL_FILE} \
  --skills_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv \
  --model_name ${ENCODER_TEXT} \
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


python src/cpp/train_cpp_enhanced_v2_tuned_weights.py \
    --data_type decorte \
    --skill_scores_file /path/to/best_fused_scores.json \
    --skill_embeddings_dir /path/to/precomputed_skill_embeddings \
    --use_learnable_pooling \
    --pooling_lr_multiplier 0.1 \
    --initial_alpha 1.0 \
    --initial_beta 1.0 \
    --initial_gamma 0.0 \
    --max_skills_per_path 400 \
    --use_text_history \
    --use_skill_text \
    --max_epochs 20 \
    --batch_size 32 \
    --device cuda:0 \
    --n_trials 50


python -m src.cpp.train_cpp_enhanced_v2_gated \
    --data_type $DATA_TYPE \
    --skill_embeddings_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index \
    --skill_scores_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --output_dir "${OUTPUT_BASE}/job_titles_skills_pjmath_desc_weighted_idf_logpool_adv_gated" \
    --run_name "job_titles_skills_pjmath_desc_weighted_idf_logpool_adv_gated" \
    --use_text_history \
    --use_skill_text \
    --use_text_description \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM \
    --n_layers $N_LAYERS \
    --dropout $DROPOUT \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
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
    --use_skill_path_log_pooling

#### Skills Only Experiments
python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --skill_embeddings_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index \
    --skill_scores_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --output_dir "${OUTPUT_BASE}/skills_only_pjmath_desc_weighted_idf_logpool" \
    --run_name "skills_only_pjmath_desc_weighted_idf_logpool" \
    --use_text_description \
    --use_skill_text \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM \
    --n_layers $N_LAYERS \
    --dropout $DROPOUT \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --patience $PATIENCE \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --num_workers 4 \
    --eval_clean_test \
    --normalize_input \
    --early_stop_metric loss \
    --seed 42 \
    --pooling_strategy weighted_idf \
    --use_skill_path_log_pooling \
    --save_scores \
    --max_epochs 50

python -m src.cpp.train_cpp_enhanced_v2_tuned_weights \
    --data_type $DATA_TYPE \
    --skill_embeddings_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index \
    --skill_scores_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --output_dir "${OUTPUT_BASE}/skills_only_pjmath_desc_tuned" \
    --run_name "skills_only_pjmath_desc_tuned" \
    --use_text_description \
    --use_skill_text \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM \
    --n_layers $N_LAYERS \
    --dropout $DROPOUT \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --patience $PATIENCE \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --num_workers 4 \
    --eval_clean_test \
    --normalize_input \
    --early_stop_metric loss \
    --seed 42 \
    --save_scores \
    --max_epochs 50 \
    --use_learnable_pooling \
    --pooling_lr_multiplier 0.1 \
    --initial_alpha 1.0 \
    --initial_beta 1.0 \
    --initial_gamma 0.0 \
    --max_skills_per_path 400

python src/cpp/train_cpp_enhanced_v2_tuned_weights.py \
    --data_type decorte \
    --skill_scores_file /path/to/best_fused_scores.json \
    --skill_embeddings_dir /path/to/precomputed_skill_embeddings \
    --use_learnable_pooling \
    --pooling_lr_multiplier 0.1 \
    --initial_alpha 1.0 \
    --initial_beta 1.0 \
    --initial_gamma 0.0 \
    --max_skills_per_path 400 \
    --use_text_history \
    --use_skill_text \
    --max_epochs 20 \
    --batch_size 32 \
    --device cuda:0 \
    --n_trials 50

python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --skill_embeddings_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index \
    --skill_scores_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --output_dir "${OUTPUT_BASE}/skills_only_pjmath_desc_weighted_idf" \
    --run_name "skills_only_pjmath_desc_weighted_idf" \
    --use_skill_text \
    --use_text_description \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM \
    --n_layers $N_LAYERS \
    --dropout $DROPOUT \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --patience $PATIENCE \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --num_workers 4 \
    --eval_clean_test \
    --normalize_input \
    --early_stop_metric loss \
    --seed 42 \
    --pooling_strategy weighted_idf \
    --save_scores \
    --max_epochs 50

## MPNET Experiments (text_encoder = skill_encoder)
python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --skill_scores_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --output_dir "${OUTPUT_BASE}/job_titles_skills_mpnet" \
    --run_name "job_titles_skills_mpnet" \
    --use_text_history \
    --use_skill_text \
    --use_text_description \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM \
    --n_layers $N_LAYERS \
    --dropout $DROPOUT \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --patience $PATIENCE \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --num_workers 4 \
    --eval_clean_test \
    --normalize_input \
    --early_stop_metric loss \
    --seed 42

python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --skill_scores_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --output_dir "${OUTPUT_BASE}/job_titles_skills_mpnet_adv_weighted_idf_logpooling" \
    --run_name "job_titles_skills_mpnet_adv_weighted_idf_logpooling" \
    --use_text_history \
    --use_skill_text \
    --use_text_description \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM \
    --n_layers $N_LAYERS \
    --dropout $DROPOUT \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
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
    --use_skill_path_log_pooling

python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --skill_scores_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --encoder_skill $ENCODER_SKILL \
    --output_dir "${OUTPUT_BASE}/job_titles_skills_mpnet_desc_weighted_idf_logpool_adv_2" \
    --run_name "job_titles_skills_mpnet_desc_weighted_idf_logpool_adv_2" \
    --use_text_history \
    --use_skill_text \
    --use_text_description \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM \
    --n_layers $N_LAYERS \
    --dropout $DROPOUT \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
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
    --use_skill_path_log_pooling

python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --skill_scores_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --encoder_skill $ENCODER_SKILL \
    --output_dir "${OUTPUT_BASE}/job_titles_skills_mpnet_desc_weighted_idf_logpool_adv_jobbert" \
    --run_name "job_titles_skills_mpnet_desc_weighted_idf_logpool_adv_jobbert" \
    --use_text_history \
    --use_skill_text \
    --use_text_description \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM \
    --n_layers $N_LAYERS \
    --dropout $DROPOUT \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
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
    --use_skill_path_log_pooling


python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --skill_scores_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --encoder_skill $ENCODER_SKILL \
    --output_dir "${OUTPUT_BASE}/job_titles_skills_mpnet_desc_weighted_idf_adv_jobbert" \
    --run_name "job_titles_skills_mpnet_desc_weighted_idf_adv_jobbert" \
    --use_text_history \
    --use_skill_text \
    --use_text_description \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM \
    --n_layers $N_LAYERS \
    --dropout $DROPOUT \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --patience $PATIENCE \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --num_workers 4 \
    --eval_clean_test \
    --normalize_input \
    --early_stop_metric loss \
    --seed 42 \
    --use_advanced \
    --pooling_strategy weighted_idf


# Ablation: Skill confidence score filtering
## two approaches to use here
# job_titles_skills_pjmath_desc_weighted_idf_adv
# job_titles_skills_pjmath_desc_weighted_idf_logpool_adv


python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --skill_embeddings_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index \
    --skill_scores_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --output_dir "${OUTPUT_BASE}/job_titles_skills_pjmath_desc_weighted_idf_adv_thres0_9" \
    --run_name "job_titles_skills_pjmath_desc_weighted_idf_adv_thres0_9" \
    --use_text_history \
    --use_skill_text \
    --use_text_description \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM \
    --n_layers $N_LAYERS \
    --dropout $DROPOUT \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --patience $PATIENCE \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --num_workers 4 \
    --eval_clean_test \
    --normalize_input \
    --early_stop_metric loss \
    --seed 42 \
    --use_advanced \
    --use_modality_weights \
    --pooling_strategy weighted_idf \
    --skill_confidence_threshold 0.9

python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --skill_embeddings_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index \
    --skill_scores_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --output_dir "${OUTPUT_BASE}/job_titles_skills_pjmath_desc_weighted_idf_logpool_adv_thres0_9" \
    --run_name "job_titles_skills_pjmath_desc_weighted_idf_logpool_adv_thres0_9" \
    --use_text_history \
    --use_skill_text \
    --use_text_description \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM \
    --n_layers $N_LAYERS \
    --dropout $DROPOUT \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --patience $PATIENCE \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --num_workers 4 \
    --eval_clean_test \
    --normalize_input \
    --early_stop_metric loss \
    --seed 42 \
    --use_advanced \
    --use_modality_weights \
    --pooling_strategy weighted_idf \
    --use_skill_path_log_pooling \
    --skill_confidence_threshold 0.9

python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --skill_embeddings_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index \
    --skill_scores_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --output_dir "${OUTPUT_BASE}/job_titles_skills_pjmath_desc_weighted_idf_logpool_adv_2" \
    --run_name "job_titles_skills_pjmath_desc_weighted_idf_logpool_adv_2" \
    --use_text_history \
    --use_skill_text \
    --use_text_description \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM \
    --n_layers $N_LAYERS \
    --dropout $DROPOUT \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --patience $PATIENCE \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --num_workers 4 \
    --eval_clean_test \
    --normalize_input \
    --early_stop_metric loss \
    --seed 42 \
    --use_advanced \
    --use_modality_weights \
    --pooling_strategy weighted_idf \
    --use_skill_path_log_pooling

# Ablation: Our fine-tuned skill encoder
python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --skill_scores_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --encoder_skill /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/decorte_embedding_final/decorte_weighted_0_8_topk_10skills_es_mpnet/best-model-mpnet \
    --output_dir "${OUTPUT_BASE}/job_titles_skills_mpnet_ft_desc_weighted_idf_logpool_adv" \
    --run_name "job_titles_skills_mpnet_ft_desc_weighted_idf_logpool_adv" \
    --use_text_history \
    --use_skill_text \
    --use_text_description \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM \
    --n_layers $N_LAYERS \
    --dropout $DROPOUT \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --patience $PATIENCE \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --num_workers 4 \
    --eval_clean_test \
    --normalize_input \
    --early_stop_metric loss \
    --seed 42 \
    --use_advanced \
    --use_modality_weights \
    --pooling_strategy weighted_idf \
    --use_skill_path_log_pooling

python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --skill_scores_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --encoder_skill /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/decorte_embedding_final/decorte_weighted_0_8_topk_80skills_gte_base/best-model \
    --output_dir "${OUTPUT_BASE}/job_titles_skills_gte_ft_desc_weighted_idf_logpool_adv" \
    --run_name "job_titles_skills_gte_ft_desc_weighted_idf_logpool_adv" \
    --use_text_history \
    --use_skill_text \
    --use_text_description \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM \
    --n_layers $N_LAYERS \
    --dropout $DROPOUT \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --patience $PATIENCE \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --num_workers 4 \
    --eval_clean_test \
    --normalize_input \
    --early_stop_metric loss \
    --seed 42 \
    --use_advanced \
    --use_modality_weights \
    --pooling_strategy weighted_idf \
    --use_skill_path_log_pooling