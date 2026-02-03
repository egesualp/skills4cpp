#!/bin/bash
#SBATCH --job-name=cpp_kw_100k_ablation        # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/cpp_kw_100k_ablation_%j.log    # Log output (%j expands to job ID)
#SBATCH --error=experiments/logs/cpp_kw_100k_ablation_%j.err     # Error output
#SBATCH --partition=mcml-dgx-a100-40x8  # Specify the partition to use
#SBATCH --gres=gpu:1                     # Number of GPUs to use
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=8                 # Number of CPUs per task
#SBATCH --time=4:00:00                   # Time limit (increased for multiple runs)

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
DATA_TYPE="karrierewege_100k"
MASTER_SKILL_FILE="results/karrierewege_esco_100k_esco_ground_truth/job_title_skills_master.csv"
ENCODER_TEXT="ElenaSenger/career-path-representation-mpnet-karrierewege"
ENCODER_SKILL="ElenaSenger/career-path-representation-mpnet-karrierewege"
OUTPUT_BASE="results/cpp/kw_esco_100k_ablation"
WANDB_PROJECT="cpp-kw-esco-100k-static-ablation"

# Static Hyperparameters
BATCH_SIZE=4096
EVAL_BATCH_SIZE=2048
MAX_EPOCHS=30
PATIENCE=5
LR=2e-5
HIDDEN_DIM=512
N_LAYERS=2
DROPOUT=0.25
USE_MODALITY_WEIGHTS=True

echo "Starting Ablation Study..."
echo "Data: $DATA_TYPE"
echo "Project: $WANDB_PROJECT"

# -----------------------------------------------------------------------------
# 1. Job Titles Only
# -----------------------------------------------------------------------------
echo -e "\n\n=== Running Experiment 1: Job Titles Only ==="
python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --master_skill_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --encoder_skill $ENCODER_SKILL \
    --output_dir "${OUTPUT_BASE}/job_titles_only" \
    --run_name "job_titles_only" \
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
    --mixed_precision \
    --num_workers 0 \
    --save_scores \
    --scores_output_dir results/cpp/kw_esco_100k_ablation/scores \


# -----------------------------------------------------------------------------
# 2. Job Titles + Descriptions
# -----------------------------------------------------------------------------
echo -e "\n\n=== Running Experiment 2: Job Titles + Descriptions ==="
python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --master_skill_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --encoder_skill $ENCODER_SKILL \
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
    --mixed_precision \
    --num_workers 0 \
    --save_scores \
    --scores_output_dir results/cpp/kw_esco_100k_ablation/scores \

# -----------------------------------------------------------------------------
# 3. Skills Only (Names, IDF)
# -----------------------------------------------------------------------------
echo -e "\n\n=== Running Experiment 3: Skills Only (Names, IDF) ==="
python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --master_skill_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --encoder_skill $ENCODER_SKILL \
    --output_dir "${OUTPUT_BASE}/skills_only_names_idf" \
    --run_name "skills_only_names_idf" \
    --use_skill_text \
    --pooling_strategy weighted_idf \
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
    --mixed_precision \
    --num_workers 0

# -----------------------------------------------------------------------------
# 4. Skills Only (Names + Desc, IDF)
# -----------------------------------------------------------------------------
echo -e "\n\n=== Running Experiment 4: Skills Only (Names + Desc, IDF) ==="
python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --master_skill_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --encoder_skill "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/with_skill_description/best-model" \
    --output_dir "${OUTPUT_BASE}/skills_only_names_desc_idf_new_encoder" \
    --run_name "skills_only_names_desc_idf_new_encoder" \
    --use_skill_text \
    --use_skill_description \
    --pooling_strategy weighted_idf \
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
    --mixed_precision \
    --num_workers 0

# -----------------------------------------------------------------------------
# 3-2. Skills Only (Names, IDF)
# -----------------------------------------------------------------------------
echo -e "\n\n=== Running Experiment 3: Skills Only (Names, IDF) ==="
python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --master_skill_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --encoder_skill "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/no_skill_descriptions/best-model" \
    --output_dir "${OUTPUT_BASE}/skills_only_names_idf_new_encoder" \
    --run_name "skills_only_names_idf_new_encoder" \
    --use_skill_text \
    --pooling_strategy weighted_idf \
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
    --mixed_precision \
    --num_workers 0

# -----------------------------------------------------------------------------
# 4-2. Skills Only (Names + Desc, IDF)
# -----------------------------------------------------------------------------
echo -e "\n\n=== Running Experiment 4: Skills Only (Names + Desc, IDF) ==="
python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --master_skill_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --encoder_skill $ENCODER_SKILL \
    --output_dir "${OUTPUT_BASE}/skills_only_names_desc_idf" \
    --run_name "skills_only_names_desc_idf" \
    --use_skill_text \
    --use_skill_description \
    --pooling_strategy weighted_idf \
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
    --mixed_precision \
    --num_workers 0

# -----------------------------------------------------------------------------
# 5. Multimodal: Job Titles + Skills (Names, IDF)
# -----------------------------------------------------------------------------
echo -e "\n\n=== Running Experiment 5: Job Titles + Skills (Names, IDF) ==="
python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --master_skill_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --encoder_skill $ENCODER_SKILL \
    --output_dir "${OUTPUT_BASE}/multimodal_job_skill_idf" \
    --run_name "multimodal_job_skill_idf" \
    --use_text_history \
    --use_text_description \
    --use_skill_description \
    --use_skill_text \
    --pooling_strategy weighted_idf \
    --use_advanced \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM \
    --n_layers $N_LAYERS \
    --dropout $DROPOUT \
    --max_epochs $MAX_EPOCHS \
    --use_modality_weights $USE_MODALITY_WEIGHTS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --patience $PATIENCE \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --mixed_precision \
    --num_workers 0

# -----------------------------------------------------------------------------
# 6. Multimodal: Job Titles + Skills (Names, Log Pooling)
# -----------------------------------------------------------------------------
echo -e "\n\n=== Running Experiment 6: Job Titles + Skills (Names, Log Pooling) ==="
python -m src.cpp.train_cpp_enhanced_v2 \
    --data_type $DATA_TYPE \
    --master_skill_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --encoder_skill $ENCODER_SKILL \
    --output_dir "${OUTPUT_BASE}/multimodal_job_skill_logpool" \
    --run_name "multimodal_job_skill_logpool" \
    --use_text_history \
    --use_skill_text \
    --pooling_strategy weighted_idf \
    --use_skill_path_log_pooling \
    --skill_path_alpha_decay 0.5 \
    --use_advanced \
    --lr $LR \
    --hidden_dim $HIDDEN_DIM \
    --n_layers $N_LAYERS \
    --dropout $DROPOUT \
    --max_epochs $MAX_EPOCHS \
    --use_modality_weights $USE_MODALITY_WEIGHTS \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --patience $PATIENCE \
    --use_wandb \
    --wandb_project $WANDB_PROJECT \
    --mixed_precision \
    --num_workers 0

# -----------------------------------------------------------------------------
# 7. V3: Last Job Skills Only (Names, IDF, Top-10)
# -----------------------------------------------------------------------------
echo -e "\n\n=== Running Experiment 7: V3 Last Job Skills Only (Names, IDF, Top-10) ==="
python -m src.cpp.train_cpp_enhanced_v3 \
    --data_type $DATA_TYPE \
    --master_skill_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --encoder_skill "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/no_skill_descriptions/best-model" \
    --output_dir "${OUTPUT_BASE}/v3_last_job_skills_names_adv" \
    --run_name "v3_last_job_skills_names" \
    --use_skill_text \
    --pooling_strategy weighted_idf \
    --use_advanced \
    --top_k_skills 20 \
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
    --mixed_precision \
    --num_workers 0

# -----------------------------------------------------------------------------
# 8. V3: Last Job Skills Only (Names + Desc, IDF, Top-10)
# -----------------------------------------------------------------------------
echo -e "\n\n=== Running Experiment 8: V3 Last Job Skills Only (Names + Desc, IDF, Top-10) ==="
python -m src.cpp.train_cpp_enhanced_v3 \
    --data_type $DATA_TYPE \
    --master_skill_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --encoder_skill "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/with_skill_description/best-model" \
    --output_dir "${OUTPUT_BASE}/v3_last_job_skills_desc_adv" \
    --run_name "v3_last_job_skills_desc" \
    --use_skill_description \
    --use_skill_text \
    --pooling_strategy weighted_idf \
    --use_advanced \
    --top_k_skills 20 \
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
    --mixed_precision \
    --num_workers 0
