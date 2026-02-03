#!/bin/bash
#SBATCH --job-name=cpp_kw_100k_ablation_3        # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/cpp_kw_100k_ablation_%j.log    # Log output (%j expands to job ID)
#SBATCH --error=experiments/logs/cpp_kw_100k_ablation_%j.err     # Error output
#SBATCH --partition=mcml-dgx-a100-40x8  # Specify the partition to use
#SBATCH --gres=gpu:1                     # Number of GPUs to use
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=8                 # Number of CPUs per task
#SBATCH --time=2:00:00                   # Time limit (increased for multiple runs)

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
    --use_advanced \
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
    --use_skill_path_log_pooling \
    --skill_path_alpha_decay 0.5 \
    --use_advanced \
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