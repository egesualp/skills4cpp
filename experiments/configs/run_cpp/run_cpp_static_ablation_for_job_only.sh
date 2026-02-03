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
MASTER_SKILL_FILE="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/karrierewege_esco_100k_esco_ground_truth/job_title_skills_master.csv"
ENCODER_TEXT="ElenaSenger/career-path-representation-mpnet-karrierewege"
ENCODER_SKILL="ElenaSenger/career-path-representation-mpnet-karrierewege"
OUTPUT_BASE="results/cpp/kw_esco_100k_ablation_2"
WANDB_PROJECT="cpp-kw-esco-100k-repro"

# Static Hyperparameters
BATCH_SIZE=16
EVAL_BATCH_SIZE=32
MAX_EPOCHS=10
PATIENCE=2
LR=2e-5
HIDDEN_DIM=512
N_LAYERS=1
DROPOUT=0.1
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
    --num_workers 0 \
    --eval_clean_test \
    --normalize_input \
    --early_stop_metric loss \
    --seed 42

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
    --num_workers 0 \
    --normalize_input \
    --early_stop_metric loss \
    --seed 42

