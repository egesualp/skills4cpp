#!/bin/bash
#SBATCH --job-name=hybrid_evals        # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/hybrid_evals_%j.log    # Log output (%j expands to job ID)
#SBATCH --error=experiments/logs/hybrid_evals_%j.err     # Error output
#SBATCH --partition=mcml-dgx-a100-40x8  # Specify the partition to use
#SBATCH --gres=gpu:1                     # Number of GPUs to use
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --time=00:30:00                   # Time limit



#
# This script runs the full 4-step evaluation for the hybrid skill model.
# It runs:
#   1. The Baseline (Sim model only)
#   2. Hybrid Re-ranking with Level 0 (Pillars)
#   3. Hybrid Re-ranking with Level 1
#   4. Hybrid Re-ranking with Level 2
#
# It will exit immediately if any command fails.
#
source /dss/dsshome1/02/ra95kix2/miniconda3/bin/activate thesis
# --- 1. Define Variables ---
# Path to your python module
PYTHON_MODULE="src.skill_mapping.v1.run_inference"

# Path to your test data
TRAIN_DATA_PATH="data/title_pairs_desc/decorte_train_pairs.csv"
VAL_DATA_PATH="data/title_pairs_desc/decorte_val_pairs.csv"
TEST_DATA_PATH="data/title_pairs_desc/decorte_test_pairs.csv"
# The SOTA Similarity model (Sim)
SKILL_ENCODER="TechWolf/JobBERT-v2"

# The base encoder used for your category probes (P(Cat))
BASE_ENCODER="all-MiniLM-L6-v2"


# Directory where your trained probes (.pt files) are stored
CHECKPOINT_DIR="models"

# --- 2. Run Experiments ---

echo "--- 1/3: Running Baseline (JobBERT-v2 Only) ---"
python -m $PYTHON_MODULE \
  --job_data_path "$TRAIN_DATA_PATH" \
  --skill_encoder_ckpt "$SKILL_ENCODER" \
  --run_name "eval_baseline_JobBERT_decorte_train" \
  --text_fields "title"

echo "--- 2/3: Running Baseline (JobBERT-v2 Only) ---"
python -m $PYTHON_MODULE \
  --job_data_path "$VAL_DATA_PATH" \
  --skill_encoder_ckpt "$SKILL_ENCODER" \
  --run_name "eval_baseline_JobBERT_decorte_val" \
  --text_fields "title"

  echo "--- 3/3: Running Baseline (JobBERT-v2 Only) ---"
python -m $PYTHON_MODULE \
  --job_data_path "$TEST_DATA_PATH" \
  --skill_encoder_ckpt "$SKILL_ENCODER" \
  --run_name "eval_baseline_JobBERT_decorte_test" \
  --text_fields "title"

echo "--- All evaluations complete. ---"
echo "Check your 'results/' folder and 'master_results_new.csv' for outputs."