#!/bin/bash
#SBATCH --job-name=cpp_cached_experiments
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/cpp_cached_%j.log
#SBATCH --error=experiments/logs/cpp_cached_%j.err
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=8
#SBATCH --time=01:00:00  # Much shorter since we're using cache!

#
# EXAMPLE: Running multiple experiments using cached embeddings
#
# This script demonstrates how to run multiple training runs using
# pre-computed embeddings, saving hours of computation time.
#
# Prerequisites:
# 1. Run the initial embedding computation (run_cpp_baseline_kw_100k.sh)
# 2. Verify cache files exist in embeddings directory
#

source /dss/dsshome1/02/ra95kix2/miniconda3/bin/activate thesis

# Common parameters (these determine which cache file is used)
DATA_TYPE="karrierewege_100k"
MASTER_SKILL_FILE="results/karrierewege_esco_100k_esco_ground_truth/job_title_skills_master.csv"
ENCODER="ElenaSenger/career-path-representation-mpnet-karrierewege"
POOLING="weighted_idf"

echo "================================================================================"
echo "Running Multiple Experiments with Cached Embeddings"
echo "================================================================================"
echo ""
echo "These experiments will load embeddings from cache (~2 seconds each)"
echo "instead of recomputing them (~2 hours each)."
echo ""

# Experiment 1: Test different hyperparameters with text-only embeddings
echo "Experiment 1: Text-only with hidden_dim=512"
python -m src.cpp.train_cpp_enhanced \
    --data_type $DATA_TYPE \
    --master_skill_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER \
    --use_text_history \
    --use_text_description \
    --pooling_strategy $POOLING \
    --max_epochs 50 \
    --batch_size 4096 \
    --hidden_dim 512 \
    --n_layers 2 \
    --output_dir results/cpp/kw_esco_100k_variants \
    --run_name kw_text_hidden512_layers2 \
    --use_wandb \
    --wandb_project "cpp-kw-esco-100k-variants" \
    --optuna \
    --n_trials 20

echo ""
echo "Experiment 2: Text-only with hidden_dim=1024"
python -m src.cpp.train_cpp_enhanced \
    --data_type $DATA_TYPE \
    --master_skill_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER \
    --use_text_history \
    --use_text_description \
    --pooling_strategy $POOLING \
    --max_epochs 50 \
    --batch_size 4096 \
    --hidden_dim 1024 \
    --n_layers 3 \
    --output_dir results/cpp/kw_esco_100k_variants \
    --run_name kw_text_hidden1024_layers3 \
    --use_wandb \
    --wandb_project "cpp-kw-esco-100k-variants" \
    --optuna \
    --n_trials 20

echo ""
echo "Experiment 3: Text-only with SGD optimizer"
python -m src.cpp.train_cpp_enhanced \
    --data_type $DATA_TYPE \
    --master_skill_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER \
    --use_text_history \
    --use_text_description \
    --pooling_strategy $POOLING \
    --max_epochs 50 \
    --batch_size 4096 \
    --hidden_dim 512 \
    --output_dir results/cpp/kw_esco_100k_variants \
    --run_name kw_text_sgd \
    --use_wandb \
    --wandb_project "cpp-kw-esco-100k-variants" \
    --optimizer sgd \
    --optuna \
    --n_trials 20

echo ""
echo "================================================================================"
echo "All experiments completed using cached embeddings!"
echo "================================================================================"
echo ""
echo "Time saved: ~6 hours (3 experiments × 2 hours each)"
echo ""
echo "Note: All three experiments used the SAME cached embeddings"
echo "because they share the same data configuration:"
echo "  - data_type: $DATA_TYPE"
echo "  - encoder: $ENCODER"
echo "  - modalities: text_history + text_description"
echo "  - pooling: $POOLING"
echo ""
echo "Different model architectures and training hyperparameters"
echo "do NOT affect the cache - they only affect training!"
echo "================================================================================"

