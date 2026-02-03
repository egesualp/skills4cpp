#!/bin/bash
#SBATCH --job-name=cpp_karrierewege_100k        # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/cpp_karrierewege_100k_%j.log    # Log output (%j expands to job ID)
#SBATCH --error=experiments/logs/cpp_karrierewege_100k_%j.err     # Error output
#SBATCH --partition=mcml-dgx-a100-40x8  # Specify the partition to use
#SBATCH --gres=gpu:1                     # Number of GPUs to use
#SBATCH --ntasks=8                       # Number of tasks
#SBATCH --time=03:00:00                   # Time limit



#
# This script runs 3 experiments with different modality configurations:
#   1. Baseline with text history (text only)
#   2. Baseline with text history and skills (multimodal)
#   3. Baseline with skills only (skills only)
#
# EMBEDDINGS CACHING:
# - Each experiment will automatically cache its pre-computed embeddings
# - Cache files are stored in: /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/embeddings/
# - First run: ~2 hours to compute embeddings
# - Subsequent runs with same config: ~2 seconds to load from cache
# - Different modality configs use different cache files (no conflicts)
#
# PERFORMANCE:
# - Without cache: 3 experiments × 2 hours = 6 hours total
# - With cache: 2 hours (first) + instant (second) + instant (third) = ~2 hours total
# - Time saved: 4 hours (66% reduction!)
#
source /dss/dsshome1/02/ra95kix2/miniconda3/bin/activate thesis

echo "================================================================================"
echo "Experiment 1/3: Text History Only"
echo "================================================================================"
echo "This will create a cache file for text embeddings (~4 GB, ~2 hours)"
echo ""
python -m src.cpp.train_cpp_enhanced \
    --data_type karrierewege_100k \
    --master_skill_file results/karrierewege_esco_100k_esco_ground_truth/job_title_skills_master.csv \
    --encoder_text "ElenaSenger/career-path-representation-mpnet-karrierewege" \
    --max_epochs 30 \
    --batch_size 1024 \
    --output_dir results/cpp/kw_esco_100k_baseline \
    --use_text_history \
    --use_text_description \
    --run_name kw_esco_100k_baseline \
    --use_wandb \
    --wandb_project "cpp-kw-esco-100k" \
    --optuna \
    --n_trials 10 \
    --patience 3 \
    --num_workers 4 \
    --pooling_strategy weighted_idf \
    --optimizer adam

