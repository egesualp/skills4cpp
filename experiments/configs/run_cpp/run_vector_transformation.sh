#!/bin/bash
#SBATCH --job-name=vector_transform
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/vector_transformation/vector_transform_%j.log
#SBATCH --error=experiments/logs/vector_transformation/vector_transform_%j.err
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=8                 # Number of CPUs per task
#SBATCH --time=03:00:00

source /dss/dsshome1/02/ra95kix2/miniconda3/bin/activate thesis

# Add src to PYTHONPATH so that 'from cpp.utils' imports in data_classes.py work
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

echo "Running vector transformation training"
# Custom hyperparameters
python src/cpp/train_vector_transformation.py \
    --data_type karrierewege_100k \
    --encoder ElenaSenger/career-path-representation-mpnet-karrierewege \
    --hidden_sizes 512 \
    --max_epochs 10 \
    --patience 2 \
    --lr 2e-5 \
    --use_wandb \
    --batch_size 16 \
    --dropout \
    --dropout_rate 0.1 \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/vector_transformation_karrierewege_100k_wo_desc \
    --run_name karrierewege_100k \
    --wandb_project vector-transformation-karrierewege-100k \
    --epoch_eval_frac 0.02 \
    --ranking_chunk_size 128 \
    --save_model