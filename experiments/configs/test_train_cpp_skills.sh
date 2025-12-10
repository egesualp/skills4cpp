#!/bin/bash
#SBATCH --job-name=embedding_finetuning        # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/embedding_finetuning_%j.log    # Log output (%j expands to job ID)
#SBATCH --error=experiments/logs/embedding_finetuning_%j.err     # Error output
#SBATCH --partition=mcml-dgx-a100-40x8  # Specify the partition to use
#SBATCH --gres=gpu:1                     # Number of GPUs to use
#SBATCH --ntasks=8                       # Number of tasks
#SBATCH --time=03:00:00                   # Time limit

echo "================================"
echo "Testing Skill-Based Training"
echo "================================"
echo ""

# Set paths
source /dss/dsshome1/02/ra95kix2/miniconda3/bin/activate thesis
# Activate environment if needed
# source /path/to/venv/bin/activate

echo "Running skill-based training test..."
echo ""

python -m src.cpp.train_cpp_skills \
    --data_type karrierewege_100k \
    --job_title_skills_csv results/karrierewege_esco_100k_esco_ground_truth/job_title_skills_master.csv \
    --skills_csv data/esco_datasets/skills_en.csv \
    --occupations_csv data/esco_datasets/occupations_en.csv \
    --model_name ElenaSenger/career-path-representation-mpnet-karrierewege \
    --alpha_decay 0.5 \
    --batch_size 128 \
    --learning_rate 2e-5 \
    --num_epochs 2 \
    --patience 2 \
    --use_skill_description \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models \
    --save_model \
    --run_name "test_skill_training" \
    --num_workers 4

echo ""
echo "================================"
echo "Test complete!"
echo "================================"

