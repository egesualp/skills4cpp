#!/bin/bash
#SBATCH --job-name=kw_100k_skills_battle        # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/kw_100k_skills_battle_%j.log    # Log output (%j expands to job ID)
#SBATCH --error=experiments/logs/kw_100k_skills_battle_%j.err     # Error output
#SBATCH --partition=mcml-dgx-a100-40x8  # Specify the partition to use
#SBATCH --gres=gpu:1                     # Number of GPUs to use
#SBATCH --ntasks=6                       # Number of tasks
#SBATCH --time=03:00:00  

source /dss/dsshome1/02/ra95kix2/miniconda3/bin/activate thesis

python -m src.cpp.decorte.skill_overlap_scoring  --data_type karrierewege_100k --master_skill_file /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/karrierewege_esco_100k_esco_ground_truth/job_title_skills_master.csv --output_dir src/cpp/decorte/results


# Common Configuration
DATA_TYPE="karrierewege_100k"
MASTER_SKILL_FILE="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/karrierewege_esco_100k_esco_ground_truth/job_title_skills_master.csv"
ENCODER_TEXT="ElenaSenger/career-path-representation-mpnet-karrierewege"
ENCODER_SKILL_OLD="ElenaSenger/career-path-representation-mpnet-karrierewege"
ENCODER_SKILL_NEW_DESC="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/with_skill_description/best-model"
ENCODER_SKILL_NEW_NO_DESC="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/no_skill_descriptions/best-model"
OUTPUT_BASE="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/kw_esco_100k_skills_battle"
WANDB_PROJECT="cpp-kw-esco-100k-skills-battle"

# Static Hyperparameters
BATCH_SIZE=4096
EVAL_BATCH_SIZE=2048
MAX_EPOCHS=30
PATIENCE=5
LR=2e-5
HIDDEN_DIM=1024
N_LAYERS=1
DROPOUT=0.1
USE_MODALITY_WEIGHTS=False

## Execution list
# ┌──────────────┬─────────────┬──────────────┐
# │ Pool/Concat  │ With Desc   │ Encoder      │
# ├──────────────┼─────────────┼──────────────┤
# │ Pool         │ Yes         │ Old          │
# │ Pool         │ Yes         │ New-Desc     │
# │ Pool         │ No          │ New-noDesc   │
# │ Concat       │ Yes         │ Old          │
# │ Concat       │ Yes         │ New-Desc     │
# │ Concat       │ No          │ New-noDesc   │
# └──────────────┴─────────────┴──────────────┘

echo -e "\n\n=== Running Experiment 4: Concat - With Desc - Old Encoder ==="
python -m src.cpp.train_cpp_enhanced_v3 \
    --data_type $DATA_TYPE \
    --master_skill_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --encoder_skill $ENCODER_SKILL_OLD \
    --output_dir "${OUTPUT_BASE}/concat_with_desc_old_encoder" \
    --run_name "concat_with_desc_old_encoder" \
    --use_skill_description \
    --use_skill_text \
    --use_text_description \
    --pooling_strategy weighted_idf \
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
    --num_workers 0 \
    --weight_decay 3.5e-6 \
    --profile_data_loading

echo -e "\n\n=== Running Experiment 5: Concat - With Desc - New Desc Encoder ==="
python -m src.cpp.train_cpp_enhanced_v3 \
    --data_type $DATA_TYPE \
    --master_skill_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --encoder_skill $ENCODER_SKILL_NEW_DESC \
    --output_dir "${OUTPUT_BASE}/concat_with_desc_new_desc_encoder" \
    --run_name "concat_with_desc_new_desc_encoder" \
    --use_skill_description \
    --use_skill_text \
    --use_text_description \
    --pooling_strategy weighted_idf \
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
    --num_workers 0 \
    --weight_decay 3.5e-6 \
    --profile_data_loading

echo -e "\n\n=== Running Experiment 6: Concat - No Desc - New No Desc Encoder ==="
python -m src.cpp.train_cpp_enhanced_v3 \
    --data_type $DATA_TYPE \
    --master_skill_file $MASTER_SKILL_FILE \
    --encoder_text $ENCODER_TEXT \
    --encoder_skill $ENCODER_SKILL_NEW_NO_DESC \
    --output_dir "${OUTPUT_BASE}/concat_no_desc_new_no_desc_encoder" \
    --run_name "concat_no_desc_new_no_desc_encoder" \
    --use_skill_text \
    --pooling_strategy weighted_idf \
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
    --num_workers 0 \
    --weight_decay 3.5e-6 \
    --profile_data_loading


