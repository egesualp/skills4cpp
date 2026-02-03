#!/bin/bash

# Define variables for paths and arguments
JOBS_CSV="/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/decorte_master_2.csv"
MODEL_DIR="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h2_soft_deep_larger_val_mpnet_expanded"
OUTPUT_DIR="${MODEL_DIR}/inference_results_soft_labels_deep_larger_val_mpnet_expanded"
TEXT_COL="raw_title"
DESC_COL="raw_description"
ISCO_LEVEL=2
DEVICE="cpu" # Changed to cpu for parallelization
NUM_WORKERS=4 # Added num_workers
ISCO_COL="iscoGroup" # NOTE: Adjust this if your ground truth ISCO column in decorte_master_2.csv is named differently

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the inference script
python -m skill_mapping.v2.isco_inference \
    --jobs_csv "$JOBS_CSV" \
    --model_dir "$MODEL_DIR" \
    --output_path "$OUTPUT_DIR" \
    --text_col "$TEXT_COL" \
    --desc_col "$DESC_COL" \
    --isco_level "$ISCO_LEVEL" \
    --device "$DEVICE" \
    --include_description \
    --compute_metrics \
    --isco_col "$ISCO_COL" \
    --num_workers "$NUM_WORKERS"

echo "Inference complete. Results saved to $OUTPUT_DIR"









