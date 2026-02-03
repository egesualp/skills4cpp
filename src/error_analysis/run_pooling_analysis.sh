#!/bin/bash
# Commands to run pooling method analysis

# Activate conda environment with required packages
source /dss/dsshome1/02/ra95kix2/miniconda3/etc/profile.d/conda.sh
conda activate thesis

# Define paths
PREDICTIONS_TEXT=/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/job_title_desc/scores/test_scores_text.pkl
PREDICTIONS_SKILL=/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte_static_v3/skills_only_mpnet_desc_optuna_2/scores/test_scores_text.pkl
PREDICTIONS_HYBRID=/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte_static_v3/job_titles_skills_mpnet_desc_2/scores/test_scores_text.pkl
MASTER_SKILL_FILE=/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_h2_pjmath/fused_predictions.json
MASTER_JOB_DATA=/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/decorte_master_3.csv
ESCO_TAXONOMY=/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/occupationSkillRelations_en.csv

OUTPUT_DIR=/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/src/error_analysis/pooling_analysis_output_v4

# Run the analysis
python /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/src/error_analysis/pooling_method_analysis.py \
    --text_scores "$PREDICTIONS_TEXT" \
    --skill_scores "$PREDICTIONS_SKILL" \
    --hybrid_scores "$PREDICTIONS_HYBRID" \
    --master_skill_file "$MASTER_SKILL_FILE" \
    --master_job_data "$MASTER_JOB_DATA" \
    --esco_taxonomy_file "$ESCO_TAXONOMY" \
    --output_dir "$OUTPUT_DIR"

echo "Analysis complete! Output saved to: $OUTPUT_DIR"
