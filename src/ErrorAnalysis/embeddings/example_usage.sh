#!/bin/bash
# Example commands for running embedding error analysis with Tasks 1, 3, and 5

# Set common paths
SKILL_SCORES="/path/to/fused_predictions.json"
PRED_TEXT="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/job_title_desc/scores/test_scores_text.pkl"
OUTPUT_DIR="src/error_analysis/embeddings/results"

# Example 1: Task 1 only (Text-Skill Redundancy) - Already working
echo "Running Task 1 only..."
python src/error_analysis/embeddings/quantify_overlap.py \
  --skill_scores_file "$SKILL_SCORES" \
  --method v3 \
  --top_k_skills 10 \
  --scoring_mode idf_only \
  --output_dir "$OUTPUT_DIR/task1_only"

# Example 2: Tasks 1 + 3 (Add Target Proximity Analysis)
echo "Running Tasks 1 + 3..."
python src/error_analysis/embeddings/quantify_overlap.py \
  --skill_scores_file "$SKILL_SCORES" \
  --method v3 \
  --top_k_skills 10 \
  --scoring_mode idf_only \
  --run_task3 \
  --output_dir "$OUTPUT_DIR/task1_3"

# Example 3: Full analysis (Tasks 1 + 3 + 5)
echo "Running full analysis (Tasks 1 + 3 + 5)..."
python src/error_analysis/embeddings/quantify_overlap.py \
  --skill_scores_file "$SKILL_SCORES" \
  --method v3 \
  --top_k_skills 10 \
  --scoring_mode idf_only \
  --use_skill_description \
  --run_task3 \
  --run_task5 \
  --predictions_text "$PRED_TEXT" \
  --output_dir "$OUTPUT_DIR/full_analysis"

# Example 4: Compare multiple models in Task 5
echo "Running Task 5 with multiple model predictions..."
python src/error_analysis/embeddings/quantify_overlap.py \
  --skill_scores_file "$SKILL_SCORES" \
  --method v3 \
  --top_k_skills 10 \
  --scoring_mode weighted \
  --importance_weight 0.8 \
  --use_skill_description \
  --run_task3 \
  --run_task5 \
  --predictions_text "/path/to/text_only_scores.pkl" \
  --predictions_skill "/path/to/skill_only_scores.pkl" \
  --predictions_hybrid "/path/to/hybrid_scores.pkl" \
  --output_dir "$OUTPUT_DIR/multi_model_comparison"

# Example 5: Analyze clean test set
echo "Running analysis on clean test set..."
python src/error_analysis/embeddings/quantify_overlap.py \
  --skill_scores_file "$SKILL_SCORES" \
  --method v3 \
  --eval_clean_test \
  --run_task3 \
  --run_task5 \
  --predictions_text "$PRED_TEXT" \
  --output_dir "$OUTPUT_DIR/clean_test"

echo "Analysis complete! Check $OUTPUT_DIR for results."
