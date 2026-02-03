#!/bin/bash
# Test command for extended embedding error analysis
# This is a ready-to-run example using actual file paths from your system

# NOTE: Update these paths based on your actual data files
SKILL_SCORES_FILE="/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/results/decorte_jobbert_v2_baseline/job_title_skills_master.csv"
PREDICTIONS_TEXT="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/job_title_desc/scores/test_scores_text.pkl"
OUTPUT_DIR="/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/src/error_analysis/embeddings/results/test_run"

echo "Running extended embedding error analysis..."
echo "Output directory: $OUTPUT_DIR"

python src/error_analysis/embeddings/quantify_overlap.py \
  --skill_scores_file "$SKILL_SCORES_FILE" \
  --data_type decorte \
  --method v3 \
  --top_k_skills 10 \
  --scoring_mode idf_only \
  --use_skill_description \
  --encoder_text "ElenaSenger/career-path-representation-mpnet-decorte" \
  --run_task3 \
  --run_task5 \
  --predictions_text "$PREDICTIONS_TEXT" \
  --output_dir "$OUTPUT_DIR" \
  --seed 42

echo ""
echo "Analysis complete!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  - overlap_stats.csv (Task 1 results)"
echo "  - task3_target_proximity.json (Task 3 results)"
echo "  - task5_correctness_patterns.json (Task 5 results)"
echo "  - visualizations/ (PNG plots at 300 DPI)"
