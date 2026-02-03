#!/bin/bash
# run_pipeline.sh - Example pipeline for LLM-based re-ranking

# Configuration
FUSION_SCORES="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/linear_fusion_sum/best_fused_scores.json"
JOBS_CSV="./data/title_pairs_desc/decorte_master.csv"
SKILLS_CSV="./data/esco_datasets/skills_en.csv"
OCCUPATIONS_CSV="./data/esco_datasets/occupations_en.csv"
OCC_SKILLS_CSV="./data/esco_datasets/occupationSkillRelations_en.csv"
OUTPUT_DIR="./outputs/llm_reranking"
ISCO_GROUPS="5120,2654"  # Optional: set to empty string for all jobs

# Create output directory
mkdir -p $OUTPUT_DIR

echo "============================================"
echo "Step 1: Run LLM Re-ranking (with ground truth preparation)"
echo "============================================"
if [ -n "$ISCO_GROUPS" ]; then
    python -m skill_mapping.v3.llm_reranker \
        --fusion_scores_json $FUSION_SCORES \
        --jobs_csv $JOBS_CSV \
        --skills_csv $SKILLS_CSV \
        --occupations_csv $OCCUPATIONS_CSV \
        --occ_skills_csv $OCC_SKILLS_CSV \
        --output_dir $OUTPUT_DIR \
        --top_k 100 \
        --max_workers 5 \
        --isco_groups $ISCO_GROUPS \
        --prepare_ground_truth
else
    python -m skill_mapping.v3.llm_reranker \
        --fusion_scores_json $FUSION_SCORES \
        --jobs_csv $JOBS_CSV \
        --skills_csv $SKILLS_CSV \
        --occupations_csv $OCCUPATIONS_CSV \
        --occ_skills_csv $OCC_SKILLS_CSV \
        --output_dir $OUTPUT_DIR \
        --top_k 100 \
        --max_workers 5 \
        --prepare_ground_truth
fi

echo ""
echo "============================================"
echo "Step 2: Evaluate Results"
echo "============================================"
python -m skill_mapping.v3.evaluate_reranking \
    --reranked_scores $OUTPUT_DIR/llm_reranked_scores_compact.json \
    --original_scores $FUSION_SCORES \
    --ground_truth $OUTPUT_DIR/ground_truth.json \
    --output_dir $OUTPUT_DIR/evaluation

echo ""
echo "============================================"
echo "Pipeline Complete!"
echo "============================================"
echo "Results saved to: $OUTPUT_DIR"
echo "Evaluation metrics: $OUTPUT_DIR/evaluation/comparison_metrics.csv"
echo "Tier analysis: $OUTPUT_DIR/evaluation/tier_analysis.json"

