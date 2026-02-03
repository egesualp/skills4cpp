python -m src.cpp.decorte.skill_overlap_scoring \
    --data_type decorte \
    --skill_scores_file /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/linear_fusion_sum/best_fused_scores.json \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skill_overlap_scores
    return train_pairs, train_job_ids, val_pairs, val_job_ids, test_pairs, test_job_ids

python -m src.cpp.decorte.skill_overlap_scoring_v2 \
  --data_type decorte \
  --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skill_overlap_scores_top50 \
  --top_k_skills 50 \
  --skill_scores_file /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/decorte_w_desc_3/similarity_scores.json \
  --splits train val test \
  --eval_clean_test

python -m src.cpp.decorte.skill_overlap_scoring_v2 \
  --data_type decorte \
  --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skill_overlap_scores_top100 \
  --top_k_skills 100 \
  --skill_scores_file /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/decorte_w_desc_3/similarity_scores.json \
  --splits test \
  --eval_clean_test

python -m src.cpp.decorte.skill_overlap_scoring_v2 \
  --data_type decorte \
  --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skill_overlap_scores_top10 \
  --top_k_skills 10 \
  --skill_scores_file /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/decorte_w_desc_3/similarity_scores.json \
  --splits test \
  --eval_clean_test

### Run from here for fused

python -m src.cpp.decorte.skill_overlap_scoring_v2 \
  --data_type decorte \
  --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skill_overlap_scores_all_isco_fused \
  --skill_scores_file /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_h2_pjmath/fused_predictions.json \
  --splits val test \
  --eval_clean_test

python -m src.cpp.decorte.skill_overlap_scoring_v2 \
  --data_type decorte \
  --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skill_overlap_scores_top1000_isco_fused \
  --top_k_skills 1000 \
  --skill_scores_file /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_h2_pjmath/fused_predictions.json \
  --splits val test \
  --eval_clean_test

# kw cp
python -m src.cpp.decorte.skill_overlap_scoring_v2 \
  --data_type karrierewege_cp \
  --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/karrierewege_cp/skill_overlap_scores_top1000_isco_fused \
  --top_k_skills 1000 \
  --skill_scores_file /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/kw_cp/v5_fused_scorer/linear_h2_kw_cp/fused_predictions.jsonl \
  --splits val test \
  --eval_clean_test \
  --save_scores

python -m src.cpp.decorte.skill_overlap_scoring_v2 \
  --data_type decorte \
  --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skill_overlap_scores_top100_isco_fused_4_h1 \
  --top_k_skills 100 \
  --skill_scores_file /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_h1/fused_predictions.json \
  --splits test \
  --eval_clean_test \
  --save_scores

python -m src.cpp.decorte.skill_overlap_scoring_v2 \
  --data_type decorte \
  --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skill_overlap_scores_top100_isco_fused_5_h2 \
  --top_k_skills 100 \
  --skill_scores_file /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/linear_h2_pjmath/fused_predictions.json \
  --splits test \
  --eval_clean_test \
  --save_scores

# kw cp
python -m src.cpp.decorte.skill_overlap_scoring_v2 \
  --data_type karrierewege_cp \
  --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/karrierewege_cp/skill_overlap_scores_top100_isco_fused \
  --top_k_skills 100 \
  --skill_scores_file /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/kw_cp/v5_fused_scorer/linear_h2_kw_cp/fused_predictions.jsonl \
  --splits test \
  --eval_clean_test \
  --save_scores

python -m src.cpp.decorte.skill_overlap_scoring_v2 \
  --data_type decorte \
  --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skill_overlap_scores_top100_isco_fused_6 \
  --top_k_skills 100 \
  --skill_scores_file /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_3/fused_predictions.json \
  --splits train val test \
  --eval_clean_test \
  --save_scores

python -m src.cpp.decorte.skill_overlap_scoring_v2 \
  --data_type decorte \
  --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skill_overlap_scores_top100_isco_fused_7 \
  --top_k_skills 100 \
  --skill_scores_file /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_3/fused_predictions.json \
  --splits test \
  --eval_clean_test \
  --save_scores

python -m src.cpp.decorte.skill_overlap_scoring_v2 \
  --data_type decorte \
  --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skill_overlap_scores_top100_isco_fused_7 \
  --top_k_skills 100 \
  --skill_scores_file /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_3/fused_predictions.json \
  --splits test \
  --eval_clean_test \
  --save_scores

python -m src.cpp.decorte.skill_overlap_scoring_v2 \
  --data_type decorte \
  --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skill_overlap_scores_top100_isco_fused_8 \
  --top_k_skills 100 \
  --skill_scores_file /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/linear_h2_pjmath/fused_predictions.json  \
  --splits test \
  --eval_clean_test \
  --save_scores

python -m src.cpp.decorte.skill_overlap_scoring_v3 \
  --data_type decorte \
  --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skill_overlap_scores_top100_isco_fused_9 \
  --top_k_skills 100 \
  --skill_scores_file /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/linear_h2_pjmath/fused_predictions.json  \
  --splits test \
  --eval_clean_test \
  --save_scores

python -m src.cpp.decorte.skill_overlap_scoring_v3 \
  --data_type decorte \
  --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skill_overlap_scores_top100_isco_fused_10 \
  --top_k_skills 100 \
  --skill_scores_file /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_3/fused_predictions.json  \
  --splits test \
  --eval_clean_test

python -m src.cpp.decorte.skill_overlap_scoring_v3 \
  --data_type decorte \
  --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skill_overlap_scores_top100_isco_fused_11 \
  --top_k_skills 100 \
  --skill_scores_file /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_h2_pjmath/fused_predictions.json  \
  --splits test \
  --eval_clean_test \
  --save_scores

python -m src.cpp.decorte.skill_overlap_scoring_v1 \
  --data_type decorte \
  --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skill_overlap_scores_top100_isco_fused_v1_old \
  --top_k_skills 100 \
  --skill_scores_file /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_h2_pjmath_old/fused_predictions.json \
  --splits test \
  --eval_clean_test \
  --save_scores


python -m src.cpp.decorte.skill_overlap_scoring_v2 \
  --data_type decorte \
  --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skill_overlap_scores_top100_isco_fused_v2_old \
  --top_k_skills 100 \
  --skill_scores_file /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_h2_pjmath_old/fused_predictions.json \
  --splits test \
  --eval_clean_test


python -m src.cpp.decorte.skill_overlap_scoring_v2 \
  --data_type decorte \
  --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skill_overlap_scores_top50_isco_fused \
  --top_k_skills 50 \
  --skill_scores_file /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_h2_pjmath/fused_predictions.json \
  --splits val test \
  --eval_clean_test

# kw_cp
python -m src.cpp.decorte.skill_overlap_scoring_v2 \
  --data_type karrierewege_cp \
  --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/karrierewege_cp/skill_overlap_scores_top50_isco_fused \
  --top_k_skills 50 \
  --skill_scores_file /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/kw_cp/v5_fused_scorer/linear_h2_kw_cp/fused_predictions.jsonl \
  --splits val test \
  --eval_clean_test

python -m src.cpp.decorte.skill_overlap_scoring_v3 \
  --data_type decorte \
  --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skill_overlap_scores_thres_0_9_isco_fused \
  --min_skill_confidence 0.9 \
  --skill_scores_file /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_h2_pjmath/fused_predictions.json \
  --splits val test \
  --eval_clean_test

# kw_cp
python -m src.cpp.decorte.skill_overlap_scoring_v3 \
  --data_type karrierewege_cp \
  --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/karrierewege_cp/skill_overlap_scores_thres_0_7_isco_fused \
  --min_skill_confidence 0.7 \
  --skill_scores_file /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/kw_cp/v5_fused_scorer/linear_h2_kw_cp/fused_predictions.jsonl \
  --splits val test \
  --eval_clean_test


python -m src.cpp.decorte.skill_overlap_scoring_v2 \
  --data_type decorte \
  --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skill_overlap_scores_top10_isco_fused \
  --top_k_skills 10 \
  --skill_scores_file /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_h2_pjmath/fused_predictions.json \
  --splits val test \
  --eval_clean_test



  python -m src.cpp.decorte.skill_overlap_scoring_v2 \
  --data_type decorte_esco \
  --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte_esco/skill_overlap_scores \
  --splits train val test \
  --eval_clean_test

python -m src.cpp.decorte.occupation_score_fuser \
    --text_scores_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/job_title_desc/scores \
    --overlap_scores_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skill_overlap_scores_top100_isco_fused_6 \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/fused_scores_linear_isco_fused_final \
    --fusion_mode linear \
    --alphas "0.0,0.3,0.5,0.7,1.0" \
    --save_scores \
    --grid_normalize

# kw_cp
python -m src.cpp.decorte.occupation_score_fuser \
    --text_scores_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/karrierewege_cp_static/job_titles_desc/scores \
    --overlap_scores_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/karrierewege_cp/skill_overlap_scores_top1000_isco_fused \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/karrierewege_cp/fused_scores_linear_isco_fused_final \
    --fusion_mode linear \
    --alphas "0.0,0.3,0.5,0.7,1.0" \
    --save_scores \
    --grid_normalize

python -m src.cpp.decorte.occupation_score_fuser \
    --text_scores_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/job_title_desc/scores \
    --overlap_scores_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skill_overlap_scores_top100_isco_fused_6 \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/fused_scores_bayesian_isco_fused_final \
    --fusion_mode bayesian \
    --weights 0.0,0.5,1.0,1.5,2.0 \
    --epsilons "1e-1,1e-2,1e-4,1e-6" \
    --grid_normalize \
    --save_scores