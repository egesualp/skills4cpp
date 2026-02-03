python -m src.error_analysis.task_1_triangle_baseline \
    --text /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/job_title_desc/scores/test_scores_text.pkl \
    --hybrid /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte_static/job_titles_skills_pjmath_desc_weighted_idf_logpool_adv_2/scores/test_scores_text.pkl \
    --skill /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skills_only_pjmath_desc_weighted_idf_logpool_adv/scores/test_scores_text.pkl

python -m src.error_analysis.task_2_gain_loss \
    --text /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/job_title_desc/scores/test_scores_text.pkl \
    --hybrid /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte_static/job_titles_skills_pjmath_desc_weighted_idf_logpool_adv_2/scores/test_scores_text.pkl \
    --skill /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skills_only_pjmath_desc_weighted_idf_logpool_adv/scores/test_scores_text.pkl

python -m src.error_analysis.task_3_magnitude_and_conflict \
    --text /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/job_title_desc/scores/test_scores_text.pkl \
    --skill /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skills_only_pjmath_desc_weighted_idf_logpool_adv/scores/test_scores_text.pkl

python -m src.error_analysis.task_4_isco_groups \
    --text /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/job_title_desc/scores/test_scores_text.pkl \
    --hybrid /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte_static/job_titles_skills_pjmath_desc_weighted_idf_logpool_adv_2/scores/test_scores_text.pkl \
    --skill /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skills_only_pjmath_desc_weighted_idf_logpool_adv/scores/test_scores_text.pkl


python -m src.error_analysis.task_5_skill_sensitivity \
    --text /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/job_title_desc/scores/test_scores_text.pkl \
    --hybrid /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte_static/job_titles_skills_pjmath_desc_weighted_idf_logpool_adv_2/scores/test_scores_text.pkl \
    --skill /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skills_only_pjmath_desc_weighted_idf_logpool_adv/scores/test_scores_text.pkl \
    --fused_json /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_h2_pjmath/fused_predictions.json

python -m src.error_analysis.task_5_2_skill_sensitivity \
    --text /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/job_title_desc/scores/test_scores_text.pkl \
    --hybrid /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte_static/job_titles_skills_pjmath_desc_weighted_idf_logpool_adv_2/scores/test_scores_text.pkl \
    --fused_json /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_h2_pjmath/fused_predictions.json