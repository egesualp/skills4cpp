python -m skill_mapping.v4.llm_fetcher \
    --fusion_scores_json /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/decorte_w_desc_3/similarity_scores.json \
    --jobs_csv ./data/title_pairs_desc/decorte_master_3.csv \
    --skills_csv ./data/esco_datasets/skills_en.csv \
    --occupations_csv ./data/esco_datasets/occupations_en.csv \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v4_llm_2 \
    --isco_groups 25

python -m skill_mapping.v4.score_adjuster \
    --llm_responses /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v4_llm/llm_responses.jsonl.gz \
    --original_scores /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/linear_fusion_sum/best_fused_scores.json \
    --ground_truth /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/llm_reranking/ground_truth.json \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v4_reranking \
    --essential_base 3.0 \
    --optional_base 2.0 \
    --irrelevant_base 1.0 \
    --epsilon 0.1


python -m skill_mapping.v4.score_adjuster \
    --llm_responses /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v4_llm/llm_responses.jsonl.gz \
    --original_scores /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/linear_fusion_sum/best_fused_scores.json \
    --ground_truth /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/llm_reranking/ground_truth.json \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v4_reranking \
    --grid_search \
    --essential_grid 1.0 1.5 2.0 2.5 3.0 \
    --optional_grid 0.5 0.8 0.9 1.0 2.0 \
    --irrelevant_grid 0.1 0.2 0.3 0.4 0.5 \
    --epsilon_grid 0.05 0.1 0.2 0.4 0.6 0.8