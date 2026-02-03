python src/skill_mapping/v5/fused_scorer.py \
    --esco_dir data/esco_datasets \
    --label_encoder /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h1_soft_deep_larger_val_mpnet/label_encoder.json \
    --task_a /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/experiments/results/infer_decorte_all_jobbert_final_2/predictions.jsonl \
    --task_b /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/decorte_w_desc_3/similarity_scores.json \
    --isco_preds /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h1_soft_deep_larger_val_mpnet_expanded/inference_results_soft_labels_deep_larger_val_mpnet_expanded/isco_predictions.json \
    --decorte_map /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/decorte_master_3.csv \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_h1 \
    --fusion_strategy multiplicative \
    --n_jobs 4 \
    --isco_level 1

python -m skill_mapping.v5.fused_scorer \
    --esco_dir data/esco_datasets \
    --label_encoder /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h2_soft_deep_larger_val_mpnet/label_encoder.json \
    --task_a /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/experiments/results/infer_decorte_all_jobbert_final_2/predictions.jsonl \
    --task_b /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/decorte_w_desc_3/similarity_scores.json \
    --isco_preds /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h2_soft_deep_larger_val_mpnet/inference_results_h2_decorte_master_2/isco_predictions.json \
    --decorte_map /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/decorte_master_3.csv \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/linear_2 \
    --fusion_strategy linear \
    --n_jobs 4 \
    --isco_level 2

=========================

python src/skill_mapping/v5/fused_scorer.py \
    --esco_dir data/esco_datasets \
    --label_encoder /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h2_soft_deep_larger_val_mpnet/label_encoder.json \
    --task_a /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/experiments/results/infer_decorte_desc_all_pjmath_1_final/predictions.jsonl \
    --task_b /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/decorte_w_desc_3/similarity_scores.json \
    --isco_preds /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h2_soft_deep_larger_val_mpnet/inference_results_h2_decorte_master_2/isco_predictions.json \
    --decorte_map /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/decorte_master_3.csv \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/linear_h2_pjmath \
    --fusion_strategy linear \
    --n_jobs 4 \
    --isco_level 2

python src/skill_mapping/v5/fused_scorer.py \
    --esco_dir data/esco_datasets \
    --label_encoder /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h2_soft_deep_larger_val_mpnet/label_encoder.json \
    --task_a /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/experiments/results/infer_decorte_desc_all_pjmath_1_final/predictions.jsonl \
    --task_b /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/decorte_w_desc_3/similarity_scores.json \
    --isco_preds /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h2_soft_deep_larger_val_mpnet/inference_results_h2_decorte_master_2/isco_predictions.json \
    --decorte_map /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/decorte_master_3.csv \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_h2_pjmath \
    --fusion_strategy multiplicative \
    --n_jobs 4 \
    --isco_level 2


python src/skill_mapping/v5/fused_scorer_opt.py \
    --esco_dir data/esco_datasets \
    --label_encoder /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/kw_cp/isco_model_h2_soft_deep_larger_val_mpnet/label_encoder.json \
    --task_a /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/experiments/results/infer_kw_cp_desc_all_jobbert_final/predictions.jsonl \
    --task_b /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/kw_cp/kw_cp_jobbert/similarity_scores.json \
    --isco_preds /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/kw_cp/isco_model_h2_soft_deep_larger_val_mpnet/inference_results_soft_labels_deep_larger_val_mpnet_kw_cp/isco_predictions.json \
    --decorte_map /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/karrierewege_plus_cp_master_3.csv \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/kw_cp/v5_fused_scorer/linear_h2_kw_cp \
    --fusion_strategy linear \
    --n_jobs 4 \
    --isco_level 2 \
    --chunk_size 20000 


# For repeated experiments: convert to Parquet first
python src/skill_mapping/v5/fused_scorer_chunked.py --convert-to-parquet \
    --task_b /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/kw_cp/kw_cp_jobbert/similarity_scores.json \
    --parquet_output /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/kw_cp/kw_cp_jobbert/similarity_scores_chunked.parquet


python src/skill_mapping/v5/fused_scorer_chunked.py \
    --esco_dir data/esco_datasets \
    --label_encoder /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/kw_cp/isco_model_h2_soft_deep_larger_val_mpnet/label_encoder.json \
    --task_a /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/experiments/results/infer_kw_cp_desc_all_jobbert_final_2/predictions.jsonl \
    --task_b_parquet /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/kw_cp/kw_cp_jobbert/similarity_scores_chunked.parquet \
    --isco_preds /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/kw_cp/isco_model_h2_soft_deep_larger_val_mpnet/inference_results_soft_labels_deep_larger_val_mpnet_kw_cp/isco_predictions.json \
    --decorte_map /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/karrierewege_plus_cp_master_3.csv \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/kw_cp/v5_fused_scorer/linear_h2_kw_cp \
    --fusion_strategy linear \
    --isco_level 2 \
    --chunk_size 5000 
