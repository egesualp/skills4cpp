python -m skill_mapping.v2.skill_indexer \
    --skills_csv data/processed/augmentation/augmented_esco_skills.csv \
    --model_name pj-mathematician/JobSkillBGE-large-en-v1.5 \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index \
    --text_column job_brief \
    --batch_size 64 \
    --device cuda

python -m skill_mapping.v2.skill_indexer \
    --skills_csv data/esco_datasets/skills_en_clean.csv \
    --model_name TechWolf/JobBERT-v2 \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index_jobbert \
    --use_raw_features \
    --desc_column description \
    --separator "</s>"

python -m skill_mapping.v2.skill_indexer \
    --skills_csv data/esco_datasets/skills_en_clean.csv \
    --model_name TechWolf/JobBERT-v2 \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index_jobbert_title_only \
    --use_raw_features
    # Note: --desc_column is omitted, so it defaults to None (Title Only)

python -m skill_mapping.v2.similarity_scorer \
    --jobs_csv data/title_pairs_desc/decorte_master_2.csv \
    --index_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index_jobbert_title_only \
    --model_path TechWolf/JobBERT-v2 \
    --output_path /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/decorte_jobbert_title_only/similarity_scores.json  \
    --use_raw_features \
    --top_k 3000 \
    --title_column "raw_title" \
    --test_metrics

python -m skill_mapping.v2.similarity_scorer \
    --jobs_csv data/title_pairs_desc/decorte_master_2.csv \
    --index_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index_jobbert \
    --model_path TechWolf/JobBERT-v2 \
    --output_path /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/decorte_jobbert/similarity_scores.json  \
    --use_raw_features \
    --top_k 3000 \
    --title_column "raw_title" \
    --separator "</s>" \
    --desc_column "raw_description" \
    --test_metrics

# Adding similarity scorer for kw cp #######
python -m skill_mapping.v2.similarity_scorer \
    --jobs_csv data/title_pairs_desc/karrierewege_plus_cp_master_3.csv \
    --index_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index_jobbert_title_only \
    --model_path TechWolf/JobBERT-v2 \
    --output_path /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/kw_cp/kw_cp_jobbert_title_only/similarity_scores.json  \
    --use_raw_features \
    --top_k 3000 \
    --title_column "raw_title" \
    --test_metrics

python -m skill_mapping.v2.similarity_scorer_opt \
    --jobs_csv data/title_pairs_desc/karrierewege_plus_cp_master_3.csv \
    --index_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index_jobbert \
    --model_path TechWolf/JobBERT-v2 \
    --output_path /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/kw_cp/kw_cp_jobbert/similarity_scores.json  \
    --use_raw_features \
    --top_k 1000 \
    --title_column "raw_title" \
    --separator "</s>" \
    --desc_column "raw_description" \
    --test_metrics \
    --save_scores \
    --id_column "job_id"
###############################################

python -m skill_mapping.v2.similarity_scorer \
    --jobs_csv data/title_pairs_desc/decorte_master_2.csv \
    --index_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index_jobbert \
    --model_path TechWolf/JobBERT-v2 \
    --output_path /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/decorte_w_desc_jobbert \
    --use_raw_features \
    --top_k 3000 \
    --separator "</s>" \
    --title_column "raw_title" \
    --desc_column "raw_description" \
    --test_metrics

# Implement different skill pools here
python -m skill_mapping.v2.similarity_scorer \
    --jobs_csv ./data/processed/augmentation/augmented_decorte_occupations_with_desc_2.csv \
    --index_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index \
    --model_path pj-mathematician/JobSkillBGE-large-en-v1.5 \
    --output_path /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/decorte_w_desc_3/similarity_scores.json \
    --text_column skill_brief \
    --id_column job_id \
    --top_k 3000 \
    --batch_size 64 \
    --device cuda \
    --load_embeddings /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/embeddings/decorte_occupations_with_desc_2 \
    --test_metrics

python -m skill_mapping.v2.category_trainer \
    --esco_path ./data/processed/master_datasets_2/master_complete_hierarchy_w_occ.csv \
    --augmented_path data/processed/augmentation/augmented_esco_occupations_2.csv \
    --val_path ./data/processed/augmentation/augmented_talent_clef_taskb_validation.csv \
    --test_path /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/category_test_split.csv \
    --model_path pj-mathematician/JobSkillBGE-large-en-v1.5 \
    --target_level 2 \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/category_model_h2_soft_simple_deep_larger_val \
    --cache_name soft_labels_simple_deep_larger_val \
    --n_trials 30 \
    --max_epochs 50 \
    --final_epochs 100 \
    --device cuda \
    --samples_per_occupation 1 \
    --soft_labels

python -m skill_mapping.v2.isco_trainer \
    --esco_path ./data/esco_datasets/occupations_en.csv \
    --augmented_path data/processed/augmentation/augmented_esco_occupations_with_expanded_alts.csv \
    --test_path /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/category_test_split_isco.csv \
    --model_path pj-mathematician/JobSkillBGE-large-en-v1.5 \
    --isco_level 2 \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h2_soft_deep_larger_val \
    --cache_name soft_labels_deep_larger_val \
    --n_trials 30 \
    --max_epochs 50 \
    --final_epochs 100 \
    --device cuda

python -m skill_mapping.v2.isco_trainer \
    --esco_path ./data/esco_datasets/occupations_en.csv \
    --test_path /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/category_test_split_isco.csv \
    --model_path ElenaSenger/career-path-representation-mpnet-decorte \
    --isco_level 2 \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h2_soft_deep_larger_val_mpnet_2 \
    --cache_name soft_labels_deep_larger_val_mpnet \
    --n_trials 30 \
    --max_epochs 50 \
    --final_epochs 100 \
    --device cuda

python -m skill_mapping.v2.isco_trainer \
    --esco_path ./data/esco_datasets/occupations_en.csv \
    --train_path /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/karrierewege_plus_cp_master_2.csv \
    --test_path /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/category_test_split_isco.csv \
    --model_path ElenaSenger/career-path-representation-mpnet-decorte \
    --isco_level 2 \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h2_soft_deep_larger_val_mpnet_kw_cp_decorte \
    --cache_name soft_labels_deep_larger_val_mpnet_kw_cp_decorte \
    --n_trials 30 \
    --max_epochs 50 \
    --final_epochs 100 \
    --device cuda

# new isco trainings
python -m skill_mapping.v2.isco_trainer \
    --train_path ./data/occupations_en_expanded.csv \
    --test_path /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/category_test_split_isco.csv \
    --model_path TechWolf/JobBERT-v2 \
    --isco_level 2 \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h2_soft_deep_larger_val_desc_jobbert \
    --cache_name soft_labels_deep_larger_val_desc_jobbert \
    --n_trials 30 \
    --max_epochs 50 \
    --final_epochs 100 \
    --device cuda \
    --include_description

python -m skill_mapping.v2.isco_trainer \
    --train_path data/occupations_en_expanded.csv \
    --text_col preferredLabel \
    --desc_col description \
    --test_path data/title_pairs_desc/category_test_split_isco.csv \
    --model_path TechWolf/JobBERT-v2 \
    --isco_level 2 \
    --test_text_col raw_title \
    --test_desc_col raw_description \
    --include_description \
    --prefix role \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h2_soft_deep_larger_val_desc_jobbert \
    --cache_name soft_labels_deep_larger_val_desc_jobbert \
    --n_trials 20 \
    --max_epochs 50 \
    --final_epochs 100 \
    --device cuda

python -m skill_mapping.v2.isco_trainer \
    --train_path /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/karrierewege_plus_cp_master_2.csv \
    --test_text_col raw_title \
    --test_desc_col raw_description \
    --test_path data/title_pairs_desc/category_test_split_isco.csv \
    --model_path TechWolf/JobBERT-v2 \
    --isco_level 2 \
    --include_description \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h2_soft_deep_larger_val_desc_jobbert_kw_cp \
    --cache_name soft_labels_deep_larger_val_desc_jobbert_kw_cp \
    --n_trials 20 \
    --max_epochs 50 \
    --final_epochs 100 \
    --device cuda

python -m skill_mapping.v2.isco_trainer \
    --text_col preferredLabel \
    --test_path data/title_pairs_desc/category_test_split_isco.csv \
    --model_path TechWolf/JobBERT-v2 \
    --isco_level 2 \
    --test_text_col raw_title \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h2_soft_deep_val_jobbert \
    --cache_name soft_labels_deep_val_jobbert \
    --n_trials 20 \
    --max_epochs 50 \
    --final_epochs 100 \
    --device cuda

# new isco trainings
# isco_model_h2_soft_deep_larger_val_mpnet_2
python -m skill_mapping.v2.isco_trainer \
    --train_path ./data/occupations_en_expanded.csv \
    --test_path /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/category_test_split_isco.csv \
    --model_path ElenaSenger/career-path-representation-mpnet-decorte \
    --isco_level 2 \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h2_soft_deep_larger_val_mpnet_expanded \
    --cache_name soft_labels_deep_larger_val_mpnet_expanded \
    --test_text_col raw_title \
    --test_desc_col raw_description \
    --text_col preferredLabel \
    --desc_col description \
    --n_trials 30 \
    --max_epochs 50 \
    --final_epochs 100 \
    --device cuda \
    --include_description

python -m skill_mapping.v2.isco_inference \
    --jobs_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/decorte_master_2.csv \
    --model_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h2_soft_deep_larger_val_mpnet \
    --output_path /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h2_soft_deep_larger_val_mpnet/inference_results_h2_decorte_master_2 \
    --text_col raw_title \
    --desc_col raw_description \
    --isco_level 2 \
    --device cpu \
    --num_workers 4 \
    --include_description \
    --compute_metrics \
    --isco_col iscoGroup \
    --prefix role \
    --esco_path /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/occupations_en.csv 



python -m skill_mapping.v2.isco_inference \
    --jobs_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/decorte_master_2.csv \
    --model_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h1_soft_deep_larger_val_mpnet \
    --output_path /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h1_soft_deep_larger_val_mpnet_expanded/inference_results_soft_labels_deep_larger_val_mpnet_expanded \
    --text_col raw_title \
    --desc_col raw_description \
    --isco_level 1 \
    --device cpu \
    --num_workers 4 \
    --include_description \
    --compute_metrics \
    --isco_col iscoGroup \
    --prefix role \
    --esco_path /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/occupations_en.csv 

#### KW CP ISCO INFERENCE
python -m skill_mapping.v2.isco_trainer \
    --train_path ./data/occupations_en.csv \
    --model_path ElenaSenger/career-path-representation-mpnet-karrierewege-cp \
    --isco_level 2 \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/kw_cp/isco_model_h2_soft_deep_larger_val_mpnet \
    --cache_name soft_labels_deep_larger_val_mpnet_kw_cp \
    --test_text_col raw_title \
    --test_desc_col raw_description \
    --text_col preferredLabel \
    --desc_col description \
    --n_trials 30 \
    --max_epochs 50 \
    --final_epochs 100 \
    --device cuda \
    --include_description \
    --test_path /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/category_test_split_isco_kw_cp.csv

python -m skill_mapping.v2.isco_inference \
    --jobs_csv /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/karrierewege_plus_cp_master_3.csv \
    --model_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/kw_cp/isco_model_h2_soft_deep_larger_val_mpnet \
    --output_path /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/kw_cp/isco_model_h2_soft_deep_larger_val_mpnet/inference_results_soft_labels_deep_larger_val_mpnet_kw_cp \
    --text_col raw_title \
    --desc_col raw_description \
    --isco_level 2 \
    --device cuda \
    --num_workers 4 \
    --include_description \
    --compute_metrics \
    --isco_col iscoGroup \
    --prefix role \
    --esco_path /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/occupations_en.csv
############################

python -m skill_mapping.v2.isco_trainer \
    --train_path ./data/occupations_en_expanded.csv \
    --test_path /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/category_test_split_isco.csv \
    --model_path ElenaSenger/career-path-representation-mpnet-decorte \
    --isco_level 2 \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h2_soft_deep_larger_val_mpnet_expanded_wo_desc \
    --cache_name soft_labels_deep_larger_val_mpnet_expanded_wo_desc \
    --test_text_col raw_title \
    --text_col preferredLabel \
    --n_trials 30 \
    --max_epochs 50 \
    --final_epochs 100 \
    --device cuda

# isco_model_h2_soft_deep_larger_val_mpnet_2
python -m skill_mapping.v2.isco_trainer \
    --train_path ./data/occupations_en.csv \
    --test_path /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/category_test_split_isco.csv \
    --model_path ElenaSenger/career-path-representation-mpnet-decorte \
    --isco_level 2 \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h2_soft_deep_larger_val_mpnet \
    --cache_name soft_labels_deep_larger_val_mpnet \
    --test_text_col raw_title \
    --test_desc_col raw_description \
    --text_col preferredLabel \
    --desc_col description \
    --n_trials 30 \
    --max_epochs 50 \
    --final_epochs 100 \
    --device cuda \
    --include_description

python -m skill_mapping.v2.isco_trainer \
    --train_path ./data/occupations_en.csv \
    --test_path /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/category_test_split_isco.csv \
    --model_path ElenaSenger/career-path-representation-mpnet-decorte \
    --isco_level 3 \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h3_soft_deep_larger_val_mpnet \
    --cache_name soft_labels_deep_larger_val_mpnet_h3 \
    --test_text_col raw_title \
    --test_desc_col raw_description \
    --text_col preferredLabel \
    --desc_col description \
    --n_trials 30 \
    --max_epochs 50 \
    --final_epochs 100 \
    --device cuda \
    --prefix role \
    --include_description

python -m skill_mapping.v2.isco_trainer \
    --train_path ./data/occupations_en.csv \
    --test_path /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/category_test_split_isco.csv \
    --model_path ElenaSenger/career-path-representation-mpnet-decorte \
    --isco_level 4 \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h4_soft_deep_larger_val_mpnet \
    --cache_name soft_labels_deep_larger_val_mpnet_h4 \
    --test_text_col raw_title \
    --test_desc_col raw_description \
    --text_col preferredLabel \
    --desc_col description \
    --n_trials 30 \
    --max_epochs 50 \
    --final_epochs 100 \
    --device cpu \
    --prefix role \
    --include_description

python -m skill_mapping.v2.isco_trainer \
    --train_path ./data/occupations_en.csv \
    --test_path /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/category_test_split_isco.csv \
    --model_path ElenaSenger/career-path-representation-mpnet-decorte \
    --isco_level 2 \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/isco_model_h2_soft_deep_larger_val_mpnet_wo_desc \
    --cache_name soft_labels_deep_larger_val_mpnet_wo_desc \
    --test_text_col raw_title \
    --text_col preferredLabel \
    --n_trials 30 \
    --max_epochs 50 \
    --final_epochs 100 \
    --device cuda


python -m skill_mapping.v2.category_trainer \
    --esco_path ./data/processed/master_datasets_2/master_complete_hierarchy_w_occ.csv \
    --augmented_path data/processed/augmentation/augmented_esco_occupations_with_expanded_alts.csv \
    --test_path /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/category_test_split.csv \
    --model_path pj-mathematician/JobSkillBGE-large-en-v1.5 \
    --test_text_col raw_title \
    --test_desc_col raw_description \
    --target_level 2 \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/category_model_h2_soft_deep_larger_val \
    --cache_name soft_labels_deep_larger_val \
    --n_trials 30 \
    --max_epochs 50 \
    --final_epochs 100 \
    --device cuda \
    --samples_per_occupation 10 \
    --soft_labels

python -m skill_mapping.v2.category_inference \
    --jobs_csv ./data/processed/augmentation/augmented_decorte_occupations_with_desc_2.csv \
    --id_col job_id \
    --model_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/category_model_h1_soft_deep_larger_val \
    --output_path decorte_w_desc_2_inference \
    --compute_metrics \
    --esco_path ./data/processed/master_datasets_2/master_complete_hierarchy_w_occ.csv \
    --text_col skill_brief \
    --embeddings_path /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/embeddings/decorte_occupations_with_desc_2


python -m skill_mapping.v2.bayesian_fuser \
    --similarity_scores_json /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/decorte_w_desc_2/similarity_scores.json \
    --category_scores_json /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/category_model_h1_soft_deep_larger_val/decorte_w_desc_2_inference/category_scores.json \
    --jobs_csv ./data/processed/augmentation/augmented_decorte_occupations_with_desc_2.csv \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/bayesian_fusion_3_sum \
    --fusion_mode bayesian \
    --weights 0.0,0.5,1.0,1.5,2.0 \
    --temperatures 0.1,0.2,0.3,0.4,0.5,1.0,2.0,5.0 \
    --thresholds 0.0,0.1,0.2,0.3,0.4,0.5 \
    --num_workers 24 \
    --aggregate sum \
    --save_strategy best

python -m skill_mapping.v2.bayesian_fuser \
    --similarity_scores_json /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/decorte_w_desc_3/similarity_scores.json \
    --category_scores_json /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/category_model_h1_soft_deep_larger_val/decorte_w_desc_2_inference/category_scores.json \
    --jobs_csv ./data/title_pairs_desc/decorte_master_3.csv \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/bayesian_fusion_w_taskA_sum \
    --fusion_mode bayesian \
    --weights 0.0,0.5,1.0,1.5,2.0 \
    --temperatures 0.1,0.2,0.3,0.4,0.5,1.0,2.0,5.0 \
    --thresholds 0.0,0.1,0.2,0.3,0.4,0.5 \
    --num_workers 24 \
    --aggregate sum \
    --save_strategy best \
    --task_a_scores_json /dss/dsshome1/02/ra95kix2/thesis/skills4cpp/experiments/results/infer_decorte_desc_all_pjmath_1_final/predictions.jsonl \
    --task_a_k 10


python -m skill_mapping.v2.bayesian_fuser \
    --similarity_scores_json /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/decorte_w_desc_3/similarity_scores.json \
    --category_scores_json /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/category_model_h1_soft_deep_larger_val/decorte_w_desc_2_inference/category_scores.json \
    --jobs_csv ./data/title_pairs_desc/decorte_master_3.csv \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v2_bayesian_fuser/bayesian_h1_max \
    --fusion_mode bayesian \
    --weights 0.0,0.5,1.0,1.5,2.0 \
    --temperatures 0.1,0.2,0.3,0.4,0.5,1.0,2.0,5.0 \
    --thresholds 0.0,0.1,0.2,0.3,0.4,0.5 \
    --num_workers 24 \
    --aggregate max \
    --top_k 3000

python -m skill_mapping.v2.bayesian_fuser \
    --similarity_scores_json /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/decorte_w_desc_3/similarity_scores.json \
    --category_scores_json /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/category_model_h1_soft_deep_larger_val/decorte_w_desc_2_inference/category_scores.json \
    --jobs_csv ./data/title_pairs_desc/decorte_master_3.csv \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v2_bayesian_fuser/linear_h1_max \
    --fusion_mode linear \
    --alphas 0.0,0.1,0.2,0.3,0.5,0.7 \
    --temperatures 0.1,0.2,0.3,0.4,0.5,1.0 \
    --thresholds 0.0,0.1,0.2,0.3,0.4,0.5 \
    --num_workers 24 \
    --aggregate max \
    --top_k 3000

python -m skill_mapping.v2.bayesian_fuser \
    --similarity_scores_json /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/decorte_w_desc_3/similarity_scores.json \
    --category_scores_json /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/category_model_h1_soft_deep_larger_val/decorte_w_desc_2_inference/category_scores.json \
    --jobs_csv ./data/title_pairs_desc/decorte_master_3.csv \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/linear_fusion_2_sum_final_2 \
    --fusion_mode linear \
    --alphas 0.0,0.1,0.3,0.7 \
    --temperatures 0.2,0.5,1.0 \
    --thresholds 0.0,0.2,0.4 \
    --num_workers 16 \
    --aggregate sum \
    --top_k 3000


python -m skill_mapping.v2.bayesian_fuser \
    --similarity_scores_json /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/decorte_w_desc_3/similarity_scores.json \
    --category_scores_json /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/category_model_h1_soft_deep_larger_val/decorte_w_desc_2_inference/category_scores.json \
    --jobs_csv ./data/title_pairs_desc/decorte_master_3.csv \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v2_bayesian_fuser/linear_h1_sum_100 \
    --fusion_mode linear \
    --alphas 0.0,0.1,0.2,0.3,0.5,0.7 \
    --temperatures 0.1,0.2,0.3,0.4,0.5,1.0 \
    --thresholds 0.0,0.1,0.2,0.3,0.4,0.5 \
    --num_workers 16 \
    --aggregate sum \
    --top_k 100 \
    --save_strategy best