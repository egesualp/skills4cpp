MASTER_SKILL_FILE="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_h2_pjmath/fused_predictions.json"
ENCODER_SKILL="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/decorte_embedding_final/decorte_weighted_0_8_topk_80skills_gte_base/best-model"
ENCODER_SKILL_MPNET=/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/models/decorte_embedding_final/decorte_weighted_0_8_topk_10skills_es_mpnet/best-model-mpnet
SKILL_EMBEDDINGS_DIR=/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index

PREDICTIONS_TEXT=/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/job_title_desc/scores/test_scores_text.pkl
PREDICTIONS_SKILL=/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte_static_v3/skills_only_mpnet_desc_optuna_2/scores/test_scores_text.pkl
PREDICTIONS_HYBRID=/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte_static_v3/job_titles_skills_mpnet_desc_2/scores/test_scores_text.pkl

python src/error_analysis/embeddings/quantify_overlap.py \
--method v3 \
--top_k_skills 10 \
--skill_selection_strategy top_k \
--scoring_mode idf_only \
--skill_scores_file $MASTER_SKILL_FILE \
--encoder_skill $ENCODER_SKILL \
--use_text_description \
--use_skill_description \
--top_k_skills 80 \
--skill_selection_strategy top_k \
--scoring_mode weighted \
--importance_weight 0.8

python src/error_analysis/embeddings/quantify_overlap.py \
--method v3 \
--top_k_skills 10 \
--skill_selection_strategy top_k \
--scoring_mode idf_only \
--skill_scores_file $MASTER_SKILL_FILE \
--encoder_skill $ENCODER_SKILL_MPNET \
--use_text_description \
--use_skill_description \
--top_k_skills 10 \
--skill_selection_strategy top_k \
--scoring_mode weighted \
--importance_weight 0.8 \
--run_task3 \
--run_task5 \
--predictions_text $PREDICTIONS_TEXT \
--predictions_skill $PREDICTIONS_SKILL \
--predictions_hybrid $PREDICTIONS_HYBRID

python src/error_analysis/embeddings/quantify_overlap.py \
--skill_scores_file /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_h2_pjma_skill_name_only/best_fused_scores.json \
--encoder_skill ElenaSenger/career-path-representation-mpnet-decorte \
--skill_embeddings_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/embeddings/skill_embeddings_mpnet_tuned

python src/error_analysis/embeddings/variance_analysis.py \
--mode v3 \
--encoder_skill $ENCODER_SKILL_MPNET \
--skill_scores_file $MASTER_SKILL_FILE \
--top_k_skills 10 \
--skill_selection_strategy top_k \
--scoring_mode weighted \
--importance_weight 0.8

python src/error_analysis/embeddings/variance_analysis.py \
--mode v2 \
--skill_embeddings_dir $SKILL_EMBEDDINGS_DIR \
--skill_scores_file $MASTER_SKILL_FILE


    parser.add_argument("--data_type", type=str, default="decorte")
    parser.add_argument("--skill_embeddings_dir", type=str, 
                       default="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/skill_index")
    parser.add_argument("--skill_scores_file", type=str, 
                       default="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_h2_pjmath/fused_predictions.json")
    parser.add_argument("--embeddings_cache_dir", type=str, 
                       default="/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/embeddings",
                       help="Directory to save/load pre-computed embeddings cache")
    parser.add_argument("--encoder_text", type=str, default="ElenaSenger/career-path-representation-mpnet-decorte")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of samples")
    parser.add_argument("--alpha", type=float, default=1.0, help="Confidence weight alpha")
    parser.add_argument("--beta", type=float, default=1.0, help="IDF weight beta")
    
    # V3 Arguments
    parser.add_argument("--mode", type=str, default="v2", choices=["v2", "v3", "both"], help="Analysis mode")
    parser.add_argument("--encoder_skill", type=str, default="", help="Skill encoder for V3")
    parser.add_argument("--top_k_skills", type=int, default=10, help="Top K skills for V3")
    parser.add_argument("--skill_selection_strategy", type=str, default="top_k", choices=["top_k", "stratified"])
    parser.add_argument("--scoring_mode", type=str, default="idf_only", choices=["idf_only", "scores_only", "weighted"])
    parser.add_argument("--importance_weight", type=float, default=0.5)
