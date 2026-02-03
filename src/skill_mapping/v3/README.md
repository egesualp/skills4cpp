# v3: LLM-based Re-ranking for Job-to-Skill Mapping

This module implements LLM-based re-ranking using GPT-4o-mini to refine skill predictions from the linear fusion model. The LLM classifies skills into three tiers (Essential, Optional, Irrelevant) and re-scores them to improve ranking quality.

## Overview

The re-ranking pipeline consists of:

1. **Input**: Top-100 candidate skills from linear fusion model
2. **LLM Classification**: GPT-4o-mini classifies each skill into tiers
3. **Re-scoring**: Tiered scoring ensures proper separation while preserving relative order
4. **Evaluation**: Compare against original scores using mAP, Recall@K, NDCG@K

## Key Features

- **Tiered Classification**: Essential > Optional > Irrelevant
- **Preserved Ordering**: Within each tier, original ranking is maintained
- **Parallel Processing**: Async/concurrent API calls for efficiency
- **ISCO Filtering**: Filter jobs by ISCO occupation groups
- **Comprehensive Evaluation**: Full metrics suite for performance analysis

## Installation

Install required dependencies:

```bash
pip install openai loguru tqdm pandas numpy
```

Set your OpenAI API key:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

## Usage

### 1. Prepare Ground Truth Labels

First, extract ground truth labels from ESCO occupation-skill mappings:

```bash
python -m skill_mapping.v3.prepare_ground_truth \
    --jobs_csv ./data/title_pairs_desc/decorte_master.csv \
    --occupations_csv ./data/esco_datasets/occupations_en.csv \
    --occ_skills_csv ./data/esco_datasets/occupationSkillRelations_en.csv \
    --output_json ./data/processed/ground_truth.json
```

With ISCO group filtering:

```bash
python -m skill_mapping.v3.prepare_ground_truth \
    --jobs_csv ./data/title_pairs_desc/decorte_master.csv \
    --occupations_csv ./data/esco_datasets/occupations_en.csv \
    --occ_skills_csv ./data/esco_datasets/occupationSkillRelations_en.csv \
    --output_json ./data/processed/ground_truth_isco_5120_2654.json \
    --isco_groups 5120,2654
```

Only essential skills:

```bash
python -m skill_mapping.v3.prepare_ground_truth \
    --jobs_csv ./data/title_pairs_desc/decorte_master.csv \
    --occupations_csv ./data/esco_datasets/occupations_en.csv \
    --occ_skills_csv ./data/esco_datasets/occupationSkillRelations_en.csv \
    --output_json ./data/processed/ground_truth_essential.json \
    --relation_type essential
```

### 2. Run LLM Re-ranking

Re-rank the top-100 skills using GPT-4o-mini:

```bash
python -m skill_mapping.v3.llm_reranker \
    --fusion_scores_json /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/linear_fusion_sum/best_fused_scores.json \
    --jobs_csv ./data/title_pairs_desc/decorte_master.csv \
    --skills_csv ./data/esco_datasets/skills_en.csv \
    --occupations_csv ./data/esco_datasets/occupations_en.csv \
    --output_dir ./outputs/llm_reranking \
    --top_k 100 \
    --max_workers 5
```

With ISCO group filtering:

```bash
python -m skill_mapping.v3.llm_reranker \
    --fusion_scores_json /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/linear_fusion_sum/best_fused_scores.json \
    --jobs_csv ./data/title_pairs_desc/decorte_master.csv \
    --skills_csv ./data/esco_datasets/skills_en.csv \
    --occupations_csv ./data/esco_datasets/occupations_en.csv \
    --output_dir ./outputs/llm_reranking_isco_filtered \
    --top_k 100 \
    --max_workers 5 \
    --isco_groups 5120,2654
```

### 3. Evaluate Results

Compare LLM re-ranked scores against original fusion scores:

```bash
python -m skill_mapping.v3.evaluate_reranking \
    --reranked_scores /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/llm_reranking/llm_reranked_scores.json \
    --original_scores /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/linear_fusion_sum/best_fused_scores.json \
    --ground_truth /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/llm_reranking/ground_truth.json \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/llm_reranking/evaluation
```

## Scoring Logic

The re-scoring formula preserves tier separation while maintaining relative order:

```
score_final = BaseScore_tier + (epsilon × OriginalScore)

where:
    BaseScore_Essential   = 3.0
    BaseScore_Optional    = 2.0
    BaseScore_Irrelevant  = 1.0
    epsilon               = 0.1
```

This ensures:
- All Essential skills rank above all Optional skills
- All Optional skills rank above all Irrelevant skills
- Within each tier, skills maintain their original relative ranking

## Output Files

### Re-ranking Output

1. **llm_reranked_scores.json**: Full results with all metadata
   - Job titles, skill names, tier classifications
   - Original and final scores/ranks

2. **llm_reranked_scores_compact.json**: Compact format (similar to fusion scores)
   - Ready for evaluation and downstream use
   - Contains: skill_uri, score, rank, tier, original_rank

### Evaluation Output

1. **comparison_metrics.csv**: Side-by-side comparison
   - mAP, Recall@K, Precision@K, NDCG@K
   - Percentage improvements

2. **tier_analysis.json**: Tier-based performance
   - How many ground truth skills fall into each tier
   - Top-K coverage for each tier

## Parameters

### LLM Re-ranking

- `--fusion_scores_json`: Path to linear fusion scores
- `--jobs_csv`: Path to decorte_master.csv
- `--skills_csv`: Path to skills_en.csv
- `--occupations_csv`: Path to occupations_en.csv
- `--output_dir`: Output directory
- `--api_key`: OpenAI API key (or use OPENAI_API_KEY env var)
- `--model`: Model name (default: gpt-4o-mini)
- `--top_k`: Number of top skills to re-rank (default: 100)
- `--max_workers`: Concurrent API calls (default: 5)
- `--isco_groups`: Comma-separated ISCO groups to filter

### Evaluation

- `--reranked_scores`: Path to LLM re-ranked scores
- `--original_scores`: Path to original fusion scores
- `--ground_truth`: Path to ground truth JSON
- `--output_dir`: Output directory

## Example Workflow

```bash
# 1. Prepare ground truth (with ISCO filtering)
python -m skill_mapping.v3.prepare_ground_truth \
    --jobs_csv ./data/title_pairs_desc/decorte_master.csv \
    --occupations_csv ./data/esco_datasets/occupations_en.csv \
    --occ_skills_csv ./data/esco_datasets/occupationSkillRelations_en.csv \
    --output_json ./data/processed/ground_truth_isco_5120_2654.json \
    --isco_groups 5120,2654

# 2. Run LLM re-ranking (with same ISCO filter)
python -m skill_mapping.v3.llm_reranker \
    --fusion_scores_json /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/linear_fusion_sum/best_fused_scores.json \
    --jobs_csv ./data/title_pairs_desc/decorte_master.csv \
    --skills_csv ./data/esco_datasets/skills_en.csv \
    --occupations_csv ./data/esco_datasets/occupations_en.csv \
    --output_dir ./outputs/llm_reranking_isco_5120_2654 \
    --isco_groups 5120,2654 \
    --max_workers 10

# 3. Evaluate results
python -m skill_mapping.v3.evaluate_reranking \
    --reranked_scores ./outputs/llm_reranking_isco_5120_2654/llm_reranked_scores_compact.json \
    --original_scores /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/linear_fusion_sum/best_fused_scores.json \
    --ground_truth ./data/processed/ground_truth_isco_5120_2654.json \
    --output_dir ./outputs/llm_reranking_isco_5120_2654/evaluation
```

## Performance Optimization

- **API Rate Limits**: Adjust `--max_workers` based on your OpenAI rate limits
- **Cost Management**: Start with a small subset using ISCO filtering
- **Caching**: Results are saved incrementally; failed jobs can be retried

## Expected Improvements

The LLM re-ranking is expected to improve:
- **mAP**: Better overall ranking quality
- **Recall@20**: More ground truth skills in top-20
- **Precision@5/10**: Higher relevance in top predictions

Key mechanism: Moving misclassified "Essential" skills from lower ranks (30-100) into the top-20.

## Architecture

```
llm_reranker.py
├── SkillCandidate: Input skill with metadata
├── RankedSkill: Output skill with tier and final score
├── LLMReranker: Handles LLM API calls and scoring
└── LLMRerankingPipeline: End-to-end orchestration

evaluate_reranking.py
├── RerankingEvaluator: Computes metrics
├── Tier analysis
└── Method comparison

prepare_ground_truth.py
└── Extract and filter ground truth labels
```

## Notes

- The script uses `response_format={"type": "json_object"}` to ensure valid JSON
- Temperature is set to 0.0 for deterministic classification
- Failed API calls are retried up to 3 times with exponential backoff
- ISCO group filtering is applied consistently across all scripts

