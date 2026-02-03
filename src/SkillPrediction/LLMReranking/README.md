# v4: Modular LLM Reranking Pipeline

Separates LLM data fetching from score adjustment for easier experimentation.

## Scripts

### 1. `llm_fetcher.py` - Fetch LLM Classifications

Sends job/skill data to GPT-4o-mini and saves all inputs/outputs.

```bash
python -m skill_mapping.v4.llm_fetcher \
    --fusion_scores_json /path/to/best_fused_scores.json \
    --jobs_csv ./data/title_pairs_desc/decorte_master.csv \
    --skills_csv ./data/esco_datasets/skills_en.csv \
    --occupations_csv ./data/esco_datasets/occupations_en.csv \
    --output_dir ./outputs/v4_llm \
    --top_k 100 \
    --max_workers 5 \
    --isco_groups 25
```

**Output:** `llm_responses.jsonl.gz` - Gzipped JSONL with full prompts and responses.

### 2. `score_adjuster.py` - Adjust Scores & Evaluate

Processes LLM outputs, adjusts scores, and compares metrics.

```bash
python -m skill_mapping.v4.score_adjuster \
    --llm_responses ./outputs/v4_llm/llm_responses.jsonl.gz \
    --original_scores /path/to/best_fused_scores.json \
    --ground_truth ./outputs/llm_reranking/ground_truth.json \
    --output_dir ./outputs/v4_reranking \
    --essential_base 3.0 \
    --optional_base 2.0 \
    --irrelevant_base 1.0 \
    --epsilon 0.1
```

**Outputs:**
- `adjusted_scores.json` - Reranked scores with tier info
- `metric_comparison.csv` - Before/after metric comparison
- `mismatch_report.json` - LLM output validation report
- `tier_analysis.json` - Tier distribution statistics

## Scoring Formula

```
final_score = base_tier + (epsilon × normalized_original_score)

where:
    base_Essential   = 3.0 (configurable)
    base_Optional    = 2.0 (configurable)
    base_Irrelevant  = 1.0 (configurable)
    epsilon          = 0.1 (configurable)
```

## Quick Start

```bash
# Step 1: Fetch LLM classifications (requires OPENAI_API_KEY)
python -m skill_mapping.v4.llm_fetcher \
    --fusion_scores_json /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/linear_fusion_sum/best_fused_scores.json \
    --jobs_csv ./data/title_pairs_desc/decorte_master.csv \
    --skills_csv ./data/esco_datasets/skills_en.csv \
    --occupations_csv ./data/esco_datasets/occupations_en.csv \
    --output_dir ./outputs/v4_llm \
    --isco_groups 25

# Step 2: Adjust scores and evaluate (no API key needed)
python -m skill_mapping.v4.score_adjuster \
    --llm_responses ./outputs/v4_llm/llm_responses.jsonl.gz \
    --original_scores /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/linear_fusion_sum/best_fused_scores.json \
    --ground_truth /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/llm_reranking/ground_truth.json \
    --output_dir ./outputs/v4_reranking
```

## Experimenting with Scoring

Re-run `score_adjuster.py` with different tier weights without calling LLM again:

```bash
# Higher penalty for Irrelevant
python -m skill_mapping.v4.score_adjuster \
    --llm_responses ./outputs/v4_llm/llm_responses.jsonl.gz \
    --original_scores /path/to/scores.json \
    --ground_truth /path/to/ground_truth.json \
    --output_dir ./outputs/v4_experiment1 \
    --essential_base 3.0 \
    --optional_base 2.0 \
    --irrelevant_base 0.5

# Larger separation between tiers
python -m skill_mapping.v4.score_adjuster \
    --llm_responses ./outputs/v4_llm/llm_responses.jsonl.gz \
    --original_scores /path/to/scores.json \
    --ground_truth /path/to/ground_truth.json \
    --output_dir ./outputs/v4_experiment2 \
    --essential_base 10.0 \
    --optional_base 5.0 \
    --irrelevant_base 1.0 \
    --epsilon 0.01
```

## File Formats

### llm_responses.jsonl.gz

Each line is a JSON object:
```json
{
  "job_id": "123",
  "job_title": "Software Engineer",
  "job_description": "...",
  "skill_candidates": [
    {"skill_uri": "...", "skill_name": "Python", "original_score": 0.85, "original_rank": 1},
    ...
  ],
  "prompt": "Given a job posting...",
  "llm_response": {
    "success": true,
    "response": {"skill_classifications": [{"skill_number": 1, "tier": "Essential"}, ...]},
    "raw_response": "..."
  }
}
```

### adjusted_scores.json

```json
{
  "metadata": {
    "tier_config": {"essential_base": 3.0, "optional_base": 2.0, ...},
    "n_jobs": 394
  },
  "scores": {
    "123": [
      {"skill_uri": "...", "original_score": 0.85, "tier": "Essential", "final_score": 3.085, "final_rank": 1},
      ...
    ]
  }
}
```






