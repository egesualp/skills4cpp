# Quick Start Guide: LLM Re-ranking for Skills4CPP

## Overview

This guide will help you quickly get started with the LLM-based re-ranking system for job-to-skill mapping.

## What This Does

The LLM re-ranking system takes the top-100 skill predictions from your linear fusion model and uses GPT-4o-mini to classify them into three tiers:
- **Essential**: Must-have skills for the job
- **Optional**: Useful but not mandatory skills
- **Irrelevant**: Skills that don't match the job

This improves ranking quality by ensuring Essential skills appear at the top, which should improve mAP and Recall@20 metrics.

## Prerequisites

1. **OpenAI API Key**: Get one from https://platform.openai.com/api-keys
2. **Linear Fusion Scores**: Complete output from the linear fusion model
3. **Python Environment**: Python 3.8+ with required packages

## Installation

```bash
# Install dependencies
pip install openai loguru tqdm pandas numpy

# Set API key
export OPENAI_API_KEY='your-api-key-here'
```

## Quick Start (2 Steps)

### Step 1: Run LLM Re-ranking (with ground truth preparation)

This sends job descriptions and candidate skills to GPT-4o-mini for classification and prepares ground truth in one go:

```bash
python -m skill_mapping.v3.llm_reranker \
    --fusion_scores_json /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/linear_fusion_sum/best_fused_scores.json \
    --jobs_csv ./data/title_pairs_desc/decorte_master.csv \
    --skills_csv ./data/esco_datasets/skills_en.csv \
    --occupations_csv ./data/esco_datasets/occupations_en.csv \
    --occ_skills_csv ./data/esco_datasets/occupationSkillRelations_en.csv \
    --output_dir /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/llm_reranking \
    --top_k 100 \
    --max_workers 8 \
    --prepare_ground_truth
```

**With ISCO group filtering** (recommended for testing on a subset):

```bash
python -m skill_mapping.v3.llm_reranker \
    --fusion_scores_json /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/linear_fusion_sum/best_fused_scores.json \
    --jobs_csv ./data/title_pairs_desc/decorte_master.csv \
    --skills_csv ./data/esco_datasets/skills_en.csv \
    --occupations_csv ./data/esco_datasets/occupations_en.csv \
    --occ_skills_csv ./data/esco_datasets/occupationSkillRelations_en.csv \
    --output_dir ./outputs/llm_reranking_test \
    --top_k 100 \
    --max_workers 5 \
    --isco_groups 5120,2654 \
    --prepare_ground_truth
```

**Important parameters:**
- `--max_workers`: Controls API concurrency (5-10 is safe for most rate limits)
- `--isco_groups`: Filter to specific ISCO groups (e.g., `--isco_groups 5120,2654`)
- `--top_k`: How many skills to re-rank (default: 100)
- `--prepare_ground_truth`: Prepare ground truth for evaluation (saves to output_dir/ground_truth.json)
- `--relation_type`: Filter ground truth by relation type (essential/optional)

### Step 2: Evaluate Results

Compare the re-ranked results against the original fusion scores:

```bash
python -m skill_mapping.v3.evaluate_reranking \
    --reranked_scores ./outputs/llm_reranking/llm_reranked_scores_compact.json \
    --original_scores /dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/linear_fusion_sum/best_fused_scores.json \
    --ground_truth ./outputs/llm_reranking/ground_truth.json \
    --output_dir ./outputs/llm_reranking/evaluation
```

This generates:
- `comparison_metrics.csv`: Side-by-side comparison of metrics
- `tier_analysis.json`: Performance breakdown by tier

## Using the Automated Pipeline Script

Alternatively, run all three steps at once:

```bash
cd src/skill_mapping/v3
bash run_pipeline.sh
```

Edit the variables at the top of `run_pipeline.sh` to configure paths and ISCO filtering.

## Understanding the Output

### Re-ranking Output

Two JSON files are created:

1. **llm_reranked_scores.json**: Full results with all details
   - Job titles and descriptions
   - Skill names and URIs
   - Tier classifications
   - Original and final scores/ranks

2. **llm_reranked_scores_compact.json**: Minimal format for evaluation
   - Same structure as fusion scores
   - Includes tier information
   - Ready for downstream use

### Evaluation Output

1. **comparison_metrics.csv**: 
   ```
   method                  | mAP    | Recall@20 | Precision@10 | ...
   Original (Linear Fusion)| 0.1054 | 0.3245    | 0.1823       | ...
   LLM Re-ranked          | 0.1289 | 0.3891    | 0.2156       | ...
   Improvement (%)         | 22.3%  | 19.9%     | 18.3%        | ...
   ```

2. **tier_analysis.json**:
   Shows how many ground truth skills fall into each tier and their top-K coverage.

## Cost Estimation

Using GPT-4o-mini (very affordable):
- ~$0.01 per 100 skills re-ranked
- For 7,930 jobs: ~$80 total
- For 100 jobs (ISCO filtered): ~$1

## Tips for Success

1. **Start Small**: Use `--isco_groups` to test on 50-100 jobs first
2. **Monitor Rate Limits**: Start with `--max_workers 5`, increase if needed
3. **Check Results**: Look at a few examples in `llm_reranked_scores.json` to verify quality
4. **ISCO Consistency**: Use same ISCO filter for ground truth prep and re-ranking

## Common Issues

### "API key not found"
```bash
export OPENAI_API_KEY='your-key-here'
```

### Rate limit errors
- Reduce `--max_workers` to 3 or lower
- The script has built-in retry logic

### Out of memory
- Process in smaller batches using ISCO filtering
- Results are saved per job, so you can split the workload

## File Locations

All files are under `src/skill_mapping/v3/`:
- `llm_reranker.py` - Main re-ranking script
- `evaluate_reranking.py` - Evaluation script
- `prepare_ground_truth.py` - Ground truth preparation
- `run_pipeline.sh` - Automated pipeline
- `README.md` - Full documentation
- `requirements.txt` - Dependencies

## Example ISCO Groups

Common ISCO occupation groups you might want to filter:
- `5120` - Cooks
- `2654` - Technical directors
- `2311` - University and higher education teachers
- `1120` - Managing directors and chief executives

Find more in `./data/esco_datasets/occupations_en.csv`.

## Next Steps

After successful re-ranking:
1. Analyze which types of skills moved up in ranking
2. Compare tier distributions across different job types
3. Use the re-ranked scores as input to your downstream tasks
4. Experiment with different prompts (modify `_build_prompt` in `llm_reranker.py`)

## Support

For issues or questions:
1. Check the full README.md in this directory
2. Review example outputs in the output directory
3. Run the test script: `python -m skill_mapping.v3.test_llm_reranker`

---

**Ready to go?** Start with Step 1 above! 🚀

**Note**: The standalone `prepare_ground_truth.py` script is still available if you need to prepare ground truth separately, but using `--prepare_ground_truth` flag with the reranker is simpler and saves storage space.

