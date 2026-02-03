"""
llm_reranker.py - LLM-based Re-ranking for Job-to-Skill Mapping

Uses GPT-4o-mini to refine top-100 skill predictions by classifying them into tiers:
    - Essential: Must-have skills for the job
    - Optional: Useful but not critical skills
    - Irrelevant: Skills that don't match the job

The script then re-scores skills so that:
    - All Essential skills rank above Optional
    - All Optional skills rank above Irrelevant
    - Within each tier, original ranking is preserved

Usage:
    python -m skill_mapping.v3.llm_reranker \
        --fusion_scores_json /path/to/best_fused_scores.json \
        --jobs_csv ./data/title_pairs_desc/decorte_master.csv \
        --skills_csv ./data/esco_datasets/skills_en.csv \
        --occupations_csv ./data/esco_datasets/occupations_en.csv \
        --output_dir ./outputs/llm_reranking \
        --api_key YOUR_OPENAI_API_KEY \
        --top_k 100 \
        --max_workers 5 \
        --isco_groups 5120,2654
"""

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple

import dotenv
import pandas as pd
from loguru import logger
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio


@dataclass
class SkillCandidate:
    """Represents a candidate skill with its metadata."""
    skill_uri: str
    skill_name: str
    skill_description: str
    original_score: float
    original_rank: int


@dataclass
class RankedSkill:
    """Represents a re-ranked skill with tier classification."""
    skill_uri: str
    skill_name: str
    original_score: float
    original_rank: int
    tier: Literal["Essential", "Optional", "Irrelevant"]
    final_score: float
    final_rank: int


class LLMReranker:
    """
    Re-ranks skills using GPT-4o-mini for tier classification.
    
    Scoring logic:
        Essential:   score_final = 3.0 + (epsilon × original_score), epsilon = 0.1
        Optional:    score_final = 2.0 + (epsilon × original_score), epsilon = 0.1
        Irrelevant:  score_final = 1.0 + (epsilon × original_score), epsilon = 0.1
    
    This ensures tier separation while preserving relative order within tiers.
    """
    
    # Base scores for each tier
    TIER_BASE_SCORES = {
        "Essential": 3.0,
        "Optional": 2.0,
        "Irrelevant": 1.0,
    }
    
    # Epsilon for preserving relative order within tiers
    EPSILON = 0.1
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """
        Initialize the LLM reranker.
        
        Args:
            api_key: OpenAI API key
            model: Model name (default: gpt-4o-mini)
            max_retries: Maximum number of retries for API calls
            retry_delay: Delay between retries in seconds
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
    def _build_prompt(
        self,
        job_title: str,
        job_description: str,
        candidates: List[SkillCandidate],
    ) -> str:
        """
        Build the prompt for LLM classification.
        
        Args:
            job_title: The job title
            job_description: The job description
            candidates: List of candidate skills
            
        Returns:
            The formatted prompt string
        """
        # Format the skill list
        skills_text = "\n".join([
            f"{i+1}. {skill.skill_name} - {skill.skill_description}"
            for i, skill in enumerate(candidates)
        ])

        num_skills = len(candidates)
        
        prompt = f"""Given a job posting and a list of {num_skills} candidate skills, classify each skill into one of three tiers:

**Essential**: Skills that are critical and must-have for this job.
**Optional**: Skills that are useful or nice-to-have but not mandatory.
**Irrelevant**: Skills that are not applicable or relevant to this job.

**Job Title**: {job_title}

**Job Description**:
{job_description}

**Candidate Skills** (with descriptions):
{skills_text}

**Instructions**:
- Analyze each skill carefully in the context of the job.
- Return a JSON object with the structure: {{"skill_classifications": [{{"skill_number": 1, "tier": "Essential"}}, ...]}}
- Include all {num_skills} skills in your response.
- Use exactly these tier names: "Essential", "Optional", or "Irrelevant".

Respond with ONLY the JSON object, no additional text."""
        
        return prompt
    
    async def _call_llm_with_retry(
        self,
        prompt: str,
        job_id: str,
    ) -> Optional[Dict]:
        """
        Call the LLM API with retry logic.
        
        Args:
            prompt: The prompt to send
            job_id: Job ID for logging
            
        Returns:
            The parsed JSON response or None if failed
        """
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert HR analyst."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.0,  # Deterministic output
                    response_format={"type": "json_object"},  # Force JSON response
                )
                
                # Parse the response
                content = response.choices[0].message.content
                result = json.loads(content)
                
                # Validate the response structure
                if "skill_classifications" not in result:
                    logger.warning(f"Job {job_id}: Missing 'skill_classifications' key in response")
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay)
                        continue
                    return None
                
                return result
                
            except json.JSONDecodeError as e:
                logger.warning(f"Job {job_id}: JSON decode error on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
                    
            except Exception as e:
                logger.error(f"Job {job_id}: API call failed on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
                    
        logger.error(f"Job {job_id}: All retry attempts failed")
        return None
    
    def _rescore_skills(
        self,
        candidates: List[SkillCandidate],
        classifications: Dict[int, str],
    ) -> List[RankedSkill]:
        """
        Re-score skills based on tier classifications.
        
        Args:
            candidates: List of candidate skills
            classifications: Mapping from skill_number to tier
            
        Returns:
            List of re-ranked skills sorted by final score
        """
        ranked_skills = []
        
        for i, candidate in enumerate(candidates):
            skill_number = i + 1
            tier = classifications.get(skill_number, "Irrelevant")
            
            # Calculate final score
            base_score = self.TIER_BASE_SCORES[tier]
            # Normalize original score to [0, 1] range
            normalized_score = candidate.original_score / max_score if max_score > 0 else 0
            final_score = base_score + (self.EPSILON * normalized_score)
            
            ranked_skills.append(RankedSkill(
                skill_uri=candidate.skill_uri,
                skill_name=candidate.skill_name,
                original_score=candidate.original_score,
                original_rank=candidate.original_rank,
                tier=tier,
                final_score=final_score,
                final_rank=0,  # Will be set after sorting
            ))
        
        # Sort by final score (descending)
        ranked_skills.sort(key=lambda x: x.final_score, reverse=True)
        
        # Assign final ranks
        for i, skill in enumerate(ranked_skills):
            skill.final_rank = i + 1
            
        return ranked_skills
    
    async def rerank_job(
        self,
        job_id: str,
        job_title: str,
        job_description: str,
        candidates: List[SkillCandidate],
    ) -> Optional[List[RankedSkill]]:
        """
        Re-rank skills for a single job using LLM.
        
        Args:
            job_id: Job ID
            job_title: Job title
            job_description: Job description
            candidates: List of candidate skills
            
        Returns:
            List of re-ranked skills or None if failed
        """
        # Build prompt
        prompt = self._build_prompt(job_title, job_description, candidates)
        
        # Call LLM
        result = await self._call_llm_with_retry(prompt, job_id)
        if result is None:
            return None
        
        # Parse classifications
        classifications = {}
        for item in result["skill_classifications"]:
            skill_num = item.get("skill_number")
            tier = item.get("tier")
            if skill_num and tier in self.TIER_BASE_SCORES:
                classifications[skill_num] = tier
        
        # Re-score skills
        ranked_skills = self._rescore_skills(candidates, classifications)
        
        return ranked_skills


class LLMRerankingPipeline:
    """
    End-to-end pipeline for LLM-based re-ranking.
    """
    
    def __init__(
        self,
        fusion_scores_json: Path,
        jobs_csv: Path,
        skills_csv: Path,
        occupations_csv: Path,
        occ_skills_csv: Path,
        output_dir: Path,
        api_key: str,
        top_k: int = 100,
        model: str = "gpt-4o-mini",
        max_workers: int = 5,
        isco_groups: Optional[List[str]] = None,
        prepare_ground_truth: bool = True,
        relation_type: Optional[str] = None,
    ):
        """
        Initialize the pipeline.
        
        Args:
            fusion_scores_json: Path to linear fusion scores
            jobs_csv: Path to decorte_master.csv
            skills_csv: Path to skills_en.csv
            occupations_csv: Path to occupations_en.csv
            occ_skills_csv: Path to occupationSkillRelations_en.csv
            output_dir: Output directory
            api_key: OpenAI API key
            top_k: Number of top skills to consider
            model: OpenAI model name
            max_workers: Maximum concurrent API calls
            isco_groups: List of ISCO groups to filter (e.g., ["5120", "2654"])
            prepare_ground_truth: Whether to prepare ground truth for evaluation
            relation_type: Filter ground truth by relation type ('essential', 'optional', or None)
        """
        self.fusion_scores_json = fusion_scores_json
        self.jobs_csv = jobs_csv
        self.skills_csv = skills_csv
        self.occupations_csv = occupations_csv
        self.occ_skills_csv = occ_skills_csv
        self.output_dir = output_dir
        self.api_key = api_key
        self.top_k = top_k
        self.model = model
        self.max_workers = max_workers
        self.isco_groups = set(isco_groups) if isco_groups else None
        self.prepare_gt = prepare_ground_truth
        self.relation_type = relation_type
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize reranker
        self.reranker = LLMReranker(api_key=api_key, model=model)
        
        # Load data
        logger.info("Loading data...")
        self.jobs_df = self._load_jobs()
        self.skills_df = self._load_skills()
        self.fusion_scores = self._load_fusion_scores()
        
        logger.info(f"Loaded {len(self.jobs_df)} jobs")
        logger.info(f"Loaded {len(self.skills_df)} skills")
        logger.info(f"Loaded fusion scores for {len(self.fusion_scores)} jobs")
        
        # Prepare ground truth if requested
        self.ground_truth = None
        if self.prepare_gt:
            self.ground_truth = self._prepare_ground_truth()
        
    def _load_jobs(self) -> pd.DataFrame:
        """Load and filter jobs dataset."""
        df = pd.read_csv(self.jobs_csv)
        
        # Filter by ISCO groups if specified
        if self.isco_groups:
            # Load occupations to get ISCO groups
            occupations = pd.read_csv(self.occupations_csv)
            occupations = occupations[['conceptUri', 'iscoGroup']]
            occupations.columns = ['esco_id', 'iscoGroup']
            
            # Merge to get ISCO groups for jobs
            df = df.merge(occupations, on='esco_id', how='left')
            
            # Filter by ISCO groups
            # Support both exact matches and 2-digit prefix filtering
            df['iscoGroup'] = df['iscoGroup'].astype(str)
            
            prefixes = [g for g in self.isco_groups if len(g) == 2]
            others = [g for g in self.isco_groups if len(g) != 2]
            
            mask = pd.Series(False, index=df.index)
            if others:
                mask |= df['iscoGroup'].isin(others)
            
            if prefixes:
                prefix_pattern = '|'.join(f'^{p}' for p in prefixes)
                mask |= df['iscoGroup'].str.contains(prefix_pattern, regex=True, na=False)
            
            df = df[mask]
            logger.info(f"Filtered to {len(df)} jobs with ISCO groups: {self.isco_groups}")
        
        return df
    
    def _load_skills(self) -> pd.DataFrame:
        """Load skills dataset."""
        df = pd.read_csv(self.skills_csv)
        df = df[['conceptUri', 'preferredLabel', 'description']]
        df.columns = ['skill_uri', 'skill_name', 'skill_description']
        df['skill_description'] = df['skill_description'].fillna('')
        return df
    
    def _load_fusion_scores(self) -> Dict[str, List[Dict]]:
        """Load fusion scores from JSON."""
        with open(self.fusion_scores_json, 'r') as f:
            data = json.load(f)
        return data['scores']
    
    def _prepare_ground_truth(self) -> Dict[str, Set[str]]:
        """
        Prepare ground truth labels from ESCO occupation-skill mappings.
        
        Returns:
            Dictionary mapping job_id to set of skill URIs
        """
        logger.info("Preparing ground truth...")
        
        # Load occupation-skill relations
        occ_skills_df = pd.read_csv(self.occ_skills_csv)
        logger.info(f"Loaded {len(occ_skills_df)} occupation-skill relations")
        
        # Filter by relation type if specified
        if self.relation_type:
            if 'relationType' in occ_skills_df.columns:
                occ_skills_df = occ_skills_df[occ_skills_df['relationType'] == self.relation_type]
                logger.info(f"Filtered to {len(occ_skills_df)} {self.relation_type} relations")
        
        # Build ground truth for jobs we have
        ground_truth = {}
        for _, row in self.jobs_df.iterrows():
            job_id = str(row['job_id'])
            esco_id = row['esco_id']
            
            # Get skills for this occupation
            occ_skills = occ_skills_df[occ_skills_df['occupationUri'] == esco_id]
            
            if not occ_skills.empty:
                skill_uris = set(occ_skills['skillUri'].values)
                ground_truth[job_id] = skill_uris
        
        logger.info(f"Built ground truth for {len(ground_truth)} jobs")
        
        # Print statistics
        if ground_truth:
            num_skills_per_job = [len(skills) for skills in ground_truth.values()]
            logger.info(f"Average skills per job: {sum(num_skills_per_job) / len(num_skills_per_job):.2f}")
            logger.info(f"Min skills per job: {min(num_skills_per_job)}")
            logger.info(f"Max skills per job: {max(num_skills_per_job)}")
        
        return ground_truth
    
    def _get_candidates_for_job(self, job_id: str) -> List[SkillCandidate]:
        """
        Get top-K candidate skills for a job.
        
        Args:
            job_id: Job ID
            
        Returns:
            List of SkillCandidate objects
        """
        if job_id not in self.fusion_scores:
            return []
        
        # Get top-K scores
        scores = self.fusion_scores[job_id][:self.top_k]
        
        candidates = []
        for score_item in scores:
            skill_uri = score_item['skill_uri']
            
            # Look up skill metadata
            skill_row = self.skills_df[self.skills_df['skill_uri'] == skill_uri]
            if skill_row.empty:
                logger.warning(f"Skill {skill_uri} not found in skills dataset")
                continue
            
            skill_name = skill_row.iloc[0]['skill_name']
            skill_description = skill_row.iloc[0]['skill_description']
            
            candidates.append(SkillCandidate(
                skill_uri=skill_uri,
                skill_name=skill_name,
                skill_description=skill_description,
                original_score=score_item['score'],
                original_rank=score_item['rank'],
            ))
        
        return candidates
    
    async def _process_job(self, job_id: str, job_row: pd.Series) -> Optional[Dict]:
        """
        Process a single job.
        
        Args:
            job_id: Job ID
            job_row: Row from jobs dataframe
            
        Returns:
            Result dictionary or None if failed
        """
        job_title = job_row['raw_title']
        job_description = job_row['raw_description']
        
        # Get candidates
        candidates = self._get_candidates_for_job(job_id)
        if not candidates:
            logger.warning(f"No candidates found for job {job_id}")
            return None
        
        # Re-rank
        ranked_skills = await self.reranker.rerank_job(
            job_id=job_id,
            job_title=job_title,
            job_description=job_description,
            candidates=candidates,
        )
        
        if ranked_skills is None:
            return None
        
        # Format result
        result = {
            'job_id': job_id,
            'job_title': job_title,
            'ranked_skills': [
                {
                    'skill_uri': skill.skill_uri,
                    'skill_name': skill.skill_name,
                    'original_score': skill.original_score,
                    'original_rank': skill.original_rank,
                    'tier': skill.tier,
                    'final_score': skill.final_score,
                    'final_rank': skill.final_rank,
                }
                for skill in ranked_skills
            ]
        }
        
        return result
    
    async def _process_jobs_batch(self, job_items: List[tuple]) -> List[Optional[Dict]]:
        """
        Process a batch of jobs concurrently.
        
        Args:
            job_items: List of (job_id, job_row) tuples
            
        Returns:
            List of results
        """
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_with_semaphore(job_id, job_row):
            async with semaphore:
                return await self._process_job(job_id, job_row)
        
        tasks = [process_with_semaphore(job_id, job_row) for job_id, job_row in job_items]
        results = await tqdm_asyncio.gather(*tasks, desc="Re-ranking jobs")
        
        return results
    
    async def run(self):
        """Run the re-ranking pipeline."""
        logger.info("Starting LLM-based re-ranking pipeline")
        start_time = time.time()
        
        # Prepare job items
        job_items = []
        for _, row in self.jobs_df.iterrows():
            job_id = str(row['job_id'])
            if job_id in self.fusion_scores:
                job_items.append((job_id, row))
        
        logger.info(f"Processing {len(job_items)} jobs")
        
        # Process all jobs
        results = await self._process_jobs_batch(job_items)
        
        # Filter out failed jobs
        successful_results = [r for r in results if r is not None]
        failed_count = len(results) - len(successful_results)
        
        logger.info(f"Successfully processed {len(successful_results)} jobs")
        if failed_count > 0:
            logger.warning(f"Failed to process {failed_count} jobs")
        
        # Save results
        output_file = self.output_dir / "llm_reranked_scores.json"
        with open(output_file, 'w') as f:
            json.dump({
                'metadata': {
                    'model': self.model,
                    'top_k': self.top_k,
                    'n_jobs': len(successful_results),
                    'n_failed': failed_count,
                    'isco_groups': list(self.isco_groups) if self.isco_groups else None,
                    'fusion_scores_source': str(self.fusion_scores_json),
                },
                'results': successful_results,
            }, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        
        # Save compact scores format (similar to fusion scores)
        compact_scores = {}
        for result in successful_results:
            job_id = result['job_id']
            compact_scores[job_id] = [
                {
                    'skill_uri': skill['skill_uri'],
                    'score': skill['final_score'],
                    'rank': skill['final_rank'],
                    'tier': skill['tier'],
                    'original_rank': skill['original_rank'],
                }
                for skill in result['ranked_skills']
            ]
        
        compact_file = self.output_dir / "llm_reranked_scores_compact.json"
        with open(compact_file, 'w') as f:
            json.dump({
                'metadata': {
                    'model': self.model,
                    'top_k': self.top_k,
                    'n_jobs': len(successful_results),
                    'isco_groups': list(self.isco_groups) if self.isco_groups else None,
                },
                'scores': compact_scores,
            }, f, indent=2)
        
        logger.info(f"Compact scores saved to {compact_file}")
        
        # Save ground truth if prepared
        if self.ground_truth:
            gt_file = self.output_dir / "ground_truth.json"
            gt_serializable = {k: list(v) for k, v in self.ground_truth.items()}
            with open(gt_file, 'w') as f:
                json.dump(gt_serializable, f, indent=2)
            logger.info(f"Ground truth saved to {gt_file}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")
        
        # Print statistics
        self._print_statistics(successful_results)
    
    def _print_statistics(self, results: List[Dict]):
        """Print tier distribution statistics."""
        tier_counts = {'Essential': 0, 'Optional': 0, 'Irrelevant': 0}
        total_skills = 0
        
        for result in results:
            for skill in result['ranked_skills']:
                tier_counts[skill['tier']] += 1
                total_skills += 1
        
        logger.info("Tier distribution:")
        for tier, count in tier_counts.items():
            pct = (count / total_skills * 100) if total_skills > 0 else 0
            logger.info(f"  {tier}: {count} ({pct:.2f}%)")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="LLM-based re-ranking for job-to-skill mapping"
    )
    
    # Input/output paths
    parser.add_argument(
        "--fusion_scores_json",
        type=Path,
        required=True,
        help="Path to linear fusion scores JSON file"
    )
    parser.add_argument(
        "--jobs_csv",
        type=Path,
        required=True,
        help="Path to decorte_master.csv"
    )
    parser.add_argument(
        "--skills_csv",
        type=Path,
        required=True,
        help="Path to skills_en.csv"
    )
    parser.add_argument(
        "--occupations_csv",
        type=Path,
        required=True,
        help="Path to occupations_en.csv"
    )
    parser.add_argument(
        "--occ_skills_csv",
        type=Path,
        required=True,
        help="Path to occupationSkillRelations_en.csv"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory"
    )
    
    # API settings
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model name (default: gpt-4o-mini)"
    )
    
    # Processing settings
    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="Number of top skills to consider (default: 100)"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="Maximum concurrent API calls (default: 5)"
    )
    parser.add_argument(
        "--isco_groups",
        type=str,
        default=None,
        help="Comma-separated list of ISCO groups to filter (e.g., '5120,2654')"
    )
    parser.add_argument(
        "--prepare_ground_truth",
        action="store_true",
        help="Prepare ground truth for evaluation (saves to output_dir/ground_truth.json)"
    )
    parser.add_argument(
        "--relation_type",
        type=str,
        choices=['essential', 'optional'],
        default=None,
        help="Filter ground truth by relation type (essential/optional, or all if not specified)"
    )
    
    args = parser.parse_args()

    dotenv.load_dotenv()

    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key must be provided via --api_key or OPENAI_API_KEY env var")
    
    # Parse ISCO groups
    isco_groups = None
    if args.isco_groups:
        isco_groups = [g.strip() for g in args.isco_groups.split(',')]
    
    # Run pipeline
    pipeline = LLMRerankingPipeline(
        fusion_scores_json=args.fusion_scores_json,
        jobs_csv=args.jobs_csv,
        skills_csv=args.skills_csv,
        occupations_csv=args.occupations_csv,
        occ_skills_csv=args.occ_skills_csv,
        output_dir=args.output_dir,
        api_key=api_key,
        top_k=args.top_k,
        model=args.model,
        max_workers=args.max_workers,
        isco_groups=isco_groups,
        prepare_ground_truth=args.prepare_ground_truth,
        relation_type=args.relation_type,
    )
    
    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())

