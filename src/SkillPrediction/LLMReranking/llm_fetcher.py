"""
llm_fetcher.py - Fetch LLM tier classifications for skill candidates

Sends prompts to GPT-4o-mini and saves inputs/outputs for later processing.
Outputs are stored in gzipped JSONL format for efficient storage.

Usage:
    python -m skill_mapping.v4.llm_fetcher \
        --fusion_scores_json /path/to/best_fused_scores.json \
        --jobs_csv ./data/title_pairs_desc/decorte_master.csv \
        --skills_csv ./data/esco_datasets/skills_en.csv \
        --occupations_csv ./data/esco_datasets/occupations_en.csv \
        --output_dir ./outputs/llm_fetcher \
        --top_k 100 \
        --max_workers 5 \
        --isco_groups 25
"""

import argparse
import asyncio
import gzip
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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


class LLMFetcher:
    """Fetches tier classifications from LLM for skill candidates."""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
    def build_prompt(
        self,
        job_title: str,
        job_description: str,
        candidates: List[SkillCandidate],
    ) -> str:
        """Build the prompt for LLM classification."""
        skills_text = "\n".join([
            f"{i+1}. {skill.skill_name} - {skill.skill_description}"
            for i, skill in enumerate(candidates)
        ])

        num_skills = len(candidates)
        
        prompt = f"""Given a job experience description from a CV and a list of {num_skills} candidate skills, classify each skill into one of three tiers:

**Essential**: The 5-8 core skills that define this role's primary 
  responsibilities. These are skills the person used daily or that were 
  critical to their main deliverables.
**Optional**: Skills that support the role, show relevant domain knowledge, 
  or represent secondary capabilities.
**Irrelevant**: Skills that are not applicable or relevant to this job.

**Job Title**: {job_title}

**Job Description**:
{job_description}

**Candidate Skills** (with descriptions):
{skills_text}

**Classification Guidelines**:
1. **Essential - Be Extremely Selective**: Ask yourself: "Could this person 
   have performed their core job duties without this skill?" If yes, it's 
   not Essential. Aim for 5-8 Essential skills maximum per job.

2. **Optional - Default Category**: When uncertain between Essential and 
   Optional, choose Optional. Skills that are domain-relevant, mentioned 
   tangentially, or represent supporting expertise belong here.

3. **Irrelevant - Reserve for Clear Mismatches**: Use this only for skills 
   that have no logical connection to the role or its domain.

**Output Requirements**:
- Return a valid JSON object: {{"skill_classifications": [{{"skill_number": 1, "tier": "Essential"}}, ...]}}
- Include all {num_skills} skills
- Use exactly these tier names: "Essential", "Optional", "Irrelevant"

Respond with ONLY the JSON object, no additional text."""
        
        return prompt
    
    async def fetch_classification(
        self,
        prompt: str,
        job_id: str,
    ) -> Dict:
        """
        Call the LLM API and return raw response.
        
        Returns:
            Dictionary with success status and response/error data
        """
        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert HR analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.0,
                    response_format={"type": "json_object"},
                )
                
                content = response.choices[0].message.content
                parsed = json.loads(content)
                
                if "skill_classifications" not in parsed:
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(self.retry_delay)
                        continue
                    return {
                        "success": False,
                        "error": "Missing 'skill_classifications' key",
                        "raw_response": content,
                    }
                
                return {
                    "success": True,
                    "response": parsed,
                    "raw_response": content,
                }
                
            except json.JSONDecodeError as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
                return {
                    "success": False,
                    "error": f"JSON decode error: {str(e)}",
                    "raw_response": content if 'content' in dir() else None,
                }
                    
            except Exception as e:
                logger.error(f"Job {job_id}: API call failed on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
                return {
                    "success": False,
                    "error": str(e),
                }
                    
        return {"success": False, "error": "All retry attempts failed"}


class LLMFetcherPipeline:
    """Pipeline for fetching LLM classifications."""
    
    def __init__(
        self,
        fusion_scores_json: Path,
        jobs_csv: Path,
        skills_csv: Path,
        occupations_csv: Path,
        output_dir: Path,
        api_key: str,
        top_k: int = 100,
        model: str = "gpt-4o-mini",
        max_workers: int = 5,
        isco_groups: Optional[List[str]] = None,
    ):
        self.fusion_scores_json = fusion_scores_json
        self.jobs_csv = jobs_csv
        self.skills_csv = skills_csv
        self.occupations_csv = occupations_csv
        self.output_dir = output_dir
        self.api_key = api_key
        self.top_k = top_k
        self.model = model
        self.max_workers = max_workers
        self.isco_groups = set(isco_groups) if isco_groups else None
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.fetcher = LLMFetcher(api_key=api_key, model=model)
        
        logger.info("Loading data...")
        self.jobs_df = self._load_jobs()
        self.skills_df = self._load_skills()
        self.fusion_scores = self._load_fusion_scores()
        
        logger.info(f"Loaded {len(self.jobs_df)} jobs")
        logger.info(f"Loaded {len(self.skills_df)} skills")
        logger.info(f"Loaded fusion scores for {len(self.fusion_scores)} jobs")
        
    def _load_jobs(self) -> pd.DataFrame:
        """Load and filter jobs dataset."""
        df = pd.read_csv(self.jobs_csv)
        
        if self.isco_groups:
            occupations = pd.read_csv(self.occupations_csv)
            occupations = occupations[['conceptUri', 'iscoGroup']]
            occupations.columns = ['esco_id', 'iscoGroup']
            
            df = df.merge(occupations, on='esco_id', how='left')
            df = df.dropna(subset='iscoGroup')
            print(f"DEBUG: Jobs DF after merge and before ISCO filtering:\n{df.head()}")
            df['iscoGroup'] = df['iscoGroup'].astype(int).astype(str)
            
            prefixes = [g for g in self.isco_groups if len(g) == 2]
            others = [g for g in self.isco_groups if len(g) != 2]
            
            print(f"DEBUG: isco_groups parameter: {self.isco_groups}")
            print(f"DEBUG: prefixes: {prefixes}, others: {others}")
            
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
        path = str(self.fusion_scores_json)

        # Support plain JSON and gzipped JSON
        open_fn = gzip.open if path.endswith(".gz") else open
        with open_fn(path, 'rt', encoding='utf-8') as f:
            data = json.load(f)

        # If the file uses a top-level "scores" key, use it
        if isinstance(data, dict) and "scores" in data:
            raw = data["scores"]
        else:
            # Otherwise assume the top-level object is the mapping job_id -> list
            raw = data

        # Normalize entries so downstream code can rely on keys: skill_uri, score, rank
        normalized: Dict[str, List[Dict]] = {}
        for job_id, items in raw.items():
            # Some files may include metadata at top-level; skip non-list entries
            if not isinstance(items, list):
                continue

            norm_items = []
            for idx, item in enumerate(items):
                # item might be a dict with varying key names; try common alternatives
                # Skill uri keys
                skill_uri = None
                for k in ("skill_uri", "conceptUri", "uri", "id"):
                    if k in item:
                        skill_uri = item[k]
                        break

                # Score keys
                score = None
                for k in ("score", "original_score", "similarity", "sim"):
                    if k in item:
                        score = item[k]
                        break

                # Rank keys
                rank = None
                for k in ("rank", "original_rank", "position", "r"):
                    if k in item:
                        rank = item[k]
                        break

                # Fallbacks
                if skill_uri is None:
                    # skip if no identifier
                    continue
                if score is None:
                    # default to 0.0 if missing
                    score = 0.0
                if rank is None:
                    # If rank missing, infer from ordering (1-based)
                    rank = idx + 1

                norm_items.append({
                    "skill_uri": skill_uri,
                    "score": score,
                    "rank": rank,
                })

            normalized[str(job_id)] = norm_items

        return normalized
    
    def _get_candidates_for_job(self, job_id: str) -> List[SkillCandidate]:
        """Get top-K candidate skills for a job."""
        if job_id not in self.fusion_scores:
            return []
        
        scores = self.fusion_scores[job_id][:self.top_k]
        
        candidates = []
        for score_item in scores:
            skill_uri = score_item['skill_uri']
            skill_row = self.skills_df[self.skills_df['skill_uri'] == skill_uri]
            
            if skill_row.empty:
                continue
            
            candidates.append(SkillCandidate(
                skill_uri=skill_uri,
                skill_name=skill_row.iloc[0]['skill_name'],
                skill_description=skill_row.iloc[0]['skill_description'],
                original_score=score_item['score'],
                original_rank=score_item['rank'],
            ))
        
        return candidates
    
    async def _process_job(self, job_id: str, job_row: pd.Series) -> Optional[Dict]:
        """Process a single job and return full input/output data."""
        job_title = job_row['raw_title']
        job_description = job_row['raw_description']
        
        candidates = self._get_candidates_for_job(job_id)
        if not candidates:
            return None
        
        # Build skill list for storage
        skill_list = [
            {
                "skill_uri": c.skill_uri,
                "skill_name": c.skill_name,
                "original_score": c.original_score,
                "original_rank": c.original_rank,
            }
            for c in candidates
        ]
        
        prompt = self.fetcher.build_prompt(job_title, job_description, candidates)
        result = await self.fetcher.fetch_classification(prompt, job_id)
        
        return {
            "job_id": job_id,
            "job_title": job_title,
            "job_description": job_description,
            "skill_candidates": skill_list,
            "prompt": prompt,
            "llm_response": result,
        }
    
    async def _process_jobs_batch(self, job_items: List[tuple]) -> List[Optional[Dict]]:
        """Process a batch of jobs concurrently."""
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_with_semaphore(job_id, job_row):
            async with semaphore:
                return await self._process_job(job_id, job_row)
        
        tasks = [process_with_semaphore(job_id, job_row) for job_id, job_row in job_items]
        results = await tqdm_asyncio.gather(*tasks, desc="Fetching LLM classifications")
        
        return results
    
    async def run(self):
        """Run the fetching pipeline."""
        logger.info("Starting LLM fetching pipeline")
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
        
        # Separate successful and failed results
        successful = [r for r in results if r is not None and r["llm_response"]["success"]]
        failed = [r for r in results if r is not None and not r["llm_response"]["success"]]
        skipped = len(results) - len(successful) - len(failed)
        
        logger.info(f"Successfully processed: {len(successful)} jobs")
        logger.info(f"Failed: {len(failed)} jobs")
        if skipped > 0:
            logger.info(f"Skipped (no candidates): {skipped} jobs")
        
        # Save results as gzipped JSONL for efficiency
        output_file = self.output_dir / "llm_responses.jsonl.gz"
        with gzip.open(output_file, 'wt', encoding='utf-8') as f:
            for result in results:
                if result is not None:
                    f.write(json.dumps(result) + '\n')
        
        logger.info(f"Results saved to {output_file}")
        
        # Save metadata
        metadata = {
            "model": self.model,
            "top_k": self.top_k,
            "n_jobs_total": len(job_items),
            "n_successful": len(successful),
            "n_failed": len(failed),
            "n_skipped": skipped,
            "isco_groups": list(self.isco_groups) if self.isco_groups else None,
            "fusion_scores_source": str(self.fusion_scores_json),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        metadata_file = self.output_dir / "fetch_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_file}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds")
        
        # Print tier distribution from successful responses
        self._print_tier_stats(successful)
    
    def _print_tier_stats(self, results: List[Dict]):
        """Print tier distribution from LLM responses."""
        tier_counts = {'Essential': 0, 'Optional': 0, 'Irrelevant': 0}
        
        for result in results:
            classifications = result["llm_response"]["response"]["skill_classifications"]
            for item in classifications:
                tier = item.get("tier")
                if tier in tier_counts:
                    tier_counts[tier] += 1
        
        total = sum(tier_counts.values())
        logger.info("Tier distribution from LLM:")
        for tier, count in tier_counts.items():
            pct = (count / total * 100) if total > 0 else 0
            logger.info(f"  {tier}: {count} ({pct:.2f}%)")


async def main():
    parser = argparse.ArgumentParser(
        description="Fetch LLM tier classifications for skill candidates"
    )
    
    parser.add_argument("--fusion_scores_json", type=Path, required=True)
    parser.add_argument("--jobs_csv", type=Path, required=True)
    parser.add_argument("--skills_csv", type=Path, required=True)
    parser.add_argument("--occupations_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--max_workers", type=int, default=5)
    parser.add_argument("--isco_groups", type=str, default=None,
                       help="Comma-separated list of ISCO groups (e.g., '25' or '2512,2513')")
    
    args = parser.parse_args()
    
    dotenv.load_dotenv()
    
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key must be provided via --api_key or OPENAI_API_KEY env var")
    
    isco_groups = None
    if args.isco_groups:
        isco_groups = [g.strip() for g in args.isco_groups.split(',')]
        logger.info(f"All isco groups: {isco_groups}")
    
    pipeline = LLMFetcherPipeline(
        fusion_scores_json=args.fusion_scores_json,
        jobs_csv=args.jobs_csv,
        skills_csv=args.skills_csv,
        occupations_csv=args.occupations_csv,
        output_dir=args.output_dir,
        api_key=api_key,
        top_k=args.top_k,
        model=args.model,
        max_workers=args.max_workers,
        isco_groups=isco_groups,
    )
    
    await pipeline.run()


if __name__ == "__main__":
    asyncio.run(main())

