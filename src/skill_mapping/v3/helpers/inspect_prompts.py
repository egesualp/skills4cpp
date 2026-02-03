import sys
import os
import asyncio
from pathlib import Path
from typing import Dict, Optional, List
import pandas as pd

# Add src to python path to allow imports
# Current file: src/skill_mapping/v3/helpers/inspect_prompts.py
# We want to add 'src' folder to sys.path
current_file = Path(__file__).resolve()
src_dir = current_file.parents[3] # src
sys.path.append(str(src_dir))

from skill_mapping.v3.llm_reranker import LLMReranker, LLMRerankingPipeline

class MockLLMReranker(LLMReranker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_count = 0
        self.max_prompts = 5

    async def _call_llm_with_retry(self, prompt: str, job_id: str) -> Optional[Dict]:
        if self.prompt_count < self.max_prompts:
            print(f"\n=== PROMPT {self.prompt_count + 1} (Job ID: {job_id}) ===\n")
            print(prompt)
            print("\n" + "="*80 + "\n")
            self.prompt_count += 1
        return None

class InspectPipeline(LLMRerankingPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Swap the reranker with our mock
        self.reranker = MockLLMReranker(api_key="dummy")

    def _load_jobs(self) -> pd.DataFrame:
        """Load and filter jobs dataset with type-safe ISCO filtering."""
        print("Loading jobs (overridden method)...")
        df = pd.read_csv(self.jobs_csv)
        
        # Filter by ISCO groups if specified
        if self.isco_groups:
            # Load occupations to get ISCO groups
            occupations = pd.read_csv(self.occupations_csv)
            occupations = occupations[['conceptUri', 'iscoGroup']]
            occupations.columns = ['esco_id', 'iscoGroup']
            
            # Ensure iscoGroup is string in both places for comparison
            # Handle NaN values and convert float-like strings (e.g. "2512.0")
            occupations['iscoGroup'] = occupations['iscoGroup'].fillna('').astype(str).str.replace(r'\.0$', '', regex=True)
            
            # Merge to get ISCO groups for jobs
            df = df.merge(occupations, on='esco_id', how='left')
            
            # Ensure DataFrame column is also string
            df['iscoGroup'] = df['iscoGroup'].fillna('').astype(str).str.replace(r'\.0$', '', regex=True)
            
            # Convert self.isco_groups to strings just in case
            target_groups = set(str(g) for g in self.isco_groups)
            
            # Filter
            original_count = len(df)
            df = df[df['iscoGroup'].isin(target_groups)]
            print(f"Filtered jobs from {original_count} to {len(df)} using ISCO groups")
            
            if len(df) == 0:
                print("DEBUG: Sample ISCO groups in data:", df['iscoGroup'].unique()[:5] if not df.empty else "Empty DataFrame")
                print("DEBUG: Target groups:", list(target_groups)[:5])
        
        return df

    async def run(self):
        print("Starting inspection pipeline...")
        
        # Prepare job items, but limit to 5
        job_items = []
        count = 0
        
        # Iterate over jobs and select valid ones (present in fusion scores)
        for _, row in self.jobs_df.iterrows():
            job_id = str(row['job_id'])
            if job_id in self.fusion_scores:
                job_items.append((job_id, row))
                count += 1
                if count >= 5:
                    break
        
        print(f"Selected {len(job_items)} jobs for prompt inspection.")
        
        if not job_items:
            print("No matching jobs found in fusion scores!")
            return

        # Process only the selected batch
        await self._process_jobs_batch(job_items)

async def main():
    # Define paths matching run_pipeline.sh
    # Assuming workspace root is the base for relative paths
    workspace_root = Path(os.getcwd())
    
    # Paths from run_pipeline.sh
    fusion_scores = Path("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/linear_fusion_sum/best_fused_scores.json")
    
    # These are relative in run_pipeline.sh, so we resolve them relative to workspace root
    jobs_csv = workspace_root / "data/title_pairs_desc/decorte_master.csv"
    skills_csv = workspace_root / "data/esco_datasets/skills_en.csv"
    occupations_csv = workspace_root / "data/esco_datasets/occupations_en.csv"
    occ_skills_csv = workspace_root / "data/esco_datasets/occupationSkillRelations_en.csv"
    
    # Dummy output dir
    output_dir = workspace_root / "outputs/inspect_prompts"
    
    print(f"Loading data from: {workspace_root}")
    print(f"Fusion scores: {fusion_scores}")
    
    # Pre-process ISCO groups to find all 4-digit codes starting with 25
    # The pipeline uses exact matching, so passing [25] won't work directly
    print("Finding ISCO groups starting with '25'...")
    try:
        occ_df = pd.read_csv(occupations_csv)
        # Ensure iscoGroup is treated as string for prefix matching
        # Handle potential float/int conversions
        occ_df['iscoGroup'] = occ_df['iscoGroup'].astype(str).str.replace(r'\.0$', '', regex=True)
        
        target_groups = occ_df[occ_df['iscoGroup'].str.startswith('25')]['iscoGroup'].unique().tolist()
        print(f"Found {len(target_groups)} ISCO groups starting with 25: {target_groups[:5]}...")
        
        if not target_groups:
            print("Warning: No ISCO groups found starting with 25! Exiting.")
            return
            
    except Exception as e:
        print(f"Error reading occupations CSV: {e}")
        return

    try:
        pipeline = InspectPipeline(
            fusion_scores_json=fusion_scores,
            jobs_csv=jobs_csv,
            skills_csv=skills_csv,
            occupations_csv=occupations_csv,
            occ_skills_csv=occ_skills_csv,
            output_dir=output_dir,
            api_key="dummy", # Not used by MockLLMReranker
            top_k=100,
            model="gpt-4o-mini",
            max_workers=1, # Serial processing for clean output
            isco_groups=target_groups, # Pass the expanded list of codes
            prepare_ground_truth=False
        )
        
        await pipeline.run()
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file: {e}")
        print("Please ensure you are running this script from the project root.")

if __name__ == "__main__":
    asyncio.run(main())
