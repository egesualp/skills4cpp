import os
import pandas as pd
from typing import List, Dict, Tuple
from datasets import Dataset

# --- CONFIGURATION ---
# Define the LLM Model for consistent referencing
LLM_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

# Define local paths for ESCO files (as per your prompt)
ESCO_DIR = './data/esco_datasets'
OCCUPATIONS_FILE = os.path.join(ESCO_DIR, 'occupations_en.csv')
SKILLS_FILE = os.path.join(ESCO_DIR, 'skills_en.csv')
RELATIONS_FILE = os.path.join(ESCO_DIR, 'occupationSkillRelations_en.csv')

# --- PROMPT TEMPLATES (for LLM inference/generation) ---
# We define the final BI-ENCODER inputs here, which rely on the generated LLM text.
# The actual LLM-generated text (llm_description) will be inserted into these templates in 'inference.py'.

# Task B Query Input (Job Title + LLM-Generated Description)
# The {llm_description} is the output of the LLM call for this job title.
JOB_AUGMENT_TEMPLATE = "Job: {job_title} [SEP] Description: {llm_description}"

# Task B Corpus Input (Skill Name + ESCO Description)
# ESCO provides the skill descriptions, so we use that directly.
SKILL_AUGMENT_TEMPLATE = "Skill: {skill_name} [SEP] Description: {esco_description}"


# --- MAIN FUNCTIONS ---

def load_and_prepare_esco_data() -> Tuple[List[Dict], List[Dict]]:
    """
    Loads ESCO files, prepares them, and extracts the unique list of items
    to be sent to the LLM for augmentation.
    """
    if not all(os.path.exists(f) for f in [OCCUPATIONS_FILE, SKILLS_FILE, RELATIONS_FILE]):
        print(f"ERROR: ESCO files not found in {ESCO_DIR}. Simulating load.")
        # --- SIMULATE DATA LOADING FOR STRUCTURE ---
        occupations = pd.DataFrame({'conceptUri': ['occ_a', 'occ_b'], 'preferredLabel': ['Software Engineer', 'Marketing Manager']})
        skills = pd.DataFrame({'conceptUri': ['skill_1', 'skill_2', 'skill_3'], 'preferredLabel': ['Python', 'Teamwork', 'Budgeting'], 'description': ['Python code development.', 'Collaborating effectively.', 'Managing financial resources.']})
        relations = pd.DataFrame({'occupatiomUri': ['occ_a', 'occ_a', 'occ_b'], 'skillUri': ['skill_1', 'skill_2', 'skill_3']})
        
        # Manually create the desired output for augmentation simulation
        job_titles_for_llm = [{'job_title': title, 'uri': uri} for uri, title in occupations[['conceptUri', 'preferredLabel']].values]
        skills_for_llm = [{'skill_name': name, 'esco_description': desc} for name, desc in skills[['preferredLabel', 'description']].values]
        # --- END SIMULATION ---
        print("Using simulated ESCO data for structure.")
        return job_titles_for_llm, skills_for_llm

    # --- REAL ESCO DATA LOADING ---
    occupations = pd.read_csv(OCCUPATIONS_FILE, usecols=['conceptUri', 'preferredLabel'])
    skills = pd.read_csv(SKILLS_FILE, usecols=['conceptUri', 'preferredLabel', 'description'])
    relations = pd.read_csv(RELATIONS_FILE, usecols=['occupationUri', 'skillUri', 'relationType'])
    
    # Rename columns for consistent merging
    occupations.rename(columns={'conceptUri': 'uri', 'preferredLabel': 'job_title'}, inplace=True)
    skills.rename(columns={'conceptUri': 'skill_uri', 'preferredLabel': 'skill_name', 'description': 'esco_description'}, inplace=True)
    
    # --- Extract Unique Job Titles for LLM Generation (Task B Query/Test Augmentation) ---
    # We use preferredLabel as the canonical title for the LLM prompt
    job_titles_for_llm = occupations[['job_title', 'uri']].drop_duplicates().to_dict('records')

    # --- Extract Unique Skills for LLM Generation (Task B Corpus) ---
    # Note: ESCO already contains good descriptions, but LLM augmentation often targets skills as well.
    # We extract the canonical name and ESCO description for later use.
    skills_for_llm = skills[['skill_name', 'esco_description']].dropna().drop_duplicates().to_dict('records')
    
    return job_titles_for_llm, skills_for_llm


def create_unaugmented_datasets(
    job_titles_list: List[Dict], 
    skills_list: List[Dict]
) -> Tuple[Dataset, Dataset]:
    """
    Creates HuggingFace Datasets from the raw lists, ready to receive LLM output.
    """
    # 1. Job Queries Dataset (The queries to be augmented by LLM)
    # We use a subset of these as 'queries' in the inference step, but the full list is the source.
    job_df = pd.DataFrame(job_titles_list).rename(columns={'job_title': 'query_title'})
    job_ds = Dataset.from_pandas(job_df)
    
    # 2. Skills Corpus Dataset (The candidates to be embedded)
    skill_df = pd.DataFrame(skills_list).rename(columns={'skill_name': 'corpus_name'})
    skill_ds = Dataset.from_pandas(skill_df)
    
    return job_ds, skill_ds

# --- PUBLIC INTERFACE ---

def get_data_for_augmentation() -> Tuple[Dataset, Dataset]:
    """
    Main function to get the raw data frames that need LLM augmentation.
    Returns: (Job Titles Dataset, Skills Corpus Dataset)
    """
    job_titles, skills_list = load_and_prepare_esco_data()
    job_ds, skill_ds = create_unaugmented_datasets(job_titles, skills_list)
    
    print("\nData Source for LLM Augmentation Ready:")
    print(f"Total Unique Job Titles (Source for Queries): {len(job_ds)}")
    print(f"Total Unique Skill Names (Source for Corpus): {len(skill_ds)}")
    
    return job_ds, skill_ds

# --- EXECUTION CHECK (Optional, for local debugging) ---
if __name__ == '__main__':
    # You would typically run a script to download the ESCO data first.
    # Example download command: wget https://esco.ec.europa.eu/api/resource/download/en/csv
    # os.makedirs(ESCO_DIR, exist_ok=True)
    # os.system(f"touch {OCCUPATIONS_FILE} {SKILLS_FILE} {RELATIONS_FILE}")
    
    import json
    
    jobs_ds, skills_ds = get_data_for_augmentation()
    
    # Print structured input samples
    print("\n" + "="*80)
    print("STRUCTURED INPUT SAMPLES")
    print("="*80)
    
    print("\n--- JOB TITLES (Query Input) ---")
    print(f"Total samples: {len(jobs_ds)}")
    print("\nSample job titles (first 3):")
    for i in range(min(3, len(jobs_ds))):
        sample = dict(jobs_ds[i])
        print(f"\n  Sample {i+1}:")
        print(f"    {json.dumps(sample, indent=6)}")
        print(f"    Template usage: {JOB_AUGMENT_TEMPLATE.format(job_title=sample.get('query_title', ''), llm_description='[LLM_GENERATED_DESCRIPTION]')}")
    
    print("\n--- SKILLS (Corpus Input) ---")
    print(f"Total samples: {len(skills_ds)}")
    print("\nSample skills (first 3):")
    for i in range(min(3, len(skills_ds))):
        sample = dict(skills_ds[i])
        print(f"\n  Sample {i+1}:")
        print(f"    {json.dumps(sample, indent=6)}")
        if 'esco_description' in sample:
            print(f"    Template usage: {SKILL_AUGMENT_TEMPLATE.format(skill_name=sample.get('corpus_name', ''), esco_description=sample.get('esco_description', ''))}")
    
    print("\n" + "="*80)
    print("Dataset Structure:")
    print("="*80)
    print(f"\nJobs Dataset features: {list(jobs_ds.features.keys())}")
    print(f"Skills Dataset features: {list(skills_ds.features.keys())}")
    
    # You can now save these unaugmented files to disk to load later
    # jobs_ds.to_json(os.path.join(ESCO_DIR, 'jobs_unaugmented.jsonl'))
    # skills_ds.to_json(os.path.join(ESCO_DIR, 'skills_unaugmented.jsonl'))