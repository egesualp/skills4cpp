import os
import sys
import pandas as pd
import asyncio
from jinja2 import Template
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio
from dotenv import load_dotenv
import time

# --- CONFIGURATION START ---

# OpenAI Configuration
# Make sure to set this environment variable before running the script
API_KEY_ENV_VAR = "OPENAI_API_KEY"
MODEL = "gpt-3.5-turbo"
TEMPERATURE = 0.7
MAX_TOKENS = 150

# Input/Output Configuration
#INPUT_PATH = "data/esco_datasets/occupations_en.csv"  # Path to your raw dataset
#OUTPUT_DIR = "data/processed/augmentation"
#OUTPUT_FILENAME = "augmented_esco_occupations.csv"
#
## The column name in the output dataframe where the generated text will be stored
#OUTPUT_COLUMN_KEY = "skill_brief"
#KEY_COLUMN = "conceptUri"
#
## Jinja2 Prompt Template
## Use {{ column_name }} to reference columns from your input dataframe
PROMPT_TEMPLATE = """
Given a job role (and its synonyms), briefly (1-2 sentences) describe the skills needed
for that job. 

Role: {{ preferredLabel }}
Synonyms: {{ altLabels }}
"""

# DECORTE
# Input/Output Configuration
INPUT_PATH = "data/title_pairs_desc/decorte_missing.csv"  # Path to your raw dataset
OUTPUT_DIR = "data/processed/augmentation"
OUTPUT_FILENAME = "augmented_decorte_missing_occupations.csv"

# The column name in the output dataframe where the generated text will be stored
OUTPUT_COLUMN_KEY = "skill_brief"
KEY_COLUMN = "job_id"

# Jinja2 Prompt Template
# Use {{ column_name }} to reference columns from your input dataframe
PROMPT_TEMPLATE = """
Given a job role (and its description), briefly (1-2 sentences) describe the skills needed
for that job. 

Role: {{ raw_title }}
Description: {{ raw_description }}
"""
#
#PROMPT_TEMPLATE = """
#Given a job role, briefly (1-2 sentences) describe the skills needed
#for that job. 
#
#Role: {{ raw_title }}
#"""

#### Skills
#INPUT_PATH = "/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv"  # Path to your raw dataset
#OUTPUT_DIR = "data/processed/augmentation"
#OUTPUT_FILENAME = "augmented_esco_skills.csv"
#
## The column name in the output dataframe where the generated text will be stored
#OUTPUT_COLUMN_KEY = "job_brief"
#KEY_COLUMN = "conceptUri"
#
PROMPT_TEMPLATE = """
Given a skill (and its synonyms), briefly (1-2 sentences) describe the job roles that
require that skill.

Skill: {{ raw_title }}
Synonyms: {{ altLabels }} 
"""

## TalentCLEF Task B Validation
#INPUT_PATH = "/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/talent_clef/TaskB/validation/validation_data.csv"
#OUTPUT_DIR = "data/processed/augmentation"
#OUTPUT_FILENAME = "augmented_talent_clef_taskb_validation.csv"
#
#OUTPUT_COLUMN_KEY = "skill_brief"
#KEY_COLUMN = "conceptUri"

#PROMPT_TEMPLATE = """
#Given a job role, briefly (1-2 sentences) describe the skills needed
#for that job. 
#
#Role: {{ job_text }}
#"""

#INPUT_PATH = "/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/occupations_en_expanded_clean.csv"
#OUTPUT_DIR = "data/processed/augmentation"
#OUTPUT_FILENAME = "augmented_esco_occupations_expanded.csv"
#
#OUTPUT_COLUMN_KEY = "skill_brief"
#KEY_COLUMN = "idx"
#
#PROMPT_TEMPLATE = """Given the specific job title {{ altLabel }} and the general context of {{ preferredLabel }}, briefly (1-2 sentences) describe the skills needed for that job."""

# Processing Configuration
# Set to None to process all rows, or an integer for testing
LIMIT_ROWS = None 
# Save progress every N rows
SAVE_INTERVAL = 10

MAX_CONCURRENT_REQUESTS = 20

# --- CONFIGURATION END ---

def get_openai_client():
    api_key = os.getenv(API_KEY_ENV_VAR)
    if not api_key:
        print(f"Error: Environment variable {API_KEY_ENV_VAR} is not set.")
        sys.exit(1)
    return AsyncOpenAI(api_key=api_key)

async def generate_text(client, prompt, model=MODEL):
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert HR analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return None

async def process_row(index, row, client, template, semaphore, df, pbar):
    async with semaphore:
        # Skip if already processed (useful for restarting)
        if pd.notna(row[OUTPUT_COLUMN_KEY]) and row[OUTPUT_COLUMN_KEY] != "":
            pbar.update(1)
            return
            
        try:
            # Create context from row data
            context = row.to_dict()

            # CUSTOM ADJUSTMENT HERE
            if type(context['altLabels']) == str:
                context['altLabels'] = ', '.join(context['altLabels'].split('\n'))

            if index == 0:
                print(context)
            
            # Render prompt
            prompt = template.render(**context)
            
            # Call API
            generated_content = await generate_text(client, prompt)
            
            if generated_content:
                df.at[index, OUTPUT_COLUMN_KEY] = generated_content
            
            # Sleep slightly to avoid rate limits if necessary
            # await asyncio.sleep(0.1) 
            
        except Exception as e:
            print(f"Error processing row {index}: {e}")
        finally:
            pbar.update(1)

async def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # 1. Setup
    print("Starting data augmentation process (Async)...")
    
    if not os.path.exists(INPUT_PATH):
        print(f"Error: Input file not found at {INPUT_PATH}")
        sys.exit(1)
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    
    client = get_openai_client()
    
    # 2. Load Data
    print(f"Loading data from {INPUT_PATH}...")
    try:
        if INPUT_PATH.endswith('.csv'):
            df = pd.read_csv(INPUT_PATH)
        elif INPUT_PATH.endswith('.parquet'):
            df = pd.read_parquet(INPUT_PATH)
        elif INPUT_PATH.endswith('.json'):
            df = pd.read_json(INPUT_PATH)
        else:
            # Default to CSV or try to infer
            df = pd.read_csv(INPUT_PATH, sep=None, engine='python')
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    # Additional filtering for duplicated keys
    df_raw = df.copy()
    #df = df.drop_duplicates(subset=KEY_COLUMN, keep='first')

    if LIMIT_ROWS:
        print(f"Limiting to first {LIMIT_ROWS} rows for testing.")
        df = df.head(LIMIT_ROWS)
    
    # Initialize output column if not present
    if OUTPUT_COLUMN_KEY not in df.columns:
        df[OUTPUT_COLUMN_KEY] = None
    
    # Compile template
    try:
        template = Template(PROMPT_TEMPLATE)
    except Exception as e:
        print(f"Error parsing Jinja2 template: {e}")
        sys.exit(1)

    # 3. Process Data
    print(f"Processing {len(df)} rows with {MAX_CONCURRENT_REQUESTS} concurrent requests...")
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    tasks = []
    pbar = tqdm_asyncio(total=len(df))

    for index, row in df.iterrows():
        task = asyncio.create_task(
            process_row(index, row, client, template, semaphore, df, pbar)
        )
        tasks.append(task)
            
    # Periodic save function
    async def periodic_save():
        while True:
            await asyncio.sleep(5)
            # Check if all tasks are done to break the loop finally
            all_done = all(t.done() for t in tasks)
            
            # Save logic
            #df_to_save = df[[KEY_COLUMN, OUTPUT_COLUMN_KEY]]
            df_to_save = df.copy()
            df_to_save.to_csv(output_path, index=False)
            
            if all_done:
                break
    
    # Run processing and saver concurrently
    saver_task = asyncio.create_task(periodic_save())
    await asyncio.gather(*tasks)
    await saver_task
    
    pbar.close()
            
    # 4. Final Save
    print(f"Saving final results to {output_path}")
    #df_to_save = df_raw.merge(df[[KEY_COLUMN, OUTPUT_COLUMN_KEY]], how='left', left_on=KEY_COLUMN, right_on=KEY_COLUMN)

    df_to_save = df.copy()
    df_to_save.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    asyncio.run(main())
