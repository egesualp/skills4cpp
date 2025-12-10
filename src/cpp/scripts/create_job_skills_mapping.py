import json
import os
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import re
from cpp.utils import replace_esco_titles


def extract_job_titles_from_dataset(data_type='decorte_esco', language='en', title_type='esco'):
    """
    Extracts unique job titles from various datasets using the Data module pattern.
    
    Args:
        data_type (str): Dataset type ('decorte', 'decorte_esco', 'karrierewege', etc.)
        language (str): Language variant for karrierewege datasets ('en', 'en_free', 'en_free_cp', 'esco_100k')
        title_type (str): Type of title to extract ('esco', 'raw', 'preferredLabel')
    
    Returns:
        set: Set of unique job titles
    """
    
    all_titles = set()
    
    if data_type in ['decorte', 'decorte_esco']:
        # Load decorte dataset
        print(f"Loading {data_type} dataset...")
        dataset = load_dataset("jensjorisdecorte/anonymous-working-histories")
        
        # Apply title replacements
        print("Applying ESCO title replacements...")
        for i in range(16):
            dataset['train'] = dataset['train'].map(lambda example: replace_esco_titles(example, i))
            dataset['validation'] = dataset['validation'].map(lambda example: replace_esco_titles(example, i))
            dataset['test'] = dataset['test'].map(lambda example: replace_esco_titles(example, i))
        
        # Extract titles
        print("Extracting unique job titles from dataset...")
        for split_name in ['train', 'validation', 'test']:
            split_data = dataset[split_name]
            for example in tqdm(split_data, desc=f"Processing {split_name}"):
                for i in range(example["number_of_experiences"]):
                    if title_type == 'esco':
                        title = example.get(f'ESCO_title_{i}')
                    else:  # raw
                        title = example.get(f'title_{i}')
                    
                    if title and not pd.isna(title):
                        all_titles.add(title.strip().lower())
    
    elif 'karrierewege' in data_type:
        # Load karrierewege dataset
        print(f"Loading karrierewege dataset with language={language}...")
        if language in ['en_free', 'esco_100k', 'en_free_cp']:
            dataset = load_dataset("ElenaSenger/Karrierewege_plus")
        elif language == 'en':
            dataset = load_dataset("ElenaSenger/Karrierewege")
        else:
            raise ValueError(f"Unsupported language: {language}")
        
        # Extract titles based on language variant
        print("Extracting unique job titles from dataset...")
        for split_name in ['train', 'validation', 'test']:
            split_data = dataset[split_name].to_pandas()
            
            if language in ['en', 'esco_100k']:
                column_name = 'preferredLabel_en'
            elif language == 'en_free':
                column_name = 'new_job_title_en_occ' if title_type == 'raw' else 'preferredLabel_en'
            elif language == 'en_free_cp':
                column_name = 'new_job_title_en_cp' if title_type == 'raw' else 'preferredLabel_en'
            else:
                raise ValueError(f"Unsupported language: {language}")
            
            for title in tqdm(split_data[column_name].dropna().unique(), desc=f"Processing {split_name}"):
                all_titles.add(title.strip().lower())
    
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")
    
    print(f"Found {len(all_titles)} unique job titles")
    return all_titles


def create_esco_ground_truth_mapping(base_dir, data_type='decorte_esco', language='en', 
                                    title_type='esco', essential_only=False, output_dir=None):
    """
    Creates job title to skills mapping using ESCO ground-truth occupation-skill relations
    from various datasets and ESCO taxonomy.
    
    Skills are exported with placeholder scores (1.0). IDF weighting will be calculated
    during training based on the actual dataset distribution.
    
    IMPORTANT: Job titles are normalized (lowercase + stripped of whitespace) to ensure
    consistent matching during training. This handles variations like:
    - "Cloud DevOps Engineer" -> "cloud devops engineer"
    - " Data Scientist " -> "data scientist"
    - "Doctor's Assistant" -> "doctor's assistant"
    
    Args:
        base_dir (str): Base directory of the project
        data_type (str): Dataset type ('decorte', 'decorte_esco', 'karrierewege', etc.)
        language (str): Language variant for karrierewege datasets
        title_type (str): Type of title to extract ('esco', 'raw')
        essential_only (bool): Only include essential skills
        output_dir (str): Custom output directory (optional)
    """
    esco_dir = os.path.join(base_dir, 'data', 'esco_datasets')
    
    if output_dir is None:
        output_suffix = f"{data_type}"
        if data_type == 'karrierewege' and language != 'en':
            output_suffix += f"_{language}"
        output_dir = os.path.join(base_dir, 'results', f'{output_suffix}_esco_ground_truth')
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ESCO files
    print("Loading ESCO taxonomy files...")
    occupations_df = pd.read_csv(os.path.join(esco_dir, 'occupations_en.csv'))
    skills_df = pd.read_csv(os.path.join(esco_dir, 'skills_en.csv'))
    relations_df = pd.read_csv(os.path.join(esco_dir, 'occupationSkillRelations_en.csv'))
    
    if essential_only:
        relations_df = relations_df[relations_df['relationType'] == 'essential']
        print(f"Filtering for essential skills only")
    
    # Create mappings
    uri_to_title = pd.Series(occupations_df.preferredLabel.values, 
                             index=occupations_df.conceptUri).to_dict()
    title_to_uri = pd.Series(occupations_df.conceptUri.values, 
                             index=occupations_df.preferredLabel).to_dict()
    skill_uri_to_name = pd.Series(skills_df.preferredLabel.values, 
                                  index=skills_df.conceptUri).to_dict()
    
    # Extract all unique job titles from the dataset
    all_job_titles = extract_job_titles_from_dataset(data_type, language, title_type)
    
    # Create job-skill mappings (without IDF - will be calculated in training script)
    print("Creating job-skill mappings...")
    master_mapping = []
    titles_without_mapping = []
    
    # Create case-insensitive title to URI mapping
    title_to_uri_lower = {k.strip().lower(): v for k, v in title_to_uri.items()}
    
    for job_title in tqdm(sorted(all_job_titles), desc="Mapping jobs to skills"):
        # Get occupation URI
        occ_uri = title_to_uri_lower.get(job_title.lower())
        
        if not occ_uri:
            titles_without_mapping.append(job_title)
            continue
        
        # Get all skills for this occupation
        occ_skills = relations_df[relations_df['occupationUri'] == occ_uri]
        
        if len(occ_skills) == 0:
            titles_without_mapping.append(job_title)
            continue
        
        # Build skills list (score is placeholder - IDF will be calculated during training)
        skills_list = []
        for _, row in occ_skills.iterrows():
            skill_uri = row['skillUri']
            skill_name = skill_uri_to_name.get(skill_uri, 'Unknown')
            
            skills_list.append({
                'skill': skill_name,
                'score': 1.0,  # Placeholder - IDF calculated in training based on dataset
                'skillUri': skill_uri
            })
        
        master_mapping.append({
            'job_title': job_title,
            'skills': skills_list
        })
    
    # Save master mapping
    suffix = "_essential" if essential_only else ""
    master_output_file = os.path.join(output_dir, f'job_title_skills_master{suffix}.json')
    with open(master_output_file, 'w') as f:
        json.dump(master_mapping, f, indent=2)
    print(f"\n✓ Saved master mapping to {master_output_file}")
    print(f"  - Mapped {len(master_mapping)} job titles")
    print(f"  - {len(titles_without_mapping)} titles without ESCO mapping")
    
    if titles_without_mapping:
        print(f"\nTitles without mapping (first 10):")
        for title in titles_without_mapping[:10]:
            print(f"  - {title}")
    
    # Also save as CSV for easy inspection (flattened format)
    csv_rows = []
    row_index = 0
    for item in master_mapping:
        job_title = item['job_title']
        for skill in item['skills']:
            csv_rows.append({
                'original_row_index': row_index,
                'job_title': job_title,
                'skill': skill['skill'],
                'score': skill['score'],  # Placeholder (1.0) - IDF calculated during training
                'skillUri': skill['skillUri']
            })
        row_index += 1
    
    csv_df = pd.DataFrame(csv_rows)
    csv_output_file = os.path.join(output_dir, f'job_title_skills_master{suffix}.csv')
    csv_df.to_csv(csv_output_file, index=False)
    print(f"✓ Saved CSV version to {csv_output_file}")
    print(f"  Note: Scores are placeholders (1.0). IDF will be calculated during training based on dataset.")
    
    return master_output_file


def create_job_skills_mapping_with_scores_and_uri(base_dir):
    """
    Creates job title to skills mapping from prediction files and job data,
    including skill scores and URIs.
    """
    prediction_dir = os.path.join(base_dir, 'results', 'decorte_jobbert_v2_baseline')
    job_data_dir = os.path.join(base_dir, 'data', 'title_pairs_desc')
    output_dir = os.path.join(base_dir, 'results', 'decorte_jobbert_v2_baseline')
    skills_file = os.path.join(base_dir, 'data', 'esco_datasets', 'skills_en.csv')
    os.makedirs(output_dir, exist_ok=True)

    # Create skill to URI mapping
    skills_df = pd.read_csv(skills_file)
    skill_to_uri = pd.Series(skills_df.conceptUri.values, index=skills_df.preferredLabel).to_dict()

    splits = ['train', 'val', 'test']
    master_mapping = []

    for split in splits:
        prediction_file = os.path.join(prediction_dir, f'eval_baseline_JobBERT_decorte_{split}_predictions.json')
        job_data_file = os.path.join(job_data_dir, f'decorte_{split}_pairs.csv')
        output_file = os.path.join(output_dir, f'job_title_skills_{split}.json')

        with open(prediction_file, 'r') as f:
            predictions = json.load(f)

        job_data = pd.read_csv(job_data_file)
        job_titles = job_data['raw_title'].tolist()

        split_mapping = []
        for job_id_str, skills_data in predictions.items():
            job_id = int(job_id_str.split('_')[1])
            if job_id < len(job_titles):
                job_title = job_titles[job_id]
                
                processed_skills = []
                for item in skills_data:
                    skill_name = item['skill']
                    processed_skills.append({
                        'skill': skill_name,
                        'score': item['score'],
                        'skillUri': skill_to_uri.get(skill_name)
                    })
                
                split_mapping.append({'job_title': job_title, 'skills': processed_skills})

        with open(output_file, 'w') as f:
            json.dump(split_mapping, f, indent=2)
        print(f"Saved {split} mapping to {output_file}")

        master_mapping.extend(split_mapping)

    master_output_file = os.path.join(output_dir, 'job_title_skills_master.json')
    with open(master_output_file, 'w') as f:
        json.dump(master_mapping, f, indent=2)
    print(f"Saved master mapping to {master_output_file}")
    
    # Also save as CSV (flattened format)
    csv_rows = []
    row_index = 0
    for item in master_mapping:
        job_title = item['job_title']
        for skill in item['skills']:
            csv_rows.append({
                'original_row_index': row_index,
                'job_title': job_title,
                'skill': skill['skill'],
                'score': skill['score'],
                'skillUri': skill['skillUri']
            })
        row_index += 1
    
    csv_df = pd.DataFrame(csv_rows)
    csv_output_file = os.path.join(output_dir, 'job_title_skills_master.csv')
    csv_df.to_csv(csv_output_file, index=False)
    print(f"Saved CSV version to {csv_output_file}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Create job-skills mapping from various datasets")
    parser.add_argument("--mode", type=str, default="ir_extracted",
                       choices=["ir_extracted", "esco_ground_truth"],
                       help="Mode: 'ir_extracted' for IR-extracted skills, 'esco_ground_truth' for ESCO taxonomy skills")
    parser.add_argument("--base_dir", type=str, default=".",
                       help="Base directory of the project")
    parser.add_argument("--source", type=str, default="decorte_esco",
                       choices=["decorte", "decorte_esco", "karrierewege", "karrierewege_occ", 
                               "karrierewege_100k", "karrierewege_cp"],
                       help="Source dataset to extract job titles from")
    parser.add_argument("--language", type=str, default="en",
                       choices=["en", "en_free", "en_free_cp", "esco_100k", "de", "de_free", "de_free_cp"],
                       help="Language variant for karrierewege datasets")
    parser.add_argument("--title_type", type=str, default="esco",
                       choices=["esco", "raw", "preferredLabel"],
                       help="Type of title to extract: 'esco' for ESCO titles, 'raw' for original job titles")
    parser.add_argument("--essential_only", action='store_true', 
                       help="Only include essential skills (esco_ground_truth mode)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Custom output directory (optional)")

    args = parser.parse_args()
    
    # Map source to language for karrierewege variants
    source_to_language = {
        'karrierewege': 'en',
        'karrierewege_occ': 'en_free',
        'karrierewege_100k': 'esco_100k',
        'karrierewege_cp': 'en_free_cp'
    }
    
    # Determine data_type and language
    if args.source in source_to_language:
        data_type = 'karrierewege'
        language = source_to_language[args.source]
    else:
        data_type = args.source
        language = args.language
    
    if args.mode == "esco_ground_truth":
        print("=" * 80)
        print("Creating ESCO Ground-Truth Skill Mapping")
        print(f"Source: {args.source} (data_type={data_type}, language={language})")
        print(f"Title type: {args.title_type}")
        print(f"Essential only: {args.essential_only}")
        print("=" * 80)
        
        output_file = create_esco_ground_truth_mapping(
            base_dir=args.base_dir,
            data_type=data_type,
            language=language,
            title_type=args.title_type,
            essential_only=args.essential_only,
            output_dir=args.output_dir
        )
        
        print(f"\n✓ Done! Use this file in training:")
        print(f"  --master_skill_file {output_file}")
    else:
        print("=" * 80)
        print("Creating IR-Extracted Skill Mapping")
        print("=" * 80)
        create_job_skills_mapping_with_scores_and_uri(args.base_dir)
        print(f"\n✓ Done!")
