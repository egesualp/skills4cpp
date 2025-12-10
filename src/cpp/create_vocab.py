import pandas as pd
import json
import os
from tqdm import tqdm

# --- Configuration ---

# Define the columns in your ESCO skills CSV you want to encode.
# Add or remove columns as needed.
META_FEATURE_COLUMNS = [
    'skillType',
    'reuseLevel',
    'pillar_label',
    'level1_label',
    #'level2_label',
    # 'level3', # etc.
]

ESCO_SKILLS_CSV_PATH = "data/processed/master_datasets_2/master_skill_complete_hierarchy.csv" # <-- Point this to your ESCO skills dataframe
OUTPUT_VOCAB_FILE = "data/processed/master_datasets_2/structured_vocab.json"
OUTPUT_SKILL_PROPERTIES_FILE = "data/processed/master_datasets_2/skill_properties_map.json"

# --- Main Script ---

def create_vocab_and_skill_map():
    """
    Scans the ESCO skills dataframe to create:
    1. A master vocabulary (JSON) for all structured meta-features.
    2. A lookup map (JSON) from skillUri to its list of meta-features.
    """
    print(f"Loading ESCO skills from: {ESCO_SKILLS_CSV_PATH}")
    try:
        df = pd.read_csv(ESCO_SKILLS_CSV_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find {ESCO_SKILLS_CSV_PATH}")
        return

    # A set to hold all unique, prefixed feature names
    global_vocab_set = set()
    
    # A dictionary to build our {skillUri: [features]} map
    skill_properties_map = {}

    print("Building vocabulary and skill property map...")
    # Iterate over the dataframe row by row
    # This correctly handles skills with multiple parent paths (multiple rows)
    for _, row in tqdm(df.iterrows(), total=len(df)):
        skill_uri = row['skillUri']
        
        if skill_uri not in skill_properties_map:
            skill_properties_map[skill_uri] = set()

        # Iterate over the meta-feature columns we defined
        for col in META_FEATURE_COLUMNS:
            if col in row and pd.notna(row[col]):
                # Create the prefixed feature name
                feature_name = f"{col}:{row[col]}"
                
                # Add to our global vocabulary
                global_vocab_set.add(feature_name)
                
                # Add this feature to this skill's property set
                skill_properties_map[skill_uri].add(feature_name)

    # --- 1. Save the Master Vocabulary ---
    
    # Sort the set for a consistent index
    sorted_vocab_list = sorted(list(global_vocab_set))
    
    # Create the final {feature_name: index} map
    structured_vocab = {feature: idx for idx, feature in enumerate(sorted_vocab_list)}

    os.makedirs(os.path.dirname(OUTPUT_VOCAB_FILE), exist_ok=True)
    with open(OUTPUT_VOCAB_FILE, 'w') as f:
        json.dump(structured_vocab, f, indent=2)
        
    print(f"\nSuccessfully created vocabulary with {len(structured_vocab)} unique features.")
    print(f"Saved to: {OUTPUT_VOCAB_FILE}")

    # --- 2. Save the Skill-to-Properties Map ---
    
    # Convert sets to lists for JSON serialization
    final_skill_properties_map = {uri: list(features) for uri, features in skill_properties_map.items()}
    
    with open(OUTPUT_SKILL_PROPERTIES_FILE, 'w') as f:
        json.dump(final_skill_properties_map, f, indent=2)
        
    print(f"Successfully created skill property map for {len(final_skill_properties_map)} skills.")
    print(f"Saved to: {OUTPUT_SKILL_PROPERTIES_FILE}")

if __name__ == "__main__":
    create_vocab_and_skill_map()