import pickle
import numpy as np
import sys
import pandas as pd

# Ensure numpy compatibility
if not hasattr(np, '_core'):
    sys.modules['numpy._core'] = np
    sys.modules['numpy._core.numeric'] = np
    sys.modules['numpy._core.multiarray'] = np

file1 = '/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/results/cpp/decorte/skill_overlap_scores_top100_isco_fused_2/test_clean_scores_skill_overlap.pkl'

with open(file1, 'rb') as f:
    d = pickle.load(f)

print(f"Target at index 4: {d['target_labels'][4]}")
print(f"Target at index 3: {d['target_labels'][3]}")

# Now check skills for this target in ESCO
occ_label = "accommodation manager"

occ_df = pd.read_csv('data/esco_datasets/occupations_en.csv')
rel_df = pd.read_csv('data/esco_datasets/occupationSkillRelations_en.csv')

# Find URI for this label
uri = occ_df[occ_df['preferredLabel'].str.strip().str.lower() == occ_label]['conceptUri'].values[0]
print(f"URI for {occ_label}: {uri}")

# Count skills
skills = rel_df[rel_df['occupationUri'] == uri]
print(f"Total skills in CSV for this URI: {len(skills)}")
print(f"Essential skills: {len(skills[skills['relationType'] == 'essential'])}")
print(f"Optional skills: {len(skills[skills['relationType'] == 'optional'])}")
