import pandas as pd
from pathlib import Path

# Paths
relations_path = "data/esco_datasets/occupationSkillRelations_en.csv"
jobs_path = "data/processed/augmentation/augmented_decorte_occupations_with_desc.csv"

# Load Relations
print(f"Loading relations from {relations_path}...")
df_rel = pd.read_csv(relations_path)
if "occupationUri" in df_rel.columns:
    rel_uris = set(df_rel["occupationUri"].unique())
else:
    print("Error: 'occupationUri' column not found in relations file.")
    exit(1)
print(f"Found {len(rel_uris)} unique occupation URIs in relations file.")

# Load Jobs
print(f"Loading jobs from {jobs_path}...")
df_jobs = pd.read_csv(jobs_path)
if "esco_id" in df_jobs.columns:
    job_uris = set(df_jobs["esco_id"].unique())
else:
    print("Error: 'esco_id' column not found in jobs file.")
    exit(1)
print(f"Found {len(job_uris)} unique ESCO IDs in jobs file.")

# Check Overlap
overlap = job_uris.intersection(rel_uris)
print(f"Overlap count: {len(overlap)}")

if len(overlap) == 0:
    print("\nWARNING: No overlap found between jobs ESCO IDs and relations file occupation URIs.")
    print("Sample Relation URI:", list(rel_uris)[0])
    print("Sample Job URI:", list(job_uris)[0])
else:
    print(f"Found {len(overlap)} matching URIs.")











