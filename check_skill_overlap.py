import pandas as pd
import json
from pathlib import Path

# Paths
relations_path = "data/esco_datasets/occupationSkillRelations_en.csv"
similarity_path = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/decorte_w_desc/similarity_scores.json"

# Load Relations Skills
print(f"Loading relations from {relations_path}...")
df_rel = pd.read_csv(relations_path)
rel_skills = set(df_rel["skillUri"].unique())
print(f"Found {len(rel_skills)} unique skill URIs in relations file.")

# Load Similarity Skills
print(f"Loading similarity scores from {similarity_path}...")
with open(similarity_path, "r") as f:
    sim_data = json.load(f)

sim_skills = set()
for skills in sim_data.values():
    for s in skills:
        sim_skills.add(s["skill_uri"])
print(f"Found {len(sim_skills)} unique skill URIs in similarity scores.")

# Check Overlap
overlap = sim_skills.intersection(rel_skills)
print(f"Skill Overlap count: {len(overlap)}")
print(f"Percentage of similarity skills in relations: {len(overlap)/len(sim_skills)*100:.2f}%")
print(f"Percentage of relations skills in similarity: {len(overlap)/len(rel_skills)*100:.2f}%")

if len(overlap) == 0:
    print("\nWARNING: No overlap found between similarity skills and relations skills.")
    print("Sample Relation Skill:", list(rel_skills)[0])
    print("Sample Similarity Skill:", list(sim_skills)[0])

