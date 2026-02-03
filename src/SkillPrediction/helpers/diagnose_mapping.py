#!/usr/bin/env python3
"""
Diagnostic script to identify why skills aren't being mapped.
"""
import json
import pandas as pd
from pathlib import Path

# Load data
print("Loading data files...")
print("-" * 80)

# 1. Load skill2group
with open("data/processed/skill2group.json") as f:
    skill2group = json.load(f)
print(f"✓ skill2group.json: {len(skill2group)} skill→group mappings")

# Sample a few
print("\n  Sample mappings:")
for i, (skill, group) in enumerate(list(skill2group.items())[:3]):
    print(f"    {skill[:60]}...")
    print(f"    → {group[:60]}...")

# 2. Load skills_per_occupations
skills_per_occ = pd.read_csv("data/skills_per_occupations.csv")
print(f"\n✓ skills_per_occupations.csv: {len(skills_per_occ)} rows")
print(f"  Unique occupations: {skills_per_occ['occupationUri'].nunique()}")
print(f"  Unique skills: {skills_per_occ['skillUri'].nunique()}")

print("\n  Sample occupationUri values:")
for uri in skills_per_occ['occupationUri'].head(3):
    print(f"    {uri}")

print("\n  Sample skillUri values:")
for uri in skills_per_occ['skillUri'].head(3):
    print(f"    {uri}")

# 3. Load one pairs file
pairs_files = list(Path("data/title_pairs_desc").glob("*.csv"))
if pairs_files:
    sample_pairs = pd.read_csv(pairs_files[0], nrows=100)
    print(f"\n✓ Sample pairs file: {pairs_files[0].name}")
    print(f"  Columns: {list(sample_pairs.columns)}")
    
    if 'esco_id' in sample_pairs.columns:
        print(f"\n  Sample esco_id values:")
        for esc_id in sample_pairs['esco_id'].dropna().head(3):
            print(f"    {esc_id}")

# 4. Check overlaps
print("\n" + "=" * 80)
print("OVERLAP ANALYSIS")
print("=" * 80)

skills_in_mapping = set(skill2group.keys())
skills_in_relations = set(skills_per_occ['skillUri'].unique())

overlap = skills_in_mapping & skills_in_relations
print(f"\nSkills in skill2group.json: {len(skills_in_mapping)}")
print(f"Skills in skills_per_occupations.csv: {len(skills_in_relations)}")
print(f"Overlap (skills in both): {len(overlap)}")
print(f"Coverage: {100 * len(overlap) / len(skills_in_relations):.2f}% of skills_per_occupations")

# 5. Check if pairs esco_id match occupations
if pairs_files and 'esco_id' in sample_pairs.columns:
    pairs_occs = set(sample_pairs['esco_id'].dropna().unique())
    file_occs = set(skills_per_occ['occupationUri'].unique())
    
    occ_overlap = pairs_occs & file_occs
    print(f"\n\nOccupation IDs in pairs (sample): {len(pairs_occs)}")
    print(f"Occupation IDs in skills_per_occupations.csv: {len(file_occs)}")
    print(f"Overlap: {len(occ_overlap)}")
    if len(pairs_occs) > 0:
        print(f"Coverage: {100 * len(occ_overlap) / len(pairs_occs):.2f}% of pairs")

# 6. Recommendations
print("\n" + "=" * 80)
print("DIAGNOSIS & RECOMMENDATIONS")
print("=" * 80)

if len(overlap) == 0:
    print("\n❌ CRITICAL: Zero overlap between skill2group and skills_per_occupations!")
    print("\n   This means build_hierarchy.py is using a different ESCO dataset")
    print("   or extracting skills from a different source.")
    print("\n   SOLUTION:")
    print("   - Check that build_hierarchy.py uses the same ESCO version")
    print("   - Verify skillsHierarchy_en.csv contains the skills you need")
    print("   - Consider using broaderRelationsSkillPillar_en.csv instead")
    
elif len(overlap) < len(skills_in_relations) * 0.5:
    print(f"\n⚠️  WARNING: Low coverage ({100 * len(overlap) / len(skills_in_relations):.1f}%)")
    print("\n   Many skills in skills_per_occupations.csv are not in skill2group.json")
    print("\n   SOLUTIONS:")
    print("   - Use a more complete hierarchy file in build_hierarchy.py")
    print("   - Filter skills_per_occupations.csv to only include mapped skills")
    print("   - Expand skill2group.json to include more skills")
    
else:
    print(f"\n✓ Good coverage: {100 * len(overlap) / len(skills_in_relations):.1f}%")
    print("\n   The mapping should work. If you still see 0 rows after mapping,")
    print("   check that esco_id values in your pairs match occupationUri values.")

print("\n" + "=" * 80)


