#!/usr/bin/env python3
"""
Explore ESCO Skill Hierarchy Structure
Uses the preprocessed outputs from build_hierarchy.py
Shows: Pillars → Groups → Skills with examples
"""
import pandas as pd
import json
from pathlib import Path

def main():
    processed_dir = Path("data/processed")
    
    print("=" * 80)
    print("ESCO SKILL HIERARCHY EXPLORER")
    print("=" * 80)
    
    # Load preprocessed data from build_hierarchy.py outputs
    print("\n[1] Loading preprocessed hierarchy data...")
    skill_lookup = pd.read_csv(processed_dir / "skill_to_group_pillar.csv")
    
    with open(processed_dir / "skill2group.json") as f:
        skill2group = json.load(f)
    
    with open(processed_dir / "group2label.json") as f:
        group2label = json.load(f)
    
    print(f"  ✓ Loaded skill_to_group_pillar.csv: {len(skill_lookup):,} rows")
    print(f"  ✓ Loaded skill2group.json: {len(skill2group):,} mappings")
    print(f"  ✓ Loaded group2label.json: {len(group2label):,} labels")
    
    # Show hierarchy structure
    print("\n" + "=" * 80)
    print("HIERARCHY STRUCTURE")
    print("=" * 80)
    print("""
    Level 0 (Top):    PILLARS (highest level categories)
                      ↓
    Level 1 (Middle): SKILL GROUPS (categories of related skills)
                      ↓
    Level 2 (Bottom): SKILLS (individual competencies)
    
    Example:
        Pillar:       "management skills"
        ├─ Group:     "supervising people"
        │  ├─ Skill:  "coordinate technical teams in artistic productions"
        │  ├─ Skill:  "manage staff"
        │  └─ Skill:  "supervise work"
        └─ Group:     "leading and motivating"
           ├─ Skill:  "inspire employees"
           └─ Skill:  "lead others"
    """)
    
    # Count hierarchy levels
    print("=" * 80)
    print("HIERARCHY STATISTICS")
    print("=" * 80)
    
    # Count pillars
    n_pillars = skill_lookup['pillar_label'].nunique()
    print(f"\nTotal Pillars: {n_pillars}")
    
    # Count groups
    n_groups = skill_lookup['group_label'].nunique()
    print(f"Total Skill Groups: {n_groups}")
    
    # Count skills
    n_skills = skill_lookup['skill_label'].nunique()
    print(f"Total Skills: {n_skills}")
    
    # Average skills per group
    avg_skills = skill_lookup.groupby('group_label').size().mean()
    print(f"\nAverage skills per group: {avg_skills:.1f}")
    
    # Show pillar distribution
    print("\n" + "=" * 80)
    print("TOP 15 PILLARS (by number of skills)")
    print("=" * 80)
    pillar_counts = skill_lookup.groupby('pillar_label').size().sort_values(ascending=False).head(15)
    for i, (pillar, count) in enumerate(pillar_counts.items(), 1):
        if pd.notna(pillar):
            print(f"{i:2d}. {pillar:65s} ({count:4d} skills)")
    
    # Show 5 concrete examples
    print("\n" + "=" * 80)
    print("5 COMPLETE HIERARCHY EXAMPLES")
    print("=" * 80)
    
    # Select 5 diverse examples from different pillars
    examples = skill_lookup.sample(n=min(5, len(skill_lookup)), random_state=42)
    
    for i, (_, row) in enumerate(examples.iterrows(), 1):
        print(f"\n{'─' * 80}")
        print(f"Example {i}:")
        print(f"{'─' * 80}")
        print(f"  🏛️  PILLAR:      {row['pillar_label']}")
        print(f"  📦 GROUP:       {row['group_label']}")
        print(f"  ⚙️  SKILL:       {row['skill_label']}")
        print(f"\n  URIs:")
        print(f"     Skill:      {row['skill_id'][:70]}...")
        print(f"     Group:      {row['group_id'][:70]}...")
        if pd.notna(row.get('pillar_id')):
            print(f"     Pillar:     {row['pillar_id'][:70]}...")
    
    # Show skills within a specific group
    print("\n" + "=" * 80)
    print("EXAMPLE: Skills in a Specific Group")
    print("=" * 80)
    
    # Pick a group with a good number of skills
    group_sizes = skill_lookup.groupby('group_label').size()
    example_group = group_sizes[group_sizes.between(10, 50)].sample(n=1, random_state=42).index[0]
    
    group_skills = skill_lookup[skill_lookup['group_label'] == example_group]
    print(f"\nGroup: '{example_group}'")
    if pd.notna(group_skills.iloc[0]['pillar_label']):
        print(f"Pillar: '{group_skills.iloc[0]['pillar_label']}'")
    print(f"Number of skills: {len(group_skills)}\n")
    
    print("Skills in this group:")
    for i, skill in enumerate(group_skills['skill_label'].head(15), 1):
        print(f"  {i:2d}. {skill}")
    
    if len(group_skills) > 15:
        print(f"  ... and {len(group_skills) - 15} more")
    
    # Show how the data structure looks
    print("\n" + "=" * 80)
    print("DATA STRUCTURE: skill_to_group_pillar.csv")
    print("=" * 80)
    print("\nColumns:")
    for col in skill_lookup.columns:
        print(f"  - {col}")
    
    print("\n5 Sample Rows:")
    print(skill_lookup[['skill_label', 'group_label', 'pillar_label']].head().to_string())
    
    # Show distribution of skills across groups
    print("\n" + "=" * 80)
    print("GROUP SIZE DISTRIBUTION")
    print("=" * 80)
    
    group_sizes = skill_lookup.groupby('group_label').size()
    print(f"\nStatistics:")
    print(f"  Min skills in a group:  {group_sizes.min()}")
    print(f"  Max skills in a group:  {group_sizes.max()}")
    print(f"  Mean skills per group:  {group_sizes.mean():.1f}")
    print(f"  Median skills per group: {group_sizes.median():.0f}")
    
    print("\nTop 10 largest groups:")
    for i, (group, size) in enumerate(group_sizes.nlargest(10).items(), 1):
        print(f"  {i:2d}. {group:60s} ({size:3d} skills)")
    
    # Show how it's used in preprocessing
    print("\n" + "=" * 80)
    print("HOW THIS CONNECTS TO PREPROCESSING")
    print("=" * 80)
    print("""
    1. skill2group.json is used to map individual skills → groups
       - When a job requires "coordinate technical teams"
       - We look it up: skill2group["http://.../coordinate-teams"]
       - Returns: "http://.../supervising-people"
    
    2. group2label.json provides human-readable labels
       - Maps group URI → readable name
       - "http://.../supervising-people" → "supervising people"
    
    3. preprocessing.py uses these to create multi-label training data
       - Job titles → skills → groups → binary columns
       - Each column = one skill group (0 or 1)
       - Model learns: title/description → which groups are relevant
    """)
    
    # Show a concrete mapping example
    print("\n" + "=" * 80)
    print("CONCRETE EXAMPLE: skill2group.json Usage")
    print("=" * 80)
    
    # Show 5 skill→group mappings
    print("\nSample skill→group mappings:")
    for i, (skill_id, group_id) in enumerate(list(skill2group.items())[:5], 1):
        # Get labels
        skill_row = skill_lookup[skill_lookup['skill_id'] == skill_id].iloc[0]
        group_label = group2label.get(group_id, "Unknown")
        print(f"\n{i}. Skill: {skill_row['skill_label']}")
        print(f"   → Group: {group_label}")
    
    print("\n" + "=" * 80)
    print("✓ Exploration complete!")
    print("=" * 80)
    print(f"\nFiles used from build_hierarchy.py output:")
    print(f"  - {processed_dir}/skill_to_group_pillar.csv")
    print(f"  - {processed_dir}/skill2group.json")
    print(f"  - {processed_dir}/group2label.json")

if __name__ == "__main__":
    main()


