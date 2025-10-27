# Build Hierarchy Guide

This guide documents the `build_hierarchy.py` script, which processes ESCO (European Skills, Competences, Qualifications and Occupations) datasets to build a hierarchical mapping of skills to groups and, ultimately, to top-level pillars.

The script is designed to handle the graph-based nature of ESCO data, where skills and groups can have **multiple parents**.

## Script Location

```
src/skill_mapping/build_hierarchy.py
```

## Input Data

The script reads the following ESCO CSV files from `data/esco_datasets/` (language can be configured via the `LANG` variable):

- **skills_{LANG}.csv**: Complete list of ESCO skills with URIs and labels.
- **skillGroups_{LANG}.csv**: Skill groups, pillars, and their metadata.
- **broaderRelationsSkillPillar_{LANG}.csv**: The core file defining the relationships (the graph) between skills, groups, and pillars.

## Output Artifacts

All outputs are saved to `data/processed/`:

### 1. skill_to_group_pillar.csv

A comprehensive lookup table where each row represents a single skill-to-group relationship. Skills with multiple parents will have multiple rows.

| Column | Description | Example |
|---|---|---|
| skill_id | URI of the skill | `http://data.europa.eu/esco/skill/...` |
| group_id | URI of the parent group | `http://data.europa.eu/esco/isced-f/...` |
| skill_label | Human-readable skill name | `"Haskell"` |
| group_label | Human-readable group name | `"software and applications development..."` |
| pillar_id | URI of the top-level pillar | `http://data.europa.eu/esco/skill/...` |
| pillar_label | Human-readable pillar name | `"knowledge"` |

**Size**: ~20,186 rows

### 2. JSON Mappings (Multi-Parent Preserved)

Several JSON files are generated to preserve the one-to-many relationships for programmatic use:

- **skill2group.json**: Maps each skill URI to a *list* of its parent group URIs.
  ```json
  {
    "http://.../[skill-uri]": [
      "http://.../[group-uri-1]",
      "http://.../[group-uri-2]"
    ]
  }
  ```
- **group2parent.json**: Maps each group URI to a *list* of its parent group/pillar URIs.
- **group2pillar.json**: Maps each group URI to a *list* of its final, top-level pillar URIs after traversing the hierarchy.
- **skill2pillar.json**: Maps each skill URI to a *list* of its final, top-level pillar URIs.
- **group2label.json**: A simple dictionary mapping group/pillar URIs to their human-readable labels.

## Data Statistics

Based on the current ESCO dataset:

- **Total skills in ESCO file**: 13,939
- **Skills mapped to ≥1 group**: 13,939
- **Skills with ≥1 pillar**: 13,653 (97.9%)
- **Orphan skills (no path to pillar)**: 286
- **Unique groups**: 2,695
- **Unique pillars**: 4

### Pillar Distribution

1.  **skills**: 10,340
2.  **knowledge**: 3,143
3.  **transversal skills and competences**: 167
4.  **language skills and knowledge**: 73

## How It Works

The script implements a graph traversal approach to handle the complex, multi-parent ESCO hierarchy:

1.  **Skill-to-Group Mapping**: Reads `broaderRelationsSkillPillar_en.csv` to build a `skill -> [groups]` mapping, preserving all parent relationships.
2.  **Group Adjacency List**: Builds a `group -> [parents]` adjacency list from the same file, which defines the main graph structure.
3.  **Pillar Identification**: Identifies the four top-level pillar nodes by matching their labels in `skillGroups_en.csv`.
4.  **Graph Traversal**: Uses a breadth-first search (`climb_to_pillars` function) to traverse upwards from each group to find all reachable pillar(s).
5.  **Skill-to-Pillar Linking**: Combines these mappings to link each skill to its ultimate pillar(s).
6.  **Artifact Generation**: Produces the flat CSV for analysis and the JSON files that preserve the one-to-many relationships.

## Usage

### Running the Script

```bash
python src/skill_mapping/build_hierarchy.py
```

### Expected Output

```
Loading ESCO data...
  ✓ Loaded 13,939 skills
  ✓ Loaded 640 skill groups (and pillar nodes)
  ✓ Loaded 20,822 broader relations

Extracting skill-to-group mappings (multi-parent)...
  ✓ Skills with at least one parent group: 13,939
Extracting group-to-parent mappings (multi-parent)...
  ✓ Groups with at least one parent: 636
  ✓ Detected 4 pillar nodes

Climbing from groups to pillars (multi-hop, multi-parent)...
Linking skills to pillars via their groups...

Building complete multi-parent skill lookup table...
  ✓ Saved skill_to_group_pillar.csv (20,186 rows)
  ...

[✓] Build complete!
```

## Testing

Run the comprehensive test suite:

```bash
python -m pytest tests/test_build_hierarchy.py -v
```

The suite includes integration and data quality tests. All **12 tests** should pass.

## Configuration

Edit the path and language constants at the top of `build_hierarchy.py`:

```python
# ==== CONFIG (edit these) ====
PATH_ESCO = Path("data/esco_datasets")   # folder with ESCO CSVs
PATH_OUT  = Path("data/processed")
LANG = "en"  # use your ESCO language code suffix (e.g., 'en', 'de', ...)
# =============================
```

## Example Usage in Code

```python
import pandas as pd
import json

# Load the complete lookup table (exploded)
df = pd.read_csv("data/processed/skill_to_group_pillar.csv")

# Find skills in the 'knowledge' pillar
knowledge_skills = df[df['pillar_label'] == 'knowledge']
print(f"Found {len(knowledge_skills)} knowledge-related skill mappings")

# Load multi-parent lookup
with open("data/processed/skill2group.json") as f:
    skill2group = json.load(f)

skill_uri = "http://data.europa.eu/esco/skill/000f1d3d-220f-4789-9c0a-cc742521fb02" # Haskell
parent_groups = skill2group.get(skill_uri)
print(f"Haskell has {len(parent_groups)} parents.")
```


