# ESCO Master Datasets Guide

This guide documents the master datasets created from ESCO data, providing comprehensive views for different analysis purposes.

## Overview

Six master datasets have been created by joining and enriching the raw ESCO CSV files:

1. **master_skills.csv** - Skill-centric with groups and occupation counts
2. **master_occupation_skills.csv** - Occupation-skill relationships with full context
3. **master_complete_skills.csv** - Complete skill dataset with full hierarchy
4. **master_skill_complete_hierarchy.csv** - Complete hierarchy with all levels (0-3) + skills in one table
5. **master_graph_nodes.csv** - All nodes for network analysis
6. **master_graph_edges.csv** - All edges for network analysis

**Location:** `data/processed/master_datasets/`

## Dataset Details

### 1. master_skills.csv

**Purpose:** Skill-centric analysis with group mappings and occupation statistics.

**Size:** 20,186 rows × 13 columns

**Note:** Skills with multiple parent groups appear in multiple rows (exploded format).

**Columns:**
- `skillUri` - Unique skill identifier (URI)
- `skillLabel` - Skill name (human-readable)
- `altLabels` - Alternative labels/names (newline-separated)
- `skillType` - Type: "knowledge" or "skill/competence"
- `reuseLevel` - Reuse level: "transversal", "cross-sector", "sector-specific", "occupation-specific"
- `status` - Status: "released", "draft", etc.
- `description` - Detailed skill description
- `definition` - Formal definition
- `groupUri` - Parent skill group URI
- `groupLabel` - Parent skill group name
- `total_occupations` - Total number of occupations requiring this skill
- `essential_occupations` - Number of occupations where skill is essential
- `optional_occupations` - Number of occupations where skill is optional

**Use Cases:**
- Analyzing skills by group
- Finding most in-demand skills (by occupation count)
- Skill classification and categorization
- Understanding skill popularity

**Example Query:**
```python
import pandas as pd

df = pd.read_csv('data/processed/master_datasets/master_skills.csv')

# Top 10 most in-demand skills
top_skills = df.nlargest(10, 'total_occupations')[['skillLabel', 'total_occupations', 'groupLabel']]

# Skills by reuse level
skills_by_level = df.groupby('reuseLevel')['skillUri'].nunique()

# Essential skills only
essential_skills = df[df['essential_occupations'] > 0]
```

---

### 2. master_occupation_skills.csv

**Purpose:** Occupation-skill relationship analysis with full context.

**Size:** 129,004 rows × 13 columns

**Note:** Each row represents one occupation-skill relationship.

**Columns:**
- `occupationUri` - Unique occupation identifier
- `occupationLabel` - Occupation name
- `iscoGroup` - ISCO classification code
- `occupationDescription` - Detailed occupation description
- `occupationCode` - ESCO occupation code
- `skillUri` - Unique skill identifier
- `skillLabel` - Skill name
- `skillType_x` - Skill type from relationship table
- `skillType_y` - Skill type from skills table
- `reuseLevel` - Skill reuse level
- `skillDescription` - Detailed skill description
- `skillGroups` - Skill group labels (pipe-separated if multiple)
- `relationType` - Relationship type: "essential" or "optional"

**Use Cases:**
- Occupation-skill matching
- Building recommendation systems
- Job profile analysis
- Skills gap analysis
- Career path mapping

**Example Query:**
```python
import pandas as pd

df = pd.read_csv('data/processed/master_datasets/master_occupation_skills.csv')

# Get all skills for a specific occupation
occupation = "data scientist"
occ_skills = df[df['occupationLabel'].str.contains(occupation, case=False, na=False)]

# Essential skills across all occupations
essential = df[df['relationType'] == 'essential'].groupby('skillLabel').size().sort_values(ascending=False).head(20)

# Occupations requiring a specific skill
skill_name = "Python"
occupations = df[df['skillLabel'].str.contains(skill_name, case=False, na=False)][['occupationLabel', 'relationType']]

# Skills by ISCO group
isco_skills = df.groupby('iscoGroup')['skillLabel'].apply(list)
```

---

### 3. master_complete_skills.csv

**Purpose:** Complete skill dataset with full multi-parent hierarchy information.

**Size:** 13,939 rows × 15 columns

**Note:** Each skill appears once with all hierarchy info concatenated.

**Columns:**
- `skillUri` - Unique skill identifier
- `skillLabel` - Skill name
- `skillType` - Type: "knowledge" or "skill/competence"
- `reuseLevel` - Reuse level
- `status` - Status
- `description` - Detailed description
- `definition` - Formal definition
- `altLabels` - Alternative labels
- `groups` - Parent group labels (pipe-separated)
- `groupUris` - Parent group URIs (pipe-separated)
- `pillars` - Top-level pillar labels (pipe-separated)
- `pillarUris` - Top-level pillar URIs (pipe-separated)
- `total_occupations` - Total occupations count
- `essential_occupations` - Essential occupations count
- `optional_occupations` - Optional occupations count

**Use Cases:**
- Full hierarchy analysis
- Pillar-based skill distribution
- Multi-parent relationship analysis
- Complete skill profiling

**Example Query:**
```python
import pandas as pd

df = pd.read_csv('data/processed/master_datasets/master_complete_skills.csv')

# Skills by pillar
knowledge_skills = df[df['pillars'].str.contains('knowledge', na=False)]
transversal_skills = df[df['pillars'].str.contains('transversal', na=False)]

# Skills with multiple groups
multi_group = df[df['groups'].str.contains('\\|', na=False)]

# Most demanded skills by pillar
for pillar in ['skills', 'knowledge', 'transversal', 'language']:
    pillar_skills = df[df['pillars'].str.contains(pillar, na=False)]
    top_10 = pillar_skills.nlargest(10, 'total_occupations')[['skillLabel', 'total_occupations']]
    print(f"\nTop 10 {pillar} skills:")
    print(top_10)
```

---

### 4. master_skill_complete_hierarchy.csv

**Purpose:** Complete denormalized hierarchy table showing Level 0 (Pillar) through Level 3 (Groups) and individual skills all in one row.

**Size:** 20,186 rows × 15 columns

**Note:** This is the ONLY table where you can see the complete hierarchy path (Pillar → Group1 → Group2 → Group3 → Skill) in a single row. Skills with multiple paths appear multiple times.

**Columns:**
- `skillUri` - Unique skill identifier
- `skillLabel` - Skill name
- `skillType` - Type: "knowledge" or "skill/competence"
- `reuseLevel` - Reuse level
- `status` - Status
- `description` - Detailed description
- `definition` - Formal definition
- `level0_uri`, `level0_label` - **Pillar** (top level)
- `level1_uri`, `level1_label` - First group level
- `level2_uri`, `level2_label` - Second group level
- `level3_uri`, `level3_label` - Third group level (closest to skill)

**Example Row:**
```
Level 0 (Pillar):  skills
Level 1 (Group):   management skills
Level 2 (Group):   supervising people
Level 3 (Group):   supervising a team or group
Skill:             manage musical staff (skill/competence, sector-specific)
```

**Use Cases:**
- Viewing complete hierarchy in a spreadsheet
- Filtering skills by any level
- Understanding skill categorization
- Building hierarchical visualizations (treemaps, sunburst charts)
- Pivot table analysis by level

**Example Query:**
```python
import pandas as pd

df = pd.read_csv('data/processed/master_datasets/master_skill_complete_hierarchy.csv')

# Get all skills under a specific pillar
knowledge_skills = df[df['level0_label'] == 'knowledge']

# Get all skills in a specific level 1 group
management_skills = df[df['level1_label'] == 'management skills']

# Count skills by pillar
pillar_counts = df.groupby('level0_label')['skillUri'].nunique()

# Find skills at different depth levels
level3_skills = df[df['level3_label'].notna()]  # Skills with 4 levels
level2_only = df[df['level2_label'].notna() & df['level3_label'].isna()]  # Skills at level 2

# Build a hierarchical path column
df['hierarchy_path'] = (
    df['level0_label'].fillna('') + ' > ' + 
    df['level1_label'].fillna('') + ' > ' + 
    df['level2_label'].fillna('') + ' > ' + 
    df['level3_label'].fillna('') + ' > ' + 
    df['skillLabel']
)
```

**Statistics:**
- Total rows: 20,186
- Unique skills: 13,939
- Skills with full hierarchy (level 0-3): 13,483 (66.8%)
- Skills with missing hierarchy: 6,463 (32.0%)

**Pillar Distribution:**
- skills: 10,340 rows
- knowledge: 3,143 rows
- transversal skills and competences: 167 rows
- language skills and knowledge: 73 rows

---

### 5. master_graph_nodes.csv

**Purpose:** All nodes (skills + occupations) for network/graph analysis.

**Size:** 16,978 rows × 6 columns

**Node Types:**
- Skills: 13,939 nodes
- Occupations: 3,039 nodes

**Columns:**
- `nodeId` - Unique node identifier (URI)
- `label` - Node name (human-readable)
- `skillType` - Skill type (null for occupation nodes)
- `reuseLevel` - Skill reuse level (null for occupation nodes)
- `nodeType` - Type: "skill" or "occupation"
- `iscoGroup` - ISCO group (null for skill nodes)

**Use Cases:**
- Building knowledge graphs
- Network analysis with NetworkX, igraph, or Neo4j
- Centrality analysis
- Community detection
- Pathfinding between occupations and skills

---

### 6. master_graph_edges.csv

**Purpose:** All relationships (edges) for network/graph analysis.

**Size:** 134,822 rows × 5 columns

**Edge Types:**
- Occupation→Skill (requires): 129,004 edges
- Skill→Skill (relates_to): 5,818 edges

**Columns:**
- `source` - Source node URI
- `target` - Target node URI
- `edgeType` - Type: "requires" or "relates_to"
- `relationship` - Relationship qualifier: "essential", "optional"
- `weight` - Edge weight (1.0=essential, 0.5=optional, 0.8/0.3 for skill relations)

**Use Cases:**
- Graph-based skill recommendations
- Occupation similarity analysis
- Skill clustering
- Career path analysis
- Visualization with Gephi, Cytoscape, or D3.js

**Example Usage:**
```python
import pandas as pd
import networkx as nx

# Load graph data
nodes = pd.read_csv('data/processed/master_datasets/master_graph_nodes.csv')
edges = pd.read_csv('data/processed/master_datasets/master_graph_edges.csv')

# Build NetworkX graph
G = nx.DiGraph()

# Add nodes
for _, row in nodes.iterrows():
    G.add_node(row['nodeId'], 
               label=row['label'],
               node_type=row['nodeType'],
               skill_type=row['skillType'],
               reuse_level=row['reuseLevel'])

# Add edges
for _, row in edges.iterrows():
    G.add_edge(row['source'], row['target'],
               edge_type=row['edgeType'],
               relationship=row['relationship'],
               weight=row['weight'])

print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# Find most central skills
skill_nodes = [n for n, d in G.nodes(data=True) if d['node_type'] == 'skill']
skill_subgraph = G.subgraph(skill_nodes)
centrality = nx.degree_centrality(skill_subgraph)
top_skills = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
```

---

## Building the Datasets

### Prerequisites

1. Raw ESCO datasets in `data/esco_datasets/`
2. Hierarchy files (generated by `build_hierarchy.py`)

### Generate All Datasets

```bash
# Step 1: Build hierarchy files (if not already done)
python src/skill_mapping/build_hierarchy.py

# Step 2: Build master datasets
python src/skill_mapping/build_master_datasets.py
```

This will create all 5 master datasets in `data/processed/master_datasets/`.

---

## Data Quality Notes

### Multiple Parents

Many skills have multiple parent groups. In `master_skills.csv`, these create duplicate rows (one per group). In `master_complete_skills.csv`, groups are concatenated with pipe separators.

### Missing Data

- **286 orphan skills** have no path to a pillar (0 parents in hierarchy)
- Some skills may have null descriptions or definitions
- Not all skills are used in occupations (some have 0 occupation counts)

### Skill Coverage

- **Skills mapped to occupations:** 13,492 out of 13,939 (96.8%)
- **Skills with essential requirement:** Most in-demand skills
- **Skills with optional requirement:** Complementary skills

---

## Common Analysis Patterns

### 1. Finding Similar Occupations

```python
# Occupations with similar skill profiles
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv('data/processed/master_datasets/master_occupation_skills.csv')

# Create occupation-skill matrix
pivot = df.pivot_table(
    index='occupationLabel',
    columns='skillLabel',
    values='relationType',
    aggfunc='first'
).fillna(0)
pivot = (pivot == 'essential').astype(int)

# Compute similarity
similarity = cosine_similarity(pivot)
```

### 2. Skill Recommendations

```python
# Given an occupation, recommend related skills
target_occ = "software developer"

occ_skills = df[df['occupationLabel'].str.contains(target_occ, case=False, na=False)]
essential_skills = occ_skills[occ_skills['relationType'] == 'essential']['skillLabel'].tolist()

# Find skills that co-occur with these
related = df[df['skillLabel'].isin(essential_skills)].groupby('occupationLabel').size()
```

### 3. Career Pathfinding

```python
# Find career paths based on skill overlap
from itertools import combinations

# Get skills for each occupation
occ_skills_dict = df.groupby('occupationLabel')['skillLabel'].apply(set).to_dict()

# Compute skill overlap between all occupation pairs
for occ1, occ2 in combinations(occ_skills_dict.keys(), 2):
    overlap = occ_skills_dict[occ1] & occ_skills_dict[occ2]
    if len(overlap) > 10:  # Significant overlap
        print(f"{occ1} → {occ2}: {len(overlap)} shared skills")
```

---

## Maintenance

### Updating Datasets

When ESCO releases new data:

1. Replace CSVs in `data/esco_datasets/`
2. Re-run `build_hierarchy.py`
3. Re-run `build_master_datasets.py`

### Version Control

- Master datasets are generated artifacts (can be in `.gitignore`)
- Keep the generation scripts under version control
- Document ESCO version used in your analysis

---

## Related Files

- `src/skill_mapping/build_hierarchy.py` - Builds hierarchy mappings
- `src/skill_mapping/build_master_datasets.py` - Builds master datasets
- `docs/build_hierarchy_guide.md` - Hierarchy building documentation
- `data/processed/` - Hierarchy JSON files
- `data/esco_datasets/` - Raw ESCO CSV files

---

## Questions?

For issues or questions about the master datasets, refer to:
- ESCO documentation: https://esco.ec.europa.eu/
- Project documentation: `docs/`
- Test files: `tests/test_build_hierarchy.py`

