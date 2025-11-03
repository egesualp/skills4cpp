# ESCO Dataset Map - Complete Reference

This document provides a comprehensive map of all ESCO datasets, showing table structures, relationships, and keys for joining tables.

## 📁 Dataset Locations

### Raw ESCO Data
**Location:** `data/esco_datasets/`

### Processed Data
**Location:** `data/processed/`

### Master Datasets
**Location:** `data/processed/master_datasets/`

---

## 🗃️ Raw ESCO Tables

### Core Entity Tables

#### 1. skills_en.csv (13,939 skills)
**Primary Key:** `conceptUri`

| Column | Type | Description |
|--------|------|-------------|
| conceptType | String | Always "KnowledgeSkillCompetence" |
| conceptUri | **URI (PK)** | Unique skill identifier |
| skillType | String | "knowledge" or "skill/competence" |
| reuseLevel | String | "transversal", "cross-sector", "sector-specific", "occupation-specific" |
| preferredLabel | String | Skill name |
| altLabels | String | Alternative labels (newline-separated) |
| hiddenLabels | String | Hidden search terms |
| status | String | "released", "draft" |
| modifiedDate | Date | Last modification |
| scopeNote | Text | Scope notes |
| definition | Text | Formal definition |
| inScheme | URI | Concept scheme URIs |
| description | Text | Detailed description |

**Relationships:**
- → `occupationSkillRelations_en.csv` (skillUri)
- → `skillSkillRelations_en.csv` (originalSkillUri, relatedSkillUri)
- → `broaderRelationsSkillPillar_en.csv` (conceptUri)

---

#### 2. occupations_en.csv (3,039 occupations)
**Primary Key:** `conceptUri`  
**Foreign Key:** `iscoGroup` → `ISCOGroups_en.csv.code`

| Column | Type | Description |
|--------|------|-------------|
| conceptType | String | Always "Occupation" |
| conceptUri | **URI (PK)** | Unique occupation identifier |
| iscoGroup | **String (FK)** | ISCO classification code |
| preferredLabel | String | Occupation name |
| altLabels | String | Alternative labels |
| hiddenLabels | String | Hidden search terms |
| status | String | Status |
| modifiedDate | Date | Last modification |
| regulatedProfessionNote | Text | Regulation notes |
| scopeNote | Text | Scope notes |
| definition | Text | Occupation definition |
| inScheme | URI | Concept scheme URIs |
| description | Text | Detailed description |
| code | String | ESCO occupation code |

**Relationships:**
- → `occupationSkillRelations_en.csv` (occupationUri)
- → `broaderRelationsOccPillar_en.csv` (conceptUri)
- ← `ISCOGroups_en.csv` (code)

---

#### 3. skillGroups_en.csv (640 groups + 4 pillars)
**Primary Key:** `conceptUri`

| Column | Type | Description |
|--------|------|-------------|
| conceptType | String | Always "SkillGroup" |
| conceptUri | **URI (PK)** | Unique group identifier |
| preferredLabel | String | Group name |
| altLabels | String | Alternative labels |
| hiddenLabels | String | Hidden search terms |
| status | String | Status |
| modifiedDate | Date | Last modification |
| scopeNote | Text | Scope notes |
| inScheme | URI | Concept scheme URIs |
| description | Text | Detailed description |
| code | String | ISCED-F code |

**Relationships:**
- → `broaderRelationsSkillPillar_en.csv` (broaderUri)

---

#### 4. ISCOGroups_en.csv (436 ISCO groups)
**Primary Key:** `conceptUri`  
**Alternate Key:** `code`

| Column | Type | Description |
|--------|------|-------------|
| conceptType | String | Always "ISCOGroup" |
| conceptUri | **URI (PK)** | Unique ISCO group identifier |
| code | **String (AK)** | ISCO code (e.g., "0", "2654") |
| preferredLabel | String | ISCO group name |
| status | String | Status |
| altLabels | String | Alternative labels |
| inScheme | URI | Concept scheme URIs |
| description | Text | Detailed description |

**Relationships:**
- → `occupations_en.csv` (iscoGroup)
- → `broaderRelationsOccPillar_en.csv` (conceptUri, broaderUri)

---

### Relationship Tables

#### 5. occupationSkillRelations_en.csv (129,004 relationships)
**Purpose:** Links occupations to required skills

**Composite Key:** (`occupationUri`, `skillUri`)

| Column | Type | Description |
|--------|------|-------------|
| occupationUri | **URI (FK)** | → occupations_en.csv.conceptUri |
| relationType | String | "essential" or "optional" |
| skillType | String | "knowledge" or "skill/competence" |
| skillUri | **URI (FK)** | → skills_en.csv.conceptUri |

**Join Pattern:**
```sql
SELECT o.preferredLabel, s.preferredLabel, osr.relationType
FROM occupations_en o
JOIN occupationSkillRelations_en osr ON o.conceptUri = osr.occupationUri
JOIN skills_en s ON osr.skillUri = s.conceptUri
WHERE osr.relationType = 'essential'
```

---

#### 6. skillSkillRelations_en.csv (5,818 relationships)
**Purpose:** Links related skills

**Composite Key:** (`originalSkillUri`, `relatedSkillUri`)

| Column | Type | Description |
|--------|------|-------------|
| originalSkillUri | **URI (FK)** | → skills_en.csv.conceptUri |
| originalSkillType | String | Type of original skill |
| relationType | String | "essential" or "optional" |
| relatedSkillType | String | Type of related skill |
| relatedSkillUri | **URI (FK)** | → skills_en.csv.conceptUri |

**Join Pattern:**
```sql
SELECT s1.preferredLabel AS skill, s2.preferredLabel AS related_skill
FROM skillSkillRelations_en ssr
JOIN skills_en s1 ON ssr.originalSkillUri = s1.conceptUri
JOIN skills_en s2 ON ssr.relatedSkillUri = s2.conceptUri
```

---

#### 7. broaderRelationsSkillPillar_en.csv (20,824 relationships)
**Purpose:** Defines skill/group hierarchy (many-to-many)

**Composite Key:** (`conceptUri`, `broaderUri`)

| Column | Type | Description |
|--------|------|-------------|
| conceptType | String | "KnowledgeSkillCompetence" or "SkillGroup" |
| conceptUri | **URI (FK)** | → skills_en.csv.conceptUri OR skillGroups_en.csv.conceptUri |
| broaderType | String | "SkillGroup" |
| broaderUri | **URI (FK)** | → skillGroups_en.csv.conceptUri |

**Note:** This creates a **directed acyclic graph (DAG)** where:
- Skills can have multiple parent groups
- Groups can have multiple parent groups
- Top-level groups are the 4 pillars

**Join Pattern:**
```sql
-- Get skill hierarchy (1 level)
SELECT s.preferredLabel AS skill, sg.preferredLabel AS group
FROM skills_en s
JOIN broaderRelationsSkillPillar_en br ON s.conceptUri = br.conceptUri
JOIN skillGroups_en sg ON br.broaderUri = sg.conceptUri
WHERE br.conceptType = 'KnowledgeSkillCompetence'
```

---

#### 8. broaderRelationsOccPillar_en.csv (3,654 relationships)
**Purpose:** Defines occupation/ISCO hierarchy

**Composite Key:** (`conceptUri`, `broaderUri`)

| Column | Type | Description |
|--------|------|-------------|
| conceptType | String | "Occupation" or "ISCOGroup" |
| conceptUri | **URI (FK)** | → occupations_en.csv.conceptUri OR ISCOGroups_en.csv.conceptUri |
| broaderType | String | "ISCOGroup" |
| broaderUri | **URI (FK)** | → ISCOGroups_en.csv.conceptUri |

---

### Pre-Flattened Views

#### 9. skillsHierarchy_en.csv (3,062 rows)
**Purpose:** Denormalized view of skill GROUP hierarchy (up to 4 levels)

**Note:** Does NOT include individual skills from `skills_en.csv`, only the group structure.

| Column | Type | Description |
|--------|------|-------------|
| Level 0 URI | URI | Pillar URI |
| Level 0 preferred term | String | Pillar name |
| Level 1 URI | URI | Top group URI |
| Level 1 preferred term | String | Top group name |
| Level 2 URI | URI | Mid-level group URI |
| Level 2 preferred term | String | Mid-level group name |
| Level 3 URI | URI | Lower-level group URI |
| Level 3 preferred term | String | Lower-level group name |
| Description | Text | Description |
| Scope note | Text | Scope note |
| Level 0-3 code | String | Codes |

**Relationship:** This is derived from `broaderRelationsSkillPillar_en.csv` and `skillGroups_en.csv`.

---

### Collection Tables (Curated Subsets)

#### 10. languageSkillsCollection_en.csv (361 skills)
#### 11. digitalSkillsCollection_en.csv (1,295 skills)
#### 12. digCompSkillsCollection_en.csv (26 skills)
#### 13. greenSkillsCollection_en.csv (605 skills)
#### 14. transversalSkillsCollection_en.csv (99 skills)
#### 15. researchSkillsCollection_en.csv (51 skills)
#### 16. researchOccupationsCollection_en.csv (130 occupations)

**Note:** These are subsets of `skills_en.csv` and `occupations_en.csv` with additional metadata.

**Additional Columns:**
- `broaderConceptUri` - Parent concept URI(s)
- `broaderConceptPT` - Parent concept preferred term(s)

---

#### 17. conceptSchemes_en.csv (Metadata table)

| Column | Type | Description |
|--------|------|-------------|
| conceptType | String | Always "ConceptScheme" |
| conceptSchemeUri | URI | URI of the concept scheme |
| preferredLabel | String | Scheme name |
| title | String | Title |
| status | String | Status |
| description | Text | Description |
| hasTopConcept | URI | List of top-level concepts |

---

## 🎯 Master Datasets (Pre-Joined)

### 18. master_skills.csv (20,186 rows × 13 cols)
**Purpose:** Skill-centric dataset with groups and occupation statistics

**Built from:** `skills_en` ⋈ `broaderRelationsSkillPillar_en` ⋈ `skillGroups_en` + occupation aggregates

**Key Columns:**
- All from `skills_en` (skillUri, skillLabel, skillType, reuseLevel, etc.)
- `groupUri`, `groupLabel` - Parent skill group
- `total_occupations`, `essential_occupations`, `optional_occupations` - Demand metrics

**Use:** Skill analysis, classification, demand analysis

---

### 19. master_occupation_skills.csv (129,004 rows × 13 cols)
**Purpose:** Complete occupation-skill relationships with context

**Built from:** `occupationSkillRelations_en` ⋈ `occupations_en` ⋈ `skills_en` + skill groups

**Key Columns:**
- All occupation details (occupationUri, occupationLabel, iscoGroup, etc.)
- All skill details (skillUri, skillLabel, skillType, reuseLevel, etc.)
- `relationType` - "essential" or "optional"
- `skillGroups` - Aggregated skill group labels

**Use:** Occupation-skill matching, job profiling, recommendations

---

### 20. master_complete_skills.csv (13,939 rows × 15 cols)
**Purpose:** Complete skill dataset with full hierarchy

**Built from:** `skills_en` + hierarchy JSON files + occupation aggregates

**Key Columns:**
- All from `skills_en`
- `groups`, `groupUris` - All parent groups (pipe-separated)
- `pillars`, `pillarUris` - All top-level pillars (pipe-separated)
- Occupation counts

**Use:** Full hierarchy analysis, multi-parent relationships, pillar distributions

---

### 21. master_graph_nodes.csv (16,978 rows × 6 cols)
**Purpose:** All nodes for network analysis

**Built from:** `skills_en` + `occupations_en`

**Node Types:**
- 13,939 skill nodes
- 3,039 occupation nodes

**Key Columns:**
- `nodeId` - URI
- `label` - Name
- `nodeType` - "skill" or "occupation"
- Relevant metadata

**Use:** Graph/network analysis, visualization

---

### 22. master_graph_edges.csv (134,822 rows × 5 cols)
**Purpose:** All relationships for network analysis

**Built from:** `occupationSkillRelations_en` + `skillSkillRelations_en`

**Edge Types:**
- 129,004 occupation→skill edges
- 5,818 skill→skill edges

**Key Columns:**
- `source`, `target` - Node URIs
- `edgeType` - "requires" or "relates_to"
- `relationship` - "essential" or "optional"
- `weight` - Numeric weight (1.0, 0.5, 0.8, 0.3)

**Use:** Graph algorithms, centrality analysis, pathfinding

---

## 🔗 Complete Relationship Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    ESCO Data Relationships                   │
└─────────────────────────────────────────────────────────────┘

OCCUPATIONS                      SKILLS
    │                              │
    │ iscoGroup                    │ conceptUri
    ▼                              ▼
ISCOGroups_en              broaderRelationsSkillPillar_en
    │                              │
    │                              │ broaderUri
    │                              ▼
    │                        skillGroups_en
    │                        (includes 4 Pillars)
    │                              │
    │                              │
    │                              ▼
    │                        skillsHierarchy_en
    │                        (flattened groups only)
    │
    ▼
broaderRelationsOccPillar_en


RELATIONSHIPS (Many-to-Many):

occupations_en ─────────────┐
                            │
                            ▼
              occupationSkillRelations_en
                            │
                            ▼
skills_en ──────────────────┘


skills_en ──────────────────┐
                            │
                            ▼
              skillSkillRelations_en
                            │
                            ▼
skills_en ──────────────────┘


MASTER DATASETS (Pre-Joined):

master_skills.csv
  = skills_en ⋈ broaderRelationsSkillPillar_en ⋈ skillGroups_en + stats

master_occupation_skills.csv
  = occupationSkillRelations_en ⋈ occupations_en ⋈ skills_en + groups

master_complete_skills.csv
  = skills_en + full hierarchy (from build_hierarchy.py) + stats

master_graph_nodes.csv
  = skills_en ∪ occupations_en

master_graph_edges.csv
  = occupationSkillRelations_en ∪ skillSkillRelations_en
```

---

## 🔑 Primary Keys & Foreign Keys Summary

| Table | Primary Key | Foreign Keys | References |
|-------|-------------|--------------|------------|
| skills_en | conceptUri | - | - |
| occupations_en | conceptUri | iscoGroup | ISCOGroups_en.code |
| skillGroups_en | conceptUri | - | - |
| ISCOGroups_en | conceptUri (code) | - | - |
| occupationSkillRelations_en | (occupationUri, skillUri) | occupationUri, skillUri | occupations_en, skills_en |
| skillSkillRelations_en | (originalSkillUri, relatedSkillUri) | both | skills_en |
| broaderRelationsSkillPillar_en | (conceptUri, broaderUri) | both | skills_en OR skillGroups_en |
| broaderRelationsOccPillar_en | (conceptUri, broaderUri) | both | occupations_en OR ISCOGroups_en |

---

## 📊 Data Statistics

### Entity Counts
- **Skills:** 13,939
- **Occupations:** 3,039
- **Skill Groups:** 640 (including 4 pillars)
- **ISCO Groups:** 436
- **Occupation-Skill Relations:** 129,004
- **Skill-Skill Relations:** 5,818
- **Skill-Group Relations:** 20,822

### Hierarchy Stats
- **Pillars:** 4 (skills, knowledge, transversal, language)
- **Skills with pillar:** 13,653 (97.9%)
- **Orphan skills:** 286 (2.1%)
- **Max group depth:** 3 levels

### Coverage
- **Skills used in occupations:** 13,492 (96.8%)
- **Essential skill relationships:** 67,622
- **Optional skill relationships:** 61,382

---

## 🛠️ Building the Datasets

### Generate Master Datasets

```bash
# Step 1: Build hierarchy (creates JSON files)
python src/skill_mapping/build_hierarchy.py

# Step 2: Build master datasets (creates CSVs)
python src/skill_mapping/build_master_datasets.py
```

**Output:**
- `data/processed/` - Hierarchy JSON files
- `data/processed/master_datasets/` - Master CSV files

---

## 📚 Related Documentation

- **Master Datasets Guide:** `docs/master_datasets_guide.md`
- **Hierarchy Building Guide:** `docs/build_hierarchy_guide.md`
- **Raw ESCO Documentation:** https://esco.ec.europa.eu/

---

## ✅ Key Takeaways

1. **skills_en.csv** and **occupations_en.csv** are the core entity tables
2. **occupationSkillRelations_en.csv** is the main relationship table (129K rows)
3. **broaderRelationsSkillPillar_en.csv** defines the skill hierarchy (DAG, not tree)
4. **skillsHierarchy_en.csv** contains groups only, NOT individual skills
5. **Master datasets** provide pre-joined, analysis-ready views
6. All relationships are URI-based (use conceptUri for joins)
7. Skills can have multiple parent groups (many-to-many)
8. Use master datasets for most analysis tasks to avoid complex joins

---

*Last Updated: October 27, 2025*


