# Dataset Statistics Report

**Generated:** 2026-01-14 01:06:59

---

## 4.1.1 DECORTE Dataset

The DECORTE dataset (Decorte et al.) contains anonymized career histories with ESCO occupation mappings.

**Source:** `jensjorisdecorte/anonymous-working-histories` (HuggingFace)


### Basic Statistics

| Metric | Value |
|--------|-------|
| Total number of career histories (resumes) | 2,164 |
| Total number of job experiences | 9,919 |
| Train split size | 1,720 |
| Validation split size | 217 |
| Test split size | 227 |

### Career Length Statistics

| Metric | Value |
|--------|-------|
| Average career length (jobs per career) | 4.58 |
| Standard deviation | 1.92 |
| Minimum career length | 2 |
| Maximum career length | 17 |
| Median career length | 4.0 |

### Career Length Percentiles

| Percentile | Value |
|------------|-------|
| 25th percentile | 3.0 |
| 50th percentile (median) | 4.0 |
| 75th percentile | 5.0 |
| 90th percentile | 7.0 |

### Career Length Distribution

| Career Length | Count |
|---------------|-------|
| 2 jobs | 96 |
| 3 jobs | 579 |
| 4 jobs | 704 |
| 5 jobs | 263 |
| 6 jobs | 195 |
| 7 jobs | 145 |
| 8 jobs | 81 |
| 9 jobs | 47 |
| 10 jobs | 29 |
| 11 jobs | 9 |
| 12 jobs | 5 |
| 13 jobs | 5 |
| 14 jobs | 3 |
| 15 jobs | 2 |
| 17 jobs | 1 |

### ESCO Coverage

| Metric | Value |
|--------|-------|
| Number of unique ESCO occupations in dataset | 1,155 |
| ESCO version used | v1.2.0 (files dated 2024-04) |

---

## 4.1.3 ESCO Taxonomy

The European Skills, Competences, Qualifications and Occupations (ESCO) taxonomy.


### Core Statistics

| Metric | Value |
|--------|-------|
| Total number of occupations | 3,039 |
| Total number of skills | 13,939 |
| Total occupation-skill relations | 129,004 |

### Essential vs Optional Skill Relations

| Relation Type | Count | Percentage |
|---------------|-------|------------|
| Essential | 67,622 | 52.4% |
| Optional | 61,382 | 47.6% |

### Skills per Occupation

| Metric | Essential Only | Optional Only | Combined |
|--------|----------------|---------------|----------|
| Average | 22.25 | 21.48 | 42.45 |
| Std Dev | - | - | 25.76 |
| Min | - | - | 7 |
| Max | - | - | 345 |
| Median | - | - | 37.0 |

### Skill Hierarchy

| Level | Count |
|-------|-------|
| L1 skill categories (pillars) | 28 |
| L2 skill categories | 156 |

### ISCO Grouping Statistics

| ISCO Level | Number of Groups |
|------------|------------------|
| 1-digit (Major groups) | 13 |
| 2-digit (Sub-major groups) | 43 |
| 3-digit (Minor groups) | 130 |
| 4-digit (Unit groups) | 433 |

### DAG Structure

| Metric | Value |
|--------|-------|
| Skills with multiple parents (before DAG expansion) | 4,761 |
| Maximum parents per skill | 9 |

---

## 4.1.4 Exploratory Data Analysis


### Occupation Distribution Across ISCO Major Groups (ESCO Taxonomy)

| ISCO Code | Major Group Name | Count |
|-----------|------------------|-------|
| 1 | Managers | 362 |
| 2 | Professionals | 872 |
| 3 | Technicians and associate professionals | 651 |
| 4 | Clerical support workers | 89 |
| 5 | Service and sales workers | 204 |
| 6 | Skilled agricultural, forestry and fishery workers | 44 |
| 7 | Craft and related trades workers | 394 |
| 8 | Plant and machine operators, and assemblers | 347 |
| 9 | Elementary occupations | 76 |

### ISCO Distribution in DECORTE Dataset (Job Experiences)

| ISCO Code | Major Group Name | Count |
|-----------|------------------|-------|
| 1 | Managers | 2,431 (24.5%) |
| 2 | Professionals | 3,141 (31.7%) |
| 3 | Technicians and associate professionals | 2,050 (20.7%) |
| 4 | Clerical support workers | 859 (8.7%) |
| 5 | Service and sales workers | 938 (9.5%) |
| 6 | Skilled agricultural, forestry and fishery workers | 12 (0.1%) |
| 7 | Craft and related trades workers | 239 (2.4%) |
| 8 | Plant and machine operators, and assemblers | 68 (0.7%) |
| 9 | Elementary occupations | 179 (1.8%) |

### Skills per Occupation Distribution

| Metric | Value |
|--------|-------|
| Mean | 42.45 |
| Median | 37.0 |
| Std Dev | 25.76 |
| Min | 7 |
| Max | 345 |

### Career Length Distribution (Percentiles)

| Percentile | Career Length |
|------------|---------------|
| 25th | 3.0 jobs |
| 50th (median) | 4.0 jobs |
| 75th | 5.0 jobs |
| 90th | 7.0 jobs |

### Transition Patterns

| Metric | Value |
|--------|-------|
| Total job transitions analyzed | 7,753 |
| Transitions within same ISCO major group | 3,592 (46.3%) |
| Transitions across ISCO major groups | 4,161 (53.7%) |

---

## Summary


This report provides comprehensive statistics for the datasets used in the thesis:

- **DECORTE**: 2,164 career histories with 9,919 job experiences
- **ESCO Taxonomy**: 3,039 occupations and 13,939 skills
- **Skill Relations**: 129,004 occupation-skill mappings
- **Career Transitions**: 46.3% within same ISCO group, 53.7% across groups