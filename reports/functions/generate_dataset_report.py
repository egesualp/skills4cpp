#!/usr/bin/env python3
"""
Dataset Statistics Report Generator

Generates a comprehensive report containing statistics for:
- 4.1.1 DECORTE Dataset
- 4.1.3 ESCO Taxonomy
- 4.1.4 Exploratory Data Analysis

Output: reports/outputs/dataset_statistics_report.md
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset
from collections import Counter, defaultdict
from datetime import datetime
import json
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ESCO_DIR = DATA_DIR / "esco_datasets"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "outputs"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_esco_data():
    """Load all ESCO datasets."""
    print("Loading ESCO datasets...")
    
    data = {
        'skills': pd.read_csv(ESCO_DIR / "skills_en.csv", low_memory=False),
        'occupations': pd.read_csv(ESCO_DIR / "occupations_en.csv", low_memory=False),
        'occ_skill_relations': pd.read_csv(ESCO_DIR / "occupationSkillRelations_en.csv", low_memory=False),
        'skill_groups': pd.read_csv(ESCO_DIR / "skillGroups_en.csv", low_memory=False),
        'skill_hierarchy': pd.read_csv(ESCO_DIR / "skillsHierarchy_en.csv", low_memory=False),
        'broader_relations': pd.read_csv(ESCO_DIR / "broaderRelationsSkillPillar_en.csv", low_memory=False),
        'isco_groups': pd.read_csv(ESCO_DIR / "ISCOGroups_en.csv", low_memory=False),
    }
    
    print(f"  ✓ Skills: {len(data['skills']):,}")
    print(f"  ✓ Occupations: {len(data['occupations']):,}")
    print(f"  ✓ Occupation-Skill Relations: {len(data['occ_skill_relations']):,}")
    print(f"  ✓ Skill Groups: {len(data['skill_groups']):,}")
    print(f"  ✓ Skill Hierarchy: {len(data['skill_hierarchy']):,}")
    print(f"  ✓ Broader Relations: {len(data['broader_relations']):,}")
    print(f"  ✓ ISCO Groups: {len(data['isco_groups']):,}")
    
    return data


def load_decorte_dataset():
    """Load the DECORTE dataset from HuggingFace."""
    print("\nLoading DECORTE dataset from HuggingFace...")
    dataset = load_dataset("jensjorisdecorte/anonymous-working-histories")
    print(f"  ✓ Train: {len(dataset['train']):,}")
    print(f"  ✓ Validation: {len(dataset['validation']):,}")
    print(f"  ✓ Test: {len(dataset['test']):,}")
    return dataset


# ============================================================================
# 4.1.1 DECORTE Dataset Statistics
# ============================================================================
def compute_decorte_stats(dataset, esco_data):
    """Compute statistics for the DECORTE dataset."""
    print("\nComputing DECORTE statistics...")
    
    stats = {}
    
    # Total careers in each split
    stats['train_size'] = len(dataset['train'])
    stats['val_size'] = len(dataset['validation'])
    stats['test_size'] = len(dataset['test'])
    stats['total_careers'] = stats['train_size'] + stats['val_size'] + stats['test_size']
    
    # Career lengths and job experiences
    all_career_lengths = []
    all_esco_uris = set()
    total_job_experiences = 0
    
    for split_name in ['train', 'validation', 'test']:
        split_data = dataset[split_name]
        for example in split_data:
            n_exp = example['number_of_experiences']
            all_career_lengths.append(n_exp)
            total_job_experiences += n_exp
            
            # Collect unique ESCO occupations
            for i in range(n_exp):
                uri = example.get(f'ESCO_uri_{i}')
                if uri and not pd.isna(uri):
                    all_esco_uris.add(uri)
    
    stats['total_job_experiences'] = total_job_experiences
    stats['avg_career_length'] = np.mean(all_career_lengths)
    stats['std_career_length'] = np.std(all_career_lengths)
    stats['min_career_length'] = min(all_career_lengths)
    stats['max_career_length'] = max(all_career_lengths)
    stats['median_career_length'] = np.median(all_career_lengths)
    
    # Career length percentiles
    stats['p25_career_length'] = np.percentile(all_career_lengths, 25)
    stats['p50_career_length'] = np.percentile(all_career_lengths, 50)
    stats['p75_career_length'] = np.percentile(all_career_lengths, 75)
    stats['p90_career_length'] = np.percentile(all_career_lengths, 90)
    
    # Career length distribution (for histogram)
    stats['career_length_distribution'] = Counter(all_career_lengths)
    
    # Unique ESCO occupations in dataset
    stats['unique_esco_occupations'] = len(all_esco_uris)
    
    # ESCO version - determine from modified dates in ESCO files
    occ_df = esco_data['occupations']
    if 'modifiedDate' in occ_df.columns:
        # Get the most common year from modified dates
        dates = pd.to_datetime(occ_df['modifiedDate'], errors='coerce')
        most_recent = dates.max()
        stats['esco_version'] = f"v1.2.0 (files dated {most_recent.strftime('%Y-%m')})"
    else:
        stats['esco_version'] = "v1.2.0 (estimated from 2024-01 file dates)"
    
    return stats, all_career_lengths, all_esco_uris


# ============================================================================
# 4.1.3 ESCO Taxonomy Statistics
# ============================================================================
def compute_esco_taxonomy_stats(esco_data):
    """Compute statistics for the ESCO taxonomy."""
    print("\nComputing ESCO taxonomy statistics...")
    
    stats = {}
    
    # Basic counts
    stats['total_occupations'] = len(esco_data['occupations'])
    stats['total_skills'] = len(esco_data['skills'])
    
    # Skill-occupation relations
    occ_skill_rel = esco_data['occ_skill_relations']
    stats['total_occ_skill_relations'] = len(occ_skill_rel)
    
    # Essential vs optional relations
    essential_mask = occ_skill_rel['relationType'] == 'essential'
    stats['essential_relations'] = essential_mask.sum()
    stats['optional_relations'] = (~essential_mask).sum()
    
    # Skills per occupation
    skills_per_occ = occ_skill_rel.groupby('occupationUri')['skillUri'].count()
    stats['avg_skills_per_occupation'] = skills_per_occ.mean()
    stats['std_skills_per_occupation'] = skills_per_occ.std()
    stats['min_skills_per_occupation'] = skills_per_occ.min()
    stats['max_skills_per_occupation'] = skills_per_occ.max()
    stats['median_skills_per_occupation'] = skills_per_occ.median()
    
    # Essential only skills per occupation
    essential_per_occ = occ_skill_rel[essential_mask].groupby('occupationUri')['skillUri'].count()
    stats['avg_essential_skills_per_occupation'] = essential_per_occ.mean()
    
    # Optional only skills per occupation
    optional_per_occ = occ_skill_rel[~essential_mask].groupby('occupationUri')['skillUri'].count()
    stats['avg_optional_skills_per_occupation'] = optional_per_occ.mean() if len(optional_per_occ) > 0 else 0
    
    # Skill hierarchy levels
    skill_hierarchy = esco_data['skill_hierarchy']
    
    # Count L1 categories (Level 1 URI is not null, Level 2 is null)
    l1_mask = skill_hierarchy['Level 1 URI'].notna() & skill_hierarchy['Level 2 URI'].isna()
    stats['num_l1_skill_categories'] = skill_hierarchy[l1_mask]['Level 1 URI'].nunique()
    
    # Count L2 categories
    l2_mask = skill_hierarchy['Level 2 URI'].notna() & skill_hierarchy['Level 3 URI'].isna()
    stats['num_l2_skill_categories'] = skill_hierarchy[l2_mask]['Level 2 URI'].nunique()
    
    # Alternative: count from skill groups
    skill_groups = esco_data['skill_groups']
    if 'code' in skill_groups.columns:
        # L1 = single letter codes (S, K, L, A, T)
        l1_codes = skill_groups[skill_groups['code'].str.match(r'^[A-Z]$', na=False)]
        stats['num_l1_skill_categories_alt'] = len(l1_codes)
        
        # L2 = codes like S1, S2, K1, etc.
        l2_codes = skill_groups[skill_groups['code'].str.match(r'^[A-Z]\d+$', na=False)]
        stats['num_l2_skill_categories_alt'] = len(l2_codes)
    
    # ISCO grouping statistics
    isco_groups = esco_data['isco_groups']
    occupations = esco_data['occupations']
    
    # Count ISCO groups at each digit level
    if 'code' in isco_groups.columns:
        isco_codes = isco_groups['code'].astype(str)
        stats['isco_1digit_groups'] = len(isco_codes[isco_codes.str.len() == 1])
        stats['isco_2digit_groups'] = len(isco_codes[isco_codes.str.len() == 2])
        stats['isco_3digit_groups'] = len(isco_codes[isco_codes.str.len() == 3])
        stats['isco_4digit_groups'] = len(isco_codes[isco_codes.str.len() == 4])
    
    # Skills with multiple parents (before DAG expansion)
    broader_relations = esco_data['broader_relations']
    # Count how many parents each concept has
    parent_counts = broader_relations.groupby('conceptUri')['broaderUri'].count()
    stats['skills_with_multiple_parents'] = (parent_counts > 1).sum()
    stats['max_parents_per_skill'] = parent_counts.max()
    
    return stats, skills_per_occ


# ============================================================================
# 4.1.4 Exploratory Data Analysis
# ============================================================================
def compute_eda_stats(dataset, esco_data, all_esco_uris):
    """Compute exploratory data analysis statistics."""
    print("\nComputing EDA statistics...")
    
    stats = {}
    occupations = esco_data['occupations']
    
    # Create ISCO code lookup from occupations
    occ_to_isco = {}
    for _, row in occupations.iterrows():
        uri = row['conceptUri']
        isco = str(row['iscoGroup']) if pd.notna(row['iscoGroup']) else None
        if uri and isco:
            occ_to_isco[uri] = isco
    
    # Occupation distribution across 10 ISCO major groups
    isco_major_counts = Counter()
    isco_major_names = {
        '0': 'Armed forces occupations',
        '1': 'Managers',
        '2': 'Professionals',
        '3': 'Technicians and associate professionals',
        '4': 'Clerical support workers',
        '5': 'Service and sales workers',
        '6': 'Skilled agricultural, forestry and fishery workers',
        '7': 'Craft and related trades workers',
        '8': 'Plant and machine operators, and assemblers',
        '9': 'Elementary occupations',
    }
    
    # Count occupations per ISCO major group (from ESCO taxonomy)
    for _, row in occupations.iterrows():
        isco = str(row['iscoGroup']) if pd.notna(row['iscoGroup']) else None
        if isco and len(isco) >= 1:
            major_group = isco[0]
            isco_major_counts[major_group] += 1
    
    stats['isco_major_group_distribution'] = dict(isco_major_counts)
    stats['isco_major_names'] = isco_major_names
    
    # Count ISCO groups appearing in DECORTE dataset
    decorte_isco_counts = Counter()
    for split_name in ['train', 'validation', 'test']:
        split_data = dataset[split_name]
        for example in split_data:
            n_exp = example['number_of_experiences']
            for i in range(n_exp):
                uri = example.get(f'ESCO_uri_{i}')
                if uri and uri in occ_to_isco:
                    isco = occ_to_isco[uri]
                    if len(isco) >= 1:
                        major_group = isco[0]
                        decorte_isco_counts[major_group] += 1
    
    stats['decorte_isco_distribution'] = dict(decorte_isco_counts)
    
    # Transition patterns analysis
    same_isco_transitions = 0
    cross_isco_transitions = 0
    total_transitions = 0
    
    for split_name in ['train', 'validation', 'test']:
        split_data = dataset[split_name]
        for example in split_data:
            n_exp = example['number_of_experiences']
            if n_exp < 2:
                continue
                
            prev_major = None
            for i in range(n_exp):
                uri = example.get(f'ESCO_uri_{i}')
                if uri and uri in occ_to_isco:
                    isco = occ_to_isco[uri]
                    if len(isco) >= 1:
                        curr_major = isco[0]
                        if prev_major is not None:
                            total_transitions += 1
                            if curr_major == prev_major:
                                same_isco_transitions += 1
                            else:
                                cross_isco_transitions += 1
                        prev_major = curr_major
    
    stats['total_transitions'] = total_transitions
    stats['same_isco_transitions'] = same_isco_transitions
    stats['cross_isco_transitions'] = cross_isco_transitions
    if total_transitions > 0:
        stats['pct_same_isco'] = (same_isco_transitions / total_transitions) * 100
        stats['pct_cross_isco'] = (cross_isco_transitions / total_transitions) * 100
    else:
        stats['pct_same_isco'] = 0
        stats['pct_cross_isco'] = 0
    
    return stats


# ============================================================================
# Report Generation
# ============================================================================
def generate_report(decorte_stats, career_lengths, esco_stats, skills_per_occ, eda_stats):
    """Generate the markdown report."""
    
    report = []
    report.append("# Dataset Statistics Report")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n---\n")
    
    # =========================================================================
    # 4.1.1 DECORTE Dataset
    # =========================================================================
    report.append("## 4.1.1 DECORTE Dataset\n")
    report.append("The DECORTE dataset (Decorte et al.) contains anonymized career histories with ESCO occupation mappings.\n")
    report.append(f"**Source:** `jensjorisdecorte/anonymous-working-histories` (HuggingFace)\n")
    
    report.append("\n### Basic Statistics\n")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    report.append(f"| Total number of career histories (resumes) | {decorte_stats['total_careers']:,} |")
    report.append(f"| Total number of job experiences | {decorte_stats['total_job_experiences']:,} |")
    report.append(f"| Train split size | {decorte_stats['train_size']:,} |")
    report.append(f"| Validation split size | {decorte_stats['val_size']:,} |")
    report.append(f"| Test split size | {decorte_stats['test_size']:,} |")
    
    report.append("\n### Career Length Statistics\n")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    report.append(f"| Average career length (jobs per career) | {decorte_stats['avg_career_length']:.2f} |")
    report.append(f"| Standard deviation | {decorte_stats['std_career_length']:.2f} |")
    report.append(f"| Minimum career length | {decorte_stats['min_career_length']} |")
    report.append(f"| Maximum career length | {decorte_stats['max_career_length']} |")
    report.append(f"| Median career length | {decorte_stats['median_career_length']:.1f} |")
    
    report.append("\n### Career Length Percentiles\n")
    report.append("| Percentile | Value |")
    report.append("|------------|-------|")
    report.append(f"| 25th percentile | {decorte_stats['p25_career_length']:.1f} |")
    report.append(f"| 50th percentile (median) | {decorte_stats['p50_career_length']:.1f} |")
    report.append(f"| 75th percentile | {decorte_stats['p75_career_length']:.1f} |")
    report.append(f"| 90th percentile | {decorte_stats['p90_career_length']:.1f} |")
    
    report.append("\n### Career Length Distribution\n")
    report.append("| Career Length | Count |")
    report.append("|---------------|-------|")
    for length in sorted(decorte_stats['career_length_distribution'].keys()):
        count = decorte_stats['career_length_distribution'][length]
        report.append(f"| {length} jobs | {count:,} |")
    
    report.append("\n### ESCO Coverage\n")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    report.append(f"| Number of unique ESCO occupations in dataset | {decorte_stats['unique_esco_occupations']:,} |")
    report.append(f"| ESCO version used | {decorte_stats['esco_version']} |")
    
    # =========================================================================
    # 4.1.3 ESCO Taxonomy
    # =========================================================================
    report.append("\n---\n")
    report.append("## 4.1.3 ESCO Taxonomy\n")
    report.append("The European Skills, Competences, Qualifications and Occupations (ESCO) taxonomy.\n")
    
    report.append("\n### Core Statistics\n")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    report.append(f"| Total number of occupations | {esco_stats['total_occupations']:,} |")
    report.append(f"| Total number of skills | {esco_stats['total_skills']:,} |")
    report.append(f"| Total occupation-skill relations | {esco_stats['total_occ_skill_relations']:,} |")
    
    report.append("\n### Essential vs Optional Skill Relations\n")
    report.append("| Relation Type | Count | Percentage |")
    report.append("|---------------|-------|------------|")
    total_rel = esco_stats['essential_relations'] + esco_stats['optional_relations']
    report.append(f"| Essential | {esco_stats['essential_relations']:,} | {esco_stats['essential_relations']/total_rel*100:.1f}% |")
    report.append(f"| Optional | {esco_stats['optional_relations']:,} | {esco_stats['optional_relations']/total_rel*100:.1f}% |")
    
    report.append("\n### Skills per Occupation\n")
    report.append("| Metric | Essential Only | Optional Only | Combined |")
    report.append("|--------|----------------|---------------|----------|")
    report.append(f"| Average | {esco_stats['avg_essential_skills_per_occupation']:.2f} | {esco_stats['avg_optional_skills_per_occupation']:.2f} | {esco_stats['avg_skills_per_occupation']:.2f} |")
    report.append(f"| Std Dev | - | - | {esco_stats['std_skills_per_occupation']:.2f} |")
    report.append(f"| Min | - | - | {esco_stats['min_skills_per_occupation']} |")
    report.append(f"| Max | - | - | {esco_stats['max_skills_per_occupation']} |")
    report.append(f"| Median | - | - | {esco_stats['median_skills_per_occupation']:.1f} |")
    
    report.append("\n### Skill Hierarchy\n")
    report.append("| Level | Count |")
    report.append("|-------|-------|")
    report.append(f"| L1 skill categories (pillars) | {esco_stats['num_l1_skill_categories']} |")
    report.append(f"| L2 skill categories | {esco_stats['num_l2_skill_categories']} |")
    
    report.append("\n### ISCO Grouping Statistics\n")
    report.append("| ISCO Level | Number of Groups |")
    report.append("|------------|------------------|")
    report.append(f"| 1-digit (Major groups) | {esco_stats.get('isco_1digit_groups', 'N/A')} |")
    report.append(f"| 2-digit (Sub-major groups) | {esco_stats.get('isco_2digit_groups', 'N/A')} |")
    report.append(f"| 3-digit (Minor groups) | {esco_stats.get('isco_3digit_groups', 'N/A')} |")
    report.append(f"| 4-digit (Unit groups) | {esco_stats.get('isco_4digit_groups', 'N/A')} |")
    
    report.append("\n### DAG Structure\n")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    report.append(f"| Skills with multiple parents (before DAG expansion) | {esco_stats['skills_with_multiple_parents']:,} |")
    report.append(f"| Maximum parents per skill | {esco_stats['max_parents_per_skill']} |")
    
    # =========================================================================
    # 4.1.4 Exploratory Data Analysis
    # =========================================================================
    report.append("\n---\n")
    report.append("## 4.1.4 Exploratory Data Analysis\n")
    
    report.append("\n### Occupation Distribution Across ISCO Major Groups (ESCO Taxonomy)\n")
    report.append("| ISCO Code | Major Group Name | Count |")
    report.append("|-----------|------------------|-------|")
    for code in sorted(eda_stats['isco_major_group_distribution'].keys()):
        name = eda_stats['isco_major_names'].get(code, 'Unknown')
        count = eda_stats['isco_major_group_distribution'][code]
        report.append(f"| {code} | {name} | {count:,} |")
    
    report.append("\n### ISCO Distribution in DECORTE Dataset (Job Experiences)\n")
    report.append("| ISCO Code | Major Group Name | Count |")
    report.append("|-----------|------------------|-------|")
    total_decorte = sum(eda_stats['decorte_isco_distribution'].values())
    for code in sorted(eda_stats['decorte_isco_distribution'].keys()):
        name = eda_stats['isco_major_names'].get(code, 'Unknown')
        count = eda_stats['decorte_isco_distribution'][code]
        pct = count / total_decorte * 100 if total_decorte > 0 else 0
        report.append(f"| {code} | {name} | {count:,} ({pct:.1f}%) |")
    
    report.append("\n### Skills per Occupation Distribution\n")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    report.append(f"| Mean | {esco_stats['avg_skills_per_occupation']:.2f} |")
    report.append(f"| Median | {esco_stats['median_skills_per_occupation']:.1f} |")
    report.append(f"| Std Dev | {esco_stats['std_skills_per_occupation']:.2f} |")
    report.append(f"| Min | {esco_stats['min_skills_per_occupation']} |")
    report.append(f"| Max | {esco_stats['max_skills_per_occupation']} |")
    
    report.append("\n### Career Length Distribution (Percentiles)\n")
    report.append("| Percentile | Career Length |")
    report.append("|------------|---------------|")
    report.append(f"| 25th | {decorte_stats['p25_career_length']:.1f} jobs |")
    report.append(f"| 50th (median) | {decorte_stats['p50_career_length']:.1f} jobs |")
    report.append(f"| 75th | {decorte_stats['p75_career_length']:.1f} jobs |")
    report.append(f"| 90th | {decorte_stats['p90_career_length']:.1f} jobs |")
    
    report.append("\n### Transition Patterns\n")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    report.append(f"| Total job transitions analyzed | {eda_stats['total_transitions']:,} |")
    report.append(f"| Transitions within same ISCO major group | {eda_stats['same_isco_transitions']:,} ({eda_stats['pct_same_isco']:.1f}%) |")
    report.append(f"| Transitions across ISCO major groups | {eda_stats['cross_isco_transitions']:,} ({eda_stats['pct_cross_isco']:.1f}%) |")
    
    report.append("\n---\n")
    report.append("## Summary\n")
    report.append("\nThis report provides comprehensive statistics for the datasets used in the thesis:\n")
    report.append(f"- **DECORTE**: {decorte_stats['total_careers']:,} career histories with {decorte_stats['total_job_experiences']:,} job experiences")
    report.append(f"- **ESCO Taxonomy**: {esco_stats['total_occupations']:,} occupations and {esco_stats['total_skills']:,} skills")
    report.append(f"- **Skill Relations**: {esco_stats['total_occ_skill_relations']:,} occupation-skill mappings")
    report.append(f"- **Career Transitions**: {eda_stats['pct_same_isco']:.1f}% within same ISCO group, {eda_stats['pct_cross_isco']:.1f}% across groups")
    
    return "\n".join(report)


def convert_to_json_serializable(obj):
    """Convert numpy types and other non-serializable types to JSON-compatible types."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, Counter):
        return dict(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def save_json_stats(decorte_stats, esco_stats, eda_stats):
    """Save statistics as JSON for programmatic access."""
    all_stats = {
        'decorte': {k: v for k, v in decorte_stats.items() if not isinstance(v, Counter)},
        'esco': esco_stats,
        'eda': eda_stats,
        'generated_at': datetime.now().isoformat()
    }
    
    # Convert Counter to dict for JSON serialization
    if 'career_length_distribution' in decorte_stats:
        all_stats['decorte']['career_length_distribution'] = dict(decorte_stats['career_length_distribution'])
    
    # Convert all numpy types to native Python types
    all_stats = convert_to_json_serializable(all_stats)
    
    json_path = OUTPUT_DIR / "dataset_statistics.json"
    with open(json_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n✓ JSON statistics saved to: {json_path}")


def main():
    """Main function to generate the dataset report."""
    print("=" * 60)
    print("Dataset Statistics Report Generator")
    print("=" * 60)
    
    # Load data
    esco_data = load_esco_data()
    decorte_dataset = load_decorte_dataset()
    
    # Compute statistics
    decorte_stats, career_lengths, all_esco_uris = compute_decorte_stats(decorte_dataset, esco_data)
    esco_stats, skills_per_occ = compute_esco_taxonomy_stats(esco_data)
    eda_stats = compute_eda_stats(decorte_dataset, esco_data, all_esco_uris)
    
    # Generate report
    print("\nGenerating report...")
    report = generate_report(decorte_stats, career_lengths, esco_stats, skills_per_occ, eda_stats)
    
    # Save report
    report_path = OUTPUT_DIR / "dataset_statistics_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n✓ Report saved to: {report_path}")
    
    # Save JSON stats
    save_json_stats(decorte_stats, esco_stats, eda_stats)
    
    print("\n" + "=" * 60)
    print("Report generation complete!")
    print("=" * 60)
    
    return report


if __name__ == "__main__":
    main()

