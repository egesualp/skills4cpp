#!/usr/bin/env python3
"""
Comprehensive Analysis Script for Skill Prediction Evaluation
Generates detailed metrics and error analysis report
"""

import json
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import sys

# File paths
SIMILARITY_SCORES_PATH = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_h2_pjmath/fused_predictions.json"
JOB_IDS_PATH = "/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/decorte_master_3.csv"
OCCUPATION_SKILLS_PATH = "/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/occupationSkillRelations_en.csv"
OCCUPATIONS_PATH = "/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/occupations_en.csv"
SKILLS_PATH = "/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/skills_en.csv"


# Input format control
# - "base": expects {job_id: [{"skill_uri": "...", "score": ..., "rank": ...}, ...]}
# - "fused": expects {job_id: [[skill_uri, score], [skill_uri, score], ...]}
PREDICTIONS_FORMAT = "fused"  # change to "fused" for fused_predictions.json


def normalize_predictions(similarity_scores, predictions_format: str):
    """
    Normalizes different prediction JSON formats into:
      Dict[job_id, List[Dict]] with at least {'skill_uri': ...}
    """
    if predictions_format not in {"base", "fused"}:
        raise ValueError(f"Unknown PREDICTIONS_FORMAT='{predictions_format}'. Use 'base' or 'fused'.")

    if predictions_format == "base":
        return similarity_scores

    # Fused format: list of [skill_uri, score]
    normalized = {}
    for job_id, preds in similarity_scores.items():
        out = []
        for rank, item in enumerate(preds, start=1):
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                skill_uri = item[0]
                score = item[1] if len(item) > 1 else None
                out.append({"skill_uri": skill_uri, "score": score, "rank": rank})
            elif isinstance(item, dict) and "skill_uri" in item:
                # tolerate already-normalized entries
                out.append(item)
        normalized[str(job_id)] = out
    return normalized


def load_data():
    """Load all necessary data files"""
    print("Loading data files...")
    
    # Load similarity scores (predictions)
    print("  - Loading similarity scores...")
    with open(SIMILARITY_SCORES_PATH, 'r') as f:
        similarity_scores = json.load(f)
    similarity_scores = normalize_predictions(similarity_scores, PREDICTIONS_FORMAT)
    
    # Load job mappings
    print("  - Loading job mappings...")
    jobs_df = pd.read_csv(JOB_IDS_PATH)
    
    # Load ground truth (occupation-skill relations)
    print("  - Loading occupation-skill relations...")
    occ_skills_df = pd.read_csv(OCCUPATION_SKILLS_PATH)
    
    # Load occupation metadata
    print("  - Loading occupation metadata...")
    occupations_df = pd.read_csv(OCCUPATIONS_PATH)
    
    # Load skill metadata
    print("  - Loading skill metadata...")
    skills_df = pd.read_csv(SKILLS_PATH)
    
    print("Data loading complete!\n")
    return similarity_scores, jobs_df, occ_skills_df, occupations_df, skills_df


def build_ground_truth(jobs_df, occ_skills_df):
    """Build ground truth mapping: job_id -> {skill_uris}, separated by essential/optional"""
    print("Building ground truth mappings...")
    
    ground_truth_all = {}
    ground_truth_essential = {}
    ground_truth_optional = {}
    
    for idx, row in jobs_df.iterrows():
        job_id = str(row['job_id'])
        esco_id = row['esco_id']
        
        # Get all skills for this occupation
        occ_skills = occ_skills_df[occ_skills_df['occupationUri'] == esco_id]
        
        all_skills = set(occ_skills['skillUri'].values)
        essential_skills = set(occ_skills[occ_skills['relationType'] == 'essential']['skillUri'].values)
        optional_skills = set(occ_skills[occ_skills['relationType'] == 'optional']['skillUri'].values)
        
        ground_truth_all[job_id] = all_skills
        ground_truth_essential[job_id] = essential_skills
        ground_truth_optional[job_id] = optional_skills
    
    print(f"Ground truth built for {len(ground_truth_all)} jobs\n")
    return ground_truth_all, ground_truth_essential, ground_truth_optional


def calculate_metrics_at_k(predicted_skills: List[str], ground_truth: Set[str], k_values: List[int]) -> Dict:
    """Calculate Precision, Recall, and F1 at various k values"""
    metrics = {}
    
    for k in k_values:
        predicted_at_k = set(predicted_skills[:k])
        
        if len(predicted_at_k) == 0:
            precision = 0.0
            recall = 0.0
            f1 = 0.0
        else:
            true_positives = len(predicted_at_k & ground_truth)
            precision = true_positives / len(predicted_at_k) if len(predicted_at_k) > 0 else 0.0
            recall = true_positives / len(ground_truth) if len(ground_truth) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics[f'P@{k}'] = precision
        metrics[f'R@{k}'] = recall
        metrics[f'F1@{k}'] = f1
    
    return metrics


def calculate_mrr(predicted_skills: List[str], ground_truth: Set[str]) -> float:
    """Calculate Mean Reciprocal Rank"""
    for i, skill in enumerate(predicted_skills, 1):
        if skill in ground_truth:
            return 1.0 / i
    return 0.0


def calculate_map(predicted_skills: List[str], ground_truth: Set[str], k: int = None) -> float:
    """Calculate Mean Average Precision"""
    if len(ground_truth) == 0:
        return 0.0
    
    if k is not None:
        predicted_skills = predicted_skills[:k]
    
    relevant_count = 0
    precision_sum = 0.0
    
    for i, skill in enumerate(predicted_skills, 1):
        if skill in ground_truth:
            relevant_count += 1
            precision_at_i = relevant_count / i
            precision_sum += precision_at_i
    
    if relevant_count == 0:
        return 0.0
    
    return precision_sum / len(ground_truth)


def evaluate_predictions(similarity_scores, ground_truth_all, ground_truth_essential, ground_truth_optional):
    """Evaluate predictions and calculate metrics"""
    print("Evaluating predictions...")
    
    k_values = [5, 10, 20, 50, 100, 200, 500, 1000]
    
    results = {
        'all_skills': defaultdict(list),
        'essential_skills': defaultdict(list),
        'optional_skills': defaultdict(list)
    }
    
    job_details = []
    
    for job_id_str, predictions in similarity_scores.items():
        if job_id_str not in ground_truth_all:
            continue
        
        # Extract predicted skill URIs in order
        predicted_skills = [pred['skill_uri'] for pred in predictions]
        
        gt_all = ground_truth_all[job_id_str]
        gt_essential = ground_truth_essential[job_id_str]
        gt_optional = ground_truth_optional[job_id_str]
        
        # Calculate metrics for all skills
        metrics_all = calculate_metrics_at_k(predicted_skills, gt_all, k_values)
        mrr_all = calculate_mrr(predicted_skills, gt_all)
        map_all = calculate_map(predicted_skills, gt_all, k=1000)
        
        for k in k_values:
            results['all_skills'][f'P@{k}'].append(metrics_all[f'P@{k}'])
            results['all_skills'][f'R@{k}'].append(metrics_all[f'R@{k}'])
            results['all_skills'][f'F1@{k}'].append(metrics_all[f'F1@{k}'])
        
        results['all_skills']['MRR'].append(mrr_all)
        results['all_skills']['MAP@1000'].append(map_all)
        
        # Calculate metrics for essential skills
        metrics_essential = calculate_metrics_at_k(predicted_skills, gt_essential, k_values)
        mrr_essential = calculate_mrr(predicted_skills, gt_essential)
        map_essential = calculate_map(predicted_skills, gt_essential, k=1000)
        
        for k in k_values:
            results['essential_skills'][f'P@{k}'].append(metrics_essential[f'P@{k}'])
            results['essential_skills'][f'R@{k}'].append(metrics_essential[f'R@{k}'])
            results['essential_skills'][f'F1@{k}'].append(metrics_essential[f'F1@{k}'])
        
        results['essential_skills']['MRR'].append(mrr_essential)
        results['essential_skills']['MAP@1000'].append(map_essential)
        
        # Calculate metrics for optional skills
        if len(gt_optional) > 0:
            metrics_optional = calculate_metrics_at_k(predicted_skills, gt_optional, k_values)
            mrr_optional = calculate_mrr(predicted_skills, gt_optional)
            map_optional = calculate_map(predicted_skills, gt_optional, k=1000)
            
            for k in k_values:
                results['optional_skills'][f'P@{k}'].append(metrics_optional[f'P@{k}'])
                results['optional_skills'][f'R@{k}'].append(metrics_optional[f'R@{k}'])
                results['optional_skills'][f'F1@{k}'].append(metrics_optional[f'F1@{k}'])
            
            results['optional_skills']['MRR'].append(mrr_optional)
            results['optional_skills']['MAP@1000'].append(map_optional)
        
        # Store job-level details for further analysis
        job_details.append({
            'job_id': job_id_str,
            'num_gt_all': len(gt_all),
            'num_gt_essential': len(gt_essential),
            'num_gt_optional': len(gt_optional),
            'mrr_all': mrr_all,
            'mrr_essential': mrr_essential,
            'mrr_optional': mrr_optional if len(gt_optional) > 0 else None,
            'map_all': map_all,
            'map_essential': map_essential,
            'map_optional': map_optional if len(gt_optional) > 0 else None,
            'r@1000_all': metrics_all['R@1000'],
            'r@1000_essential': metrics_essential['R@1000'],
            'r@1000_optional': metrics_optional['R@1000'] if len(gt_optional) > 0 else None,
        })
    
    print(f"Evaluated {len(job_details)} jobs\n")
    return results, job_details


def analyze_top_skills(similarity_scores, ground_truth_all, ground_truth_essential, ground_truth_optional, skills_df):
    """Analyze which types of skills appear at the top of predictions"""
    print("Analyzing skill types in top predictions...")
    
    top_k_values = [1, 5, 10, 20, 50, 100]
    
    analysis = {}
    
    for k in top_k_values:
        essential_count = 0
        optional_count = 0
        not_relevant_count = 0
        total = 0
        
        for job_id_str, predictions in similarity_scores.items():
            if job_id_str not in ground_truth_all:
                continue
            
            predicted_skills = [pred['skill_uri'] for pred in predictions[:k]]
            
            gt_essential = ground_truth_essential[job_id_str]
            gt_optional = ground_truth_optional[job_id_str]
            
            for skill in predicted_skills:
                total += 1
                if skill in gt_essential:
                    essential_count += 1
                elif skill in gt_optional:
                    optional_count += 1
                else:
                    not_relevant_count += 1
        
        analysis[k] = {
            'essential': essential_count,
            'optional': optional_count,
            'not_relevant': not_relevant_count,
            'total': total,
            'essential_pct': 100 * essential_count / total if total > 0 else 0,
            'optional_pct': 100 * optional_count / total if total > 0 else 0,
            'not_relevant_pct': 100 * not_relevant_count / total if total > 0 else 0,
        }
    
    print("Skill type analysis complete\n")
    return analysis


def analyze_by_occupation_group(similarity_scores, jobs_df, occupations_df, ground_truth_all, ground_truth_essential, ground_truth_optional):
    """Analyze performance by occupation group (ISCO group)"""
    print("Analyzing by occupation group...")
    
    # Merge job data with occupation data to get ISCO groups
    jobs_with_isco = jobs_df.merge(
        occupations_df[['conceptUri', 'iscoGroup', 'preferredLabel']], 
        left_on='esco_id', 
        right_on='conceptUri', 
        how='left'
    )
    
    # Group by ISCO group
    isco_results = defaultdict(lambda: defaultdict(list))
    
    for idx, row in jobs_with_isco.iterrows():
        job_id = str(row['job_id'])
        isco_group = str(row['iscoGroup'])[:2] if pd.notna(row['iscoGroup']) else 'Unknown'  # First 2 digits
        
        if job_id not in similarity_scores or job_id not in ground_truth_all:
            continue
        
        predictions = similarity_scores[job_id]
        predicted_skills = [pred['skill_uri'] for pred in predictions]
        
        gt_all = ground_truth_all[job_id]
        gt_essential = ground_truth_essential[job_id]
        
        # Calculate key metrics
        metrics_all = calculate_metrics_at_k(predicted_skills, gt_all, [1000])
        mrr_all = calculate_mrr(predicted_skills, gt_all)
        map_all = calculate_map(predicted_skills, gt_all, k=1000)
        
        isco_results[isco_group]['R@1000'].append(metrics_all['R@1000'])
        isco_results[isco_group]['MRR'].append(mrr_all)
        isco_results[isco_group]['MAP@1000'].append(map_all)
        isco_results[isco_group]['count'].append(1)
    
    # Calculate averages per ISCO group
    isco_summary = []
    for isco_group, metrics in isco_results.items():
        isco_summary.append({
            'ISCO_Group': isco_group,
            'Count': sum(metrics['count']),
            'Avg_R@1000': np.mean(metrics['R@1000']),
            'Avg_MRR': np.mean(metrics['MRR']),
            'Avg_MAP@1000': np.mean(metrics['MAP@1000']),
        })
    
    isco_df = pd.DataFrame(isco_summary).sort_values('ISCO_Group')
    
    print(f"Analyzed {len(isco_df)} occupation groups\n")
    return isco_df


def analyze_by_skill_type(similarity_scores, jobs_df, skills_df, ground_truth_all):
    """Analyze performance by skill type (knowledge vs skill/competence) and reuse level"""
    print("Analyzing by skill type...")
    
    # Create skill metadata lookup
    skill_meta = skills_df.set_index('conceptUri')[['skillType', 'reuseLevel']].to_dict('index')
    
    # Track performance by skill type
    type_results = defaultdict(lambda: defaultdict(list))
    
    for job_id_str, predictions in similarity_scores.items():
        if job_id_str not in ground_truth_all:
            continue
        
        predicted_skills = [pred['skill_uri'] for pred in predictions]
        gt_all = ground_truth_all[job_id_str]
        
        # Check top 100 predictions
        for i, skill_uri in enumerate(predicted_skills[:100], 1):
            if skill_uri in skill_meta:
                skill_type = skill_meta[skill_uri].get('skillType', 'unknown')
                reuse_level = skill_meta[skill_uri].get('reuseLevel', 'unknown')
                
                is_relevant = 1 if skill_uri in gt_all else 0
                
                type_results[skill_type]['relevant'].append(is_relevant)
                type_results[skill_type]['total'].append(1)
                type_results[skill_type]['rank'].append(i)
                
                type_results[reuse_level]['relevant'].append(is_relevant)
                type_results[reuse_level]['total'].append(1)
                type_results[reuse_level]['rank'].append(i)
    
    # Calculate summary statistics
    type_summary = []
    for category, metrics in type_results.items():
        if category and category != 'unknown':
            type_summary.append({
                'Category': category,
                'Total_Predictions': sum(metrics['total']),
                'Relevant_Predictions': sum(metrics['relevant']),
                'Relevance_Rate': 100 * sum(metrics['relevant']) / sum(metrics['total']),
                'Avg_Rank': np.mean(metrics['rank']),
            })
    
    type_df = pd.DataFrame(type_summary).sort_values('Relevance_Rate', ascending=False)
    
    print("Skill type analysis complete\n")
    return type_df


def generate_report(results, job_details, top_skills_analysis, isco_df, skill_type_df, jobs_df, occupations_df):
    """Generate comprehensive report"""
    
    report = []
    report.append("=" * 100)
    report.append("COMPREHENSIVE SKILL PREDICTION EVALUATION REPORT")
    report.append("=" * 100)
    report.append("")
    
    # 1. Overall Metrics Summary
    report.append("=" * 100)
    report.append("1. OVERALL METRICS SUMMARY")
    report.append("=" * 100)
    report.append("")
    
    for skill_type in ['all_skills', 'essential_skills', 'optional_skills']:
        if not results[skill_type]:
            continue
        
        report.append(f"\n{skill_type.replace('_', ' ').upper()}")
        report.append("-" * 100)
        
        # Precision @ k
        report.append("\nPrecision @ k:")
        for k in [5, 10, 20, 50, 100, 200, 500, 1000]:
            metric_name = f'P@{k}'
            if metric_name in results[skill_type]:
                avg_value = np.mean(results[skill_type][metric_name])
                report.append(f"  P@{k:4d} = {avg_value:.4f}")
        
        # Recall @ k
        report.append("\nRecall @ k:")
        for k in [5, 10, 20, 50, 100, 200, 500, 1000]:
            metric_name = f'R@{k}'
            if metric_name in results[skill_type]:
                avg_value = np.mean(results[skill_type][metric_name])
                report.append(f"  R@{k:4d} = {avg_value:.4f}")
        
        # F1 @ k
        report.append("\nF1 @ k:")
        for k in [5, 10, 20, 50, 100, 200, 500, 1000]:
            metric_name = f'F1@{k}'
            if metric_name in results[skill_type]:
                avg_value = np.mean(results[skill_type][metric_name])
                report.append(f"  F1@{k:4d} = {avg_value:.4f}")
        
        # MRR and MAP
        report.append("\nRanking Metrics:")
        if 'MRR' in results[skill_type]:
            avg_mrr = np.mean(results[skill_type]['MRR'])
            report.append(f"  MRR      = {avg_mrr:.4f}")
        if 'MAP@1000' in results[skill_type]:
            avg_map = np.mean(results[skill_type]['MAP@1000'])
            report.append(f"  MAP@1000 = {avg_map:.4f}")
        
        report.append("")
    
    # 2. Essential vs Optional Skills Comparison
    report.append("\n" + "=" * 100)
    report.append("2. ESSENTIAL vs OPTIONAL SKILLS - KEY COMPARISON")
    report.append("=" * 100)
    report.append("")
    
    comparison_metrics = ['R@10', 'R@50', 'R@100', 'R@1000', 'MRR', 'MAP@1000']
    report.append(f"{'Metric':<15} {'Essential':>12} {'Optional':>12} {'Difference':>12} {'% Diff':>10}")
    report.append("-" * 100)
    
    for metric in comparison_metrics:
        if metric in results['essential_skills'] and metric in results['optional_skills']:
            essential_val = np.mean(results['essential_skills'][metric])
            optional_val = np.mean(results['optional_skills'][metric])
            diff = essential_val - optional_val
            pct_diff = 100 * diff / essential_val if essential_val > 0 else 0
            report.append(f"{metric:<15} {essential_val:>12.4f} {optional_val:>12.4f} {diff:>12.4f} {pct_diff:>9.1f}%")
    
    # 3. Top-K Skill Type Analysis
    report.append("\n" + "=" * 100)
    report.append("3. SKILL TYPE DISTRIBUTION IN TOP-K PREDICTIONS")
    report.append("=" * 100)
    report.append("")
    report.append("Analysis: What types of skills appear at the top of our predictions?")
    report.append("")
    report.append(f"{'Top-K':<10} {'Essential':>12} {'Optional':>12} {'Not Relevant':>15} {'Essential %':>12} {'Optional %':>12}")
    report.append("-" * 100)
    
    for k in [1, 5, 10, 20, 50, 100]:
        if k in top_skills_analysis:
            data = top_skills_analysis[k]
            report.append(
                f"Top-{k:<5} "
                f"{data['essential']:>12,} "
                f"{data['optional']:>12,} "
                f"{data['not_relevant']:>15,} "
                f"{data['essential_pct']:>11.1f}% "
                f"{data['optional_pct']:>11.1f}%"
            )
    
    # 4. Occupation Group Analysis (ISCO)
    report.append("\n" + "=" * 100)
    report.append("4. PERFORMANCE BY OCCUPATION GROUP (ISCO)")
    report.append("=" * 100)
    report.append("")
    
    # ISCO group names mapping
    isco_names = {
        '11': 'Chief Executives, Senior Officials and Legislators',
        '12': 'Administrative and Commercial Managers',
        '13': 'Production and Specialized Services Managers',
        '14': 'Hospitality, Retail and Other Services Managers',
        '21': 'Science and Engineering Professionals',
        '22': 'Health Professionals',
        '23': 'Teaching Professionals',
        '24': 'Business and Administration Professionals',
        '25': 'Information and Communications Technology Professionals',
        '26': 'Legal, Social and Cultural Professionals',
        '31': 'Science and Engineering Associate Professionals',
        '32': 'Health Associate Professionals',
        '33': 'Business and Administration Associate Professionals',
        '34': 'Legal, Social, Cultural and Related Associate Professionals',
        '35': 'Information and Communications Technicians',
        '41': 'General and Keyboard Clerks',
        '42': 'Customer Services Clerks',
        '43': 'Numerical and Material Recording Clerks',
        '44': 'Other Clerical Support Workers',
        '51': 'Personal Service Workers',
        '52': 'Sales Workers',
        '53': 'Personal Care Workers',
        '54': 'Protective Services Workers',
        '61': 'Market-oriented Skilled Agricultural Workers',
        '62': 'Market-oriented Skilled Forestry, Fishery and Hunting Workers',
        '63': 'Subsistence Farmers, Fishers, Hunters and Gatherers',
        '71': 'Building and Related Trades Workers',
        '72': 'Metal, Machinery and Related Trades Workers',
        '73': 'Handicraft and Printing Workers',
        '74': 'Electrical and Electronic Trades Workers',
        '75': 'Food Processing, Wood Working, Garment and Other Craft Workers',
        '81': 'Stationary Plant and Machine Operators',
        '82': 'Assemblers',
        '83': 'Drivers and Mobile Plant Operators',
        '91': 'Cleaners and Helpers',
        '92': 'Agricultural, Forestry and Fishery Labourers',
        '93': 'Labourers in Mining, Construction, Manufacturing and Transport',
        '94': 'Food Preparation Assistants',
        '95': 'Street and Related Sales and Service Workers',
        '96': 'Refuse Workers and Other Elementary Workers',
    }
    
    report.append(f"{'ISCO':<6} {'Count':>7} {'R@1000':>10} {'MRR':>10} {'MAP@1000':>10} {'Description':<60}")
    report.append("-" * 100)
    
    for idx, row in isco_df.iterrows():
        isco = row['ISCO_Group']
        desc = isco_names.get(isco, 'Unknown')
        report.append(
            f"{isco:<6} "
            f"{row['Count']:>7} "
            f"{row['Avg_R@1000']:>10.4f} "
            f"{row['Avg_MRR']:>10.4f} "
            f"{row['Avg_MAP@1000']:>10.4f} "
            f"{desc:<60}"
        )
    
    # 5. Skill Category Analysis
    report.append("\n" + "=" * 100)
    report.append("5. PERFORMANCE BY SKILL CATEGORY")
    report.append("=" * 100)
    report.append("")
    report.append("Analysis: Which skill categories are predicted more accurately?")
    report.append("")
    report.append(f"{'Category':<35} {'Total Pred.':>12} {'Relevant':>12} {'Relevance %':>12} {'Avg Rank':>12}")
    report.append("-" * 100)
    
    for idx, row in skill_type_df.iterrows():
        report.append(
            f"{row['Category']:<35} "
            f"{row['Total_Predictions']:>12,} "
            f"{row['Relevant_Predictions']:>12,} "
            f"{row['Relevance_Rate']:>11.1f}% "
            f"{row['Avg_Rank']:>12.1f}"
        )
    
    # 6. Statistical Summary
    report.append("\n" + "=" * 100)
    report.append("6. STATISTICAL SUMMARY")
    report.append("=" * 100)
    report.append("")
    
    jobs_details_df = pd.DataFrame(job_details)
    
    report.append("Ground Truth Skills per Job:")
    report.append(f"  All Skills      - Mean: {jobs_details_df['num_gt_all'].mean():.1f}, "
                 f"Median: {jobs_details_df['num_gt_all'].median():.1f}, "
                 f"Std: {jobs_details_df['num_gt_all'].std():.1f}")
    report.append(f"  Essential       - Mean: {jobs_details_df['num_gt_essential'].mean():.1f}, "
                 f"Median: {jobs_details_df['num_gt_essential'].median():.1f}, "
                 f"Std: {jobs_details_df['num_gt_essential'].std():.1f}")
    report.append(f"  Optional        - Mean: {jobs_details_df['num_gt_optional'].mean():.1f}, "
                 f"Median: {jobs_details_df['num_gt_optional'].median():.1f}, "
                 f"Std: {jobs_details_df['num_gt_optional'].std():.1f}")
    
    report.append("\nRecall@1000 Distribution:")
    report.append(f"  All Skills      - Mean: {jobs_details_df['r@1000_all'].mean():.4f}, "
                 f"Median: {jobs_details_df['r@1000_all'].median():.4f}, "
                 f"Std: {jobs_details_df['r@1000_all'].std():.4f}")
    report.append(f"  Essential       - Mean: {jobs_details_df['r@1000_essential'].mean():.4f}, "
                 f"Median: {jobs_details_df['r@1000_essential'].median():.4f}, "
                 f"Std: {jobs_details_df['r@1000_essential'].std():.4f}")
    
    # 7. Key Insights
    report.append("\n" + "=" * 100)
    report.append("7. KEY INSIGHTS AND OBSERVATIONS")
    report.append("=" * 100)
    report.append("")
    
    # Calculate some key insights
    r1000_all = np.mean(results['all_skills']['R@1000'])
    r1000_essential = np.mean(results['essential_skills']['R@1000'])
    r1000_optional = np.mean(results['optional_skills']['R@1000']) if results['optional_skills']['R@1000'] else 0
    
    mrr_all = np.mean(results['all_skills']['MRR'])
    mrr_essential = np.mean(results['essential_skills']['MRR'])
    
    essential_in_top10 = top_skills_analysis[10]['essential_pct']
    optional_in_top10 = top_skills_analysis[10]['optional_pct']
    
    report.append(f"1. Overall Coverage:")
    report.append(f"   - The model achieves {r1000_all:.1%} recall at top-1000 for all skills")
    report.append(f"   - Essential skills have {r1000_essential:.1%} recall, while optional have {r1000_optional:.1%}")
    report.append(f"   - This suggests {'better' if r1000_essential > r1000_optional else 'similar'} performance on essential skills")
    report.append("")
    
    report.append(f"2. Ranking Quality:")
    report.append(f"   - MRR of {mrr_all:.4f} indicates the first relevant skill appears around rank {1/mrr_all:.1f} on average")
    report.append(f"   - For essential skills specifically, first appearance is around rank {1/mrr_essential:.1f}")
    report.append("")
    
    report.append(f"3. Top Predictions Analysis:")
    report.append(f"   - {essential_in_top10:.1f}% of top-10 predictions are essential skills")
    report.append(f"   - {optional_in_top10:.1f}% of top-10 predictions are optional skills")
    report.append(f"   - This indicates the model {'strongly prioritizes' if essential_in_top10 > 50 else 'struggles to prioritize'} essential skills")
    report.append("")
    
    # Best and worst ISCO groups
    best_isco = isco_df.nlargest(3, 'Avg_R@1000')
    worst_isco = isco_df.nsmallest(3, 'Avg_R@1000')
    
    report.append(f"4. Occupation Group Performance:")
    report.append(f"   Best performing groups:")
    for idx, row in best_isco.iterrows():
        isco_desc = isco_names.get(row['ISCO_Group'], 'Unknown')
        report.append(f"     - {row['ISCO_Group']} ({isco_desc}): R@1000={row['Avg_R@1000']:.4f}")
    report.append(f"   Worst performing groups:")
    for idx, row in worst_isco.iterrows():
        isco_desc = isco_names.get(row['ISCO_Group'], 'Unknown')
        report.append(f"     - {row['ISCO_Group']} ({isco_desc}): R@1000={row['Avg_R@1000']:.4f}")
    report.append("")
    
    # Skill type insights
    best_skill_type = skill_type_df.nlargest(3, 'Relevance_Rate')
    report.append(f"5. Skill Category Performance:")
    report.append(f"   Most accurately predicted categories:")
    for idx, row in best_skill_type.iterrows():
        report.append(f"     - {row['Category']}: {row['Relevance_Rate']:.1f}% relevant at rank {row['Avg_Rank']:.1f}")
    report.append("")
    
    report.append("=" * 100)
    report.append("END OF REPORT")
    report.append("=" * 100)
    
    return "\n".join(report)


def main():
    """Main execution function"""
    print("\n" + "=" * 100)
    print("SKILL PREDICTION EVALUATION - COMPREHENSIVE ANALYSIS")
    print("=" * 100 + "\n")
    
    # Load data
    similarity_scores, jobs_df, occ_skills_df, occupations_df, skills_df = load_data()
    
    # Build ground truth
    ground_truth_all, ground_truth_essential, ground_truth_optional = build_ground_truth(jobs_df, occ_skills_df)
    
    # Evaluate predictions
    results, job_details = evaluate_predictions(
        similarity_scores, 
        ground_truth_all, 
        ground_truth_essential, 
        ground_truth_optional
    )
    
    # Analyze top skills
    top_skills_analysis = analyze_top_skills(
        similarity_scores, 
        ground_truth_all, 
        ground_truth_essential, 
        ground_truth_optional,
        skills_df
    )
    
    # Analyze by occupation group
    isco_df = analyze_by_occupation_group(
        similarity_scores, 
        jobs_df, 
        occupations_df, 
        ground_truth_all,
        ground_truth_essential,
        ground_truth_optional
    )
    
    # Analyze by skill type
    skill_type_df = analyze_by_skill_type(
        similarity_scores,
        jobs_df,
        skills_df,
        ground_truth_all
    )
    
    # Generate report
    print("Generating comprehensive report...\n")
    report = generate_report(
        results, 
        job_details, 
        top_skills_analysis, 
        isco_df, 
        skill_type_df,
        jobs_df,
        occupations_df
    )
    
    # Print report
    print(report)
    
    # Save report to file
    output_path = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/decorte_w_desc_3/evaluation_report.txt"
    with open(output_path, 'w') as f:
        f.write(report)
    
    print(f"\n\nReport saved to: {output_path}")
    
    # Save detailed results as CSV
    job_details_df = pd.DataFrame(job_details)
    job_details_path = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_h2_pjmath/job_level_metrics.csv"
    job_details_df.to_csv(job_details_path, index=False)
    print(f"Job-level metrics saved to: {job_details_path}")
    
    isco_path = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_h2_pjmath/isco_group_metrics.csv"
    isco_df.to_csv(isco_path, index=False)
    print(f"ISCO group metrics saved to: {isco_path}")
    
    skill_type_path = "/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2/outputs/v5_fused_scorer/multiplicative_h2_pjmath/skill_category_metrics.csv"
    skill_type_df.to_csv(skill_type_path, index=False)
    print(f"Skill category metrics saved to: {skill_type_path}")
    
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE!")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()







