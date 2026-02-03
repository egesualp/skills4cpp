"""
evaluate_reranking.py - Evaluation metrics for LLM re-ranking

Compares the LLM re-ranked results against:
1. Original linear fusion scores
2. Ground truth labels

Computes metrics:
- mAP (Mean Average Precision)
- Recall@K (K=5, 10, 20, 50)
- NDCG@K
- Precision@K

Usage:
    python -m skill_mapping.v3.evaluate_reranking \
        --reranked_scores ./outputs/llm_reranking/llm_reranked_scores_compact.json \
        --original_scores ./outputs/linear_fusion_sum/best_fused_scores.json \
        --ground_truth ./data/processed/ground_truth.csv \
        --output_dir ./outputs/llm_reranking/evaluation
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
from loguru import logger


class RerankingEvaluator:
    """Evaluates re-ranking performance."""
    
    def __init__(
        self,
        reranked_scores: Dict[str, List[Dict]],
        original_scores: Dict[str, List[Dict]],
        ground_truth: Dict[str, Set[str]],
    ):
        """
        Initialize evaluator.
        
        Args:
            reranked_scores: Re-ranked scores by job_id
            original_scores: Original scores by job_id
            ground_truth: Ground truth skills by job_id
        """
        self.reranked_scores = reranked_scores
        self.original_scores = original_scores
        self.ground_truth = ground_truth
        
        # Find common job IDs
        self.job_ids = (
            set(reranked_scores.keys()) &
            set(original_scores.keys()) &
            set(ground_truth.keys())
        )
        logger.info(f"Evaluating on {len(self.job_ids)} common jobs")
    
    def _average_precision(self, ranked_skills: List[str], true_skills: Set[str]) -> float:
        """
        Calculate Average Precision for a single job.
        
        Args:
            ranked_skills: List of skill URIs in ranked order
            true_skills: Set of ground truth skill URIs
            
        Returns:
            Average Precision score
        """
        if len(true_skills) == 0:
            return 0.0
        
        relevant_count = 0
        precision_sum = 0.0
        
        for i, skill in enumerate(ranked_skills):
            if skill in true_skills:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
        
        return precision_sum / len(true_skills)
    
    def _recall_at_k(self, ranked_skills: List[str], true_skills: Set[str], k: int) -> float:
        """
        Calculate Recall@K for a single job.
        
        Args:
            ranked_skills: List of skill URIs in ranked order
            true_skills: Set of ground truth skill URIs
            k: Cutoff rank
            
        Returns:
            Recall@K score
        """
        if len(true_skills) == 0:
            return 0.0
        
        top_k_skills = set(ranked_skills[:k])
        relevant_retrieved = len(top_k_skills & true_skills)
        
        return relevant_retrieved / len(true_skills)
    
    def _precision_at_k(self, ranked_skills: List[str], true_skills: Set[str], k: int) -> float:
        """
        Calculate Precision@K for a single job.
        
        Args:
            ranked_skills: List of skill URIs in ranked order
            true_skills: Set of ground truth skill URIs
            k: Cutoff rank
            
        Returns:
            Precision@K score
        """
        top_k_skills = set(ranked_skills[:k])
        relevant_retrieved = len(top_k_skills & true_skills)
        
        return relevant_retrieved / k
    
    def _ndcg_at_k(self, ranked_skills: List[str], true_skills: Set[str], k: int) -> float:
        """
        Calculate NDCG@K for a single job.
        
        Args:
            ranked_skills: List of skill URIs in ranked order
            true_skills: Set of ground truth skill URIs
            k: Cutoff rank
            
        Returns:
            NDCG@K score
        """
        import numpy as np
        
        # Calculate DCG
        dcg = 0.0
        for i, skill in enumerate(ranked_skills[:k]):
            if skill in true_skills:
                # Relevance = 1 for relevant, 0 for irrelevant
                dcg += 1.0 / np.log2(i + 2)  # i+2 because i is 0-indexed
        
        # Calculate ideal DCG
        num_relevant = min(len(true_skills), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def evaluate_method(self, scores: Dict[str, List[Dict]], method_name: str) -> Dict:
        """
        Evaluate a scoring method.
        
        Args:
            scores: Scores by job_id
            method_name: Name of the method
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating {method_name}...")
        
        ap_scores = []
        recall_at_5 = []
        recall_at_10 = []
        recall_at_20 = []
        recall_at_50 = []
        precision_at_5 = []
        precision_at_10 = []
        precision_at_20 = []
        precision_at_50 = []
        ndcg_at_5 = []
        ndcg_at_10 = []
        ndcg_at_20 = []
        ndcg_at_50 = []
        
        for job_id in self.job_ids:
            # Get ranked skills
            skill_scores = scores[job_id]
            ranked_skills = [item['skill_uri'] for item in skill_scores]
            
            # Get ground truth
            true_skills = self.ground_truth[job_id]
            
            # Calculate metrics
            ap_scores.append(self._average_precision(ranked_skills, true_skills))
            recall_at_5.append(self._recall_at_k(ranked_skills, true_skills, 5))
            recall_at_10.append(self._recall_at_k(ranked_skills, true_skills, 10))
            recall_at_20.append(self._recall_at_k(ranked_skills, true_skills, 20))
            recall_at_50.append(self._recall_at_k(ranked_skills, true_skills, 50))
            precision_at_5.append(self._precision_at_k(ranked_skills, true_skills, 5))
            precision_at_10.append(self._precision_at_k(ranked_skills, true_skills, 10))
            precision_at_20.append(self._precision_at_k(ranked_skills, true_skills, 20))
            precision_at_50.append(self._precision_at_k(ranked_skills, true_skills, 50))
            ndcg_at_5.append(self._ndcg_at_k(ranked_skills, true_skills, 5))
            ndcg_at_10.append(self._ndcg_at_k(ranked_skills, true_skills, 10))
            ndcg_at_20.append(self._ndcg_at_k(ranked_skills, true_skills, 20))
            ndcg_at_50.append(self._ndcg_at_k(ranked_skills, true_skills, 50))
        
        # Calculate means
        metrics = {
            'method': method_name,
            'mAP': sum(ap_scores) / len(ap_scores),
            'Recall@5': sum(recall_at_5) / len(recall_at_5),
            'Recall@10': sum(recall_at_10) / len(recall_at_10),
            'Recall@20': sum(recall_at_20) / len(recall_at_20),
            'Recall@50': sum(recall_at_50) / len(recall_at_50),
            'Precision@5': sum(precision_at_5) / len(precision_at_5),
            'Precision@10': sum(precision_at_10) / len(precision_at_10),
            'Precision@20': sum(precision_at_20) / len(precision_at_20),
            'Precision@50': sum(precision_at_50) / len(precision_at_50),
            'NDCG@5': sum(ndcg_at_5) / len(ndcg_at_5),
            'NDCG@10': sum(ndcg_at_10) / len(ndcg_at_10),
            'NDCG@20': sum(ndcg_at_20) / len(ndcg_at_20),
            'NDCG@50': sum(ndcg_at_50) / len(ndcg_at_50),
        }
        
        return metrics
    
    def compare_methods(self) -> pd.DataFrame:
        """
        Compare original and re-ranked methods.
        
        Returns:
            DataFrame with comparison results
        """
        original_metrics = self.evaluate_method(self.original_scores, "Original (Linear Fusion)")
        reranked_metrics = self.evaluate_method(self.reranked_scores, "LLM Re-ranked")
        
        # Calculate improvements
        improvement_metrics = {
            'method': 'Improvement (%)',
        }
        for key in original_metrics:
            if key == 'method':
                continue
            original_val = original_metrics[key]
            reranked_val = reranked_metrics[key]
            if original_val > 0:
                improvement = ((reranked_val - original_val) / original_val) * 100
                improvement_metrics[key] = improvement
            else:
                improvement_metrics[key] = 0.0
        
        # Create DataFrame
        df = pd.DataFrame([original_metrics, reranked_metrics, improvement_metrics])
        
        return df
    
    def analyze_tier_performance(self) -> Dict:
        """
        Analyze performance by tier classification.
        
        Returns:
            Dictionary with tier-based analysis
        """
        logger.info("Analyzing tier performance...")
        
        tier_stats = {
            'Essential': {'in_top5': 0, 'in_top10': 0, 'in_top20': 0, 'total': 0},
            'Optional': {'in_top5': 0, 'in_top10': 0, 'in_top20': 0, 'total': 0},
            'Irrelevant': {'in_top5': 0, 'in_top10': 0, 'in_top20': 0, 'total': 0},
        }
        
        for job_id in self.job_ids:
            skill_scores = self.reranked_scores[job_id]
            true_skills = self.ground_truth[job_id]
            
            for item in skill_scores:
                skill_uri = item['skill_uri']
                tier = item.get('tier', 'Unknown')
                rank = item['rank']
                
                if skill_uri in true_skills and tier in tier_stats:
                    tier_stats[tier]['total'] += 1
                    if rank <= 5:
                        tier_stats[tier]['in_top5'] += 1
                    if rank <= 10:
                        tier_stats[tier]['in_top10'] += 1
                    if rank <= 20:
                        tier_stats[tier]['in_top20'] += 1
        
        # Calculate percentages
        for tier in tier_stats:
            total = tier_stats[tier]['total']
            if total > 0:
                tier_stats[tier]['pct_in_top5'] = (tier_stats[tier]['in_top5'] / total) * 100
                tier_stats[tier]['pct_in_top10'] = (tier_stats[tier]['in_top10'] / total) * 100
                tier_stats[tier]['pct_in_top20'] = (tier_stats[tier]['in_top20'] / total) * 100
        
        return tier_stats


def load_ground_truth(ground_truth_path: Path) -> Dict[str, Set[str]]:
    """
    Load ground truth labels.
    
    Args:
        ground_truth_path: Path to ground truth CSV or JSON
        
    Returns:
        Dictionary mapping job_id to set of skill URIs
    """
    if ground_truth_path.suffix == '.json':
        with open(ground_truth_path, 'r') as f:
            data = json.load(f)
        return {k: set(v) for k, v in data.items()}
    else:
        # Assume CSV format: job_id, skill_uri
        df = pd.read_csv(ground_truth_path)
        gt = defaultdict(set)
        for _, row in df.iterrows():
            gt[str(row['job_id'])].add(row['skill_uri'])
        return dict(gt)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate LLM re-ranking performance"
    )
    
    parser.add_argument(
        "--reranked_scores",
        type=Path,
        required=True,
        help="Path to LLM re-ranked scores JSON"
    )
    parser.add_argument(
        "--original_scores",
        type=Path,
        required=True,
        help="Path to original fusion scores JSON"
    )
    parser.add_argument(
        "--ground_truth",
        type=Path,
        required=True,
        help="Path to ground truth labels (CSV or JSON)"
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for evaluation results"
    )
    
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    
    with open(args.reranked_scores, 'r') as f:
        reranked_data = json.load(f)
        reranked_scores = reranked_data['scores']
    
    with open(args.original_scores, 'r') as f:
        original_data = json.load(f)
        original_scores = original_data['scores']
    
    ground_truth = load_ground_truth(args.ground_truth)
    
    # Create evaluator
    evaluator = RerankingEvaluator(
        reranked_scores=reranked_scores,
        original_scores=original_scores,
        ground_truth=ground_truth,
    )
    
    # Compare methods
    comparison_df = evaluator.compare_methods()
    
    # Print results
    logger.info("\n" + "="*80)
    logger.info("EVALUATION RESULTS")
    logger.info("="*80)
    print(comparison_df.to_string(index=False))
    
    # Save results
    comparison_file = args.output_dir / "comparison_metrics.csv"
    comparison_df.to_csv(comparison_file, index=False)
    logger.info(f"\nComparison saved to {comparison_file}")
    
    # Analyze tier performance
    tier_stats = evaluator.analyze_tier_performance()
    
    logger.info("\n" + "="*80)
    logger.info("TIER PERFORMANCE ANALYSIS")
    logger.info("="*80)
    for tier, stats in tier_stats.items():
        logger.info(f"\n{tier}:")
        logger.info(f"  Total ground truth skills: {stats['total']}")
        if stats['total'] > 0:
            logger.info(f"  In Top-5:  {stats['in_top5']} ({stats['pct_in_top5']:.2f}%)")
            logger.info(f"  In Top-10: {stats['in_top10']} ({stats['pct_in_top10']:.2f}%)")
            logger.info(f"  In Top-20: {stats['in_top20']} ({stats['pct_in_top20']:.2f}%)")
    
    # Save tier stats
    tier_file = args.output_dir / "tier_analysis.json"
    with open(tier_file, 'w') as f:
        json.dump(tier_stats, f, indent=2)
    logger.info(f"\nTier analysis saved to {tier_file}")


if __name__ == "__main__":
    main()


