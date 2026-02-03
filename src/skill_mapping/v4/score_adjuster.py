"""
score_adjuster.py - Adjust scores based on LLM tier classifications and evaluate metrics

Takes LLM outputs from llm_fetcher.py, adjusts skill scores, and compares 
metrics before and after LLM reranking.

Features:
- Mismatch detection between LLM skill numbers and expected skills
- Configurable tier weights for score adjustment
- Full metric comparison (mAP, Recall@K, Precision@K, NDCG@K)

Usage:
    python -m skill_mapping.v4.score_adjuster \
        --llm_responses ./outputs/llm_fetcher/llm_responses.jsonl.gz \
        --original_scores ./outputs/linear_fusion_sum/best_fused_scores.json \
        --ground_truth ./outputs/llm_reranking/ground_truth.json \
        --output_dir ./outputs/v4_reranking
"""

import argparse
import gzip
import json
import itertools
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class TierConfig:
    """Configuration for tier-based scoring."""
    essential_base: float = 3.0
    optional_base: float = 2.0
    irrelevant_base: float = 1.0
    epsilon: float = 0.1
    
    def get_final_score(self, tier: str, original_score: float, max_score: float = 1.0) -> float:
        """Calculate final score based on tier and original score."""
        base_scores = {
            "Essential": self.essential_base,
            "Optional": self.optional_base,
            "Irrelevant": self.irrelevant_base,
        }
        base = base_scores.get(tier, self.irrelevant_base)
        normalized = original_score / max_score if max_score > 0 else 0
        return base + (self.epsilon * normalized)


class MismatchReport:
    """Tracks mismatches between LLM output and expected skills."""
    
    def __init__(self):
        self.total_jobs = 0
        self.jobs_with_mismatches = 0
        self.total_skills_expected = 0
        self.total_skills_matched = 0
        self.mismatch_details = []
    
    def add_job_result(self, job_id: str, expected: int, matched: int, mismatched_nums: List[int]):
        """Record matching result for a job."""
        self.total_jobs += 1
        self.total_skills_expected += expected
        self.total_skills_matched += matched
        
        if mismatched_nums:
            self.jobs_with_mismatches += 1
            self.mismatch_details.append({
                "job_id": job_id,
                "expected": expected,
                "matched": matched,
                "mismatched_skill_numbers": mismatched_nums,
            })
    
    def print_summary(self):
        """Print mismatch summary."""
        logger.info("=== Mismatch Report ===")
        logger.info(f"Total jobs processed: {self.total_jobs}")
        logger.info(f"Jobs with mismatches: {self.jobs_with_mismatches}")
        logger.info(f"Total skills expected: {self.total_skills_expected}")
        logger.info(f"Total skills matched: {self.total_skills_matched}")
        match_rate = (self.total_skills_matched / self.total_skills_expected * 100
                      if self.total_skills_expected > 0 else 0)
        logger.info(f"Match rate: {match_rate:.2f}%")
        
        if self.mismatch_details:
            logger.warning(f"First 5 jobs with mismatches:")
            for detail in self.mismatch_details[:5]:
                logger.warning(f"  Job {detail['job_id']}: {detail['matched']}/{detail['expected']} matched, "
                             f"missing skill_numbers: {detail['mismatched_skill_numbers'][:5]}...")
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_jobs": self.total_jobs,
            "jobs_with_mismatches": self.jobs_with_mismatches,
            "total_skills_expected": self.total_skills_expected,
            "total_skills_matched": self.total_skills_matched,
            "match_rate_pct": (self.total_skills_matched / self.total_skills_expected * 100
                               if self.total_skills_expected > 0 else 0),
            "mismatch_details": self.mismatch_details,
        }


class MetricsCalculator:
    """Calculate retrieval metrics."""
    
    @staticmethod
    def average_precision(ranked_skills: List[str], true_skills: Set[str]) -> float:
        """Calculate Average Precision for a single job."""
        if not true_skills:
            return 0.0
        
        relevant_count = 0
        precision_sum = 0.0
        
        for i, skill in enumerate(ranked_skills):
            if skill in true_skills:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
        
        return precision_sum / len(true_skills)
    
    @staticmethod
    def recall_at_k(ranked_skills: List[str], true_skills: Set[str], k: int) -> float:
        """Calculate Recall@K for a single job."""
        if not true_skills:
            return 0.0
        top_k = set(ranked_skills[:k])
        return len(top_k & true_skills) / len(true_skills)
    
    @staticmethod
    def precision_at_k(ranked_skills: List[str], true_skills: Set[str], k: int) -> float:
        """Calculate Precision@K for a single job."""
        top_k = set(ranked_skills[:k])
        return len(top_k & true_skills) / k
    
    @staticmethod
    def ndcg_at_k(ranked_skills: List[str], true_skills: Set[str], k: int) -> float:
        """Calculate NDCG@K for a single job."""
        dcg = 0.0
        for i, skill in enumerate(ranked_skills[:k]):
            if skill in true_skills:
                dcg += 1.0 / np.log2(i + 2)
        
        num_relevant = min(len(true_skills), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(num_relevant))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def recall_upper_bound_at_k(num_true_skills: int, k: int) -> float:
        """Calculate Upper Bound for Recall@K (min(k, n) / n)."""
        if num_true_skills <= 0:
            return 0.0
        return min(k, num_true_skills) / num_true_skills

    @staticmethod
    def mrr(ranked_skills: List[str], true_skills: Set[str], k: Optional[int] = None) -> float:
        """Calculate Reciprocal Rank (MRR component) for a single job."""
        skills_to_check = ranked_skills[:k] if k else ranked_skills
        for i, skill in enumerate(skills_to_check):
            if skill in true_skills:
                return 1.0 / (i + 1)
        return 0.0


class ScoreAdjuster:
    """Adjusts scores based on LLM tier classifications."""
    
    def __init__(
        self,
        llm_responses_path: Path,
        original_scores_path: Path,
        ground_truth_path: Path,
        output_dir: Path,
        tier_config: Optional[TierConfig] = None,
    ):
        self.llm_responses_path = llm_responses_path
        self.original_scores_path = original_scores_path
        self.ground_truth_path = ground_truth_path
        self.output_dir = output_dir
        self.tier_config = tier_config or TierConfig()
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Loading data...")
        self.llm_data = self._load_llm_responses()
        self.original_scores = self._load_original_scores()
        self.ground_truth = self._load_ground_truth()
        
        logger.info(f"Loaded LLM responses for {len(self.llm_data)} jobs")
        logger.info(f"Loaded original scores for {len(self.original_scores)} jobs")
        logger.info(f"Loaded ground truth for {len(self.ground_truth)} jobs")
    
    def _load_llm_responses(self) -> List[Dict]:
        """Load LLM responses from gzipped JSONL."""
        data = []
        with gzip.open(self.llm_responses_path, 'rt', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line.strip())
                if record["llm_response"]["success"]:
                    data.append(record)
        return data
    
    def _load_original_scores(self) -> Dict[str, List[Dict]]:
        """Load original fusion scores."""
        with open(self.original_scores_path, 'r') as f:
            data = json.load(f)
        return data['scores']
    
    def _load_ground_truth(self) -> Dict[str, Set[str]]:
        """Load ground truth labels."""
        with open(self.ground_truth_path, 'r') as f:
            data = json.load(f)
        return {k: set(v) for k, v in data.items()}
    
    def _match_classifications_to_skills(
        self,
        job_id: str,
        skill_candidates: List[Dict],
        classifications: List[Dict],
    ) -> Tuple[Dict[str, str], List[int]]:
        """
        Match LLM classifications to skill URIs.
        
        Returns:
            Tuple of (skill_uri -> tier mapping, list of unmatched skill numbers)
        """
        # Create mapping from skill_number to expected skill_uri
        num_to_uri = {i + 1: skill["skill_uri"] for i, skill in enumerate(skill_candidates)}
        
        skill_tiers = {}
        matched_nums = set()
        
        for item in classifications:
            skill_num = item.get("skill_number")
            tier = item.get("tier")
            
            if skill_num and skill_num in num_to_uri and tier in ["Essential", "Optional", "Irrelevant"]:
                skill_tiers[num_to_uri[skill_num]] = tier
                matched_nums.add(skill_num)
        
        # Find mismatched (unmatched) skill numbers
        expected_nums = set(num_to_uri.keys())
        mismatched = sorted(expected_nums - matched_nums)
        
        return skill_tiers, mismatched
    
    def adjust_scores(self, tier_config: Optional[TierConfig] = None) -> Tuple[Dict[str, List[Dict]], MismatchReport]:
        """
        Adjust scores for all jobs based on LLM classifications.
        
        Returns:
            Tuple of (adjusted scores by job_id, mismatch report)
        """
        config = tier_config or self.tier_config
        
        adjusted_scores = {}
        mismatch_report = MismatchReport()
        
        for record in self.llm_data:
            job_id = record["job_id"]
            skill_candidates = record["skill_candidates"]
            classifications = record["llm_response"]["response"]["skill_classifications"]
            
            # Match classifications to skills
            skill_tiers, mismatched = self._match_classifications_to_skills(
                job_id, skill_candidates, classifications
            )
            
            # Record mismatch data
            mismatch_report.add_job_result(
                job_id=job_id,
                expected=len(skill_candidates),
                matched=len(skill_tiers),
                mismatched_nums=mismatched,
            )
            
            # Find max score for normalization
            max_score = max(s["original_score"] for s in skill_candidates) if skill_candidates else 1.0
            
            # Calculate adjusted scores
            adjusted = []
            for skill in skill_candidates:
                skill_uri = skill["skill_uri"]
                tier = skill_tiers.get(skill_uri, "Irrelevant")
                final_score = config.get_final_score(
                    tier, skill["original_score"], max_score
                )
                
                adjusted.append({
                    "skill_uri": skill_uri,
                    "original_score": skill["original_score"],
                    "original_rank": skill["original_rank"],
                    "tier": tier,
                    "final_score": final_score,
                })
            
            # Sort by final score descending
            adjusted.sort(key=lambda x: x["final_score"], reverse=True)
            
            # Assign final ranks
            for i, item in enumerate(adjusted):
                item["final_rank"] = i + 1
            
            adjusted_scores[job_id] = adjusted
        
        return adjusted_scores, mismatch_report
    
    def evaluate_metrics(
        self,
        adjusted_scores: Dict[str, List[Dict]],
    ) -> pd.DataFrame:
        """
        Evaluate and compare metrics before and after adjustment.
        
        Returns:
            DataFrame with comparison results
        """
        logger.info("Evaluating metrics...")
        
        # Find common jobs
        common_jobs = (
            set(adjusted_scores.keys()) &
            set(self.original_scores.keys()) &
            set(self.ground_truth.keys())
        )
        logger.info(f"Evaluating on {len(common_jobs)} common jobs")
        
        k_values = [5, 10, 20, 50]
        
        # Calculate metrics for original and adjusted
        metrics = {"original": {}, "adjusted": {}}
        
        # Upper bounds cache (only need once since GT is constant per job)
        upper_bounds = {k: [] for k in k_values}
        
        for method, scores in [("original", self.original_scores), ("adjusted", adjusted_scores)]:
            ap_scores, mrr_scores = [], []
            recall = {k: [] for k in k_values}
            precision = {k: [] for k in k_values}
            ndcg = {k: [] for k in k_values}
            
            for job_id in common_jobs:
                job_scores = scores[job_id]
                ranked_skills = [item['skill_uri'] for item in job_scores]
                true_skills = self.ground_truth[job_id]
                
                # Calculate upper bounds only in first pass (original)
                if method == "original":
                    for k in k_values:
                        upper_bounds[k].append(
                            MetricsCalculator.recall_upper_bound_at_k(len(true_skills), k)
                        )

                ap_scores.append(MetricsCalculator.average_precision(ranked_skills, true_skills))
                mrr_scores.append(MetricsCalculator.mrr(ranked_skills, true_skills))
                
                for k in k_values:
                    recall[k].append(MetricsCalculator.recall_at_k(ranked_skills, true_skills, k))
                    precision[k].append(MetricsCalculator.precision_at_k(ranked_skills, true_skills, k))
                    ndcg[k].append(MetricsCalculator.ndcg_at_k(ranked_skills, true_skills, k))
            
            metrics[method] = {
                "mAP": np.mean(ap_scores),
                "MRR": np.mean(mrr_scores),
            }
            for k in k_values:
                metrics[method][f"Recall@{k}"] = np.mean(recall[k])
                metrics[method][f"Precision@{k}"] = np.mean(precision[k])
                metrics[method][f"NDCG@{k}"] = np.mean(ndcg[k])
        
        # Add upper bounds to metrics (as a separate key or part of both?)
        # Let's add them as a separate "method" or just available for reporting
        avg_upper_bounds = {f"Recall@{k}": np.mean(upper_bounds[k]) for k in k_values}
        
        # Build comparison dataframe
        rows = []
        for method in ["original", "adjusted"]:
            row = {"method": "Original (Fusion)" if method == "original" else "LLM Adjusted"}
            row.update(metrics[method])
            rows.append(row)
        
        # Add improvement row
        improvement = {"method": "Improvement (%)"}
        for key in metrics["original"]:
            orig = metrics["original"][key]
            adj = metrics["adjusted"][key]
            if orig > 0:
                improvement[key] = ((adj - orig) / orig) * 100
            else:
                improvement[key] = 0.0
        rows.append(improvement)

        # Add Upper Bound row
        ub_row = {"method": "Upper Bound (Recall)"}
        for k in k_values:
             ub_row[f"Recall@{k}"] = avg_upper_bounds[f"Recall@{k}"]
        rows.append(ub_row)
        
        # Add Achievement Rate row (for Adjusted)
        ach_row = {"method": "Achievement Rate (%) (Adj vs UB)"}
        for k in k_values:
            ub = avg_upper_bounds[f"Recall@{k}"]
            adj = metrics["adjusted"][f"Recall@{k}"]
            ach_row[f"Recall@{k}"] = (adj / ub * 100) if ub > 0 else 0.0
        rows.append(ach_row)
        
        return pd.DataFrame(rows)
    
    def analyze_tier_distribution(self, adjusted_scores: Dict[str, List[Dict]]) -> Dict:
        """Analyze tier distribution across all jobs."""
        tier_counts = {'Essential': 0, 'Optional': 0, 'Irrelevant': 0}
        tier_in_gt = {'Essential': 0, 'Optional': 0, 'Irrelevant': 0}
        
        for job_id, skills in adjusted_scores.items():
            true_skills = self.ground_truth.get(job_id, set())
            
            for skill in skills:
                tier = skill['tier']
                tier_counts[tier] += 1
                if skill['skill_uri'] in true_skills:
                    tier_in_gt[tier] += 1
        
        total = sum(tier_counts.values())
        
        return {
            "tier_counts": tier_counts,
            "tier_percentages": {t: (c / total * 100) if total > 0 else 0 
                                 for t, c in tier_counts.items()},
            "tier_ground_truth_counts": tier_in_gt,
            "tier_precision": {t: (tier_in_gt[t] / tier_counts[t] * 100) if tier_counts[t] > 0 else 0
                               for t in tier_counts},
        }
    
    def run_grid_search(
        self,
        param_grid: Dict[str, List[float]],
        metric: str = "mAP"
    ):
        """Run grid search over tier parameters."""
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))
        
        logger.info(f"Starting grid search with {len(combinations)} combinations...")
        
        best_score = -1.0
        best_config = None
        results = []
        
        # Suppress detailed logs during grid search
        logger.disable("skill_mapping.v4.score_adjuster") # Temporarily disable logs? Or just don't log inside loop.
        # Better: just use print for progress bar or minimal logging.
        
        total_combos = len(combinations)
        
        for i, combo in enumerate(combinations):
            config_dict = dict(zip(keys, combo))
            
            # Create config mixing default/current with grid values
            current_vals = {
                "essential_base": self.tier_config.essential_base,
                "optional_base": self.tier_config.optional_base,
                "irrelevant_base": self.tier_config.irrelevant_base,
                "epsilon": self.tier_config.epsilon,
            }
            current_vals.update(config_dict)
            
            # Check constraint: essential >= optional >= irrelevant
            if not (current_vals["essential_base"] >= current_vals["optional_base"] >= current_vals["irrelevant_base"]):
                continue
            
            current_config = TierConfig(**current_vals)
            
            # Run adjustment (disable logging if possible, or just accept it)
            # Since adjust_scores logs "Adjusting scores...", we might want to silence it or make it quieter.
            # But we can't easily change logging level here without affecting everything.
            # We'll just run it.
            adj_scores, _ = self.adjust_scores(tier_config=current_config)
            
            # Evaluate
            df = self.evaluate_metrics(adj_scores)
            
            # Get metric for "LLM Adjusted"
            try:
                row = df[df["method"] == "LLM Adjusted"].iloc[0]
                score = float(row[metric])
                
                # Capture all metrics for this run
                run_metrics = row.to_dict()
                run_metrics.pop("method", None) # Remove 'method' column
                
                # Add config params
                full_result = config_dict.copy()
                full_result.update(run_metrics)
                results.append(full_result)
                
            except (IndexError, KeyError):
                logger.warning(f"Could not find {metric} in results for config {config_dict}")
                score = 0.0
            
            # Print progress every 10 or so
            if (i + 1) % 10 == 0 or (i + 1) == total_combos:
                print(f"Progress: {i+1}/{total_combos} | Best {metric}: {best_score:.4f}", end="\r")
            
            if score > best_score:
                best_score = score
                best_config = current_config
        
        print("") # Newline after progress
        
        logger.info(f"Grid search complete. Best {metric}: {best_score:.4f}")
        logger.info(f"Best config: {best_config}")
        
        # Save results to CSV
        if results:
            results_df = pd.DataFrame(results)
            # Sort by metric
            results_df = results_df.sort_values(by=metric, ascending=False)
            
            csv_path = self.output_dir / "grid_search_results.csv"
            results_df.to_csv(csv_path, index=False)
            logger.info(f"Grid search results saved to {csv_path}")
            
            # Print top 10
            print("\nTop 10 Configurations:")
            print(results_df.head(10).to_string(index=False))

        # Update self.tier_config to the best one
        if best_config:
            self.tier_config = best_config
            
        return best_config

    def run(self):
        """Run the full adjustment and evaluation pipeline."""
        logger.info("Starting score adjustment pipeline...")
        
        # Adjust scores
        adjusted_scores, mismatch_report = self.adjust_scores()
        
        # Print mismatch report
        mismatch_report.print_summary()
        
        # Evaluate metrics
        comparison_df = self.evaluate_metrics(adjusted_scores)
        
        # Print results
        logger.info("\n" + "=" * 80)
        logger.info("METRIC COMPARISON")
        logger.info("=" * 80)
        print(comparison_df.to_string(index=False))
        
        # Analyze tier distribution
        tier_stats = self.analyze_tier_distribution(adjusted_scores)
        
        logger.info("\n" + "=" * 80)
        logger.info("TIER DISTRIBUTION")
        logger.info("=" * 80)
        for tier in ['Essential', 'Optional', 'Irrelevant']:
            count = tier_stats['tier_counts'][tier]
            pct = tier_stats['tier_percentages'][tier]
            gt_count = tier_stats['tier_ground_truth_counts'][tier]
            precision = tier_stats['tier_precision'][tier]
            logger.info(f"  {tier}: {count} ({pct:.1f}%) - {gt_count} in ground truth ({precision:.1f}% precision)")
        
        # Save outputs
        self._save_outputs(adjusted_scores, mismatch_report, comparison_df, tier_stats)
        
        logger.info("\nPipeline completed!")
    
    def _save_outputs(
        self,
        adjusted_scores: Dict[str, List[Dict]],
        mismatch_report: MismatchReport,
        comparison_df: pd.DataFrame,
        tier_stats: Dict,
    ):
        """Save all outputs."""
        # Save adjusted scores (compact format)
        scores_file = self.output_dir / "adjusted_scores.json"
        with open(scores_file, 'w') as f:
            json.dump({
                "metadata": {
                    "tier_config": {
                        "essential_base": self.tier_config.essential_base,
                        "optional_base": self.tier_config.optional_base,
                        "irrelevant_base": self.tier_config.irrelevant_base,
                        "epsilon": self.tier_config.epsilon,
                    },
                    "n_jobs": len(adjusted_scores),
                },
                "scores": adjusted_scores,
            }, f, indent=2)
        logger.info(f"Adjusted scores saved to {scores_file}")
        
        # Save mismatch report
        mismatch_file = self.output_dir / "mismatch_report.json"
        with open(mismatch_file, 'w') as f:
            json.dump(mismatch_report.to_dict(), f, indent=2)
        logger.info(f"Mismatch report saved to {mismatch_file}")
        
        # Save comparison metrics
        metrics_file = self.output_dir / "metric_comparison.csv"
        comparison_df.to_csv(metrics_file, index=False)
        logger.info(f"Metric comparison saved to {metrics_file}")
        
        # Save tier statistics
        tier_file = self.output_dir / "tier_analysis.json"
        with open(tier_file, 'w') as f:
            json.dump(tier_stats, f, indent=2)
        logger.info(f"Tier analysis saved to {tier_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Adjust scores based on LLM tier classifications and evaluate metrics"
    )
    
    parser.add_argument("--llm_responses", type=Path, required=True,
                       help="Path to LLM responses (llm_responses.jsonl.gz)")
    parser.add_argument("--original_scores", type=Path, required=True,
                       help="Path to original fusion scores JSON")
    parser.add_argument("--ground_truth", type=Path, required=True,
                       help="Path to ground truth JSON")
    parser.add_argument("--output_dir", type=Path, required=True,
                       help="Output directory")
    
    # Tier configuration
    parser.add_argument("--essential_base", type=float, default=3.0)
    parser.add_argument("--optional_base", type=float, default=2.0)
    parser.add_argument("--irrelevant_base", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    
    # Grid search arguments
    parser.add_argument("--grid_search", action="store_true", help="Enable grid search")
    parser.add_argument("--essential_grid", type=float, nargs="+", help="Grid for essential_base")
    parser.add_argument("--optional_grid", type=float, nargs="+", help="Grid for optional_base")
    parser.add_argument("--irrelevant_grid", type=float, nargs="+", help="Grid for irrelevant_base")
    parser.add_argument("--epsilon_grid", type=float, nargs="+", help="Grid for epsilon")
    parser.add_argument("--metric", type=str, default="mAP", help="Metric to optimize (default: mAP)")
    
    args = parser.parse_args()
    
    tier_config = TierConfig(
        essential_base=args.essential_base,
        optional_base=args.optional_base,
        irrelevant_base=args.irrelevant_base,
        epsilon=args.epsilon,
    )
    
    adjuster = ScoreAdjuster(
        llm_responses_path=args.llm_responses,
        original_scores_path=args.original_scores,
        ground_truth_path=args.ground_truth,
        output_dir=args.output_dir,
        tier_config=tier_config,
    )
    
    if args.grid_search:
        # Define grid
        grid = {}
        if args.essential_grid: grid["essential_base"] = args.essential_grid
        if args.optional_grid: grid["optional_base"] = args.optional_grid
        if args.irrelevant_grid: grid["irrelevant_base"] = args.irrelevant_grid
        if args.epsilon_grid: grid["epsilon"] = args.epsilon_grid
        
        # If no specific grid provided, use defaults
        if not grid:
             grid = {
                 "essential_base": [3.0, 4.0, 5.0],
                 "optional_base": [2.0, 2.5, 3.0],
                 "irrelevant_base": [1.0, 1.2, 1.5],
                 "epsilon": [0.05, 0.1, 0.2]
             }
        
        adjuster.run_grid_search(grid, metric=args.metric)
    
    adjuster.run()


if __name__ == "__main__":
    main()

