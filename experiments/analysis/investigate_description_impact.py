"""
Investigation Script: Why Adding Descriptions Doesn't Help Metrics

This script analyzes three potential reasons why including job descriptions
in career path prediction might not improve (or even hurt) performance:

1. **Token Truncation**: Check if descriptions exceed the encoder's context length
2. **Semantic Redundancy**: Compare embeddings of title-only vs title+description
3. **Answer Leakage**: Check if target labels appear in input text (especially with subspans)

Additionally, it compares augmented data (subspans) vs clean data to understand
why metrics differ significantly between them.

Usage:
    python experiments/analysis/investigate_description_impact.py

Author: Investigation script for ablation study analysis
"""

import os
import sys
import re
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

try:
    from src.cpp.data_classes import Data
    from src.cpp.utils import SEP_TOKEN
except ImportError:
    # Fallback: try relative import from src
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
    from cpp.data_classes import Data
    from cpp.utils import SEP_TOKEN

# ============================================================================
# CONSTANTS
# ============================================================================

ENCODER_NAME = "ElenaSenger/career-path-representation-mpnet-karrierewege"
DATA_TYPE = "karrierewege_100k"
OUTPUT_DIR = Path("experiments/analysis/reports")


@dataclass
class AnalysisResult:
    """Container for analysis results."""
    name: str
    value: Any
    description: str


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data_pairs(data_type: str, consider_subspans: bool, load_clean_test: bool = False):
    """Load data pairs from the dataset."""
    data = Data(
        DATA_TYPE=data_type, 
        ONLY_TITLES=False,  # Load full descriptions
        LOAD_CLEAN_TEST=load_clean_test,
        consider_subspans=consider_subspans
    )
    
    train_pairs, val_pairs, test_pairs = data.get_data(stage='embedding_finetuning')
    
    if load_clean_test and data.test_pairs_clean:
        test_pairs_clean = data.test_pairs_clean
        return train_pairs, val_pairs, test_pairs, test_pairs_clean
    
    return train_pairs, val_pairs, test_pairs, None


def extract_titles_from_pairs(pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Extract just the titles from full document pairs."""
    title_pairs = []
    for doc1, doc2 in pairs:
        # Extract titles from doc1 (history)
        titles = re.findall(r"role: (.*?)\n", doc1)
        history_titles = SEP_TOKEN.join(titles)
        
        # Extract target title from doc2
        target_match = re.search(r"esco role: (.*?)\n", doc2)
        target_title = target_match.group(1) if target_match else ""
        
        title_pairs.append((history_titles, target_title))
    
    return title_pairs


# ============================================================================
# 1. TOKEN LENGTH ANALYSIS
# ============================================================================

def analyze_token_lengths(pairs: List[Tuple[str, str]], tokenizer, max_seq_length: int, 
                          name: str = "data") -> Dict[str, Any]:
    """
    Analyze token lengths and truncation rates.
    
    Returns:
        Dictionary with detailed statistics
    """
    print(f"\n{'='*60}")
    print(f"📊 TOKEN LENGTH ANALYSIS: {name}")
    print(f"{'='*60}")
    
    doc1_lengths = []
    doc2_lengths = []
    truncated_doc1 = 0
    truncated_doc2 = 0
    
    # Track how much is being truncated
    truncated_tokens_doc1 = []  # tokens lost per truncated sample
    truncated_tokens_doc2 = []
    
    for doc1, doc2 in tqdm(pairs, desc=f"Tokenizing {name}"):
        # Tokenize
        tokens1 = tokenizer.encode(doc1, add_special_tokens=True)
        tokens2 = tokenizer.encode(doc2, add_special_tokens=True)
        
        len1, len2 = len(tokens1), len(tokens2)
        doc1_lengths.append(len1)
        doc2_lengths.append(len2)
        
        if len1 > max_seq_length:
            truncated_doc1 += 1
            truncated_tokens_doc1.append(len1 - max_seq_length)
        
        if len2 > max_seq_length:
            truncated_doc2 += 1
            truncated_tokens_doc2.append(len2 - max_seq_length)
    
    n_samples = len(pairs)
    
    # Compute statistics
    results = {
        "name": name,
        "n_samples": n_samples,
        "max_seq_length": max_seq_length,
        "doc1": {
            "mean": np.mean(doc1_lengths),
            "median": np.median(doc1_lengths),
            "std": np.std(doc1_lengths),
            "min": np.min(doc1_lengths),
            "max": np.max(doc1_lengths),
            "p90": np.percentile(doc1_lengths, 90),
            "p95": np.percentile(doc1_lengths, 95),
            "p99": np.percentile(doc1_lengths, 99),
            "truncated_count": truncated_doc1,
            "truncated_pct": 100 * truncated_doc1 / n_samples,
            "avg_tokens_lost": np.mean(truncated_tokens_doc1) if truncated_tokens_doc1 else 0,
            "max_tokens_lost": np.max(truncated_tokens_doc1) if truncated_tokens_doc1 else 0,
        },
        "doc2": {
            "mean": np.mean(doc2_lengths),
            "median": np.median(doc2_lengths),
            "std": np.std(doc2_lengths),
            "min": np.min(doc2_lengths),
            "max": np.max(doc2_lengths),
            "p90": np.percentile(doc2_lengths, 90),
            "p95": np.percentile(doc2_lengths, 95),
            "p99": np.percentile(doc2_lengths, 99),
            "truncated_count": truncated_doc2,
            "truncated_pct": 100 * truncated_doc2 / n_samples,
            "avg_tokens_lost": np.mean(truncated_tokens_doc2) if truncated_tokens_doc2 else 0,
            "max_tokens_lost": np.max(truncated_tokens_doc2) if truncated_tokens_doc2 else 0,
        }
    }
    
    # Print results
    print(f"\nEncoder max_seq_length: {max_seq_length}")
    print(f"Total samples: {n_samples:,}")
    
    print(f"\n📄 Input (doc1/history) statistics:")
    print(f"   Mean tokens:   {results['doc1']['mean']:.1f} ± {results['doc1']['std']:.1f}")
    print(f"   Median tokens: {results['doc1']['median']:.1f}")
    print(f"   Range:         [{results['doc1']['min']}, {results['doc1']['max']}]")
    print(f"   Percentiles:   p90={results['doc1']['p90']:.0f}, p95={results['doc1']['p95']:.0f}, p99={results['doc1']['p99']:.0f}")
    print(f"   🔴 Truncated:  {truncated_doc1:,} ({results['doc1']['truncated_pct']:.2f}%)")
    if truncated_doc1 > 0:
        print(f"   Avg tokens lost per truncated sample: {results['doc1']['avg_tokens_lost']:.1f}")
        print(f"   Max tokens lost: {results['doc1']['max_tokens_lost']}")
    
    print(f"\n🎯 Target (doc2) statistics:")
    print(f"   Mean tokens:   {results['doc2']['mean']:.1f} ± {results['doc2']['std']:.1f}")
    print(f"   Median tokens: {results['doc2']['median']:.1f}")
    print(f"   Range:         [{results['doc2']['min']}, {results['doc2']['max']}]")
    print(f"   Percentiles:   p90={results['doc2']['p90']:.0f}, p95={results['doc2']['p95']:.0f}, p99={results['doc2']['p99']:.0f}")
    print(f"   🔴 Truncated:  {truncated_doc2:,} ({results['doc2']['truncated_pct']:.2f}%)")
    
    return results


# ============================================================================
# 2. SEMANTIC REDUNDANCY ANALYSIS
# ============================================================================

def analyze_semantic_redundancy(full_pairs: List[Tuple[str, str]], 
                                title_pairs: List[Tuple[str, str]],
                                encoder, 
                                sample_size: int = 10000) -> Dict[str, Any]:
    """
    Compare cosine similarity between title-only and title+description embeddings.
    
    High similarity = descriptions are redundant (don't add new semantic info)
    Low similarity = descriptions add significant semantic information
    """
    print(f"\n{'='*60}")
    print(f"🔬 SEMANTIC REDUNDANCY ANALYSIS")
    print(f"{'='*60}")
    
    # Sample if too large
    if len(full_pairs) > sample_size:
        indices = np.random.choice(len(full_pairs), sample_size, replace=False)
        full_pairs = [full_pairs[i] for i in indices]
        title_pairs = [title_pairs[i] for i in indices]
        print(f"Sampling {sample_size:,} pairs for analysis")
    
    # Extract doc1 (history) from both
    full_histories = [p[0] for p in full_pairs]
    title_histories = [p[0] for p in title_pairs]
    
    # Extract doc2 (target) from both
    full_targets = [p[1] for p in full_pairs]
    title_targets = [p[1] for p in title_pairs]
    
    print("\nEncoding history texts (title+desc)...")
    full_history_emb = encoder.encode(full_histories, convert_to_numpy=True, 
                                       show_progress_bar=True, batch_size=256)
    
    print("Encoding history texts (title only)...")
    title_history_emb = encoder.encode(title_histories, convert_to_numpy=True, 
                                        show_progress_bar=True, batch_size=256)
    
    print("Encoding target texts (title+desc)...")
    full_target_emb = encoder.encode(full_targets, convert_to_numpy=True, 
                                      show_progress_bar=True, batch_size=256)
    
    print("Encoding target texts (title only)...")
    title_target_emb = encoder.encode(title_targets, convert_to_numpy=True, 
                                       show_progress_bar=True, batch_size=256)
    
    # Compute cosine similarities
    def cosine_sim_batch(A, B):
        """Compute row-wise cosine similarity between two matrices."""
        A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)
        B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
        return np.sum(A_norm * B_norm, axis=1)
    
    # Similarity between title-only and title+desc for history
    history_sim = cosine_sim_batch(full_history_emb, title_history_emb)
    
    # Similarity between title-only and title+desc for target
    target_sim = cosine_sim_batch(full_target_emb, title_target_emb)
    
    results = {
        "n_samples": len(full_pairs),
        "history_similarity": {
            "mean": float(np.mean(history_sim)),
            "median": float(np.median(history_sim)),
            "std": float(np.std(history_sim)),
            "min": float(np.min(history_sim)),
            "max": float(np.max(history_sim)),
            "p10": float(np.percentile(history_sim, 10)),
            "p25": float(np.percentile(history_sim, 25)),
        },
        "target_similarity": {
            "mean": float(np.mean(target_sim)),
            "median": float(np.median(target_sim)),
            "std": float(np.std(target_sim)),
            "min": float(np.min(target_sim)),
            "max": float(np.max(target_sim)),
            "p10": float(np.percentile(target_sim, 10)),
            "p25": float(np.percentile(target_sim, 25)),
        }
    }
    
    print(f"\n📊 Cosine Similarity: Title-only vs Title+Description")
    print(f"\n   HISTORY (doc1):")
    print(f"   Mean similarity:   {results['history_similarity']['mean']:.4f}")
    print(f"   Median similarity: {results['history_similarity']['median']:.4f}")
    print(f"   Std:               {results['history_similarity']['std']:.4f}")
    print(f"   Range:             [{results['history_similarity']['min']:.4f}, {results['history_similarity']['max']:.4f}]")
    
    print(f"\n   TARGET (doc2):")
    print(f"   Mean similarity:   {results['target_similarity']['mean']:.4f}")
    print(f"   Median similarity: {results['target_similarity']['median']:.4f}")
    print(f"   Std:               {results['target_similarity']['std']:.4f}")
    print(f"   Range:             [{results['target_similarity']['min']:.4f}, {results['target_similarity']['max']:.4f}]")
    
    # Interpretation
    print(f"\n💡 INTERPRETATION:")
    if results['history_similarity']['mean'] > 0.9:
        print(f"   ⚠️  HIGH redundancy: Descriptions add very little semantic information")
        print(f"      (mean cosine sim = {results['history_similarity']['mean']:.4f})")
    elif results['history_similarity']['mean'] > 0.8:
        print(f"   ⚡ MODERATE redundancy: Descriptions add some information")
        print(f"      (mean cosine sim = {results['history_similarity']['mean']:.4f})")
    else:
        print(f"   ✅ LOW redundancy: Descriptions add significant semantic information")
        print(f"      (mean cosine sim = {results['history_similarity']['mean']:.4f})")
    
    return results


# ============================================================================
# 3. ANSWER LEAKAGE ANALYSIS
# ============================================================================

def extract_target_title(doc2: str) -> str:
    """Extract just the target job title from doc2."""
    match = re.search(r"esco role: (.*?)\n", doc2)
    return match.group(1).strip().lower() if match else ""


def extract_history_titles(doc1: str) -> List[str]:
    """Extract all job titles from the history."""
    return [t.strip().lower() for t in re.findall(r"role: (.*?)\n", doc1)]


def analyze_answer_leakage(pairs: List[Tuple[str, str]], name: str = "data") -> Dict[str, Any]:
    """
    Analyze if the target label appears in the input text ("answer leakage").
    
    This is especially relevant for subspan-augmented data where shorter
    subsequences might have the target appearing in the history.
    """
    print(f"\n{'='*60}")
    print(f"🔍 ANSWER LEAKAGE ANALYSIS: {name}")
    print(f"{'='*60}")
    
    direct_leaks = []  # Target title appears exactly in history
    partial_leaks = []  # Target appears anywhere in input text
    history_lengths = []
    leaky_examples = []
    
    for i, (doc1, doc2) in enumerate(tqdm(pairs, desc="Checking leakage")):
        target_title = extract_target_title(doc2)
        history_titles = extract_history_titles(doc1)
        history_length = len(history_titles)
        history_lengths.append(history_length)
        
        # Direct title-level leakage
        direct_leak = target_title in history_titles
        direct_leaks.append(direct_leak)
        
        # Broader text-level leakage
        partial_leak = target_title in doc1.lower()
        partial_leaks.append(partial_leak)
        
        # Collect examples
        if direct_leak and len(leaky_examples) < 10:
            leaky_examples.append({
                "target": target_title,
                "history_titles": history_titles,
                "history_length": history_length,
            })
    
    n_samples = len(pairs)
    n_direct_leaks = sum(direct_leaks)
    n_partial_leaks = sum(partial_leaks)
    
    # Analyze leakage by history length
    leak_by_length = {}
    for length, leak in zip(history_lengths, direct_leaks):
        if length not in leak_by_length:
            leak_by_length[length] = {"total": 0, "leaked": 0}
        leak_by_length[length]["total"] += 1
        if leak:
            leak_by_length[length]["leaked"] += 1
    
    results = {
        "name": name,
        "n_samples": n_samples,
        "direct_leakage": {
            "count": n_direct_leaks,
            "rate": n_direct_leaks / n_samples,
            "percentage": 100 * n_direct_leaks / n_samples,
        },
        "partial_leakage": {
            "count": n_partial_leaks,
            "rate": n_partial_leaks / n_samples,
            "percentage": 100 * n_partial_leaks / n_samples,
        },
        "history_length_stats": {
            "mean": np.mean(history_lengths),
            "median": np.median(history_lengths),
            "min": int(np.min(history_lengths)),
            "max": int(np.max(history_lengths)),
        },
        "leak_by_length": {k: v for k, v in sorted(leak_by_length.items())[:10]},
        "examples": leaky_examples,
    }
    
    print(f"\nTotal samples: {n_samples:,}")
    print(f"\n🚨 DIRECT LEAKAGE (target title in history):")
    print(f"   Leaked samples: {n_direct_leaks:,} ({results['direct_leakage']['percentage']:.2f}%)")
    
    print(f"\n📝 PARTIAL LEAKAGE (target string anywhere in input):")
    print(f"   Leaked samples: {n_partial_leaks:,} ({results['partial_leakage']['percentage']:.2f}%)")
    
    print(f"\n📊 History length statistics:")
    print(f"   Mean:   {results['history_length_stats']['mean']:.2f}")
    print(f"   Median: {results['history_length_stats']['median']:.0f}")
    print(f"   Range:  [{results['history_length_stats']['min']}, {results['history_length_stats']['max']}]")
    
    print(f"\n📈 Leakage rate by history length (first 10):")
    for length, stats in sorted(leak_by_length.items())[:10]:
        rate = 100 * stats["leaked"] / stats["total"]
        print(f"   Length {length}: {stats['leaked']}/{stats['total']} ({rate:.1f}%)")
    
    if leaky_examples:
        print(f"\n🔎 Example leaked samples:")
        for i, ex in enumerate(leaky_examples[:3]):
            print(f"\n   Example {i+1}:")
            print(f"   Target: '{ex['target']}'")
            print(f"   History ({ex['history_length']} jobs): {ex['history_titles']}")
    
    return results


# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def generate_report(results: Dict[str, Any], output_path: Path):
    """Generate a comprehensive markdown report."""
    
    report = []
    report.append("# Investigation Report: Why Adding Descriptions Doesn't Help")
    report.append("")
    report.append(f"**Dataset:** {DATA_TYPE}")
    report.append(f"**Encoder:** {ENCODER_NAME}")
    report.append(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Executive Summary
    report.append("## Executive Summary")
    report.append("")
    
    # Token Length Summary
    if "token_analysis" in results:
        ta = results["token_analysis"]
        full_trunc = ta["with_desc"]["doc1"]["truncated_pct"]
        title_trunc = ta["title_only"]["doc1"]["truncated_pct"]
        
        report.append(f"### 1. Token Truncation")
        report.append(f"- **Encoder max_seq_length:** {ta['with_desc']['max_seq_length']}")
        report.append(f"- **Title+Description truncation rate:** {full_trunc:.2f}%")
        report.append(f"- **Title-only truncation rate:** {title_trunc:.2f}%")
        
        if full_trunc > 50:
            report.append(f"- **⚠️ CRITICAL:** Over half of samples are truncated with descriptions!")
        elif full_trunc > 20:
            report.append(f"- **⚡ Warning:** Significant truncation occurring with descriptions")
        report.append("")
    
    # Semantic Redundancy Summary
    if "semantic_redundancy" in results:
        sr = results["semantic_redundancy"]
        sim = sr["history_similarity"]["mean"]
        
        report.append(f"### 2. Semantic Redundancy")
        report.append(f"- **Mean cosine similarity (title vs title+desc):** {sim:.4f}")
        
        if sim > 0.95:
            report.append(f"- **⚠️ VERY HIGH redundancy:** Descriptions add almost no new information")
        elif sim > 0.90:
            report.append(f"- **⚠️ HIGH redundancy:** Descriptions add very little new information")
        elif sim > 0.80:
            report.append(f"- **⚡ Moderate redundancy:** Descriptions add some information")
        else:
            report.append(f"- **✅ Low redundancy:** Descriptions add meaningful information")
        report.append("")
    
    # Answer Leakage Summary
    if "leakage_analysis" in results:
        la = results["leakage_analysis"]
        subspan_leak = la.get("with_subspans", {}).get("direct_leakage", {}).get("percentage", 0)
        clean_leak = la.get("clean_test", {}).get("direct_leakage", {}).get("percentage", 0)
        
        report.append(f"### 3. Answer Leakage")
        report.append(f"- **Subspan data leakage rate:** {subspan_leak:.2f}%")
        report.append(f"- **Clean test data leakage rate:** {clean_leak:.2f}%")
        
        if subspan_leak > 30:
            report.append(f"- **⚠️ CRITICAL:** High answer leakage in subspan data!")
            report.append(f"  - This explains the large gap between subspan and clean test performance")
        report.append("")
    
    # Detailed Analysis Sections
    report.append("---")
    report.append("")
    report.append("## Detailed Analysis")
    report.append("")
    
    # Token Analysis Details
    if "token_analysis" in results:
        report.append("### Token Length Analysis")
        report.append("")
        report.append("| Metric | Title+Desc (doc1) | Title-only (doc1) |")
        report.append("|--------|-------------------|-------------------|")
        
        ta = results["token_analysis"]
        td = ta["with_desc"]["doc1"]
        to = ta["title_only"]["doc1"]
        
        report.append(f"| Mean tokens | {td['mean']:.1f} | {to['mean']:.1f} |")
        report.append(f"| Median tokens | {td['median']:.1f} | {to['median']:.1f} |")
        report.append(f"| P95 tokens | {td['p95']:.0f} | {to['p95']:.0f} |")
        report.append(f"| Max tokens | {td['max']} | {to['max']} |")
        report.append(f"| Truncated % | {td['truncated_pct']:.2f}% | {to['truncated_pct']:.2f}% |")
        report.append(f"| Avg tokens lost | {td['avg_tokens_lost']:.1f} | {to['avg_tokens_lost']:.1f} |")
        report.append("")
    
    # Leakage Details
    if "leakage_analysis" in results:
        report.append("### Leakage by History Length")
        report.append("")
        
        la = results["leakage_analysis"]
        if "with_subspans" in la:
            leak_by_len = la["with_subspans"].get("leak_by_length", {})
            if leak_by_len:
                report.append("| History Length | Total Samples | Leaked | Leak Rate |")
                report.append("|----------------|---------------|--------|-----------|")
                for length, stats in sorted(leak_by_len.items())[:10]:
                    rate = 100 * stats["leaked"] / stats["total"]
                    report.append(f"| {length} | {stats['total']:,} | {stats['leaked']:,} | {rate:.1f}% |")
                report.append("")
    
    # Conclusions
    report.append("---")
    report.append("")
    report.append("## Conclusions and Recommendations")
    report.append("")
    
    conclusions = []
    
    if "token_analysis" in results:
        ta = results["token_analysis"]
        full_trunc = ta["with_desc"]["doc1"]["truncated_pct"]
        if full_trunc > 20:
            conclusions.append(
                f"1. **Token truncation is significant ({full_trunc:.1f}%)**: With descriptions, "
                f"many samples exceed the encoder's context window. This means valuable information "
                f"at the end of long career histories is being lost."
            )
    
    if "semantic_redundancy" in results:
        sim = results["semantic_redundancy"]["history_similarity"]["mean"]
        if sim > 0.85:
            conclusions.append(
                f"2. **Descriptions are semantically redundant (sim={sim:.4f})**: The encoder "
                f"already captures most of the semantic content from job titles alone. "
                f"Descriptions mostly add noise rather than signal."
            )
    
    if "leakage_analysis" in results:
        la = results["leakage_analysis"]
        subspan_leak = la.get("with_subspans", {}).get("direct_leakage", {}).get("percentage", 0)
        clean_leak = la.get("clean_test", {}).get("direct_leakage", {}).get("percentage", 0)
        if subspan_leak > 5:
            conclusions.append(
                f"3. **Answer leakage explains performance gap**: Subspan augmentation causes "
                f"{subspan_leak:.1f}% leakage (vs {clean_leak:.1f}% in clean test). "
                f"The model learns to exploit this pattern, inflating subspan test metrics."
            )
    
    for c in conclusions:
        report.append(c)
        report.append("")
    
    # Save report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\n📄 Report saved to: {output_path}")
    
    return '\n'.join(report)


def main():
    """Main analysis function."""
    print("="*70)
    print("INVESTIGATION: Why Adding Descriptions Doesn't Help Metrics")
    print("="*70)
    
    # Results container
    results = {}
    
    # Load encoder
    print(f"\n📦 Loading encoder: {ENCODER_NAME}")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(ENCODER_NAME)
    max_seq_length = encoder.max_seq_length
    tokenizer = encoder.tokenizer
    
    print(f"   Encoder max_seq_length: {max_seq_length}")
    print(f"   Embedding dimension: {encoder.get_sentence_embedding_dimension()}")
    
    # =========================================================================
    # Load Data
    # =========================================================================
    print(f"\n📂 Loading data: {DATA_TYPE}")
    
    # Load with subspans (augmented)
    print("   Loading augmented data (with subspans)...")
    train_subspan, val_subspan, test_subspan, _ = load_data_pairs(
        DATA_TYPE, consider_subspans=True, load_clean_test=False
    )
    print(f"   Train: {len(train_subspan):,}, Val: {len(val_subspan):,}, Test: {len(test_subspan):,}")
    
    # Load clean test (no subspans)
    print("   Loading clean data (no subspans)...")
    train_clean, val_clean, test_clean, _ = load_data_pairs(
        DATA_TYPE, consider_subspans=False, load_clean_test=False
    )
    print(f"   Train: {len(train_clean):,}, Val: {len(val_clean):,}, Test: {len(test_clean):,}")
    
    # Combine train+val for analysis
    all_subspan = train_subspan + val_subspan
    all_clean = train_clean + val_clean
    
    # Extract title-only versions
    print("   Extracting title-only pairs...")
    title_only_subspan = extract_titles_from_pairs(all_subspan)
    title_only_clean = extract_titles_from_pairs(all_clean)
    
    # =========================================================================
    # 1. TOKEN LENGTH ANALYSIS
    # =========================================================================
    print("\n" + "="*70)
    print("ANALYSIS 1: TOKEN LENGTH & TRUNCATION")
    print("="*70)
    
    # Sample for efficiency
    sample_size = min(50000, len(all_subspan))
    indices = np.random.choice(len(all_subspan), sample_size, replace=False)
    sampled_full = [all_subspan[i] for i in indices]
    sampled_title = [title_only_subspan[i] for i in indices]
    
    # Analyze with descriptions
    token_results_full = analyze_token_lengths(
        sampled_full, tokenizer, max_seq_length, "Title + Description"
    )
    
    # Analyze title-only
    token_results_title = analyze_token_lengths(
        sampled_title, tokenizer, max_seq_length, "Title Only"
    )
    
    results["token_analysis"] = {
        "with_desc": token_results_full,
        "title_only": token_results_title,
    }
    
    # =========================================================================
    # 2. SEMANTIC REDUNDANCY ANALYSIS
    # =========================================================================
    print("\n" + "="*70)
    print("ANALYSIS 2: SEMANTIC REDUNDANCY")
    print("="*70)
    
    semantic_results = analyze_semantic_redundancy(
        sampled_full, sampled_title, encoder, sample_size=10000
    )
    results["semantic_redundancy"] = semantic_results
    
    # =========================================================================
    # 3. ANSWER LEAKAGE ANALYSIS
    # =========================================================================
    print("\n" + "="*70)
    print("ANALYSIS 3: ANSWER LEAKAGE")
    print("="*70)
    
    # Analyze subspan data
    leakage_subspan = analyze_answer_leakage(all_subspan, "Subspan-Augmented Data")
    
    # Analyze clean test
    leakage_clean = analyze_answer_leakage(test_clean, "Clean Test Data")
    
    results["leakage_analysis"] = {
        "with_subspans": leakage_subspan,
        "clean_test": leakage_clean,
    }
    
    # =========================================================================
    # GENERATE REPORT
    # =========================================================================
    print("\n" + "="*70)
    print("GENERATING REPORT")
    print("="*70)
    
    output_path = OUTPUT_DIR / "description_impact_investigation.md"
    report = generate_report(results, output_path)
    
    # Save raw results as JSON
    json_path = OUTPUT_DIR / "description_impact_results.json"
    
    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj
    
    with open(json_path, 'w') as f:
        json.dump(convert_to_serializable(results), f, indent=2)
    print(f"📊 Raw results saved to: {json_path}")
    
    # Print final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n1. TOKEN TRUNCATION:")
    print(f"   - With descriptions: {results['token_analysis']['with_desc']['doc1']['truncated_pct']:.2f}% truncated")
    print(f"   - Title-only: {results['token_analysis']['title_only']['doc1']['truncated_pct']:.2f}% truncated")
    
    print(f"\n2. SEMANTIC REDUNDANCY:")
    print(f"   - Mean cosine similarity: {results['semantic_redundancy']['history_similarity']['mean']:.4f}")
    
    print(f"\n3. ANSWER LEAKAGE:")
    print(f"   - Subspan data: {results['leakage_analysis']['with_subspans']['direct_leakage']['percentage']:.2f}%")
    print(f"   - Clean test: {results['leakage_analysis']['clean_test']['direct_leakage']['percentage']:.2f}%")
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()





