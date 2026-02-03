import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from loguru import logger

# Configure logging
logger.remove()
logger.add(
    "logs/debug_truncation.log",
    format="{time} | {level} | {message}",
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    enqueue=True
)
logger.add(
    sys.stdout,
    format="<green>{time}</green> | <level>{message}</level>",
    level="INFO"
)

# Adjust path to import modules from skills4cpp
# Assuming this script is at skills4cpp/src/cpp/helpers/
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from src.cpp.data_classes import Data
    from src.cpp.utils import SEP_TOKEN
    import re
except ImportError as e:
    logger.error(f"Could not import Data class. Ensure you are in the project root. Error: {e}")
    sys.exit(1)

# Report output directory
REPORTS_DIR = project_root / "experiments" / "analysis" / "reports"

def extract_last_job_from_doc1(doc1: str) -> dict:
    """
    Extract the last job (role and description) from doc1.
    doc1 format: "role: X \n description: Y <SEP> role: Z \n description: W ..."
    Returns dict with 'role' and 'description' keys.
    """
    # Split by SEP_TOKEN to get individual jobs
    jobs = doc1.split(SEP_TOKEN)
    if not jobs:
        return {"role": None, "description": None}
    
    last_job = jobs[-1].strip()
    
    # Extract role and description using regex
    role_match = re.search(r"role:\s*(.+?)\s*\n", last_job)
    desc_match = re.search(r"description:\s*(.+?)$", last_job, re.DOTALL)
    
    role = role_match.group(1).strip() if role_match else None
    description = desc_match.group(1).strip() if desc_match else None
    
    return {"role": role, "description": description}


def extract_target_job_from_doc2(doc2: str) -> dict:
    """
    Extract the target job (esco role and description) from doc2.
    doc2 format: "esco role: X \n description: Y"
    Returns dict with 'role' and 'description' keys.
    """
    role_match = re.search(r"esco role:\s*(.+?)\s*\n", doc2)
    desc_match = re.search(r"description:\s*(.+?)$", doc2, re.DOTALL)
    
    role = role_match.group(1).strip() if role_match else None
    description = desc_match.group(1).strip() if desc_match else None
    
    return {"role": role, "description": description}


def analyze_last_job_equals_target(name: str, pairs: list) -> dict:
    """
    Analyze how many samples have the last job in doc1 equal to doc2 (target).
    This checks if the training signal is "predict the current job" vs "predict next job".
    """
    logger.info(f"Analyzing last-job-equals-target for {name} set...")
    
    if len(pairs) == 0:
        return {"total": 0, "matches": 0, "match_rate": 0.0}
    
    matches = 0
    role_matches = 0
    match_samples = []
    non_match_samples = []
    
    for doc1, doc2 in tqdm(pairs, desc=f"Checking {name}"):
        last_job = extract_last_job_from_doc1(doc1)
        target_job = extract_target_job_from_doc2(doc2)
        
        # Check if roles match (case-insensitive comparison)
        if last_job["role"] and target_job["role"]:
            if last_job["role"].lower() == target_job["role"].lower():
                role_matches += 1
                matches += 1
                if len(match_samples) < 2:
                    match_samples.append((last_job, target_job, doc1, doc2))
            else:
                if len(non_match_samples) < 2:
                    non_match_samples.append((last_job, target_job, doc1, doc2))
    
    match_rate = (matches / len(pairs)) * 100 if len(pairs) > 0 else 0
    
    logger.info(f"  Last job == Target: {matches}/{len(pairs)} ({match_rate:.2f}%)")
    
    # Print sample matches
    if match_samples:
        logger.info(f"\n  Sample MATCHES (last job == target):")
        for i, (last_job, target_job, doc1, doc2) in enumerate(match_samples, 1):
            logger.info(f"    Match {i}:")
            logger.info(f"      Last job role: {last_job['role']}")
            logger.info(f"      Target role: {target_job['role']}")
            logger.info(f"      Doc1 (truncated): {doc1[:200]}...")
            logger.info(f"      Doc2: {doc2[:200]}...")
    
    # Print sample non-matches
    if non_match_samples:
        logger.info(f"\n  Sample NON-MATCHES (last job != target):")
        for i, (last_job, target_job, doc1, doc2) in enumerate(non_match_samples, 1):
            logger.info(f"    Non-match {i}:")
            logger.info(f"      Last job role: {last_job['role']}")
            logger.info(f"      Target role: {target_job['role']}")
            logger.info(f"      Doc1 (truncated): {doc1[:200]}...")
            logger.info(f"      Doc2: {doc2[:200]}...")
    
    return {
        "total": len(pairs),
        "matches": matches,
        "match_rate": match_rate
    }


def analyze_split(name, pairs, tokenizer, context_lengths):
    """Analyze truncation for a specific data split."""
    logger.info(f"Analyzing {name} set ({len(pairs)} pairs)...")
    
    if len(pairs) == 0:
        logger.info(f"  Skipping {name} set (empty).")
        logger.info("-" * 40)
        return {"truncation": {}, "last_job_analysis": {}}
    
    doc_lengths = []
    
    # 1. Compute all true lengths once
    for doc1, _ in tqdm(pairs, desc=f"Tokenizing {name}"):
        # Tokenize without truncation to get true length
        # verbose=False to avoid clutter
        tokens = tokenizer.encode(doc1, add_special_tokens=True)
        doc_lengths.append(len(tokens))
    
    doc_lengths = np.array(doc_lengths)
    total_tokens = np.sum(doc_lengths)
    
    # 2. Analyze against thresholds
    truncation_results = {}
    for max_len in context_lengths:
        truncated_mask = doc_lengths > max_len
        num_truncated = np.sum(truncated_mask)
        truncation_rate = (num_truncated / len(pairs)) * 100
        
        # Calculate tokens lost
        # For truncated docs: lost = length - max_len
        # For non-truncated: lost = 0
        tokens_lost = np.sum(doc_lengths[truncated_mask] - max_len)
        
        avg_tokens_lost_per_truncated = tokens_lost / num_truncated if num_truncated > 0 else 0
        total_percent_lost = (tokens_lost / total_tokens) * 100 if total_tokens > 0 else 0
        
        logger.info(f"  [Max Len: {max_len}]")
        logger.info(f"    Truncated: {num_truncated}/{len(pairs)} ({truncation_rate:.2f}%)")
        logger.info(f"    Avg tokens lost (truncated): {avg_tokens_lost_per_truncated:.1f}")
        logger.info(f"    Total tokens lost: {tokens_lost} ({total_percent_lost:.2f}%)")
        
        truncation_results[max_len] = {
            "truncated_count": int(num_truncated),
            "truncation_rate": float(truncation_rate),
            "avg_lost_truncated": float(avg_tokens_lost_per_truncated),
            "total_percent_lost": float(total_percent_lost)
        }
    
    # 3. Analyze last job == target
    last_job_analysis = analyze_last_job_equals_target(name, pairs)
    
    # 4. Additional statistics
    stats = {
        "num_pairs": len(pairs),
        "doc_length_stats": {
            "min": int(np.min(doc_lengths)),
            "max": int(np.max(doc_lengths)),
            "mean": float(np.mean(doc_lengths)),
            "median": float(np.median(doc_lengths)),
            "std": float(np.std(doc_lengths)),
            "percentile_25": float(np.percentile(doc_lengths, 25)),
            "percentile_75": float(np.percentile(doc_lengths, 75)),
            "percentile_90": float(np.percentile(doc_lengths, 90)),
            "percentile_95": float(np.percentile(doc_lengths, 95)),
            "percentile_99": float(np.percentile(doc_lengths, 99)),
        },
        "total_tokens": int(total_tokens)
    }
    
    logger.info("-" * 40)
    return {
        "truncation": truncation_results,
        "last_job_analysis": last_job_analysis,
        "statistics": stats
    }

def save_report(report: dict, report_name: str = "truncation_impact_report"):
    """Save the report as both JSON and Markdown files."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save JSON report
    json_path = REPORTS_DIR / f"{report_name}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"JSON report saved to: {json_path}")
    
    # Generate and save Markdown report
    md_path = REPORTS_DIR / f"{report_name}.md"
    md_content = generate_markdown_report(report)
    with open(md_path, "w") as f:
        f.write(md_content)
    logger.info(f"Markdown report saved to: {md_path}")


def generate_markdown_report(report: dict) -> str:
    """Generate a human-readable Markdown report."""
    lines = []
    lines.append("# Truncation Impact Analysis Report\n")
    lines.append(f"**Generated:** {report['metadata']['timestamp']}\n")
    
    # Metadata section
    lines.append("## Metadata\n")
    meta = report['metadata']
    lines.append(f"- **Encoder Model:** `{meta['encoder_model']}`")
    lines.append(f"- **Model Max Sequence Length:** {meta['model_max_seq_length']}")
    lines.append(f"- **Data Type:** {meta['data_type']}")
    lines.append(f"- **Only Titles:** {meta['only_titles']}")
    lines.append(f"- **Consider Subspans:** {meta['consider_subspans']}")
    lines.append(f"- **Context Lengths Tested:** {meta['context_lengths_tested']}")
    lines.append("")
    
    # Dataset overview
    lines.append("## Dataset Overview\n")
    lines.append("| Split | Samples |")
    lines.append("|-------|---------|")
    total = 0
    for split_name in ["train", "validation", "test"]:
        if split_name in report["results"]:
            n = report["results"][split_name]["statistics"]["num_pairs"]
            total += n
            lines.append(f"| {split_name.capitalize()} | {n:,} |")
    lines.append(f"| **Total** | **{total:,}** |")
    lines.append("")
    
    # Last Job == Target Analysis
    lines.append("## Last Job Equals Target Analysis\n")
    lines.append("This checks how many samples have the last job in the career history (doc1) equal to the target job (doc2).\n")
    lines.append("| Split | Matches | Total | Match Rate |")
    lines.append("|-------|---------|-------|------------|")
    for split_name in ["train", "validation", "test"]:
        if split_name in report["results"]:
            lj = report["results"][split_name]["last_job_analysis"]
            lines.append(f"| {split_name.capitalize()} | {lj['matches']:,} | {lj['total']:,} | {lj['match_rate']:.2f}% |")
    lines.append("")
    
    # Truncation analysis per split
    lines.append("## Truncation Analysis\n")
    for split_name in ["train", "validation", "test"]:
        if split_name not in report["results"]:
            continue
        
        split_data = report["results"][split_name]
        lines.append(f"### {split_name.capitalize()} Set\n")
        
        # Doc length statistics
        stats = split_data["statistics"]["doc_length_stats"]
        lines.append("**Token Length Statistics:**\n")
        lines.append(f"- Min: {stats['min']}, Max: {stats['max']}")
        lines.append(f"- Mean: {stats['mean']:.1f} ± {stats['std']:.1f}")
        lines.append(f"- Median: {stats['median']:.1f}")
        lines.append(f"- Percentiles: 25th={stats['percentile_25']:.0f}, 75th={stats['percentile_75']:.0f}, 90th={stats['percentile_90']:.0f}, 95th={stats['percentile_95']:.0f}, 99th={stats['percentile_99']:.0f}")
        lines.append("")
        
        # Truncation table
        lines.append("**Truncation by Context Length:**\n")
        lines.append("| Max Length | Truncated | Rate | Avg Tokens Lost | Total % Lost |")
        lines.append("|------------|-----------|------|-----------------|--------------|")
        for max_len, trunc_data in split_data["truncation"].items():
            lines.append(f"| {max_len} | {trunc_data['truncated_count']:,} | {trunc_data['truncation_rate']:.2f}% | {trunc_data['avg_lost_truncated']:.1f} | {trunc_data['total_percent_lost']:.2f}% |")
        lines.append("")
    
    return "\n".join(lines)


def analyze_truncation_impact(config_name: str, encoder_text_model_name: str = None):
    logger.info(f"Starting truncation impact analysis.")

    if not encoder_text_model_name:
        embedding_model_name = "ElenaSenger/career-path-representation-mpnet-decorte"
        logger.info(f"No encoder model name provided. Using default: {embedding_model_name}")
    else:
        embedding_model_name = encoder_text_model_name
        logger.info(f"Using specified encoder model: {embedding_model_name}")

    logger.info(f"Initializing SentenceTransformer model: {embedding_model_name}...")
    try:
        model = SentenceTransformer(embedding_model_name)
        tokenizer = model.tokenizer
        model_max_seq_length = model.max_seq_length
        logger.info(f"Successfully loaded model. Max sequence length: {model_max_seq_length}")
    except Exception as e:
        logger.error(f"Could not load SentenceTransformer model {embedding_model_name}. Error: {e}")
        return

    logger.info(f"Loading 'decorte' data (full history with descriptions)...")
    
    # Load data with subspans=True to see full history lengths
    data_loader_instance = Data(DATA_TYPE='decorte', 
                                ONLY_TITLES=False, 
                                consider_subspans=True, 
                                LOAD_CLEAN_TEST=False)
    
    logger.info(f"DEBUG: Raw train_pairs in Data instance: {len(data_loader_instance.train_pairs)}")
    if len(data_loader_instance.train_pairs) > 0:
        sample_pair = data_loader_instance.train_pairs[0]
        logger.info(f"DEBUG: Sample pair[0] (truncated): {sample_pair[0][:100]}...")
        logger.info(f"DEBUG: SEP_TOKEN present? {SEP_TOKEN in sample_pair[0]}")
    
    # Get all splits
    # Note: get_data returns (train, val, test) when LOAD_CLEAN_TEST=False
    try:
        train_pairs, val_pairs, test_pairs = data_loader_instance.get_data(stage='transformation_finetuning')
    except ValueError:
        # Handle case where it might return 4 values if default changed
        results = data_loader_instance.get_data(stage='transformation_finetuning')
        train_pairs = results[0]
        val_pairs = results[1]
        test_pairs = results[2]

    logger.info("Dataset Statistics (after get_data):")
    logger.info(f"  Train: {len(train_pairs):,} pairs")
    logger.info(f"  Val:   {len(val_pairs):,} pairs")
    logger.info(f"  Test:  {len(test_pairs):,} pairs")
    logger.info(f"  Total: {len(train_pairs) + len(val_pairs) + len(test_pairs):,} pairs")
    
    # Note on max_len default in utils.py if counts are lower than expected
    logger.info("  Note: If counts are lower than expected (~1M), check 'max_len' parameter in skills4cpp/src/cpp/utils.py (default=16 per profile).")
    logger.info("")

    # Define thresholds
    context_lengths_to_test = [128, 256, 512]
    if model_max_seq_length not in context_lengths_to_test:
        context_lengths_to_test.append(model_max_seq_length)
    context_lengths_to_test = sorted(list(set(context_lengths_to_test)))

    logger.info(f"Evaluation Context Lengths: {context_lengths_to_test}\n")

    # Initialize report structure
    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "encoder_model": embedding_model_name,
            "model_max_seq_length": model_max_seq_length,
            "data_type": "decorte",
            "only_titles": False,
            "consider_subspans": True,
            "context_lengths_tested": context_lengths_to_test,
            "sep_token": SEP_TOKEN,
        },
        "results": {}
    }

    # Analyze each split
    report["results"]["train"] = analyze_split("Train", train_pairs, tokenizer, context_lengths_to_test)
    report["results"]["validation"] = analyze_split("Validation", val_pairs, tokenizer, context_lengths_to_test)
    report["results"]["test"] = analyze_split("Test", test_pairs, tokenizer, context_lengths_to_test)

    # Save the report
    save_report(report, "truncation_impact_report")
    
    logger.info("Analysis complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Investigate truncation impact on career path data.")
    parser.add_argument(
        "--encoder_model",
        type=str,
        default="ElenaSenger/career-path-representation-mpnet-decorte-esco",
        help="Name or path of the SentenceTransformer encoder model to use for tokenization.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="decorte_esco.yaml",
        help="Configuration file name (e.g., decorte_esco.yaml). Not directly used for data loading here."
    )
    args = parser.parse_args()
    analyze_truncation_impact(config_name=args.config, encoder_text_model_name=args.encoder_model)
