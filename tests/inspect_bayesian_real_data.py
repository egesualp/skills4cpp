import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Set

# Add src to path
sys.path.append(os.path.abspath("src"))

from skill_mapping.v2.bayesian_fuser import (
    load_similarity_scores,
    load_category_scores_raw,
    build_skill_to_categories,
    prepare_vectorized_data,
    compute_fused_scores_vectorized,
    FusionConfig,
    VectorizedData
)

def main():
    # Paths to real data
    base_path_ext = Path("/dss/dssmcmlfs01/pr74ze/pr74ze-dss-0001/ra95kix2")
    sim_scores_path = base_path_ext / "outputs/decorte_w_desc/similarity_scores.json"
    cat_scores_path = base_path_ext / "outputs/category_model_h1_soft_deep_larger_val/decorte_w_desc_inference/category_scores.json"
    hierarchy_path = Path("data/processed/master_datasets_2/master_skill_complete_hierarchy.csv")

    print(f"Loading hierarchy from {hierarchy_path}...")
    skill_to_cats = build_skill_to_categories(hierarchy_path)
    
    print(f"Loading similarity scores from {sim_scores_path}...")
    # Load fully to find a good example, then slice
    with open(sim_scores_path, 'r') as f:
        full_sim_scores = json.load(f)
        
    print(f"Loading category scores from {cat_scores_path}...")
    with open(cat_scores_path, 'r') as f:
        full_cat_scores = json.load(f)

    # Find a job with a skill that has multiple categories
    target_job_id = None
    target_skill_uri = None
    
    print("Searching for a job with a multi-category skill...")
    
    # First find skills with multiple categories
    multi_cat_skills = {uri: cats for uri, cats in skill_to_cats.items() if len(cats) > 1}
    print(f"Found {len(multi_cat_skills)} skills with >1 categories.")
    
    # Iterate through jobs to find one that lists one of these skills
    for job_id, skills in full_sim_scores.items():
        if job_id not in full_cat_scores:
            continue
            
        for skill_entry in skills[:10]: # Check top 10 skills
            uri = skill_entry["skill_uri"]
            if uri in multi_cat_skills:
                target_job_id = job_id
                target_skill_uri = uri
                break
        if target_job_id:
            break
            
    if not target_job_id:
        print("Could not find a job with a multi-category skill in top 10 candidates. Picking first job.")
        target_job_id = list(full_sim_scores.keys())[0]
    
    print(f"\n=== Selected Job: {target_job_id} ===")
    if target_skill_uri:
        print(f"Target multi-category skill: {target_skill_uri}")
        print(f"Categories: {skill_to_cats[target_skill_uri]}")
    
    # Slice data for this job
    sim_scores_subset = {target_job_id: full_sim_scores[target_job_id]}
    cat_scores_subset = {target_job_id: full_cat_scores[target_job_id]}
    
    # Display Raw Data
    print("\n--- Raw Data (Subset) ---")
    print(f"Job Category Scores (Top 5):")
    # Sort category scores by score desc
    sorted_cats = sorted(cat_scores_subset[target_job_id], key=lambda x: x['score'], reverse=True)
    for c in sorted_cats[:5]:
        print(f"  {c['category']}: {c['score']}")
        
    print(f"\nCandidate Skills (Top 5):")
    for s in sim_scores_subset[target_job_id][:5]:
        cats = skill_to_cats.get(s['skill_uri'], set())
        print(f"  {s['skill_uri']} (Score: {s['score']:.4f}) -> Cats: {cats}")

    # Prepare Vectorized Data
    print("\n=== Running prepare_vectorized_data ===")
    vd = prepare_vectorized_data(sim_scores_subset, cat_scores_subset, skill_to_cats)
    
    print(f"Vectorized Job IDs: {vd.job_ids}")
    print(f"Number of skills in matrix: {len(vd.skill_uris)}")
    print(f"Number of categories: {len(vd.category_names)}")
    
    # Inspect Vectorized Data for the target skill
    if target_skill_uri:
        try:
            skill_idx = vd.skill_uris.index(target_skill_uri)
            print(f"\nVerifying Matrix for {target_skill_uri} (Index {skill_idx}):")
            cat_vector = vd.skill_category_matrix[skill_idx]
            nonzero_indices = np.nonzero(cat_vector)[0]
            print(f"  Non-zero category indices: {nonzero_indices}")
            found_cats = [vd.category_names[i] for i in nonzero_indices]
            print(f"  Mapped Categories in Matrix: {set(found_cats)}")
            print(f"  Expected Categories: {skill_to_cats[target_skill_uri]}")
        except ValueError:
            print(f"Skill {target_skill_uri} not found in vectorized data (should not happen)")

    # Test Aggregation
    print("\n=== Testing Aggregation Modes ===")
    
    # We use alpha=1.0, Temperature=1.0 to see the pure aggregated probability
    modes = ["max", "sum"]
    
    for mode in modes:
        print(f"\n--- Mode: {mode.upper()} ---")
        config = FusionConfig(
            fusion_mode="linear",
            alpha=1.0,
            temperature=1.0,
            threshold=0.0,
            aggregate_mode=mode
        )
        
        _, top_k_indices, top_k_scores, _ = compute_fused_scores_vectorized(
            vd, config, top_k=5, aggregate_mode=mode
        )
        
        # Get results
        indices = top_k_indices[0]
        scores = top_k_scores[0]
        
        print(f"Top 5 Skills by Aggregated Category Score:")
        for idx, score in zip(indices, scores):
            skill_uri = vd.skill_uris[idx]
            cats = skill_to_cats.get(skill_uri, set())
            print(f"  {skill_uri}: {score:.6f} (Cats: {cats})")
            
            # If this is our target skill, show calculation details
            if skill_uri == target_skill_uri:
                print("    [Verification]")
                # Get category probabilities for this job
                # We need to manually calculate probabilities from logits
                # Using the exact same logic as in the script
                logits = vd.category_logits[0] # job 0
                # T=1.0
                scaled_logits = logits # / 1.0
                # Softmax
                scaled_logits = scaled_logits - scaled_logits.max()
                exp_logits = np.exp(scaled_logits)
                probs = exp_logits / exp_logits.sum()
                
                skill_cats = skill_to_cats.get(skill_uri, set())
                cat_probs = []
                for cat in skill_cats:
                    if cat in vd.category_names:
                        cat_idx = vd.category_names.index(cat)
                        p = probs[cat_idx]
                        cat_probs.append((cat, p))
                        print(f"      P({cat}) = {p:.6f}")
                    else:
                        print(f"      P({cat}) = not in category list")
                
                vals = [p for _, p in cat_probs]
                if not vals:
                    agg = 0.0
                elif mode == "max":
                    agg = max(vals)
                else:
                    agg = sum(vals)
                print(f"      Calculated {mode}: {agg:.6f}")
                print(f"      Script Output:   {score:.6f}")

if __name__ == "__main__":
    main()

