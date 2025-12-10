"""
Diagnostic script to understand why skill-based predictions have low metrics.

This script will:
1. Analyze the distribution and discriminative power of skill embeddings
2. Compute baseline performance (random, nearest neighbor)
3. Check for data leakage or evaluation issues
4. Visualize embedding spaces
"""

import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sentence_transformers import SentenceTransformer
from cpp.data_classes import Data
from data_loaders import (
    load_job_and_skill_data,
    precompute_target_embeddings,
    extract_unique_skills_from_dataset,
    precompute_skill_embeddings,
    _pooled_skill_vec_optimized,
    _extract_skill_infos
)

print("="*80)
print("DIAGNOSTIC ANALYSIS: Understanding Low Skill-Based Prediction Metrics")
print("="*80)

# Configuration
DATA_TYPE = 'decorte_esco'
MASTER_SKILL_FILE = 'results/decorte_esco_ground_truth/job_title_skills_master.csv'
ESCO_SKILLS_FILE = 'data/esco_datasets/skills_en.csv'
ENCODER_NAME = 'ElenaSenger/career-path-representation-mpnet-decorte'
POOLING_STRATEGY = 'weighted_idf'
USE_SKILL_DESCRIPTION = True

print("\n[1/6] Loading data and models...")
encoder = SentenceTransformer(ENCODER_NAME)
embed_dim = encoder.get_sentence_embedding_dimension()

job_skill_map, esco_skill_text_map, _ = load_job_and_skill_data(
    MASTER_SKILL_FILE, ESCO_SKILLS_FILE, 
    skill_properties_file='data/processed/master_datasets_2/skill_properties_map.json',
    pooling_strategy=POOLING_STRATEGY, alpha=1.0, beta=1.0
)

data = Data(DATA_TYPE=DATA_TYPE, ONLY_TITLES=True)
train_pairs, val_pairs, test_pairs = data.get_data(stage='transformation_finetuning')
all_pairs = train_pairs + val_pairs + test_pairs

print(f"  ✓ Loaded {len(all_pairs)} data pairs")
print(f"  ✓ Embedding dimension: {embed_dim}")

print("\n[2/6] Analyzing target label distribution...")
# Count how many times each target appears
target_counts = Counter([pair[1] for pair in all_pairs])
print(f"  → Total unique targets: {len(target_counts)}")
print(f"  → Most common target: '{target_counts.most_common(1)[0][0]}' ({target_counts.most_common(1)[0][1]} times)")
print(f"  → Least common targets: {len([k for k,v in target_counts.items() if v == 1])} appear only once")

# Check if any targets are impossible to predict (only in test, not in train)
train_targets = set([pair[1] for pair in train_pairs])
test_targets = set([pair[1] for pair in test_pairs])
unseen_targets = test_targets - train_targets
print(f"  → Targets in test but not in train: {len(unseen_targets)} ({len(unseen_targets)/len(test_targets)*100:.1f}%)")
if len(unseen_targets) > 0:
    print(f"    ⚠️  WARNING: Model cannot predict these targets!")

print("\n[3/6] Computing skill embeddings for all samples...")
# Pre-compute skill embeddings
unique_skills = extract_unique_skills_from_dataset(all_pairs, job_skill_map)
print(f"  → Extracting {len(unique_skills)} unique skills...")

# Show examples of text being encoded
print("\n  📝 SHOWING EXAMPLES OF TEXT BEING ENCODED:")
print("  " + "="*76)
sample_skills = list(unique_skills)[:3]
for i, skill_uri in enumerate(sample_skills):
    if skill_uri in esco_skill_text_map:
        skill_data = esco_skill_text_map[skill_uri]
        if USE_SKILL_DESCRIPTION:
            text_to_encode = f"role: {skill_data['name']} \n description: {skill_data['desc']}"
        else:
            text_to_encode = skill_data['name']
        
        print(f"\n  Example {i+1}:")
        print(f"  Skill URI: {skill_uri[:60]}...")
        print(f"  Text being encoded (length={len(text_to_encode)}):")
        print(f"  ---START---")
        print(f"  {text_to_encode[:200]}" + ("..." if len(text_to_encode) > 200 else ""))
        print(f"  ---END---")
print("  " + "="*76)

skill_embedding_map = precompute_skill_embeddings(
    unique_skills, encoder, esco_skill_text_map, USE_SKILL_DESCRIPTION
)

# Compute aggregated skill vectors for each sample
skill_vectors = []
input_texts = []
target_texts = []

print("\n  📝 SHOWING EXAMPLE OF SKILL AGGREGATION FOR ONE SAMPLE:")
print("  " + "="*76)
first_sample_done = False

for history_doc, target_doc in tqdm(all_pairs, desc="  → Computing skill vectors"):
    infos = _extract_skill_infos(history_doc, job_skill_map)
    
    # Show detailed example for first non-empty sample
    if not first_sample_done and len(infos) > 0:
        print(f"\n  Input history: '{history_doc[:50]}{'...' if len(history_doc) > 50 else ''}'")
        print(f"  Target: '{target_doc}'")
        print(f"  Number of skills: {len(infos)}")
        print(f"  Pooling strategy: {POOLING_STRATEGY}")
        print(f"\n  Individual skill texts being aggregated (showing first 3):")
        
        for i, info in enumerate(infos[:3]):
            skill_uri = info['skillUri']
            if skill_uri in esco_skill_text_map:
                skill_data = esco_skill_text_map[skill_uri]
                if USE_SKILL_DESCRIPTION:
                    text = f"role: {skill_data['name']} \n description: {skill_data['desc']}"
                else:
                    text = skill_data['name']
                
                weight = info['score']
                if POOLING_STRATEGY == 'weighted_idf':
                    idf = info.get('idf', 0)
                    weight = (info['score'] ** 1.0) * (idf ** 1.0)
                
                print(f"\n  Skill {i+1}/{len(infos)}:")
                print(f"    Weight: {weight:.4f}")
                if POOLING_STRATEGY == 'weighted_idf':
                    print(f"    Score: {info['score']:.4f}, IDF: {info.get('idf', 0):.4f}")
                print(f"    Text (length={len(text)}):")
                print(f"    '{text[:150]}{'...' if len(text) > 150 else ''}'")
        
        if len(infos) > 3:
            print(f"\n  ... and {len(infos)-3} more skills")
        
        print(f"\n  → These {len(infos)} skill embeddings will be WEIGHTED AVERAGED into ONE vector")
        print("  " + "="*76)
        first_sample_done = True
    
    vec = _pooled_skill_vec_optimized(
        infos, skill_embedding_map, POOLING_STRATEGY, 
        alpha=1.0, beta=1.0, embed_dim=embed_dim, debug=False
    )
    skill_vectors.append(vec)
    input_texts.append(history_doc)
    target_texts.append(target_doc)

skill_vectors = np.array(skill_vectors)
print(f"\n  ✓ Skill vectors shape: {skill_vectors.shape}")

# Check how many are zero vectors
zero_count = np.sum(np.all(skill_vectors == 0, axis=1))
print(f"  → Zero vectors: {zero_count} ({zero_count/len(skill_vectors)*100:.1f}%)")

print("\n[4/6] Computing target embeddings...")
unique_targets = list(set(target_texts))

print("\n  📝 SHOWING EXAMPLES OF TARGET TEXT BEING ENCODED:")
print("  " + "="*76)
for i, target in enumerate(unique_targets[:3]):
    print(f"\n  Target {i+1}:")
    print(f"  Text being encoded (length={len(target)}):")
    print(f"  ---START---")
    print(f"  {target}")
    print(f"  ---END---")
print("  " + "="*76)

Y_target_dict = precompute_target_embeddings(encoder, unique_targets, show_progress=False)
target_vectors = np.array([Y_target_dict[t] for t in target_texts])
print(f"\n  ✓ Target vectors shape: {target_vectors.shape}")

print("\n[5/6] Analyzing embedding distributions...")
# Skill vector statistics
print("  Skill embeddings (input):")
print(f"    - Mean norm: {np.linalg.norm(skill_vectors, axis=1).mean():.4f}")
print(f"    - Std norm: {np.linalg.norm(skill_vectors, axis=1).std():.4f}")
print(f"    - Mean value: {skill_vectors.mean():.6f}")
print(f"    - Std value: {skill_vectors.std():.6f}")

print("  Target embeddings (what we're trying to predict):")
print(f"    - Mean norm: {np.linalg.norm(target_vectors, axis=1).mean():.4f}")
print(f"    - Std norm: {np.linalg.norm(target_vectors, axis=1).std():.4f}")
print(f"    - Mean value: {target_vectors.mean():.6f}")
print(f"    - Std value: {target_vectors.std():.6f}")

# Show direct similarity between inputs and their targets
print("\n  💡 Direct input→target similarity (what model needs to learn):")
direct_sims = [cosine_similarity(skill_vectors[i:i+1], target_vectors[i:i+1])[0, 0] 
               for i in range(min(100, len(skill_vectors)))]
print(f"    - Mean: {np.mean(direct_sims):.4f}")
print(f"    - Std: {np.std(direct_sims):.4f}")
print(f"    - Min: {np.min(direct_sims):.4f}")
print(f"    - Max: {np.max(direct_sims):.4f}")

if np.mean(direct_sims) < 0.1:
    print("    ⚠️  Very low similarity! Input and target embeddings are in different spaces.")
    print("       This makes the task extremely difficult.")

# Check discriminative power: how similar are inputs with same vs different targets?
print("\n[6/6] Analyzing discriminative power...")
print("  Computing pairwise similarities (this may take a while)...")

# Sample to make computation tractable
sample_size = min(1000, len(skill_vectors))
indices = np.random.choice(len(skill_vectors), sample_size, replace=False)
skill_sample = skill_vectors[indices]
target_sample = [target_texts[i] for i in indices]

# Compute similarities
same_target_sims = []
diff_target_sims = []

for i in tqdm(range(len(skill_sample)), desc="  → Comparing samples"):
    for j in range(i+1, min(i+100, len(skill_sample))):  # Compare with next 100 samples
        sim = cosine_similarity(skill_sample[i:i+1], skill_sample[j:j+1])[0, 0]
        if target_sample[i] == target_sample[j]:
            same_target_sims.append(sim)
        else:
            diff_target_sims.append(sim)

if len(same_target_sims) > 0:
    print(f"\n  Skill similarity when targets are SAME:")
    print(f"    - Mean: {np.mean(same_target_sims):.4f}")
    print(f"    - Std: {np.std(same_target_sims):.4f}")
    print(f"    - Count: {len(same_target_sims)}")
else:
    print(f"\n  ⚠️  No pairs with same target found in sample")

print(f"\n  Skill similarity when targets are DIFFERENT:")
print(f"    - Mean: {np.mean(diff_target_sims):.4f}")
print(f"    - Std: {np.std(diff_target_sims):.4f}")
print(f"    - Count: {len(diff_target_sims)}")

if len(same_target_sims) > 0:
    separation = np.mean(same_target_sims) - np.mean(diff_target_sims)
    print(f"\n  → Separation: {separation:.4f}")
    if separation < 0.05:
        print(f"    ⚠️  WARNING: Very low separation! Skill embeddings may not be discriminative.")
    elif separation < 0.15:
        print(f"    ⚠️  Separation is low. Task is challenging.")
    else:
        print(f"    ✓ Good separation. Skill embeddings are discriminative.")

print("\n[7/6] Computing baseline performance: Nearest Neighbor...")
# Split into train and test
test_indices = list(range(len(train_pairs) + len(val_pairs), len(all_pairs)))
train_indices = list(range(len(train_pairs) + len(val_pairs)))

X_train = skill_vectors[train_indices]
y_train = [target_texts[i] for i in train_indices]
X_test = skill_vectors[test_indices]
y_test = [target_texts[i] for i in test_indices]

# Compute target embeddings for unique train targets
unique_train_targets = list(set(y_train))
train_target_embeddings = np.array([Y_target_dict[t] for t in unique_train_targets])

print(f"  → Train size: {len(X_train)}, Test size: {len(X_test)}")
print(f"  → Unique train targets: {len(unique_train_targets)}")

# For each test sample, find most similar training target
ranks = []
for i in tqdm(range(len(X_test)), desc="  → Evaluating NN baseline"):
    if np.all(X_test[i] == 0):
        ranks.append(len(unique_train_targets))  # Worst rank
        continue
    
    # Compute similarity to all training targets
    sims = cosine_similarity(X_test[i:i+1], train_target_embeddings)[0]
    
    # Rank targets by similarity
    ranked_indices = np.argsort(-sims)
    ranked_targets = [unique_train_targets[idx] for idx in ranked_indices]
    
    # Find rank of correct target
    try:
        rank = ranked_targets.index(y_test[i]) + 1
    except ValueError:
        rank = len(unique_train_targets) + 1  # Not in training set
    ranks.append(rank)

# Compute metrics
mrr = np.mean([1/r for r in ranks])
r_at_1 = np.mean([1 if r == 1 else 0 for r in ranks])
r_at_5 = np.mean([1 if r <= 5 else 0 for r in ranks])
r_at_10 = np.mean([1 if r <= 10 else 0 for r in ranks])
r_at_20 = np.mean([1 if r <= 20 else 0 for r in ranks])

print("\n" + "="*80)
print("BASELINE PERFORMANCE (Nearest Neighbor on Skill Embeddings)")
print("="*80)
print(f"MRR:   {mrr:.4f}")
print(f"R@1:   {r_at_1:.4f}")
print(f"R@5:   {r_at_5:.4f}")
print(f"R@10:  {r_at_10:.4f}")
print(f"R@20:  {r_at_20:.4f}")
print("="*80)

print("\n💡 INTERPRETATION:")
print("-" * 80)
if mrr < 0.02:
    print("❌ CRITICAL: Even the baseline performs terribly.")
    print("   This suggests the skill embeddings lack discriminative power.")
    print("   Possible causes:")
    print("   1. Skills are too generic/common across different jobs")
    print("   2. The aggregation (weighted averaging) loses too much information")
    print("   3. The task is fundamentally too difficult with only skill data")
elif mrr < 0.1:
    print("⚠️  WARNING: Baseline is poor but not terrible.")
    print("   The neural model (MRR: 0.0107) is performing WORSE than this baseline.")
    print("   Possible causes:")
    print("   1. Model architecture is too simple")
    print("   2. Training hyperparameters need tuning")
    print("   3. More epochs needed")
else:
    print("✓ Good news: The baseline works reasonably well.")
    print("  The neural model just needs better training/architecture.")

print("\n📊 RECOMMENDATIONS:")
print("-" * 80)
if zero_count > len(skill_vectors) * 0.1:
    print(f"1. Address zero vectors: {zero_count} samples have no skills")
if len(unseen_targets) > 0:
    print(f"2. Fix data split: {len(unseen_targets)} test targets never seen in training")
if len(same_target_sims) > 0 and separation < 0.05:
    print("3. Improve skill representation:")
    print("   - Try different pooling strategies")
    print("   - Use skill sequences instead of averaging")
    print("   - Add job title text as additional signal")
if mrr > 0.05:
    print("4. Improve model:")
    print("   - Try deeper/wider architecture")
    print("   - Tune hyperparameters (learning rate, batch size)")
    print("   - Train for more epochs")
print("\n")

