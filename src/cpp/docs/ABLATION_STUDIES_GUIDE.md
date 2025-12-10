# Ablation Studies Guide - Modality Experiments

## Overview

The enhanced training script now supports **modality on/off controls** for systematic ablation studies. Perfect for your thesis experiments!

## Modality Control Flags

### Enable/Disable Modalities

```bash
--use_text_history      # Job history text (default: ON)
--use_skill_text        # Skill text features (default: ON)
--use_structured        # Structured meta-features (default: ON)

# To disable:
--no_text_history       # Exclude job history
--no_skill_text         # Exclude skills
--no_structured         # Exclude meta-features
```

### Automatic Architecture Selection

The script **automatically** selects the appropriate architecture:
- **1 modality** → Simple concatenation mode (unless `--use_advanced` is set)
- **2+ modalities** → Multi-modal fusion mode (auto-enables `--use_advanced`)

## Your Thesis Experiments

### Experiment 1: Job Path Only
```bash
python -m src.cpp.train_cpp_enhanced \
    --no_skill_text \
    --no_structured \
    --output_dir results/exp1_jobpath_only \
    --n_trials 50
```

**Expected behavior:**
- ✓ Uses simple architecture (1 modality)
- ✓ Only text history encoder active
- ✓ Baseline performance

### Experiment 2: Skills Only
```bash
python -m src.cpp.train_cpp_enhanced \
    --no_text_history \
    --no_structured \
    --output_dir results/exp2_skills_only \
    --n_trials 50
```

**Expected behavior:**
- ✓ Uses simple architecture (1 modality)
- ✓ Only skill text encoder active
- ✓ Shows importance of skills alone

### Experiment 3: Job Path + Skills
```bash
python -m src.cpp.train_cpp_enhanced \
    --no_structured \
    --output_dir results/exp3_jobpath_skills \
    --n_trials 50
```

**Expected behavior:**
- ✓ **Automatically** uses multi-modal architecture (2 modalities)
- ✓ Text + skill encoders with shared hyperparameters
- ✓ Linear fusion head
- ✓ Shows combined benefit

### Experiment 4: Job Path + Skills + Meta Features (Full)
```bash
python -m src.cpp.train_cpp_enhanced \
    --output_dir results/exp4_full_model \
    --n_trials 50
```

**Expected behavior:**
- ✓ **Automatically** uses multi-modal architecture (3 modalities)
- ✓ All encoders active
- ✓ Shared hyperparameters across all modalities
- ✓ Best performance (hopefully!)

## Advanced Variations

### Job Path + Meta Features (No Skills)
```bash
python -m src.cpp.train_cpp_enhanced \
    --no_skill_text \
    --output_dir results/jobpath_meta \
    --n_trials 50
```

### Skills + Meta Features (No Job Path)
```bash
python -m src.cpp.train_cpp_enhanced \
    --no_text_history \
    --output_dir results/skills_meta \
    --n_trials 50
```

### Meta Features Only
```bash
python -m src.cpp.train_cpp_enhanced \
    --no_text_history \
    --no_skill_text \
    --output_dir results/meta_only \
    --n_trials 50
```

## Complete Thesis Experiment Script

Save this as `run_ablation_studies.sh`:

```bash
#!/bin/bash

# Thesis Ablation Studies
# Systematically tests all modality combinations

BASE_DIR="results/thesis_ablations"
N_TRIALS=50
MAX_EPOCHS=30
BATCH_SIZE=32
POOLING="weighted_idf"

echo "=========================================="
echo "Starting Thesis Ablation Studies"
echo "=========================================="

# Experiment 1: Job Path Only
echo "[1/7] Running: Job Path Only..."
python -m src.cpp.train_cpp_enhanced \
    --no_skill_text \
    --no_structured \
    --pooling_strategy $POOLING \
    --n_trials $N_TRIALS \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --output_dir "${BASE_DIR}/1_jobpath_only"

# Experiment 2: Skills Only
echo "[2/7] Running: Skills Only..."
python -m src.cpp.train_cpp_enhanced \
    --no_text_history \
    --no_structured \
    --pooling_strategy $POOLING \
    --n_trials $N_TRIALS \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --output_dir "${BASE_DIR}/2_skills_only"

# Experiment 3: Meta Features Only
echo "[3/7] Running: Meta Features Only..."
python -m src.cpp.train_cpp_enhanced \
    --no_text_history \
    --no_skill_text \
    --pooling_strategy $POOLING \
    --n_trials $N_TRIALS \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --output_dir "${BASE_DIR}/3_meta_only"

# Experiment 4: Job Path + Skills
echo "[4/7] Running: Job Path + Skills..."
python -m src.cpp.train_cpp_enhanced \
    --no_structured \
    --pooling_strategy $POOLING \
    --n_trials $N_TRIALS \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --output_dir "${BASE_DIR}/4_jobpath_skills"

# Experiment 5: Job Path + Meta
echo "[5/7] Running: Job Path + Meta Features..."
python -m src.cpp.train_cpp_enhanced \
    --no_skill_text \
    --pooling_strategy $POOLING \
    --n_trials $N_TRIALS \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --output_dir "${BASE_DIR}/5_jobpath_meta"

# Experiment 6: Skills + Meta
echo "[6/7] Running: Skills + Meta Features..."
python -m src.cpp.train_cpp_enhanced \
    --no_text_history \
    --pooling_strategy $POOLING \
    --n_trials $N_TRIALS \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --output_dir "${BASE_DIR}/6_skills_meta"

# Experiment 7: All Features (Full Model)
echo "[7/7] Running: Full Model (All Features)..."
python -m src.cpp.train_cpp_enhanced \
    --pooling_strategy $POOLING \
    --n_trials $N_TRIALS \
    --max_epochs $MAX_EPOCHS \
    --batch_size $BATCH_SIZE \
    --output_dir "${BASE_DIR}/7_full_model"

echo "=========================================="
echo "All Experiments Complete!"
echo "=========================================="

# Generate comparison table
python -m src.cpp.compare_ablation_results "${BASE_DIR}"
```

## Collecting Results

### Extract Test Metrics

```python
import torch
import pandas as pd
import glob

results = []

for exp_dir in glob.glob("results/thesis_ablations/*/"):
    checkpoint_path = f"{exp_dir}/final_model.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        
        exp_name = exp_dir.split('/')[-2]
        metrics = checkpoint['test_metrics']
        
        results.append({
            'Experiment': exp_name,
            'MRR': metrics['MRR'],
            'R@1': metrics['R@1'],
            'R@5': metrics['R@5'],
            'R@10': metrics['R@10'],
            'R@20': metrics['R@20'],
        })

df = pd.DataFrame(results)
df = df.sort_values('MRR', ascending=False)
print(df.to_markdown(index=False))
df.to_csv('results/thesis_ablations/comparison.csv', index=False)
```

### Example Output Table

```
| Experiment          | MRR    | R@1    | R@5    | R@10   | R@20   |
|---------------------|--------|--------|--------|--------|--------|
| 7_full_model        | 0.5678 | 0.3456 | 0.6789 | 0.7890 | 0.8567 |
| 4_jobpath_skills    | 0.5234 | 0.3012 | 0.6345 | 0.7456 | 0.8234 |
| 5_jobpath_meta      | 0.4890 | 0.2789 | 0.5987 | 0.7123 | 0.8012 |
| 1_jobpath_only      | 0.4567 | 0.2456 | 0.5678 | 0.6890 | 0.7890 |
| 6_skills_meta       | 0.4345 | 0.2234 | 0.5456 | 0.6789 | 0.7678 |
| 2_skills_only       | 0.3890 | 0.1890 | 0.4789 | 0.6123 | 0.7345 |
| 3_meta_only         | 0.2456 | 0.1123 | 0.3456 | 0.4789 | 0.6234 |
```

## Interpreting Results for Thesis

### Research Questions You Can Answer

1. **Which single modality is most important?**
   - Compare: jobpath_only vs skills_only vs meta_only
   - Shows individual contribution

2. **Do modalities complement each other?**
   - Compare: (jobpath + skills) vs (jobpath_only + skills_only)
   - If combined > sum of parts, they're complementary!

3. **What's the marginal benefit of each modality?**
   - Job path: full_model vs skills_meta
   - Skills: full_model vs jobpath_meta
   - Meta: full_model vs jobpath_skills

4. **Is the multi-modal architecture effective?**
   - The script automatically uses it for 2+ modalities
   - Compare performance gains from fusion

## Architecture Details Per Experiment

### Single Modality Experiments (1, 2, 3)
```
Input → Simple MLP → Output
```
- Architecture: SimpleConcatModel
- Search space: 4 hyperparameters
- No fusion needed

### Two Modality Experiments (4, 5, 6)
```
Modality 1 → Encoder → hidden
Modality 2 → Encoder → hidden  (shared hyperparameters)
  ↓
Concatenate → Linear → Output
```
- Architecture: MultiModalCPPModel (auto-enabled)
- Search space: 5-6 hyperparameters
- Shared projections

### Three Modality Experiment (7)
```
Text → Encoder → hidden
Skills → Encoder → hidden       (shared hyperparameters)
Meta → Encoder → hidden         (shared hyperparameters)
  ↓
Concatenate → Linear → Output
```
- Architecture: MultiModalCPPModel
- Search space: 5-6 hyperparameters
- Optimal fusion strategy

## Additional Experiments

### With Different Skill Encoders
```bash
# Test if domain-specific encoder helps
python -m src.cpp.train_cpp_enhanced \
    --encoder_skill "sentence-transformers/all-MiniLM-L6-v2" \
    --output_dir results/different_skill_encoder
```

### With Skill Descriptions
```bash
# Compare name-only vs name+description
python -m src.cpp.train_cpp_enhanced \
    --use_skill_description \
    --output_dir results/with_skill_descriptions
```

### Different Pooling Strategies
```bash
for pooling in mean weighted_mean weighted_idf; do
    python -m src.cpp.train_cpp_enhanced \
        --pooling_strategy $pooling \
        --output_dir "results/pooling_${pooling}"
done
```

## Tips for Thesis Writing

### Document Your Setup
```python
# Save configuration for each experiment
with open(f'{output_dir}/config.txt', 'w') as f:
    f.write(f"Experiment: {exp_name}\n")
    f.write(f"Modalities: text={use_text}, skills={use_skills}, meta={use_meta}\n")
    f.write(f"Architecture: {'multi-modal' if use_advanced else 'simple'}\n")
    f.write(f"Date: {datetime.now()}\n")
```

### Create Visualizations
```python
import matplotlib.pyplot as plt

# Bar chart comparing MRR across experiments
df.plot(x='Experiment', y='MRR', kind='bar')
plt.ylabel('Mean Reciprocal Rank')
plt.title('Ablation Study Results')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/ablation_comparison.png')
```

### Statistical Significance
```python
# Run each experiment 3-5 times with different seeds
for seed in [42, 123, 456]:
    python -m src.cpp.train_cpp_enhanced \
        --seed $seed \
        --output_dir "results/exp1_seed_${seed}"
```

## Common Issues

### "At least one modality must be enabled"
**Cause:** All modalities disabled  
**Solution:** Enable at least one modality

### "Warning: Using multi-modal architecture with only one modality"
**Cause:** `--use_advanced` set with single modality  
**Solution:** Remove `--use_advanced` or enable more modalities

### Different number of parameters across experiments
**Expected!** Single modality experiments have fewer parameters than multi-modal ones.

## Next Steps

1. ✅ Run single-modality baselines (Exp 1-3)
2. ✅ Run two-modality combinations (Exp 4-6)
3. ✅ Run full model (Exp 7)
4. ✅ Compare results and analyze
5. ✅ Write thesis section with findings!

---

**Good luck with your thesis! 🎓**



