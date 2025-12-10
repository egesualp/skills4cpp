# Thesis Experiments - Quick Reference

## Your 4 Core Experiments

### 📊 Experiment 1: Job Path Only
```bash
python -m src.cpp.train_cpp_enhanced \
    --no_skill_text --no_structured \
    --output_dir results/thesis/exp1_jobpath_only \
    --n_trials 50
```
**Tests:** Baseline performance using only career history text

---

### 📊 Experiment 2: Skills Only
```bash
python -m src.cpp.train_cpp_enhanced \
    --no_text_history --no_structured \
    --output_dir results/thesis/exp2_skills_only \
    --n_trials 50
```
**Tests:** Can skills alone predict next job?

---

### 📊 Experiment 3: Job Path + Skills
```bash
python -m src.cpp.train_cpp_enhanced \
    --no_structured \
    --output_dir results/thesis/exp3_jobpath_skills \
    --n_trials 50
```
**Tests:** Complementary benefit of combining text and skills  
**Note:** Auto-enables multi-modal architecture ✓

---

### 📊 Experiment 4: Job Path + Skills + Meta Features (Full)
```bash
python -m src.cpp.train_cpp_enhanced \
    --output_dir results/thesis/exp4_full_model \
    --n_trials 50
```
**Tests:** Best performance with all modalities  
**Note:** Auto-enables multi-modal architecture ✓

---

## Architecture Selection (Automatic!)

| Experiment | Active Modalities | Architecture | Why |
|------------|-------------------|--------------|-----|
| Exp 1 | 1 (text) | Simple | Single input, no fusion needed |
| Exp 2 | 1 (skills) | Simple | Single input, no fusion needed |
| Exp 3 | 2 (text + skills) | **Multi-modal** ✓ | Multiple inputs benefit from fusion |
| Exp 4 | 3 (text + skills + meta) | **Multi-modal** ✓ | Multiple inputs benefit from fusion |

**The script handles this automatically!** You don't need to worry about `--use_advanced`.

---

## Expected Output Format

Each experiment produces:

```
results/thesis/expX_name/
├── final_model.pt          # Trained model + metrics
└── optuna_study.pkl        # Optimization history
```

`final_model.pt` contains:
```python
{
    'test_metrics': {
        'MRR': 0.5678,
        'R@1': 0.3456,
        'R@5': 0.6789,
        'R@10': 0.7890,
        'R@20': 0.8567
    },
    'best_params': {...},
    'model_state_dict': {...}
}
```

---

## Run All Experiments (Sequential)

```bash
#!/bin/bash
cd /dss/dsshome1/02/ra95kix2/thesis/skills4cpp

# Experiment 1: Job Path Only
python -m src.cpp.train_cpp_enhanced \
    --no_skill_text --no_structured \
    --output_dir results/thesis/exp1_jobpath_only \
    --n_trials 50 --max_epochs 30 --device cuda

# Experiment 2: Skills Only  
python -m src.cpp.train_cpp_enhanced \
    --no_text_history --no_structured \
    --output_dir results/thesis/exp2_skills_only \
    --n_trials 50 --max_epochs 30 --device cuda

# Experiment 3: Job Path + Skills
python -m src.cpp.train_cpp_enhanced \
    --no_structured \
    --output_dir results/thesis/exp3_jobpath_skills \
    --n_trials 50 --max_epochs 30 --device cuda

# Experiment 4: Full Model
python -m src.cpp.train_cpp_enhanced \
    --output_dir results/thesis/exp4_full_model \
    --n_trials 50 --max_epochs 30 --device cuda

echo "All experiments complete! Results in results/thesis/"
```

---

## Extract Results for Thesis

```python
import torch
import pandas as pd

experiments = [
    ('exp1_jobpath_only', 'Job Path Only'),
    ('exp2_skills_only', 'Skills Only'),
    ('exp3_jobpath_skills', 'Job Path + Skills'),
    ('exp4_full_model', 'Full Model'),
]

results = []
for exp_dir, exp_name in experiments:
    path = f"results/thesis/{exp_dir}/final_model.pt"
    checkpoint = torch.load(path)
    metrics = checkpoint['test_metrics']
    
    results.append({
        'Experiment': exp_name,
        'MRR': f"{metrics['MRR']:.4f}",
        'R@1': f"{metrics['R@1']:.4f}",
        'R@5': f"{metrics['R@5']:.4f}",
        'R@10': f"{metrics['R@10']:.4f}",
        'R@20': f"{metrics['R@20']:.4f}",
    })

df = pd.DataFrame(results)
print("\n=== THESIS RESULTS ===")
print(df.to_markdown(index=False))
print("\nSaved to: results/thesis/summary.csv")
df.to_csv('results/thesis/summary.csv', index=False)
```

---

## Example Thesis Results Table

| Experiment | MRR | R@1 | R@5 | R@10 | R@20 |
|------------|-----|-----|-----|------|------|
| Full Model | 0.5678 | 0.3456 | 0.6789 | 0.7890 | 0.8567 |
| Job Path + Skills | 0.5234 | 0.3012 | 0.6345 | 0.7456 | 0.8234 |
| Job Path Only | 0.4567 | 0.2456 | 0.5678 | 0.6890 | 0.7890 |
| Skills Only | 0.3890 | 0.1890 | 0.4789 | 0.6123 | 0.7345 |

---

## Key Findings for Thesis

### 1. Individual Modality Importance
Compare Exp 1 vs Exp 2:
- Which single modality is stronger?
- By how much?

### 2. Synergy Between Modalities
Compare Exp 3 vs (Exp 1 + Exp 2):
- Is combined > sum of parts?
- Shows complementary information!

### 3. Value of Meta-Features
Compare Exp 4 vs Exp 3:
- Marginal benefit of adding structured features
- Cost/benefit analysis

### 4. Multi-Modal Architecture Effectiveness
- Experiments 3 & 4 automatically use fusion
- Shows benefit of late fusion strategy

---

## Thesis Writing Tips

### Results Section
```markdown
## 5.2 Ablation Study Results

We conducted a systematic ablation study to assess the contribution 
of each modality. Table X shows the test set performance.

The full model (Exp 4) achieved the best performance with MRR=0.5678,
demonstrating that all modalities contribute valuable information.

Job path text alone (Exp 1) achieved MRR=0.4567, while skills alone 
(Exp 2) achieved MRR=0.3890, indicating that career history provides 
more predictive signal than isolated skill information.

Notably, combining job path and skills (Exp 3, MRR=0.5234) 
outperformed the sum of individual contributions, suggesting 
complementary information between modalities...
```

### Discussion Points
1. Why does X modality perform better than Y?
2. What complementary information do modalities provide?
3. Is the complexity of multi-modal fusion justified?
4. What are the practical implications?

---

## Advanced Variations

### With Different Encoders
```bash
# Test if specialized skill encoder helps
python -m src.cpp.train_cpp_enhanced \
    --encoder_skill "sentence-transformers/all-MiniLM-L6-v2" \
    --output_dir results/thesis/exp4b_different_encoder
```

### With Skill Descriptions
```bash
# Compare skill names vs names+descriptions
python -m src.cpp.train_cpp_enhanced \
    --use_skill_description \
    --output_dir results/thesis/exp4c_with_descriptions
```

### Different Pooling
```bash
# Compare pooling strategies
for pooling in mean weighted_mean weighted_idf; do
    python -m src.cpp.train_cpp_enhanced \
        --pooling_strategy $pooling \
        --output_dir "results/thesis/exp4d_pooling_${pooling}"
done
```

---

## Troubleshooting

**Issue:** GPU out of memory  
**Solution:** `--batch_size 16 --num_workers 2`

**Issue:** Want faster testing  
**Solution:** `--n_trials 10 --max_epochs 10` (for debugging)

**Issue:** Need reproducibility  
**Solution:** Set random seeds in your script

---

## Time Estimates (50 trials, 30 epochs each)

- **Exp 1 (text only):** ~2-3 hours
- **Exp 2 (skills only):** ~1-2 hours  
- **Exp 3 (text + skills):** ~3-4 hours
- **Exp 4 (all features):** ~4-5 hours

**Total:** ~12-15 hours for all experiments

**Recommendation:** Run overnight or use `nohup`:
```bash
nohup bash run_all_experiments.sh > experiments.log 2>&1 &
```

---

## Checklist for Thesis

- [ ] Run all 4 core experiments
- [ ] Extract and tabulate results
- [ ] Create comparison visualizations
- [ ] Document experimental setup
- [ ] Analyze ablation findings
- [ ] Write results section
- [ ] Discuss implications
- [ ] Include in thesis appendix

---

**Ready to run your thesis experiments? Just copy-paste the commands above!** 🎓



