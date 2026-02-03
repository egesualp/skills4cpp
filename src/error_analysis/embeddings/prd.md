# Task: Embedding Error Analysis for Career Path Prediction

## Context
I'm analyzing why skill-enhanced career path prediction doesn't improve over text-only baselines. I need to compare career text embeddings vs. skill embeddings to understand signal redundancy and discriminative power.

## Data Sources
- Career text embeddings: encoded using `ElenaSenger/career-path-representation-mpnet-decorte` (768d)
- Skill embeddings: top-10 skills from last job concatenated as single string, encoded with same MPNet model (768d)  
- Target occupation embeddings: ESCO occupation descriptions encoded with same MPNet model (768d)
- Test set: DECORTE dataset, clean test split (N=227) and augmented test split (N=1802)

## Required Outputs

### Task 1: Text-Skill Embedding Redundancy
**Question:** Do text embeddings already capture skill-relevant information?
**Method:** 
- Compute pairwise cosine similarity between `v_C` (career text) and `h_C` (skill embedding) for each sample
- Report: mean, std, min, max, percentiles (25, 50, 75, 90, 99)
- Generate histogram visualization

### Task 3: Target Occupation Proximity  
**Question:** Which embedding type places samples closer to correct target occupation?
**Method:**
- For each test sample compute:
  - `sim_text = cosine_similarity(v_C, v_target)`
  - `sim_skill = cosine_similarity(h_C, v_target)`
- Report distribution statistics for both
- Compute paired difference: `delta = sim_text - sim_skill`
- Statistical test: paired t-test or Wilcoxon signed-rank
- Generate scatter plot: sim_text vs sim_skill with diagonal reference line

### Task 5: Correct vs Incorrect Prediction Patterns
**Question:** Do embedding characteristics differ between correctly and incorrectly predicted samples?
**Method:**
- Split samples by top-1 prediction correctness (from MLP text-only baseline)
- For each group (correct/incorrect), compute:
  - Text-skill similarity (from Task 1)
  - Text-to-target similarity (from Task 3)
  - Skill-to-target similarity (from Task 3)
- Statistical comparison between groups (t-test or Mann-Whitney U)
- Generate box plots comparing distributions

## Technical Requirements

1. **Input files needed:**
   - `text_embeddings.pkl` or load from model
   - `skill_embeddings.pkl` or generate from skill predictions
   - `target_occ_embeddings.pkl` or load ESCO embeddings
   - `predictions.pkl` with top-1 correctness labels

2. **Script structure:**
```python
# embedding_error_analysis.py

class EmbeddingAnalyzer:
    def __init__(self, config):
        self.load_embeddings()
        self.load_predictions()
    
    def task1_redundancy_analysis(self) -> dict:
        """Compute text-skill similarity statistics"""
        pass
    
    def task3_target_proximity(self) -> dict:
        """Compare distance to target occupation"""
        pass
    
    def task5_correctness_patterns(self) -> dict:
        """Analyze patterns by prediction correctness"""
        pass
    
    def generate_report(self):
        """Compile all results into summary table and figures"""
        pass

if __name__ == "__main__":
    analyzer = EmbeddingAnalyzer(config)
    results = {
        'task1': analyzer.task1_redundancy_analysis(),
        'task3': analyzer.task3_target_proximity(),
        'task5': analyzer.task5_correctness_patterns(),
    }
    analyzer.generate_report()
```

3. **Output format:**
   - Summary statistics as pandas DataFrame (markdown-friendly)
   - Figures saved as PNG (300 dpi)
   - Console logging with loguru
   - Final summary printed to console

4. **Libraries:** numpy, pandas, scipy.stats, sklearn.metrics.pairwise, matplotlib, seaborn, loguru

## Example Output Format
```
=== Task 1: Text-Skill Redundancy ===
| Metric | Value |
|--------|-------|
| Mean similarity | 0.313 |
| Std | 0.174 |
| ...

=== Task 3: Target Proximity ===
| Metric | Text→Target | Skill→Target | Delta |
|--------|-------------|--------------|-------|
| Mean | 0.XXX | 0.XXX | 0.XXX |
| ...

Paired t-test: t=X.XX, p=X.XXXX
Interpretation: [Text/Skill] embeddings are significantly closer to targets.

=== Task 5: Correctness Patterns ===
| Metric | Correct (n=XX) | Incorrect (n=XX) | p-value |
|--------|----------------|------------------|---------|
| Text-Skill sim | 0.XXX | 0.XXX | 0.XXX |
| ...
```

## Notes
- Both embedding types are 768d from same encoder, so direct cosine similarity is valid
- Use cosine similarity (not distance) for consistency with CPP evaluation
- Handle any NaN/missing values gracefully
- Add assertions to verify embedding dimensions match

## Technical Requirements
- Use exact same data loading and preprocessing as in CPP (/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/src/cpp/train_cpp_enhanced_v3.py)
- Utilize caching mechanism from train_cpp_enhanced_v3.py
- Use same required arguments (top-k, scoring mode, importance weight, etc.)
- Log regularly steps, log a sample skill document format etc.