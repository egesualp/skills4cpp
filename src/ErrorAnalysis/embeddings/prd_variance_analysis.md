## Cursor Prompt for Pooling Embedding Variance Analysis

```markdown
# Task: Embedding Variance Analysis for Skill Pooling Method

## Context
I'm analyzing why skill-enhanced career path prediction doesn't improve over text-only baselines. I have two embedding types from different encoders with different dimensions, and I need to compare their variance/spread characteristics to understand if skill embeddings are "collapsed" (low diversity).

## Data Sources
- Text embeddings: `text_emb` - shape (1802, 768) from MPNet-Decorte encoder
- Skill embeddings (pooled): `skill_emb_pooled` - shape (1802, 1024) from pjmath-BGE encoder
- These are from the DECORTE dataset augmented test split

## Key Challenge
Embeddings have different dimensions (768 vs 1024), so direct cosine similarity between paired samples is not possible. Analysis must use methods that work across different dimensions.

## Required Analyses

### Analysis 1: PCA Explained Variance
**Goal:** Compare how concentrated/spread the variance is in each embedding space.
**Method:**
- Fit PCA with 50 components on each embedding matrix
- Compare cumulative explained variance for top 10, 20, 50 components
- If skill embeddings have higher concentration (e.g., top 10 PC explain 80%+ variance), they are more "collapsed"

### Analysis 2: 2D PCA Visualization
**Goal:** Visually compare the spread/structure of both embedding spaces.
**Method:**
- Project each embedding to 2D using PCA
- Create side-by-side scatter plots
- Include explained variance percentages in titles

### Analysis 3: Pairwise Similarity Distribution
**Goal:** Measure how similar samples are to each other within each space.
**Method:**
- Compute pairwise cosine similarity matrix for each embedding type
- Extract upper triangle (exclude diagonal)
- Compare distributions: mean, std, histogram
- High mean pairwise similarity = embeddings are collapsed/similar to each other

### Analysis 4: CKA (Centered Kernel Alignment)
**Goal:** Measure structural similarity between the two embedding spaces despite different dimensions.
**Method:**
```python
def cka_linear(X, Y):
    """
    Compute CKA between matrices of different dimensions.
    X: (n_samples, dim1), Y: (n_samples, dim2)
    """
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)
    
    XXT = X @ X.T  # (N, N) Gram matrix
    YYT = Y @ Y.T  # (N, N) Gram matrix
    
    hsic_xy = np.sum(XXT * YYT)
    hsic_xx = np.sum(XXT * XXT)
    hsic_yy = np.sum(YYT * YYT)
    
    return hsic_xy / (np.sqrt(hsic_xx) * np.sqrt(hsic_yy))
```
- Output is scalar between 0 (orthogonal) and 1 (identical structure)

## Technical Requirements
- Use the exact same data loading and preprocessing that we used in src/cpp/train_cpp_enhanced_v2.py.
- You can also use the cached embeddings.
- After obtaining embedding vectors, run this analysis. 


### Input files:
```python
# Load embeddings - adjust paths as needed
text_emb = np.load("path/to/text_embeddings.npy")  # (1802, 768)
skill_emb_pooled = np.load("path/to/skill_embeddings_pooled.npy")  # (1802, 1024)
```

### Script structure:
```python
# embedding_variance_analysis.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from loguru import logger

class EmbeddingVarianceAnalyzer:
    def __init__(self, text_emb, skill_emb):
        self.text_emb = text_emb
        self.skill_emb = skill_emb
        self.results = {}
        
    def analyze_pca_variance(self, n_components=50):
        """Compare explained variance concentration."""
        pass
    
    def analyze_pairwise_similarity(self, n_sample=500):
        """Compare within-space pairwise similarity distributions."""
        pass
    
    def compute_cka(self):
        """Compute CKA between text and skill spaces."""
        pass
    
    def generate_visualization(self, save_path):
        """Create 3-panel figure: PCA text, PCA skill, similarity histograms."""
        pass
    
    def generate_report(self):
        """Print summary statistics."""
        pass

if __name__ == "__main__":
    # Load data
    text_emb = np.load("text_embeddings.npy")
    skill_emb = np.load("skill_embeddings_pooled.npy")
    
    analyzer = EmbeddingVarianceAnalyzer(text_emb, skill_emb)
    analyzer.analyze_pca_variance()
    analyzer.analyze_pairwise_similarity()
    analyzer.compute_cka()
    analyzer.generate_visualization("embedding_variance_pooled.png")
    analyzer.generate_report()
```

### Output format:

**Console output:**
```
============================================================
EMBEDDING VARIANCE ANALYSIS (Pooling Method)
============================================================
Embedding Shapes:
  Text:  (1802, 768)
  Skill: (1802, 1024)

PCA Explained Variance (Cumulative):
  Components    Text        Skill
  Top 10        XX.X%       XX.X%
  Top 20        XX.X%       XX.X%
  Top 50        XX.X%       XX.X%

Pairwise Cosine Similarity (within-space):
  Text:  mean=0.XXX, std=0.XXX, min=0.XXX, max=0.XXX
  Skill: mean=0.XXX, std=0.XXX, min=0.XXX, max=0.XXX

CKA Similarity (cross-space): 0.XXXX

Interpretation:
  - [Auto-generate based on results]
============================================================
```

**Visualization:** Single figure with 3 panels saved as PNG (300 dpi)
- Panel 1: Text embeddings 2D PCA scatter
- Panel 2: Skill embeddings 2D PCA scatter  
- Panel 3: Overlaid histograms of pairwise similarity distributions

### Libraries:
numpy, matplotlib, sklearn, loguru

## Expected Insights

If skill embeddings are "collapsed":
- Top 10 PCA components explain >70% variance (vs <50% for text)
- Pairwise similarity mean >0.5 (vs <0.3 for text)
- CKA might still be moderate (0.3-0.5) because structure is preserved but compressed

This would explain why skills don't help: they lack the diversity to discriminate between different career paths.

## Notes
- Use sampling (n=500) for pairwise similarity if full computation is slow
- Handle any numerical issues (division by zero in CKA)
- Add colorbar or density coloring to scatter plots if helpful
- Include axis labels and clear titles on all plots
