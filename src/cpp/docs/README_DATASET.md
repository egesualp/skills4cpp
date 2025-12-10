# On-the-Fly Career Path Dataset

This directory contains an efficient on-the-fly dataset implementation for Career Path Prediction (CPP) that generates embeddings during training rather than pre-computing them.

## Overview

**Problem**: Pre-computing embeddings for all training samples requires significant storage space (gigabytes of `.npy` files).

**Solution**: Use PyTorch's `Dataset` and `DataLoader` to generate embeddings on-the-fly during training, leveraging multi-core processing for efficiency.

## Key Components

### 1. `cpp_dataset.py`
Contains the main `CareerPathDataset` class that:
- Generates embeddings dynamically for each sample
- Supports multiple pooling strategies (mean, weighted_mean, weighted_idf)
- Handles text variations (skill names vs. names + descriptions)
- Provides flexible feature modality selection

### 2. `data_loaders.py`
Shared utility functions for loading:
- Vocabularies for structured features
- Job-to-skill mappings
- ESCO skill text (names and descriptions)
- Skill meta-features (properties)
- Pre-computed target embeddings (still efficient!)

### 3. `example_training_with_dataset.py`
Complete working example showing how to:
- Load all required data
- Create Dataset instances
- Wrap them in DataLoaders
- Use them in a training loop

## Usage Example

```python
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

from src.cpp.data_classes import Data
from src.cpp.cpp_dataset import CareerPathDataset, collate_career_path_batch
from src.cpp.data_loaders import (
    load_all_vocabs,
    load_job_and_skill_data,
    precompute_target_embeddings
)

# Load encoder
encoder = SentenceTransformer("ElenaSenger/career-path-representation-mpnet-decorte")

# Load all helper maps
all_vocabs = load_all_vocabs("data/processed/master_datasets_2/")
job_skill_map, esco_skill_text_map, skill_properties_map = load_job_and_skill_data(
    master_skill_file="results/decorte_jobbert_v2_baseline/job_title_skills_master.csv",
    esco_skills_file="data/esco_datasets/skills_en.csv",
    skill_properties_file="data/processed/master_datasets_2/skill_properties_map.json",
    pooling_strategy="weighted_idf",
    alpha=1.0,
    beta=1.0
)

# Load raw text pairs
data = Data(DATA_TYPE='decorte')
train_pairs, val_pairs, test_pairs = data.get_data(stage='transformation_finetuning')

# Pre-compute target embeddings (one-time, much smaller than all training data)
Y_target_dict = precompute_target_embeddings(encoder, list(data.labels))

# Create datasets
train_dataset = CareerPathDataset(
    data_pairs=train_pairs,
    encoder=encoder,
    Y_target_dict=Y_target_dict,
    job_skill_map=job_skill_map,
    esco_skill_text_map=esco_skill_text_map,
    skill_properties_map=skill_properties_map,
    all_vocabs=all_vocabs,
    use_skill_description=False,
    pooling_strategy="weighted_idf",
    alpha=1.0,
    beta=1.0,
)

val_dataset = CareerPathDataset(
    data_pairs=val_pairs,
    encoder=encoder,
    Y_target_dict=Y_target_dict,
    job_skill_map=job_skill_map,
    esco_skill_text_map=esco_skill_text_map,
    skill_properties_map=skill_properties_map,
    all_vocabs=all_vocabs,
    use_skill_description=False,
    pooling_strategy="weighted_idf",
    alpha=1.0,
    beta=1.0,
)

# Wrap in DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    collate_fn=collate_career_path_batch,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_career_path_batch,
    pin_memory=True
)

# Use in training loop
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        # batch is a dict with keys:
        # - 'h_text': [batch_size, embed_dim]
        # - 'h_skill_text': [batch_size, embed_dim]
        # - 'h_structured_structured': [batch_size, vocab_size]
        # - 'y': [batch_size, embed_dim]
        
        outputs = model(batch)
        loss = criterion(outputs, batch['y'])
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

## Configuration Options

### Feature Modalities
- `include_text`: Include career history text embeddings
- `include_skill_text`: Include pooled skill text embeddings
- `include_structured`: Include multi-hot structured features

### Pooling Strategies
- `"mean"`: Simple average of skill embeddings
- `"weighted_mean"`: Weighted by confidence scores
- `"weighted_idf"`: Weighted by (confidence^alpha) * (IDF^beta)

### Text Variations
- `use_skill_description=False`: Use only skill names
- `use_skill_description=True`: Use "role: [name] \n description: [desc]"

## Performance Tips

1. **Multi-core Processing**: Set `num_workers > 0` in DataLoader for parallel embedding generation
2. **Batch Size**: Larger batches for validation/test (no gradients to store)
3. **Pin Memory**: Set `pin_memory=True` for faster GPU transfer
4. **Pre-compute Targets**: Target embeddings are pre-computed once (much smaller than all training data)

## Comparison: Pre-computed vs. On-the-Fly

### Pre-computed (`generate_embeddings.py`)
✅ Faster training (embeddings already computed)  
✅ Deterministic (same embeddings every time)  
❌ **Huge storage requirement** (GBs of `.npy` files)  
❌ Less flexible (need to regenerate for different configs)

### On-the-Fly (`cpp_dataset.py`)
✅ **Minimal storage** (only raw text pairs)  
✅ Very flexible (change configs without regenerating)  
✅ Multi-core processing for efficiency  
⚠️ Slightly slower training (embedding overhead)  
⚠️ Non-deterministic (minor embedding variations)

## Migration Guide

If you're currently using pre-computed embeddings:

**Before:**
```python
# Load pre-computed .npy files
train_text = np.load("train_text.npy")
train_skill_text = np.load("train_skill_text.npy")
train_y = np.load("train_y.npy")
```

**After:**
```python
# Create on-the-fly dataset
train_dataset = CareerPathDataset(train_pairs, encoder, ...)
train_loader = DataLoader(train_dataset, batch_size=32, num_workers=4)

for batch in train_loader:
    # batch['h_text'], batch['h_skill_text'], batch['y']
    ...
```

## Running the Example

```bash
cd /dss/dsshome1/02/ra95kix2/thesis/skills4cpp

# Run with default settings
python -m src.cpp.example_training_with_dataset

# Run with custom configuration
python -m src.cpp.example_training_with_dataset \
    --pooling_strategy weighted_idf \
    --alpha 1.0 \
    --beta 1.0 \
    --use_skill_description \
    --batch_size 32 \
    --num_workers 4
```

## Notes

- The encoder model is **shared** across all workers (read-only), so memory usage is efficient
- Target embeddings (`Y_target_dict`) are pre-computed once since there are far fewer unique targets than training samples
- Structured features are generated as sparse multi-hot vectors
- The dataset automatically filters out samples with missing target embeddings



