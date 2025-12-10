# Quick Start Guide - On-the-Fly Dataset

## TL;DR - Copy-Paste Template

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

# 1. Load encoder
encoder = SentenceTransformer("ElenaSenger/career-path-representation-mpnet-decorte")

# 2. Load helper maps
all_vocabs = load_all_vocabs("data/processed/master_datasets_2/")
job_skill_map, esco_skill_text_map, skill_properties_map = load_job_and_skill_data(
    master_skill_file="results/decorte_jobbert_v2_baseline/job_title_skills_master.csv",
    esco_skills_file="data/esco_datasets/skills_en.csv",
    skill_properties_file="data/processed/master_datasets_2/skill_properties_map.json",
    pooling_strategy="weighted_idf",
    alpha=1.0,
    beta=1.0
)

# 3. Load data
data = Data(DATA_TYPE='decorte')
train_pairs, val_pairs, test_pairs = data.get_data(stage='transformation_finetuning')
Y_target_dict = precompute_target_embeddings(encoder, list(data.labels))

# 4. Create datasets
train_dataset = CareerPathDataset(
    train_pairs, encoder, Y_target_dict, job_skill_map,
    esco_skill_text_map, skill_properties_map, all_vocabs,
    use_skill_description=False,
    pooling_strategy="weighted_idf",
    alpha=1.0, beta=1.0
)

val_dataset = CareerPathDataset(
    val_pairs, encoder, Y_target_dict, job_skill_map,
    esco_skill_text_map, skill_properties_map, all_vocabs,
    use_skill_description=False,
    pooling_strategy="weighted_idf",
    alpha=1.0, beta=1.0
)

# 5. Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, 
                         num_workers=4, collate_fn=collate_career_path_batch)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,
                       num_workers=4, collate_fn=collate_career_path_batch)

# 6. Train!
for batch in train_loader:
    # batch keys: 'h_text', 'h_skill_text', 'h_structured_structured', 'y'
    outputs = model(batch)
    loss = criterion(outputs, batch['y'])
    # ... backprop ...
```

## Command-Line Quick Start

```bash
# Test the dataset
python -m src.cpp.test_dataset

# Run example
python -m src.cpp.example_training_with_dataset --batch_size 32 --num_workers 4

# Start training
python -m src.cpp.train_cpp --pooling_strategy weighted_idf --num_epochs 10
```

## Dataset Configuration Options

| Parameter | Options | Description |
|-----------|---------|-------------|
| `pooling_strategy` | `"mean"`, `"weighted_mean"`, `"weighted_idf"` | How to pool skill embeddings |
| `use_skill_description` | `True`, `False` | Include skill descriptions in text |
| `alpha` | float (default: 1.0) | Exponent for confidence score |
| `beta` | float (default: 1.0) | Exponent for IDF score |
| `include_text` | `True`, `False` | Include text history features |
| `include_skill_text` | `True`, `False` | Include skill text features |
| `include_structured` | `True`, `False` | Include structured features |

## DataLoader Configuration

| Parameter | Recommended | Description |
|-----------|-------------|-------------|
| `batch_size` | 32-64 | Smaller for training, larger for validation |
| `num_workers` | 4-8 | Number of parallel workers (0 for debugging) |
| `shuffle` | True (train), False (val) | Shuffle data |
| `pin_memory` | True (GPU), False (CPU) | Faster GPU transfer |
| `collate_fn` | `collate_career_path_batch` | **Required** batch collation |

## Batch Structure

```python
batch = {
    'h_text': Tensor([batch_size, 768]),              # History text embedding
    'h_skill_text': Tensor([batch_size, 768]),        # Pooled skill text embedding
    'h_structured_structured': Tensor([batch_size, vocab_size]),  # Multi-hot structured
    'y': Tensor([batch_size, 768])                    # Target job embedding
}
```

## File Reference

| File | Purpose |
|------|---------|
| `cpp_dataset.py` | Main Dataset class |
| `data_loaders.py` | Data loading utilities |
| `train_cpp.py` | Training script template |
| `example_training_with_dataset.py` | Complete working example |
| `test_dataset.py` | Test script |
| `README_DATASET.md` | Full documentation |

## Common Commands

```bash
# Basic training
python -m src.cpp.train_cpp

# With IDF weighting
python -m src.cpp.train_cpp --pooling_strategy weighted_idf --alpha 1.0 --beta 1.0

# With skill descriptions
python -m src.cpp.train_cpp --use_skill_description

# Last job only
python -m src.cpp.train_cpp --last_job_only

# Custom batch size and workers
python -m src.cpp.train_cpp --batch_size 64 --num_workers 8

# GPU training
python -m src.cpp.train_cpp --device cuda
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Slow DataLoader | Increase `num_workers` (4-8) |
| Out of memory | Decrease `batch_size` or `num_workers` |
| Worker crashes | Set `num_workers=0`, check errors |
| Import errors | Check Python path, run from repo root |

## Performance Tips

1. ✅ Use `num_workers > 0` for parallel processing
2. ✅ Use larger batch size for validation (no gradients)
3. ✅ Set `pin_memory=True` when using GPU
4. ✅ Pre-compute target embeddings (already done in examples)
5. ✅ Cache encoders in workers (handled automatically)

## Next Steps

1. **First time**: Run `python -m src.cpp.test_dataset`
2. **Learn usage**: Run `python -m src.cpp.example_training_with_dataset`
3. **Start training**: Run `python -m src.cpp.train_cpp`
4. **Customize**: Edit `train_cpp.py` for your needs

---

**Need more details?** See `SUMMARY_ON_THE_FLY_DATASET.md` or `README_DATASET.md`

