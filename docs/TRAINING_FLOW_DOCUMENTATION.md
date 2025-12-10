# Career Path Prediction Training Flow - Complete Documentation

## Overview
This document provides a detailed walkthrough of the **Enhanced Career Path Prediction Training Script** (`train_cpp_enhanced.py`), explaining every step, function call, data preprocessing operation, and their purposes in the natural order of execution.

---

## Table of Contents
1. [Script Entry Point](#1-script-entry-point)
2. [Argument Parsing](#2-argument-parsing)
3. [Multiprocessing Configuration](#3-multiprocessing-configuration)
4. [Modality Validation & Architecture Selection](#4-modality-validation--architecture-selection)
5. [Weights & Biases Initialization](#5-weights--biases-initialization)
6. [Step 1: Load Encoder Models](#6-step-1-load-encoder-models)
7. [Step 2: Load Vocabularies and Skill Mappings](#7-step-2-load-vocabularies-and-skill-mappings)
8. [Step 3: Load Career Path Data](#8-step-3-load-career-path-data)
9. [Step 4: Pre-compute Target Embeddings](#9-step-4-pre-compute-target-embeddings)
10. [Step 4b: Pre-compute Input Embeddings (NEW)](#10-step-4b-pre-compute-input-embeddings-new)
11. [Step 5: Create PyTorch Datasets](#11-step-5-create-pytorch-datasets)
12. [Step 6: Hyperparameter Optimization](#12-step-6-hyperparameter-optimization)
13. [Step 7: Final Model Training](#13-step-7-final-model-training)
14. [Model Architectures](#14-model-architectures)
15. [Training & Evaluation Functions](#15-training--evaluation-functions)
16. [Data Flow Summary](#16-data-flow-summary)

---

## 1. Script Entry Point

### Function: `if __name__ == "__main__": main()`
**Location:** Line 989-990

When you run the script with `python train_cpp_enhanced.py [args]`, the execution starts here:

```python
if __name__ == "__main__":
    main()
```

This calls the `main()` function which orchestrates the entire training pipeline.

---

## 2. Argument Parsing

### Function: `parse_args()`
**Location:** Lines 459-531  
**Called from:** `main()` at line 539  
**Purpose:** Parse command-line arguments to configure the training run

#### Key Arguments Parsed:

**Data Configuration:**
- `--data_type`: Dataset to use (e.g., "decorte", "karrierewege")
- `--master_skill_file`: CSV linking job titles to skill URIs and confidence scores
- `--esco_skills_file`: ESCO skill descriptions
- `--vocab_dir`: Directory containing vocabulary JSON files for structured features
- `--skill_properties_file`: JSON mapping skills to meta-features

**Encoder Configuration:**
- `--encoder_text`: SentenceTransformer model for encoding job history text
- `--encoder_skill`: Optional separate encoder for skills (defaults to same as text)

**Feature Configuration:**
- `--use_text_description`: Include job descriptions (vs. titles only)
- `--use_skill_description`: Include skill descriptions (vs. names only)
- `--last_job_only`: Filter to single-job histories
- `--pooling_strategy`: How to pool skill embeddings ("mean", "weighted_mean", "weighted_idf")
- `--alpha`, `--beta`: Exponents for weighted_idf pooling

**Modality Selection (for ablation studies):**
- `--use_text_history`: Include job history text features
- `--use_skill_text`: Include skill text features
- `--use_structured`: Include structured meta-features

**Architecture:**
- `--use_advanced`: Use multi-modal architecture (auto-enabled for 2+ modalities)

**Training Configuration:**
- `--optuna`: Run Optuna hyperparameter optimization
- `--n_trials`: Number of Optuna trials
- `--max_epochs`: Maximum epochs per trial
- `--patience`: Early stopping patience
- `--batch_size`, `--num_workers`, `--device`

**Static Hyperparameters (when not using Optuna):**
- `--lr`: Learning rate
- `--hidden_dim`: Hidden layer dimension
- `--n_layers`: Number of layers
- `--dropout`: Dropout rate
- `--use_modality_weights`: Use learnable modality weights

**Output:**
- `--output_dir`: Where to save model checkpoints
- `--results_csv_path`: CSV file to append experiment results
- `--use_wandb`: Enable Weights & Biases logging
- `--run_name`: Name for this experiment run

---

## 3. Multiprocessing Configuration

### Code Block: Lines 541-548
**Purpose:** Fix CUDA multiprocessing issues with DataLoader workers

```python
try:
    multiprocessing.set_start_method('spawn', force=True)
    logger.info("🖥️  CUDA multiprocessing set to 'spawn'")
except RuntimeError:
    # Already set, which is fine
    pass
```

**Why:** PyTorch's DataLoader uses multiprocessing. On Linux, the default 'fork' method can cause issues with CUDA. Setting to 'spawn' ensures each worker process has a clean CUDA context.

---

## 4. Modality Validation & Architecture Selection

### Code Block: Lines 551-572
**Purpose:** Validate at least one modality is enabled and auto-select architecture

#### Validation:
```python
n_active_modalities = sum([args.use_text_history, args.use_skill_text, args.use_structured])
if n_active_modalities == 0:
    raise ValueError("At least one modality must be enabled!")
```

#### Auto-Architecture Selection:
- If 2+ modalities: Auto-enable `MultiModalCPPModel` (unless explicitly disabled)
- If 1 modality: Use `SimpleConcatModel` (unless advanced explicitly requested)

**Why:** Multi-modal architecture has separate encoders per modality with a fusion layer, better for learning modality-specific representations. Simple concatenation is more efficient for single-modality experiments.

---

## 5. Weights & Biases Initialization

### Code Block: Lines 575-583
**Purpose:** Initialize experiment tracking (optional)

```python
if WANDB_AVAILABLE and args.use_wandb:
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=args,
        name=args.run_name,
        reinit=True
    )
```

**Why:** Track experiments, hyperparameters, and metrics in the cloud for comparison and analysis.

---

## 6. Step 1: Load Encoder Models

### Code Block: Lines 588-603
**Function Called:** `SentenceTransformer()` from sentence-transformers library

```python
encoder_text = SentenceTransformer(args.encoder_text)

if args.encoder_skill:
    encoder_skill = SentenceTransformer(args.encoder_skill)
    skill_text_dim = encoder_skill.get_sentence_embedding_dimension()
else:
    encoder_skill = encoder_text
    skill_text_dim = encoder_text.get_sentence_embedding_dimension()

text_dim = encoder_text.get_sentence_embedding_dimension()
```

**What Happens:**
1. Load pre-trained SentenceTransformer model for encoding job history text
2. Optionally load separate encoder for skills (or reuse the same one)
3. Get embedding dimensions (typically 768 for BERT-based models)

**Data Generated:**
- `encoder_text`: Model that converts text strings → dense vectors
- `encoder_skill`: Model for skill text embeddings
- `text_dim`: Dimension of text embeddings (e.g., 768)
- `skill_text_dim`: Dimension of skill embeddings (e.g., 768)

**Why:** These encoders convert textual job descriptions and skill names into dense vector representations that neural networks can process.

---

## 7. Step 2: Load Vocabularies and Skill Mappings

### Code Block: Lines 606-618
**Functions Called:**
1. `load_all_vocabs()` from `data_loaders.py`
2. `load_job_and_skill_data()` from `data_loaders.py`

### 7.1: Load Vocabularies

#### Function: `load_all_vocabs(vocab_dir)`
**Location:** `data_loaders.py`, lines 18-40  
**Purpose:** Load vocabulary mappings for structured features

**What it Does:**
1. Scans `vocab_dir` for all files ending in `_vocab.json`
2. Loads each JSON file into a dictionary
3. Returns `all_vocabs`: `{feature_name: {value: index}}`

**Example Output:**
```python
{
    'structured': {
        'skillType:knowledge': 0,
        'skillType:skill': 1,
        'reuseLevel:cross-sector': 2,
        'reuseLevel:sector-specific': 3,
        # ... more entries
    }
}
```

**Dimension Calculation:**
```python
structured_dim = sum(len(vocab) for vocab in all_vocabs.values())
```

If there are 100 unique structured features, `structured_dim = 100`.

### 7.2: Load Job-Skill Mappings

#### Function: `load_job_and_skill_data()`
**Location:** `data_loaders.py`, lines 43-142  
**Purpose:** Load mappings between jobs and skills, with scoring information

**Inputs:**
- `master_skill_file`: CSV with columns `[job_title, skillUri, score]`
- `esco_skills_file`: CSV with ESCO skill names and descriptions
- `skill_properties_file`: JSON mapping skills to meta-features
- `pooling_strategy`, `alpha`, `beta`: Controls IDF calculation

**What it Does:**

1. **Load Master Skill File** (lines 69-79):
   - Reads CSV linking job titles to skill URIs with confidence scores
   - Example row: `{"job_title": "software engineer", "skillUri": "http://...", "score": 0.85}`

2. **Calculate IDF Scores** (lines 82-101) - **Only if `pooling_strategy == "weighted_idf"`**:
   ```python
   N_occ = df['job_title'].nunique()  # Total unique job titles
   skill_n_occ = df.groupby('skillUri')['job_title'].nunique()  # Jobs per skill
   idf_map = np.log((N_occ + 1) / (skill_n_occ + 1))  # IDF formula
   df['idf'] = df['skillUri'].map(idf_map)
   ```
   
   **Why IDF?** Skills appearing in many jobs (e.g., "communication") are less distinctive than rare skills (e.g., "quantum computing"). IDF down-weights common skills.

3. **Build Job→Skill Map** (lines 103-115):
   ```python
   job_skill_map = {
       'software engineer': [
           {'skillUri': 'http://...', 'score': 0.85, 'idf': 2.3},
           {'skillUri': 'http://...', 'score': 0.72, 'idf': 1.1},
           # ...
       ],
       # ... more jobs
   }
   ```

4. **Load ESCO Skill Text** (lines 117-131):
   ```python
   esco_skill_text_map = {
       'http://data.europa.eu/esco/skill/...': {
           'name': 'Python programming',
           'desc': 'The skill to write programs in Python...'
       },
       # ... more skills
   }
   ```

5. **Load Skill Properties** (lines 133-140):
   ```python
   skill_properties_map = {
       'http://data.europa.eu/esco/skill/...': [
           'skillType:skill',
           'reuseLevel:cross-sector'
       ],
       # ... more skills
   }
   ```

**Returns:**
- `job_skill_map`: Maps job titles → list of skill dicts with scores/IDF
- `esco_skill_text_map`: Maps skill URIs → skill names and descriptions
- `skill_properties_map`: Maps skill URIs → list of meta-feature strings

**Why These Mappings:**
- **job_skill_map**: Links career history text to relevant skills
- **esco_skill_text_map**: Provides text to embed for each skill
- **skill_properties_map**: Provides structured categorical features

---

## 8. Step 3: Load Career Path Data

### Code Block: Lines 621-634
**Functions Called:**
1. `Data()` class constructor from `data_classes.py`
2. `data.get_data(stage='transformation_finetuning')`

### 8.1: Data Class Initialization

#### Class: `Data(DATA_TYPE, ONLY_TITLES)`
**Location:** `data_classes.py`, lines 24-42  
**Purpose:** Load and preprocess career path sequences

**What Happens:**
```python
data = Data(DATA_TYPE=args.data_type, ONLY_TITLES=not args.use_text_description)
```

**Initialization Steps:**
1. Calls `__load_data()` (lines 44-77)
2. Based on `DATA_TYPE`, calls appropriate loader:
   - `'decorte'` → `load_prepare_decorte()`
   - `'karrierewege'` → `load_prepare_karrierewege()`
   - etc.

**Data Format Loaded:**
Each loader returns `(train_pairs, val_pairs, test_pairs)` where each pair is:
```python
(
    "role: software developer\n description: Develops software...\n role: senior developer\n description: Leads teams...",
    "role: engineering manager\n description: Manages engineering teams..."
)
```

Or if `ONLY_TITLES=True`:
```python
("software developer <SEP> senior developer", "engineering manager")
```

### 8.2: Get Processed Data

#### Method: `data.get_data(stage='transformation_finetuning')`
**Location:** `data_classes.py`, lines 122-150  
**Purpose:** Apply stage-specific preprocessing

**For stage='transformation_finetuning':**
1. If `ONLY_TITLES=True`: Extract titles using regex `r"role: (.*?)\n"`
2. Apply `__minus_last()`: Remove last segment from history (lines 80-98)
   - Input: `"job1 <SEP> job2 <SEP> job3"`
   - Output: `"job1 <SEP> job2"` (predict job3)

**Why Minus Last:**
This creates the prediction task: given first N-1 jobs, predict the Nth job.

### 8.3: Optional Last-Job-Only Filtering

```python
if args.last_job_only:
    train_pairs = [pair for pair in train_pairs if SEP_TOKEN not in pair[0]]
    val_pairs = [pair for pair in val_pairs if SEP_TOKEN not in pair[0]]
    test_pairs = [pair for pair in test_pairs if SEP_TOKEN not in pair[0]]
```

**Why:** For ablation studies comparing single-job vs. multi-job histories.

**Data After This Step:**
- `train_pairs`: List of `(history, target)` tuples
- `val_pairs`: Validation pairs
- `test_pairs`: Test pairs

Example:
```python
train_pairs = [
    ("data analyst", "senior data analyst"),
    ("software developer <SEP> senior developer", "team lead"),
    # ... thousands more
]
```

---

## 9. Step 4: Pre-compute Target Embeddings

### Code Block: Lines 637-643
**Function Called:** `precompute_target_embeddings()`

#### Function: `precompute_target_embeddings(encoder, labels)`
**Location:** `data_loaders.py`, lines 145-161  
**Purpose:** Pre-compute embeddings for all possible target jobs

**What Happens:**
```python
actual_labels = list(set([pair[1] for pair in train_pairs + val_pairs + test_pairs]))
Y_target_dict = precompute_target_embeddings(encoder_text, actual_labels, show_progress=True)
Y_target_all = np.array(list(Y_target_dict.values()))
output_dim = Y_target_all.shape[1]
```

**Step-by-Step:**
1. **Extract Unique Targets:**
   ```python
   actual_labels = ['senior data analyst', 'team lead', 'engineering manager', ...]
   # Say we have 500 unique target jobs
   ```

2. **Encode All Targets:**
   ```python
   target_embeddings = encoder.encode(actual_labels, convert_to_numpy=True)
   # Shape: [500, 768]
   ```

3. **Create Dictionary:**
   ```python
   Y_target_dict = {
       'senior data analyst': array([0.1, -0.2, ..., 0.5]),  # 768-dim vector
       'team lead': array([0.3, 0.1, ..., -0.2]),
       # ... 500 entries
   }
   ```

4. **Create Matrix:**
   ```python
   Y_target_all = np.array([...])  # Shape: [500, 768]
   output_dim = 768  # Embedding dimension
   ```

**Why Pre-compute:**
- Target embeddings are fixed (not learned)
- Computing once saves time during training
- Used for:
  1. Training loss (comparing predictions to targets)
  2. Evaluation metrics (finding nearest neighbors)

---

## 10. Step 4b: Pre-compute Input Embeddings (NEW)

### Code Block: Lines 793-819
**Function Called:** `precompute_input_embeddings()` from `data_loaders.py`

**Purpose:** Pre-compute text history and skill text embeddings for all samples to avoid redundant encoding during training.

### Why This Optimization?

Previously, the dataset computed embeddings **on-the-fly** during training:
- Each epoch re-encoded the same job histories and skills
- With weighted_idf pooling, this involved encoding hundreds of skills per sample
- Training was slow due to redundant encoder calls

**Solution:** Pre-compute embeddings once before training starts.

### Function: `precompute_input_embeddings()`

**Location:** `data_loaders.py`, lines 164-280  
**Purpose:** Batch-encode all text histories and aggregate skill embeddings upfront

**What Happens:**

```python
train_pairs, train_h_text, train_h_skill = precompute_input_embeddings(
    train_pairs, Y_target_dict, encoder_text, encoder_skill,
    job_skill_map, esco_skill_text_map,
    use_skill_description=args.use_skill_description,
    pooling_strategy=args.pooling_strategy, 
    alpha=args.alpha, 
    beta=args.beta,
    use_text_history=args.use_text_history, 
    use_skill_text=args.use_skill_text
)
```

### Step-by-Step Process:

**1. Extract Valid Pairs** (lines 194-201):
```python
valid_pairs = []
for history_doc, target_doc in data_pairs:
    if target_doc in Y_target_dict:
        valid_pairs.append((history_doc, target_doc))
```

**2. Pre-compute Text History Embeddings** (lines 204-211):
```python
if use_text_history:
    history_texts = [pair[0] for pair in valid_pairs]
    # Batch encode all histories at once
    h_text_vectors = encoder_text.encode(
        history_texts, 
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=True
    )
    # Result: [n_samples, 768]
```

**3. Pre-compute Skill Text Embeddings** (lines 214-276):

For each sample:
```python
# Extract job titles from history
raw_titles = re.findall(r"role: (.*?)\n", history_doc)

# Aggregate skills from all jobs
skill_info_list = []
for title in raw_titles:
    if title.strip() in job_skill_map:
        skill_info_list.extend(job_skill_map[title.strip()])

# Build skill texts and weights
strings_to_embed = []
weights_for_pooling = []

for skill_info in skill_info_list:
    skill_uri = skill_info['skillUri']
    skill_data = esco_skill_text_map[skill_uri]
    
    # Format skill text
    if use_skill_description:
        text = f"role: {skill_data['name']} \n description: {skill_data['desc']}"
    else:
        text = skill_data['name']
    strings_to_embed.append(text)
    
    # Calculate weight
    if pooling_strategy == "weighted_idf":
        weight = (skill_info['score'] ** alpha) * (skill_info['idf'] ** beta)
    elif pooling_strategy == "weighted_mean":
        weight = skill_info['score']
    else:  # "mean"
        weight = 1.0
    weights_for_pooling.append(weight)

# Encode all skills for this sample
skill_embeddings = encoder_skill.encode(strings_to_embed, convert_to_numpy=True)
# Shape: [n_skills, 768]

# Weighted average pooling
h_skill = np.average(skill_embeddings, axis=0, weights=weights_for_pooling)
# Shape: [768]

h_skill_vectors.append(h_skill)
```

Repeat for all samples:
```python
h_skill_vectors = np.array(h_skill_vectors)  # [n_samples, 768]
```

**Returns:**
- `valid_pairs`: Filtered list of (history, target) tuples
- `h_text_vectors`: Pre-computed text embeddings `[n_samples, 768]` or `None`
- `h_skill_vectors`: Pre-computed skill embeddings `[n_samples, 768]` or `None`

### Applied to All Splits:

```python
# Train set
train_pairs, train_h_text, train_h_skill = precompute_input_embeddings(train_pairs, ...)

# Validation set
val_pairs, val_h_text, val_h_skill = precompute_input_embeddings(val_pairs, ...)

# Test set
test_pairs, test_h_text, test_h_skill = precompute_input_embeddings(test_pairs, ...)
```

### Performance Impact:

**Before (on-the-fly encoding):**
- Each training epoch: Encode all samples again
- 10 epochs × 10,000 samples = 100,000 encoding operations

**After (pre-computed):**
- One-time encoding: 10,000 samples
- Training just uses pre-computed arrays
- **5-10x speedup** in dataset iteration

**Memory Trade-off:**
- Store embeddings: ~60MB for 10,000 samples (768-dim float32)
- Much faster training: Worth the memory cost

---

## 11. Step 5: Create PyTorch Datasets

### Code Block: Lines 941-1021
**Functions Called:**
1. `CareerPathDataset()` class instantiation (3 times: train, val, test)
2. `DataLoader()` from PyTorch

### 11.1: CareerPathDataset Class

#### Class: `CareerPathDataset`
**Location:** `cpp_dataset.py`, lines 15-299  
**Purpose:** PyTorch Dataset that can use pre-computed embeddings OR generate features on-the-fly

**Initialization Parameters:**
- `data_pairs`: List of (history, target) tuples
- `encoder`: SentenceTransformer for text history
- `Y_target_dict`: Pre-computed target embeddings
- `job_skill_map`: Job → skills mapping
- `esco_skill_text_map`: Skill URI → text
- `skill_properties_map`: Skill URI → meta-features
- `all_vocabs`: Vocabularies for structured features
- `use_skill_description`: Include skill descriptions?
- `pooling_strategy`: "mean", "weighted_mean", or "weighted_idf"
- `alpha`, `beta`: Pooling exponents
- `encoder_skill`: Optional separate skill encoder
- `include_text`, `include_skill_text`, `include_structured`: Feature toggles
- **`pre_h_text`**: (NEW) Pre-computed text embeddings array `[n_samples, 768]` or `None`
- **`pre_h_skill_text`**: (NEW) Pre-computed skill embeddings array `[n_samples, 768]` or `None`

**Initialization Steps:**
1. **Store all configurations** (lines 67-83)
2. **Store pre-computed embeddings** (NEW):
   ```python
   self.pre_h_text = pre_h_text  # [n_samples, 768] or None
   self.pre_h_skill_text = pre_h_skill_text  # [n_samples, 768] or None
   ```
3. **Pre-compute dimensions** (lines 89-93)
4. **Create zero vectors for padding** (lines 96-100) - returned when no skills found
5. **Filter valid samples** (lines 103-114) - remove pairs with missing target embeddings

**Dataset Creation with Pre-computed Embeddings:**

```python
train_dataset = CareerPathDataset(
    data_pairs=train_pairs,
    encoder=encoder_text,
    Y_target_dict=Y_target_dict,
    job_skill_map=job_skill_map,
    esco_skill_text_map=esco_skill_text_map,
    skill_properties_map=skill_properties_map,
    all_vocabs=all_vocabs,
    use_skill_description=args.use_skill_description,
    pooling_strategy=args.pooling_strategy,
    alpha=args.alpha,
    beta=args.beta,
    encoder_skill=encoder_skill,
    include_text=args.use_text_history,
    include_skill_text=args.use_skill_text,
    include_structured=args.use_structured,
    pre_h_text=train_h_text,        # ← Pre-computed text embeddings
    pre_h_skill_text=train_h_skill,  # ← Pre-computed skill embeddings
)
```

### 11.2: Dataset __getitem__ Method

#### Method: `CareerPathDataset.__getitem__(idx)`
**Location:** `cpp_dataset.py`, lines 120-182  
**Purpose:** Generate features for a single training sample

**Called:** Automatically by PyTorch DataLoader during training

**NEW BEHAVIOR:** Uses pre-computed embeddings when available, otherwise computes on-the-fly.

**Example Flow for one sample:**

**Input:**
```python
history_doc = "role: data analyst\n description: Analyzes data...\n role: senior analyst\n description: Leads analysis..."
target_doc = "role: data scientist\n description: Builds ML models..."
```

**Step 1: Generate h_text (Text History Embedding)**

**With Pre-computed Embeddings (NEW):**
```python
if self.include_text:
    if self.pre_h_text is not None:
        # Use pre-computed embedding directly
        h_text = self.pre_h_text[idx]
        features['h_text'] = torch.from_numpy(h_text).float()
    else:
        # Fall back to on-the-fly encoding
        h_text = self.encoder.encode(history_doc, convert_to_numpy=True)
        features['h_text'] = torch.from_numpy(h_text).float()
    # Shape: [768]
```

**Step 2: Get Target y**
```python
y_vector = self.Y_target_dict[target_doc]
features['y'] = torch.from_numpy(y_vector).float()
# Shape: [768]
```

**Step 3: Extract Skills from History** (lines 150-156)
```python
raw_titles_in_history = re.findall(r"role: (.*?)\n", history_doc)
# Result: ['data analyst', 'senior analyst']

skill_info_list = []
for title in raw_titles_in_history:
    if title.strip() in self.job_skill_map:
        skill_info_list.extend(self.job_skill_map[title.strip()])

# skill_info_list now contains all skills from both jobs:
# [
#     {'skillUri': 'http://...', 'score': 0.9, 'idf': 2.1},
#     {'skillUri': 'http://...', 'score': 0.8, 'idf': 1.5},
#     # ... 20 skills total
# ]
```

**Step 4: Generate h_skill_text (Pooled Skill Embedding)**

**With Pre-computed Embeddings (NEW):**
```python
if self.include_skill_text:
    if self.pre_h_skill_text is not None:
        # Use pre-computed skill embedding directly
        h_skill_text = self.pre_h_skill_text[idx]
        features['h_skill_text'] = torch.from_numpy(h_skill_text).float()
    else:
        # Fall back to on-the-fly computation (lines 169-174)
        # Calls _generate_skill_text_embedding(skill_info_list)
```

**On-the-fly Computation (when pre-computed not available):**

Calls `_generate_skill_text_embedding(skill_info_list)` (lines 184-235):

```python
strings_to_embed = []
weights_for_pooling = []

for skill_info in skill_info_list:
    skill_uri = skill_info['skillUri']
    skill_text = self.esco_skill_text_map[skill_uri]
    
    # Format text based on configuration
    if self.use_skill_description:
        text = f"role: {skill_text['name']} \n description: {skill_text['desc']}"
    else:
        text = skill_text['name']
    strings_to_embed.append(text)
    
    # Calculate weight based on pooling strategy
    if self.pooling_strategy == "mean":
        weights_for_pooling.append(1.0)
    elif self.pooling_strategy == "weighted_mean":
        weights_for_pooling.append(skill_info['score'])
    elif self.pooling_strategy == "weighted_idf":
        c_i = skill_info['score']
        idf_i = skill_info['idf']
        weight = (c_i ** self.alpha) * (idf_i ** self.beta)
        weights_for_pooling.append(weight)

# Encode all skill texts
skill_embeddings = self.encoder_skill.encode(strings_to_embed, convert_to_numpy=True)
# Shape: [20, 768] (20 skills, 768-dim embeddings)

# Pool with weights
weights = np.array(weights_for_pooling)
h_skill_text = np.average(skill_embeddings, axis=0, weights=weights)
# Shape: [768] (single pooled vector)
```

**Step 5: Generate h_structured (Multi-hot Features)** (lines 177-180)

Calls `_generate_structured_features(skill_info_list)` (lines 237-264):

```python
structured_vectors = {
    'structured': np.zeros(100, dtype=np.float32)  # 100 = structured_dim
}

for skill_info in skill_info_list:
    skill_uri = skill_info['skillUri']
    
    if skill_uri in self.skill_properties_map:
        features = self.skill_properties_map[skill_uri]
        # features = ['skillType:skill', 'reuseLevel:cross-sector']
        
        for feature_string in features:
            if feature_string in self.all_vocabs['structured']:
                idx = self.all_vocabs['structured'][feature_string]
                structured_vectors['structured'][idx] = 1.0

# Result: Multi-hot vector where 1.0 indicates presence of that meta-feature
# Shape: [100]
```

**Final Output (same whether pre-computed or on-the-fly):**
```python
{
    'h_text': tensor([0.1, -0.2, ..., 0.5]),           # [768]
    'h_skill_text': tensor([0.3, 0.1, ..., -0.1]),     # [768]
    'h_structured_structured': tensor([0, 1, 1, ...]), # [100]
    'y': tensor([0.2, -0.1, ..., 0.3])                 # [768]
}
```

**Performance Comparison:**

| Mode | Speed per Epoch | Notes |
|------|----------------|-------|
| On-the-fly | ~5 minutes | Re-encodes every sample each epoch |
| Pre-computed | ~30 seconds | Just loads from memory arrays |

### 11.3: DataLoader Creation

```python
train_loader = DataLoader(
    train_dataset, 
    batch_size=32, 
    shuffle=True,
    num_workers=4, 
    collate_fn=collate_career_path_batch,
    pin_memory=True
)
```

**Function:** `collate_career_path_batch(batch)`  
**Location:** `cpp_dataset.py`, lines 267-288  
**Purpose:** Stack individual samples into batches

**Example:**
```python
# Input: List of 32 sample dicts
batch = [sample1, sample2, ..., sample32]

# Output: Single dict with batched tensors
{
    'h_text': tensor([[...], [...], ...]),           # [32, 768]
    'h_skill_text': tensor([[...], [...], ...]),     # [32, 768]
    'h_structured_structured': tensor([[...], ...]), # [32, 100]
    'y': tensor([[...], [...], ...])                 # [32, 768]
}
```

---

## 12. Step 6: Hyperparameter Optimization

### Code Block: Lines 1064-1180
**Choice:** Optuna optimization OR static hyperparameters

### Option A: Optuna Optimization (if `args.optuna=True`)

#### Function: `objective(trial, ...)`
**Location:** Lines 343-447  
**Purpose:** Single Optuna trial - train model with suggested hyperparameters

**Flow for Each Trial:**

**1. Suggest Hyperparameters** (lines 350-396):
```python
# Multi-modal architecture
hidden_dim = trial.suggest_categorical("hidden_dim", [256, 512, 768])
n_layers = trial.suggest_int("n_layers", 1, 3)
dropout = trial.suggest_float("dropout", 0.1, 0.5)
use_modality_weights = trial.suggest_categorical("use_modality_weights", [True, False])
lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
```

**2. Build Model** with suggested hyperparameters

**3. Training Loop** (lines 406-444):
```python
for epoch in range(args.max_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_metrics = evaluate(model, val_loader, Y_target_all, device, criterion)
    val_mrr = val_metrics['MRR']
    
    # Report to Optuna for pruning
    trial.report(val_mrr, epoch)
    if trial.should_prune():
        raise optuna.exceptions.TrialPruned()
    
    # Early stopping
    if val_mrr > best_val_mrr:
        best_val_mrr = val_mrr
        epochs_no_improve = 0
        best_epoch = epoch + 1
    else:
        epochs_no_improve += 1
    
    if epochs_no_improve >= args.patience:
        break

return best_val_mrr  # Optuna tries to maximize this
```

**4. Run Study** (lines 728-744):
```python
study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner()
)

study.optimize(
    lambda trial: objective(trial, train_loader, val_loader, Y_target_all, ...),
    n_trials=50,
    callbacks=[logger_callback]
)
```

**Optuna's Strategy:**
- Trial 1: Try random hyperparameters
- Trial 2-N: Bayesian optimization suggests promising hyperparameters
- Pruning: Stop bad trials early (if worse than median of previous trials)

**Best Parameters Retrieved:**
```python
hidden_dim = study.best_params["hidden_dim"]
n_layers = study.best_params["n_layers"]
dropout = study.best_params["dropout"]
lr = study.best_params["lr"]
use_modality_weights = study.best_params["use_modality_weights"]
final_epochs = study.best_trial.user_attrs["best_epoch"] + 1
```

### Option B: Static Hyperparameters (if `args.optuna=False`)

**Flow** (lines 764-836):
1. Use hyperparameters from command-line arguments
2. Build model with static hyperparameters
3. Train with early stopping to find optimal epoch count
4. Record `best_epoch` for final training

---

## 13. Step 7: Final Model Training

### Code Block: Lines 1183-1362
**Purpose:** Train final model on train+val with best hyperparameters

### 13.1: Combine Train and Validation (with Pre-computation)

**NEW:** Pre-compute embeddings for combined dataset:

```python
combined_pairs = train_pairs + val_pairs

# Pre-compute embeddings for combined data
combined_pairs, combined_h_text, combined_h_skill = precompute_input_embeddings(
    combined_pairs, Y_target_dict, encoder_text, encoder_skill,
    job_skill_map, esco_skill_text_map,
    use_skill_description=args.use_skill_description,
    pooling_strategy=args.pooling_strategy, 
    alpha=args.alpha, 
    beta=args.beta,
    use_text_history=args.use_text_history, 
    use_skill_text=args.use_skill_text
)

# Create dataset with pre-computed embeddings
combined_dataset = CareerPathDataset(
    data_pairs=combined_pairs,
    encoder=encoder_text,
    Y_target_dict=Y_target_dict,
    job_skill_map=job_skill_map,
    esco_skill_text_map=esco_skill_text_map,
    skill_properties_map=skill_properties_map,
    all_vocabs=all_vocabs,
    use_skill_description=args.use_skill_description,
    pooling_strategy=args.pooling_strategy,
    alpha=args.alpha,
    beta=args.beta,
    encoder_skill=encoder_skill,
    include_text=args.use_text_history,
    include_skill_text=args.use_skill_text,
    include_structured=args.use_structured,
    pre_h_text=combined_h_text,        # ← Pre-computed
    pre_h_skill_text=combined_h_skill,  # ← Pre-computed
)

combined_loader = DataLoader(...)
```

**Why:** Use all available training data for final model since we've already selected hyperparameters.

### 13.2: Build Final Model

```python
if args.use_advanced:
    final_model = MultiModalCPPModel(
        text_dim=text_dim,
        skill_text_dim=skill_text_dim,
        structured_dim=structured_dim,
        hidden_dim=hidden_dim,  # Best from Optuna
        n_layers=n_layers,
        dropout=dropout,
        output_dim=output_dim,
        use_modality_weights=use_modality_weights,
        use_text=args.use_text_history,
        use_skill=args.use_skill_text,
        use_struct=args.use_structured
    ).to(device)
```

### 13.3: Train Final Model

```python
optimizer = optim.Adam(final_model.parameters(), lr=lr)
criterion = nn.CosineEmbeddingLoss()

for epoch in range(final_epochs):
    train_epoch(final_model, combined_loader, optimizer, criterion, device)
```

### 13.4: Evaluate on Test Set

```python
test_metrics = evaluate(final_model, test_loader, Y_target_all, device)
```

Returns:
```python
{
    'MRR': 0.45,
    'R@1': 0.28,
    'R@5': 0.58,
    'R@10': 0.71,
    'R@20': 0.82
}
```

### 13.5: Save Model and Results

```python
checkpoint = {
    'model_state_dict': final_model.state_dict(),
    'hidden_dim': hidden_dim,
    'n_layers': n_layers,
    'dropout': dropout,
    'lr': lr,
    'use_modality_weights': use_modality_weights,
    'test_metrics': test_metrics,
    'args': vars(args)
}
torch.save(checkpoint, os.path.join(args.output_dir, 'final_model_.pt'))
```

### 13.6: Append Results to CSV

```python
results_data = {
    'timestamp': '2025-11-23 14:30:00',
    'run_name': 'cpp_enhanced',
    'architecture': 'MultiModal',
    'text_history': True,
    'skill_text': True,
    'structured': True,
    'final_epochs': 5,
    'lr': 0.0001,
    'hidden_dim': 512,
    'test_MRR': 0.45,
    'test_R@1': 0.28,
    # ... more metrics
}
results_df = pd.DataFrame([results_data])
results_df.to_csv(args.results_csv_path, mode='a', header=False)
```

**Why:** Track all experiments in a single CSV for easy comparison.

---

## 14. Model Architectures

### Architecture 1: MultiModalCPPModel

#### Class: `MultiModalCPPModel`
**Location:** Lines 83-170  
**Purpose:** Late fusion with separate encoders per modality

**Architecture Diagram:**
```
Input Features:
├── h_text [768]           → Text Encoder (hidden_dim → hidden_dim → ...)
├── h_skill_text [768]     → Skill Encoder (hidden_dim → hidden_dim → ...)
└── h_structured [100]     → Struct Encoder (hidden_dim → hidden_dim → ...)
         ↓                            ↓                          ↓
    [hidden_dim]               [hidden_dim]                [hidden_dim]
         ↓                            ↓                          ↓
    Optional: × alpha_text      × alpha_skill              × alpha_struct
         ↓                            ↓                          ↓
         └────────────────────────────┴──────────────────────────┘
                                      ↓
                            Concatenate [hidden_dim * 3]
                                      ↓
                            Fusion Linear → [output_dim=768]
                                      ↓
                                Final Embedding
```

**Forward Pass Example:**
```python
def forward(self, batch):
    # Encode each modality separately
    h_text_encoded = self.text_encoder(batch['h_text'])           # [32, 512]
    h_skill_encoded = self.skill_encoder(batch['h_skill_text'])  # [32, 512]
    h_struct_encoded = self.struct_encoder(batch['h_structured_structured'])  # [32, 512]
    
    # Optional learnable weights
    if self.use_modality_weights:
        h_text_encoded = self.alpha_text * h_text_encoded
        h_skill_encoded = self.alpha_skill * h_skill_encoded
        h_struct_encoded = self.alpha_struct * h_struct_encoded
    
    # Concatenate and fuse
    fused = torch.cat([h_text_encoded, h_skill_encoded, h_struct_encoded], dim=1)  # [32, 1536]
    output = self.fusion_head(fused)  # [32, 768]
    
    return output
```

**Shared Encoder Architecture:**
```python
def _build_encoder(self, input_dim, hidden_dim, n_layers, dropout):
    layers = []
    current_dim = input_dim
    
    for _ in range(n_layers):
        layers.extend([
            nn.Linear(current_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ])
        current_dim = hidden_dim
    
    return nn.Sequential(*layers)
```

### Architecture 2: SimpleConcatModel

#### Class: `SimpleConcatModel`
**Location:** Lines 173-223  
**Purpose:** Early fusion by concatenating all features immediately

**Architecture Diagram:**
```
Input Features:
├── h_text [768]
├── h_skill_text [768]
└── h_structured [100]
         ↓
    Concatenate [1636]
         ↓
    Linear → ReLU → Dropout
         ↓
    [hidden_dim]
         ↓
    (Repeat n_layers times)
         ↓
    Final Linear → [output_dim=768]
         ↓
    Final Embedding
```

**Forward Pass Example:**
```python
def forward(self, batch):
    # Concatenate all features immediately
    features = []
    features.append(batch['h_text'])           # [32, 768]
    features.append(batch['h_skill_text'])     # [32, 768]
    features.append(batch['h_structured_structured'])  # [32, 100]
    
    x = torch.cat(features, dim=1)  # [32, 1636]
    return self.model(x)            # [32, 768]
```

---

## 15. Training & Evaluation Functions

### Function: `train_epoch(model, dataloader, optimizer, criterion, device)`
**Location:** Lines 285-305  
**Purpose:** Train model for one epoch

**Flow:**
```python
model.train()
total_loss = 0

for batch in dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    
    optimizer.zero_grad()
    y_pred = model(batch)  # [32, 768]
    
    # CosineEmbeddingLoss(y_pred, y_true, target)
    # target=1 means y_pred and y_true should be similar
    target = torch.ones(y_pred.size(0)).to(device)  # [32]
    loss = criterion(y_pred, batch['y'], target)
    
    loss.backward()
    optimizer.step()
    
    total_loss += loss.item()

return total_loss / len(dataloader)
```

**CosineEmbeddingLoss:**
- For each sample: `loss_i = 1 - cos_sim(y_pred_i, y_true_i)`
- Encourages predictions to be similar (high cosine similarity) to targets

### Function: `evaluate(model, dataloader, Y_target_all, device, criterion)`
**Location:** Lines 308-336  
**Purpose:** Evaluate model and compute ranking metrics

**Flow:**
```python
model.eval()
all_y_pred = []
all_y_true = []

with torch.no_grad():
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        y_pred = model(batch)
        
        all_y_pred.append(y_pred.cpu().numpy())
        all_y_true.append(batch['y'].cpu().numpy())

y_pred_vectors = np.concatenate(all_y_pred)  # [n_test, 768]
y_true_vectors = np.concatenate(all_y_true)  # [n_test, 768]

metrics = calculate_ranking_metrics(y_pred_vectors, y_true_vectors, Y_target_all)
return metrics
```

### Function: `calculate_ranking_metrics(...)`
**Location:** Lines 229-278  
**Purpose:** Calculate MRR and Recall@K

**Algorithm:**

**1. Calculate Similarity Matrix:**
```python
sim_matrix = cosine_similarity(y_pred_vectors, Y_target_all)
# Shape: [n_test, n_targets]
# sim_matrix[i, j] = cosine similarity between prediction i and target j
```

**2. Sort by Similarity:**
```python
sorted_indices = np.argsort(sim_matrix, axis=1)[:, ::-1]
# sorted_indices[i] = [index of most similar target, 2nd most similar, ...]
```

**3. Find True Target Ranks:**
```python
for i, y_true in enumerate(y_true_vectors):
    true_index = np.where((Y_target_all == y_true).all(axis=1))[0][0]
    rank_list = list(sorted_indices[i])
    rank = rank_list.index(true_index) + 1  # 1-indexed rank
    reciprocal_ranks.append(1.0 / rank)
```

**Example:**
- Prediction for sample i
- True target is "senior data scientist"
- Model's ranked predictions: ["data scientist", "ML engineer", "senior data scientist", ...]
- True target is at rank 3
- Reciprocal rank = 1/3 = 0.333

**4. Calculate MRR:**
```python
mrr = np.mean(reciprocal_ranks)
```

**5. Calculate Recall@K:**
```python
for k in [1, 5, 10, 20]:
    hits = 0
    for i, true_idx in enumerate(true_target_indices):
        if true_idx in sorted_indices[i, :k]:
            hits += 1
    recall_at_k[f'R@{k}'] = hits / len(true_target_indices)
```

**Interpretation:**
- **MRR = 0.45**: On average, true target is in top 2-3 predictions
- **R@1 = 0.28**: 28% of samples have correct target as top prediction
- **R@5 = 0.58**: 58% of samples have correct target in top 5
- **R@10 = 0.71**: 71% of samples have correct target in top 10

---

## 16. Data Flow Summary

### Complete Pipeline Visualization (UPDATED)

```
1. RAW DATA
   ├── CSV: job_title → skillUri + score
   ├── CSV: skillUri → name + description
   ├── JSON: skillUri → meta-features
   └── Career paths: [(history, target), ...]

2. PREPROCESSING
   ├── Load encoders (SentenceTransformer)
   ├── Build job_skill_map
   ├── Calculate IDF scores (optional)
   └── Pre-compute target embeddings

3. PRE-COMPUTE INPUT EMBEDDINGS (NEW - Performance Optimization)
   For train/val/test sets:
   ├── Extract all history texts → [n_samples]
   ├── Batch encode histories → h_text_all [n_samples, 768]
   └── For each sample:
       ├── Extract titles from history
       ├── Lookup skills for each title
       ├── Get skill text from ESCO
       ├── Encode skills → [n_skills, 768]
       ├── Pool with weights → [768]
       └── Store in h_skill_all [n_samples, 768]

4. DATASET GENERATION (Fast - Uses Pre-computed)
   For each sample at index i:
   ├── h_text: pre_h_text[i] → [768]            (pre-computed ✓)
   ├── h_skill_text: pre_h_skill[i] → [768]    (pre-computed ✓)
   ├── Structured features:
   │   ├── Extract titles from history
   │   ├── Lookup skills for each title
   │   ├── Get meta-features for each skill
   │   ├── Convert to multi-hot vector → [100]
   │   └── h_structured
   └── y: Y_target_dict[target] → [768]         (pre-computed ✓)

5. MODEL TRAINING
   ├── Batch: {h_text, h_skill_text, h_structured, y}
   ├── Forward pass: model(batch) → y_pred [batch_size, 768]
   ├── Loss: CosineEmbeddingLoss(y_pred, y)
   └── Backward pass: update weights

6. EVALUATION
   ├── Collect all predictions: [n_test, 768]
   ├── Compare to all targets: [n_targets, 768]
   ├── Rank by cosine similarity
   └── Calculate MRR, Recall@K

7. OUTPUT
   ├── Trained model checkpoint
   ├── Test metrics
   └── Results appended to CSV
```

### Feature Dimensions Example

For a typical run with pre-computed embeddings:

```
Pre-computed Arrays (stored in memory):
├── train_h_text:      [10000, 768]   (all training histories)
├── train_h_skill:     [10000, 768]   (all training skills)
├── val_h_text:        [2000, 768]    (all validation histories)
├── val_h_skill:       [2000, 768]    (all validation skills)
└── Y_target_all:      [500, 768]     (all possible targets)

Batch Retrieved from Dataset:
├── h_text:                    [batch_size=32, 768]  (indexed from pre-computed)
├── h_skill_text:              [32, 768]              (indexed from pre-computed)
└── h_structured_structured:   [32, 100]             (computed on-the-fly)

After Multi-Modal Encoders:
├── text_encoded:   [32, 512]  (hidden_dim=512)
├── skill_encoded:  [32, 512]
└── struct_encoded: [32, 512]

After Concatenation:
└── fused:          [32, 1536]  (512 * 3)

After Fusion Head:
└── y_pred:         [32, 768]  (output_dim=768)

Target:
└── y:              [32, 768]

Loss:
└── CosineEmbeddingLoss(y_pred, y) → scalar
```

---

## Key Design Decisions

### 1. Why Pre-compute Input Embeddings? (NEW)
- **Performance Critical:** Encoding is the bottleneck during training
- **Redundant Work:** Same histories/skills re-encoded every epoch
- **10x Speedup:** Pre-computation reduces epoch time from ~5min to ~30sec
- **Memory Trade-off:** ~60MB per 10,000 samples is acceptable
- **Still Flexible:** Can re-compute if changing pooling strategy

**When NOT to Pre-compute:**
- Experimenting with different pooling strategies (would need to re-compute)
- Very large datasets where memory is constrained
- Using data augmentation that modifies text

### 2. Why Pre-compute Target Embeddings?
- **Fixed:** Target embeddings don't change during training
- **Efficiency:** Compute once, use repeatedly for loss and evaluation
- **Ranking:** Need all targets for MRR/Recall calculation

### 3. Why Multi-Modal Architecture?
- **Separate Encoders:** Each modality has different characteristics
- **Learnable Fusion:** Model learns how to combine modalities
- **Better Performance:** Captures modality-specific patterns

### 4. Why CosineEmbeddingLoss?
- **Interpretable:** Directly optimizes cosine similarity
- **Ranking-Friendly:** High similarity = high rank
- **Stable:** Works well with normalized embeddings

### 5. Why IDF Weighting?
- **Skill Importance:** Rare skills are more distinctive
- **Down-weight Common:** Generic skills less informative
- **Empirical:** Often improves performance

---

## Conclusion

This training pipeline:

1. **Loads** pre-trained encoders and skill mappings
2. **Preprocesses** career path data into (history, target) pairs
3. **Pre-computes** embeddings for performance optimization (NEW):
   - Text history embeddings (batch encoding)
   - Weighted skill text embeddings (pooled per sample)
   - Target job embeddings (for ranking)
4. **Generates** multi-modal features efficiently:
   - Uses pre-computed embeddings (10x faster)
   - Computes structured features on-the-fly (lightweight)
5. **Trains** neural networks to predict next career steps
6. **Optimizes** hyperparameters with Optuna
7. **Evaluates** with ranking metrics (MRR, Recall@K)
8. **Saves** models and logs results for reproducibility

The entire system is designed for:
- **Flexibility:** Ablation studies with modality toggles
- **Efficiency:** Pre-computed embeddings eliminate bottlenecks
- **Performance:** Multi-modal fusion with optimized hyperparameters
- **Scalability:** Fast enough for large-scale experiments



