# Bug Analysis: Skills Not Being Extracted

## Summary
The training script is returning all-zero skill embeddings because of a mismatch between data preprocessing and skill extraction logic.

## Root Cause

###1. **Data Format Mismatch**

**When `use_text_description=False` (current configuration):**
- Line 726 in `train_cpp_enhanced_debug.py`: `ONLY_TITLES=not args.use_text_description`
- This sets `ONLY_TITLES=True`
- The `_extract_titles()` method (data_classes.py:114-120) extracts just job titles:
  ```python
  # Input: "esco role: cook \n description: ..."
  # Output: "cook "
  ```

**Skill extraction expects formatted text:**
- `_extract_skill_infos()` in data_loaders.py:165 uses regex: `r"role: (.*?)\n"`
- It expects: `"role: cook \n description: ..."`
- It receives: `"cook "` (plain title with no "role:" prefix)
- Result: **NO matches found → empty skill list → zero vector**

### 2. **Academic Advisor Skill Count (31 vs 155)**

**User's concern:** Master file shows 155 skills but debug shows 31.

**Reality:** The master file only has **31 skills** for "academic advisor":
```bash
$ grep "academic advisor" results/decorte_esco_ground_truth/job_title_skills_master.csv | wc -l
31
```
The user may have confused row indices (0-31) or looked at a different data source.

### 3. **Skill Text Format Issues**

**Lines 726-727 in logs show:**
```
{'name': 'manage musical staff', 'desc': 'Assign and manage staff tasks...'}
```

This is **NOT a bug** - it's just debug output showing the dictionary structure stored in `esco_skill_text_map`. The actual encoding happens in `_pooled_skill_vec()` which properly formats it as:
```python
text = f"role: {st['name']} \n description: {st['desc']}"
```

However, this code never runs because no skills are found in step 1!

### 4. **Zero Vectors Throughout Pipeline**

Evidence from logs:
- Line 798: "⚠️  No skills found for this job - returning zero vector"
- Line 816-817: `First embedding (first 10 values): [0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]`
- Line 868: `h_skill_text` stats: `min=0.0000, max=0.0000, mean=0.0000`

## Verification

Test output shows the data format difference:

**ONLY_TITLES=False (correct format):**
```
Doc1: "esco role: cook \n description: Cooks are culinary..."
Doc2: "esco role: cook \n description: ..."
```

**ONLY_TITLES=True (problematic format):**
```
Doc1: "cook "
Doc2: "cook "
```

## Impact

With all-zero skill embeddings:
- Model receives NO skill information
- Performance is severely degraded (MRR: 0.0072, R@1: 0.0000)
- Model cannot learn skill-based patterns

## Solution Options

### Option 1: Don't Use ONLY_TITLES When Using Skills (RECOMMENDED)
In `train_cpp_enhanced_debug.py`:
```python
# Line 726: Change from
data = Data(DATA_TYPE=args.data_type, ONLY_TITLES=not args.use_text_description)

# To:
should_use_titles_only = (not args.use_text_description) and (not args.use_skill_text)
data = Data(DATA_TYPE=args.data_type, ONLY_TITLES=should_use_titles_only)
```

**Rationale:** If we're using skill features, we NEED the formatted documents to extract skills.

### Option 2: Fix _extract_skill_infos() to Handle Both Formats
In `data_loaders.py`:
```python
def _extract_skill_infos(history_doc: str, job_skill_map: Dict[str, List[Dict]]) -> List[Dict]:
    """Extract skill information from a history document."""
    # First try formatted text
    titles = re.findall(r"role: (.*?)\n", history_doc)
    
    # If no matches, assume it's a plain title
    if not titles:
        # Handle plain titles (e.g., "cook " or "cook <SEP>head chef ")
        titles = [t.strip() for t in history_doc.split(SEP_TOKEN) if t.strip()]
    
    infos = []
    for t in titles:
        if t.strip() in job_skill_map:
            infos.extend(job_skill_map[t.strip()])
    return infos
```

**Rationale:** Make the function robust to both input formats.

### Option 3: Add Validation
Add a check in the training script:
```python
if args.use_skill_text and not args.use_text_description:
    logger.warning("⚠️  use_skill_text requires formatted documents!")
    logger.warning("    Setting use_text_description=True to preserve formatting")
    args.use_text_description = True
```

## Recommended Fix

**Implement Option 1** as it's the cleanest solution and makes the logic explicit:
- If using skill features → need formatted documents → `ONLY_TITLES=False`
- If NOT using skills AND not using text descriptions → can extract titles → `ONLY_TITLES=True`

This ensures that the data format matches what the downstream code expects.

## Additional Issues Found

1. **FutureWarning in data_loaders.py:96**
   ```python
   # Current:
   df['idf'].fillna(0, inplace=True)
   
   # Should be:
   df = df.copy()
   df['idf'] = df['idf'].fillna(0)
   ```

2. **Regex in _extract_titles() is fragile**
   - Line 114: `r"role: (.*?)\n"` matches "esco role:" by accident
   - Should explicitly match "esco role:" for decorte_esco data
   - Or use a more robust extraction method




