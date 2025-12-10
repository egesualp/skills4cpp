# Fixes Applied to Skill Extraction Pipeline

## Date: November 21, 2025

## Issues Identified and Fixed

### 1. **CRITICAL: Zero Skill Embeddings Due to Data Format Mismatch**

**Problem:**
- When `--use_skill_text` was used WITHOUT `--use_text_description`, the code set `ONLY_TITLES=True`
- This extracted plain job titles (e.g., `"cook "`) instead of formatted documents (e.g., `"role: cook\n description: ..."`)
- The skill extraction function `_extract_skill_infos()` expected formatted text with `"role: ...\n"` pattern
- Result: NO skills were found → all-zero skill embeddings → poor model performance

**Impact:**
- MRR: 0.0072 (extremely poor)
- R@1: 0.0000 (no correct predictions)
- All skill-based features were zero vectors

**Fix Applied:**
Modified `train_cpp_enhanced_debug.py` and `train_cpp_enhanced.py` (line ~726):

```python
# Before (BROKEN):
data = Data(DATA_TYPE=args.data_type, ONLY_TITLES=not args.use_text_description)

# After (FIXED):
should_use_titles_only = (not args.use_text_description) and (not args.use_skill_text)
data = Data(DATA_TYPE=args.data_type, ONLY_TITLES=should_use_titles_only)
```

**Logic:**
- `ONLY_TITLES=True` ONLY when BOTH conditions are met:
  1. NOT using text descriptions
  2. NOT using skill features
- If using skills → keep formatted documents → skills can be extracted

### 2. **Robustness: Enhanced Skill Extraction Function**

**Problem:**
- `_extract_skill_infos()` only handled formatted text
- Failed silently when given plain titles

**Fix Applied:**
Modified `data_loaders.py` (lines 163-177):

```python
def _extract_skill_infos(history_doc: str, job_skill_map: Dict[str, List[Dict]]) -> List[Dict]:
    """Extract skill information from a history document.
    
    Handles both formatted documents and plain titles.
    """
    # First try to extract from formatted text
    titles = re.findall(r"role: (.*?)\n", history_doc)
    
    # If no matches found, assume it's a plain title or SEP-separated titles
    if not titles:
        from cpp import utils
        titles = [t.strip() for t in history_doc.split(utils.SEP_TOKEN) if t.strip()]
    
    infos = []
    for t in titles:
        title_clean = t.strip()
        if title_clean in job_skill_map:
            infos.extend(job_skill_map[title_clean])
    return infos
```

**Benefits:**
- Now works with both formatted and plain text
- Provides fallback protection
- More robust to different input formats

### 3. **Fixed FutureWarning in Pandas**

**Problem:**
- Line 96 in `data_loaders.py` caused FutureWarning:
  ```
  df['idf'].fillna(0, inplace=True)
  ```
- Warning about chained assignment behavior change in Pandas 3.0

**Fix Applied:**
Modified `data_loaders.py` (lines 93-97):

```python
# Before:
df['idf'] = df['skillUri'].map(idf_map)
df['idf'].fillna(0, inplace=True)

# After:
df = df.copy()  # Avoid SettingWithCopyWarning
df['idf'] = df['skillUri'].map(idf_map)
df['idf'] = df['idf'].fillna(0)
```

## Verification

### Test Results
All tests passed ✓:
1. Formatted text extraction: ✓
2. Plain title extraction: ✓
3. SEP_TOKEN-separated titles: ✓
4. Empty document handling: ✓
5. Unknown title handling: ✓
6. ONLY_TITLES logic (4 scenarios): ✓

### Expected Performance Improvement

With these fixes, the model should now:
- ✓ Successfully extract skills from job titles
- ✓ Generate non-zero skill embeddings
- ✓ Learn skill-based patterns
- ✓ Achieve significantly better performance metrics

### Next Steps

1. **Re-run the training** with the same command:
   ```bash
   python -m src.cpp.train_cpp_enhanced_debug \
       --data_type decorte_esco \
       --master_skill_file results/decorte_esco_ground_truth/job_title_skills_master.csv \
       --max_epochs 3 \
       --batch_size 4096 \
       --output_dir results/cpp/decorte_esco_skills_only \
       --use_skill_text \
       --use_skill_description \
       --run_name test_skills_only_FIXED \
       --patience 2 \
       --num_workers 0 \
       --debug
   ```

2. **Verify the logs show**:
   - ✓ Non-zero skill embeddings (not all zeros)
   - ✓ Skills being found for jobs (not "No skills found")
   - ✓ Skill text formatting examples showing proper format
   - ✓ Better performance metrics (MRR > 0.01, R@1 > 0)

3. **Compare performance**:
   - Before: MRR: 0.0072, R@1: 0.0000
   - After: Expected MRR > 0.05, R@1 > 0.01

## Files Modified

1. `/src/cpp/train_cpp_enhanced_debug.py` (line ~726)
2. `/src/cpp/train_cpp_enhanced.py` (line ~622)
3. `/src/cpp/data_loaders.py` (lines 93-97, 163-177)

## Files Created

1. `/BUG_ANALYSIS.md` - Detailed analysis of the bug
2. `/FIXES_APPLIED.md` - This file
3. `/test_skill_extraction_fix.py` - Test suite for verification

## Question Answers

### Q1: "In master_skill_file, for 'academic advisor' I see 155 skills. But output claims 31. Why?"

**A:** The master file actually contains **31 skills** for "academic advisor", not 155. 
```bash
$ grep "academic advisor" results/decorte_esco_ground_truth/job_title_skills_master.csv | wc -l
31
```
The confusion may have come from row indices or looking at a different data source.

### Q2: "Text format check doesn't seem well. Are we not formatting skill texts correctly?"

**A:** The text format in the debug output (lines 726-727) showing `{'name': '...', 'desc': '...'}` was **NOT a bug** - it was just showing the dictionary structure stored in `esco_skill_text_map`. The actual encoding code properly formats it as `"role: name\n description: desc"`, but this code was never reached because no skills were being extracted due to the ONLY_TITLES bug.

### Q3: "Debugger says 'No skills found for this job' - are we using correct data type?"

**A:** The data type (`decorte_esco`) was correct. The issue was the preprocessing step:
- Data was in format: `"esco role: cook\n description: ..."`
- But `ONLY_TITLES=True` extracted just: `"cook "`
- Skill extraction expected: `"role: cook\n ..."`
- Mismatch → no skills found

Now fixed! The correct format is preserved when using skills.

### Q4: "Do you see any other issue in this output?"

**A:** Yes, identified and fixed:
1. ✓ **Main bug**: ONLY_TITLES logic causing zero skill embeddings
2. ✓ **FutureWarning**: Pandas chained assignment warning
3. Potential improvement: The regex `r"role: (.*?)\n"` in `_extract_titles()` accidentally matches "esco role:" - while it works, it's fragile. Consider making it more explicit in future.

## Conclusion

The root cause was a logic error in determining when to extract only titles vs. keeping formatted documents. The fix ensures that whenever skill features are requested, the formatted documents are preserved so skills can be properly extracted. This should dramatically improve model performance.




