# Investigation Report: Why Adding Descriptions Doesn't Help

**Dataset:** karrierewege_100k
**Encoder:** ElenaSenger/career-path-representation-mpnet-karrierewege
**Generated:** 2025-12-22 15:35:16

## Executive Summary

### 1. Token Truncation
- **Encoder max_seq_length:** 384
- **Title+Description truncation rate:** 16.08%
- **Title-only truncation rate:** 0.00%

### 2. Semantic Redundancy
- **Mean cosine similarity (title vs title+desc):** 0.6248
- **✅ Low redundancy:** Descriptions add meaningful information

### 3. Answer Leakage
- **Subspan data leakage rate:** 100.00%
- **Clean test data leakage rate:** 100.00%
- **⚠️ CRITICAL:** High answer leakage in subspan data!
  - This explains the large gap between subspan and clean test performance

---

## Detailed Analysis

### Token Length Analysis

| Metric | Title+Desc (doc1) | Title-only (doc1) |
|--------|-------------------|-------------------|
| Mean tokens | 248.9 | 23.7 |
| Median tokens | 195.0 | 19.0 |
| P95 tokens | 593 | 57 |
| Max tokens | 2312 | 197 |
| Truncated % | 16.08% | 0.00% |
| Avg tokens lost | 184.8 | 0.0 |

### Leakage by History Length

| History Length | Total Samples | Leaked | Leak Rate |
|----------------|---------------|--------|-----------|
| 2 | 362,153 | 362,153 | 100.0% |
| 3 | 272,153 | 272,153 | 100.0% |
| 4 | 182,153 | 182,153 | 100.0% |
| 5 | 123,969 | 123,969 | 100.0% |
| 6 | 85,459 | 85,459 | 100.0% |
| 7 | 59,056 | 59,056 | 100.0% |
| 8 | 41,219 | 41,219 | 100.0% |
| 9 | 28,878 | 28,878 | 100.0% |
| 10 | 20,396 | 20,396 | 100.0% |
| 11 | 14,470 | 14,470 | 100.0% |

---

## Conclusions and Recommendations

3. **Answer leakage explains performance gap**: Subspan augmentation causes 100.0% leakage (vs 100.0% in clean test). The model learns to exploit this pattern, inflating subspan test metrics.
