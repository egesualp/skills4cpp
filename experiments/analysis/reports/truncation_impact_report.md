# Truncation Impact Analysis Report

**Generated:** 2025-12-30T18:50:19.015937

## Metadata

- **Encoder Model:** `ElenaSenger/career-path-representation-mpnet-decorte-esco`
- **Model Max Sequence Length:** 384
- **Data Type:** decorte
- **Only Titles:** False
- **Consider Subspans:** True
- **Context Lengths Tested:** [128, 256, 384, 512]

## Dataset Overview

| Split | Samples |
|-------|---------|
| Train | 13,484 |
| Validation | 1,558 |
| Test | 1,802 |
| **Total** | **16,844** |

## Last Job Equals Target Analysis

This checks how many samples have the last job in the career history (doc1) equal to the target job (doc2).

| Split | Matches | Total | Match Rate |
|-------|---------|-------|------------|
| Train | 310 | 13,484 | 2.30% |
| Validation | 17 | 1,558 | 1.09% |
| Test | 41 | 1,802 | 2.28% |

## Truncation Analysis

### Train Set

**Token Length Statistics:**

- Min: 9, Max: 1726
- Mean: 223.5 ± 174.1
- Median: 176.0
- Percentiles: 25th=100, 75th=294, 90th=447, 95th=564, 99th=848

**Truncation by Context Length:**

| Max Length | Truncated | Rate | Avg Tokens Lost | Total % Lost |
|------------|-----------|------|-----------------|--------------|
| 128 | 8,693 | 64.47% | 174.6 | 50.37% |
| 256 | 4,242 | 31.46% | 168.5 | 23.72% |
| 384 | 1,967 | 14.59% | 169.8 | 11.08% |
| 512 | 921 | 6.83% | 171.2 | 5.23% |

### Validation Set

**Token Length Statistics:**

- Min: 13, Max: 1377
- Mean: 215.4 ± 186.6
- Median: 160.0
- Percentiles: 25th=91, 75th=275, 90th=435, 95th=585, 99th=985

**Truncation by Context Length:**

| Max Length | Truncated | Rate | Avg Tokens Lost | Total % Lost |
|------------|-----------|------|-----------------|--------------|
| 128 | 945 | 60.65% | 176.7 | 49.75% |
| 256 | 447 | 28.69% | 183.0 | 24.37% |
| 384 | 207 | 13.29% | 204.6 | 12.62% |
| 512 | 107 | 6.87% | 215.4 | 6.87% |

### Test Set

**Token Length Statistics:**

- Min: 14, Max: 1540
- Mean: 229.5 ± 175.0
- Median: 185.0
- Percentiles: 25th=104, 75th=305, 90th=456, 95th=560, 99th=826

**Truncation by Context Length:**

| Max Length | Truncated | Rate | Avg Tokens Lost | Total % Lost |
|------------|-----------|------|-----------------|--------------|
| 128 | 1,207 | 66.98% | 175.6 | 51.24% |
| 256 | 601 | 33.35% | 166.6 | 24.21% |
| 384 | 283 | 15.70% | 161.0 | 11.01% |
| 512 | 124 | 6.88% | 163.4 | 4.90% |
