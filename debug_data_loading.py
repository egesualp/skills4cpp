
import pandas as pd
from pathlib import Path

# Config matching the user's yaml
file_path = "/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/karrierewege_plus_cp_master_3.csv"
ground_truth_col = "esco_id"
group_by_col = "job_id"
text_col = "raw_title"

print(f"Loading {file_path}...")
df = pd.read_csv(file_path)
print(f"Total rows: {len(df)}")
print(f"Unique {group_by_col}: {df[group_by_col].nunique()}")

cols_to_check = [group_by_col, ground_truth_col, text_col]
if 'esco_title' in df.columns:
    cols_to_check.append('esco_title')
    print("Column 'esco_title' is present and will be checked for NaNs.")
else:
    print("Column 'esco_title' is NOT present.")

print(f"Columns to check for NaN: {cols_to_check}")

# Check naive dropna
df_dropped = df.dropna(subset=cols_to_check)
print(f"Rows after dropna(subset={cols_to_check}): {len(df_dropped)}")
print(f"Unique {group_by_col} after dropna: {df_dropped[group_by_col].nunique()}")
print(f"Dropped rows count: {len(df) - len(df_dropped)}")

# Detailed breakdown
print("\n--- Detailed NaNs Breakdown ---")
for col in cols_to_check:
    nans = df[col].isna().sum()
    print(f"NaNs in '{col}': {nans}")
    
# Combined breakdown
nans_any = df[cols_to_check].isna().any(axis=1).sum()
print(f"Rows with any NaN in checks: {nans_any}")

# Specific intersection
if 'esco_title' in cols_to_check:
    missing_esco_title = df['esco_title'].isna() & df['esco_id'].notna()
    print(f"Rows with missing esco_title but present esco_id: {missing_esco_title.sum()}")

missing_raw_title = df['raw_title'].isna()
print(f"Rows with missing raw_title: {missing_raw_title.sum()}")

# Check if job_ids are lost because all their rows were dropped
valid_job_ids = set(df_dropped[group_by_col].unique())
all_job_ids = set(df[group_by_col].unique())
lost_job_ids = all_job_ids - valid_job_ids
print(f"\nTotal lost unique job_ids: {len(lost_job_ids)}")

if len(lost_job_ids) > 0:
    print("Sample lost job_ids:", list(lost_job_ids)[:5])
    # Inspect why the first lost job_id was lost
    sample_id = list(lost_job_ids)[0]
    print(f"\nInspecting lost job_id: {sample_id}")
    sample_rows = df[df[group_by_col] == sample_id]
    print(sample_rows[cols_to_check])
