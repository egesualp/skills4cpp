
import pandas as pd
import numpy as np
from datasets import load_dataset
import re
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

def remove_title_prefix(row):
    desc = row['new_job_description_en_cp']
    title = row['new_job_title_en_cp']
    if isinstance(desc, str) and isinstance(title, str):
        prefix = f"{title}: "
        if desc.startswith(prefix):
            return desc[len(prefix):]
    return desc

def fill_missing_title(row):
    """If title is '...' and desc contains a 'job_title: ...' pattern, set title from there."""
    title = row['new_job_title_en_cp']
    desc = row['new_job_description_en_cp']
    if title == '...' and isinstance(desc, str):
        # Search for the pattern '<something>: <rest of desc>' at the start
        # Try extracting the prefix before the first colon-space
        m = re.match(r"([^:]+):", desc.strip())
        if m:
            extracted_title = m.group(1).strip()
            if extracted_title and extracted_title != "...":
                return extracted_title
    return title

def remove_leading_dots(title):
    """Remove three dots if they appear at the start of the job title."""
    if isinstance(title, str) and title.startswith("..."):
        if title != "...":
            return title[3:].lstrip()
    return title

def generate_figure_4_1(career_lengths, isco_counts_counter):
    print("\n" + "="*60)
    print("FIGURE 4.1: DATASET CHARACTERISTICS")
    print("="*60)
    
    # Setup styling similar to provided script
    try:
        sns.set_theme(style="whitegrid")
    except:
        pass # Fallback if set_theme fails or old seaborn

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Figure 4.1: Karrierewege Plus Dataset Characteristics', fontsize=12, fontweight='bold', y=0.995)

    # Define consistent color palette
    primary_color = sns.color_palette("muted")[0]  # steelblue-like
    secondary_color = sns.color_palette("muted")[3]  # red-like
    tertiary_color = sns.color_palette("muted")[1]  # orange-like
    highlight_color = sns.color_palette("muted")[2]  # green-like
    gray_color = 'gray'

    # Subplot (a): Career Length Distribution
    ax_a = axes[0]
    # career_lengths is a pandas Series of counts, we need the values
    lengths_values = career_lengths.values
    mean_length = np.mean(lengths_values)
    median_length = np.median(lengths_values)

    # Histogram
    counts, bins, patches = ax_a.hist(lengths_values, bins=range(int(lengths_values.min()), int(lengths_values.max())+2), 
                                        color=primary_color, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Add vertical lines for mean and median
    ax_a.axvline(mean_length, color=secondary_color, linestyle='--', linewidth=2, label=f'Mean: {mean_length:.1f}')
    ax_a.axvline(median_length, color=tertiary_color, linestyle='--', linewidth=2, label=f'Median: {median_length:.0f}')

    # Add shaded region for 25th-75th percentiles
    p25, p75 = np.percentile(lengths_values, [25, 75])
    ax_a.axvspan(p25, p75, alpha=0.2, color=gray_color, label=f'IQR: {p25:.0f}-{p75:.0f}')

    ax_a.set_xlabel('Career Length (Number of Jobs)', fontsize=9)
    ax_a.set_ylabel('Frequency (Number of Careers)', fontsize=9)
    ax_a.set_title('(a) Career Length Distribution', fontsize=10, fontweight='bold')
    ax_a.legend(loc='upper right', fontsize=8)
    ax_a.tick_params(axis='both', labelsize=8)
    ax_a.grid(axis='y', alpha=0.3)

    # Subplot (b): ISCO Major Group Distribution
    ax_b = axes[1]
    
    # Prepare ISCO counts dataframe from Counter
    # Ensure all keys 0-9 are present for consistent plotting
    data = {'isco_group': [], 'count': []}
    for i in range(10):
        key = str(i)
        data['isco_group'].append(i)
        data['count'].append(isco_counts_counter.get(key, 0)) # keys in counter are strings '1', '2' etc.
        
    isco_df = pd.DataFrame(data).set_index('isco_group').sort_index()

    isco_labels_map = {
        0: '0: Armed Forces',
        1: '1: Managers',
        2: '2: Professionals',
        3: '3: Technicians',
        4: '4: Clerical Support',
        5: '5: Service/Sales',
        6: '6: Agriculture',
        7: '7: Craft/Trades',
        8: '8: Machine Operators',
        9: '9: Elementary'
    }
    
    isco_labels = [isco_labels_map[i] for i in isco_df.index]
    total_jobs = isco_df['count'].sum()
    if total_jobs > 0:
        percentages = (isco_df['count'].values / total_jobs * 100)
    else:
        percentages = np.zeros(len(isco_df))

    # Color bars based on percentage (highlight underrepresented <5%)
    colors = [highlight_color if pct < 5 else primary_color for pct in percentages]

    bars = ax_b.barh(isco_labels, isco_df['count'].values, color=colors, edgecolor='black', linewidth=0.5)

    # Add percentage labels
    for i, (count, pct) in enumerate(zip(isco_df['count'].values, percentages)):
        ax_b.text(count + (total_jobs*0.01), i, f'{pct:.1f}%', va='center', fontsize=8, style='italic')

    ax_b.set_xlabel('Count', fontsize=9)
    ax_b.set_ylabel('ISCO Major Group', fontsize=9)
    ax_b.set_title('(b) ISCO Major Group Distribution', fontsize=10, fontweight='bold')
    ax_b.tick_params(axis='both', labelsize=8)
    
    # Format x-axis labels with 'k'
    from matplotlib.ticker import FuncFormatter
    def k_formatter(x, pos):
        if x >= 1000:
            return f'{int(x/1000)}k'
        return f'{int(x)}'
    ax_b.xaxis.set_major_formatter(FuncFormatter(k_formatter))
    
    ax_b.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    # Save figure in both formats
    # Use absolute path or current dir
    out_pdf = Path('figure_4_1_dataset_characteristics.pdf')
    out_png = Path('figure_4_1_dataset_characteristics.png')
    
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')

    print("\n✓ Figure 4.1 generated successfully")
    print(f"✓ Saved as PDF ({out_pdf}) and PNG ({out_png}) (300 DPI)")
    print("="*60)

def main():
    print("Loading ElenaSenger/Karrierewege_plus dataset...")
    # Load the dataset - loading 'train' split by default as done in utils.py for similar tasks
    # But for statistics we might want to look at all splits or just train. 
    # The notebook looked at train, validation, and test for Decorte stats.
    
    # Check current cache or download
    try:
        dataset = load_dataset("ElenaSenger/Karrierewege_plus")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Combine splits if available, or just use what is there.
    # Decorte had train/val/test. usage in notebook:
    # total_careers = len(decorte_dataset['train']) + len(decorte_dataset['validation']) + len(decorte_dataset['test'])
    
    dfs = []
    print(f"Dataset keys: {dataset.keys()}")
    for split in dataset.keys():
        print(f"Processing split: {split}")
        dfs.append(dataset[split].to_pandas())
    
    df = pd.concat(dfs, ignore_index=True)
    
    print(f"Initial row count: {len(df)}")
    
    print("Dataset Columns:", df.columns.tolist())
    print("First row:", df.iloc[0].to_dict())


    # --- Load ESCO Occupations for ISCO Mapping ---
    print("\n--- Loading ESCO Occupations for ISCO Mapping ---")
    esco_occupations_path = Path("/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/occupations_en.csv")
    if not esco_occupations_path.exists():
        print(f"Error: ESCO occupations file not found at {esco_occupations_path}")
    else:
        occupations = pd.read_csv(esco_occupations_path, low_memory=False)
        print("ESCO Occupations Columns:", occupations.columns.tolist())
        print("First ESCO Occupation:", occupations.iloc[0].to_dict())
        
        # Create mapping from conceptUri to iscoGroup
        # Assuming 'conceptUri' and 'iscoGroup' are the column names - verify with output
        # If columns are different, this will need adjustment in next step
        # Based on notebook, we need to map to ISCO code. 
        # In notebook: occupations['code'] is used for ISCO levels. Let's see if 'iscoGroup' or 'code' holds the ISCO code.
    

    # --- Load ESCO Occupations for ISCO Mapping ---
    print("\n--- Loading ESCO Occupations & Master Lookup ---")
    esco_occupations_path = Path("/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/occupations_en.csv")
    master_csv_path = Path("/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/karrierewege_plus_cp_master_3.csv")
    
    # 1. Map ESCO URI -> ISCO Group
    esco_to_isco = {}
    if esco_occupations_path.exists():
        print("Loading ESCO occupations...")
        occ_df = pd.read_csv(esco_occupations_path, low_memory=False)
        # Verify columns
        uri_col = 'conceptUri'
        isco_col = 'iscoGroup'
        if uri_col in occ_df.columns and isco_col in occ_df.columns:
            for _, row in occ_df.iterrows():
                if pd.notna(row[uri_col]) and pd.notna(row[isco_col]):
                    esco_to_isco[row[uri_col]] = str(row[isco_col])
            print(f"Loaded {len(esco_to_isco)} ESCO->ISCO mappings.")
        else:
            print(f"Warning: Expected columns {uri_col}, {isco_col} not found in ESCO occupations file.")
    else:
        print(f"Error: ESCO occupations file not found at {esco_occupations_path}")

    # 2. Map (Title, Description) -> ESCO URI (from Master CSV)
    CONTENT_TO_ESCO = {}
    if master_csv_path.exists():
        print("Loading Master CSV for dataset lookup...")
        master_df = pd.read_csv(master_csv_path)
        for _, row in master_df.iterrows():
            # Create a lookup key consistent with how we will access the dataset
            # Using raw_title and raw_description from master, which should match dataset
            t = str(row['raw_title']).strip()
            d = str(row['raw_description']).strip()
            # Handle potential NaN
            if pd.isna(row['raw_title']): t = ""
            if pd.isna(row['raw_description']): d = ""
            
            key = (t, d)
            if pd.notna(row['esco_id']):
                CONTENT_TO_ESCO[key] = row['esco_id']
        print(f"Loaded {len(CONTENT_TO_ESCO)} dataset->ESCO mappings.")
    else:
        print(f"Error: Master CSV not found at {master_csv_path}")

    # 3. Calculate ISCO Stats for the Dataset
    print("\n--- Calculating ISCO Distribution for Dataset ---")
    isco_counts = {}
    # We use the original columns from the dataframe BEFORE preprocessing changes them (if they do)
    # Actually, we should iterate the dataframe now.
    # Note: The dataframe 'new_job_title_en_cp' and 'new_job_description_en_cp' might need cleaning to match Master CSV?
    # Based on observation, Master CSV has 'Housekeeper' and description without prefix.
    # The dataset columns seem to match `raw_title` and `raw_description`.
    
    missing_isco_count = 0
    mapped_count = 0
    
    # To speed up, we can use apply or iterate tuples
    # We need to be careful: the previous code modifies _dataset_df columns in place.
    # We should perform this analysis on a copy or before modification.
    # Since we are inserting this BEFORE preprocessing steps (lines 141+), the dataframe is still 'raw'.
    
    # Using a loop might be slow for 500k rows, but simple to debug.
    # Optimization: Create a key column and map.
    
    # Define helper to create key
    def make_key(row):
        t = str(row['new_job_title_en_cp']).strip()
        d = str(row['new_job_description_en_cp']).strip()
        if pd.isna(row['new_job_title_en_cp']): t = ""
        if pd.isna(row['new_job_description_en_cp']): d = ""
        return (t, d)

    # Let's collect ISCO groups
    dataset_isco_first_digits = []
    
    # Batch processing or direct iteration
    # Let's just iterate, 500k is manageable in python if simple dict lookup
    
    keys = zip(df['new_job_title_en_cp'], df['new_job_description_en_cp'])
    
    for t_val, d_val in keys:
        t = str(t_val).strip() if pd.notna(t_val) else ""
        d = str(d_val).strip() if pd.notna(d_val) else ""
        
        # Clean description by removing "{title}: " prefix to match Master CSV format
        # This mirrors clean_description_remove_prefix logic
        if t and d.startswith(f"{t}: "):
            d = d[len(t)+2:].strip()
        
        k = (t, d)
        esco_id = CONTENT_TO_ESCO.get(k)
        
        if esco_id:
            isco_code = esco_to_isco.get(esco_id)
            if isco_code:
                # Take first digit for Major Group
                major_group = isco_code[0]
                dataset_isco_first_digits.append(major_group)
                mapped_count += 1
            else:
                # Found ESCO ID but no ISCO mapping?
                missing_isco_count += 1
        else:
            missing_isco_count += 1
            
    print(f"Mapped {mapped_count} experiences to ISCO Major Groups.")
    print(f"Missing mapping for {missing_isco_count} experiences.")
    
    # 4. Print Distribution
    from collections import Counter
    if dataset_isco_first_digits:
        counts = Counter(dataset_isco_first_digits)
        total = sum(counts.values())
        print("\nOccupation Distribution Across ISCO Major Groups (Karrierewege Plus Dataset):")
        # ISCO Major Group Names (Hardcoded for display)
        isco_names = {
            '0': 'Armed forces occupations',
            '1': 'Managers',
            '2': 'Professionals',
            '3': 'Technicians and associate professionals',
            '4': 'Clerical support workers',
            '5': 'Service and sales workers',
            '6': 'Skilled agricultural, forestry and fishery workers',
            '7': 'Craft and related trades workers',
            '8': 'Plant and machine operators, and assemblers',
            '9': 'Elementary occupations',
        }
        
        for grp in sorted(counts.keys()):
            c = counts[grp]
            pct = (c / total) * 100
            name = isco_names.get(grp, 'Unknown')
            print(f"  {grp} - {name}: {c:,} ({pct:.1f}%)")
    else:
        print("No ISCO codes could be mapped for distribution analysis.")

    # 5. Taxonomy Distribution (ESCO Occupations)
    print("\nOccupation Distribution Across ISCO Major Groups (ESCO Taxonomy):")
    # Using the occupations dataframe loaded earlier
    if 'occupations' in locals() and not occupations.empty:
        # Extract first digit from iscoGroup column
        # Handle mixed types if any, ensure string
        taxonomy_isco_first = str(occupations['iscoGroup']).strip() # Wait, this is wrong. Need to apply to series.
        
        # Valid ISCO codes are usually 4 digits. Major group is 1st digit.
        valid_occupations = occupations[occupations['iscoGroup'].notna()].copy()
        valid_occupations['major_group'] = valid_occupations['iscoGroup'].astype(str).str[0]
        
        tax_counts = valid_occupations['major_group'].value_counts().to_dict()
        tax_total = sum(tax_counts.values())
        
        for grp in sorted(tax_counts.keys()):
            # Filter to ensure we only have digits 0-9
            if grp in isco_names:
                c = tax_counts[grp]
                pct = (c / tax_total) * 100
                name = isco_names.get(grp, 'Unknown')
                print(f"  {grp} - {name}: {c:,} ({pct:.1f}%)")
    else:
        print("Occupations dataframe not available for taxonomy analysis.")

    # --- Preprocessing steps ---
    print("Applying preprocessing steps...")
    
    # 1. Clean titles (remove dots)
    df.loc[:, 'new_job_title_en_cp'] = df.loc[:, 'new_job_title_en_cp'].apply(remove_leading_dots)
    
    # 2. Fill missing titles
    df.loc[:,'new_job_title_en_cp'] = df.apply(fill_missing_title, axis=1)
    
    # 3. Remove title prefix from description
    df.loc[:, 'new_job_description_en_cp'] = df.apply(remove_title_prefix, axis=1)
    
    # 4. Filter out resumes that still have '...' as title?
    # User logic: c_ids_no_title = df.query('new_job_title_en_cp == "..."')._id.unique()
    # df = df.query('_id not in @c_ids_no_title')
    
    c_ids_no_title = df.query('new_job_title_en_cp == "..."')['_id'].unique()
    print(f"Found {len(c_ids_no_title)} resumes with '...' titles to exclude.")
    
    df_clean = df.query('_id not in @c_ids_no_title').copy()
    print(f"Row count after filtering: {len(df_clean)}")

    # Statistics
    print("\n--- Karrierewege Plus Dataset Statistics ---")
    
    # 1. Total number of career histories (resumes)
    # Assuming _id represents the resume/user ID
    total_resumes = df_clean['_id'].nunique()
    print(f"Total number of career histories (resumes): {total_resumes:,}")
    
    # 2. Total number of job experiences
    total_job_experiences = len(df_clean)
    print(f"Total number of job experiences: {total_job_experiences:,}")
    
    # 3. Unique number of job experiences (Title + Description)
    # We should consider uniqueness based on the cleaned title and description
    unique_df = df_clean.drop_duplicates(subset=['new_job_title_en_cp', 'new_job_description_en_cp'])
    unique_exp = len(unique_df)
    print(f"Unique number of job experiences: {unique_exp:,}")
    
    # Check for NaN values in title/desc to avoid counting them as unique?
    # The snippet used drop_duplicates which handles NaNs effectively (Na=Na).
    
    # Optional: Print some examples like the notebook does
    print("\nSome example unique experiences:")
    print(unique_df[['new_job_title_en_cp', 'new_job_description_en_cp']].head(5))

    # --- Additional Requested Metrics ---
    print("\n--- Additional Requested Metrics ---")

    # 4. Official data split proportions
    print("\n[Official Data Split Proportions]")
    # We iterated over keys earlier, let's reconstruct since we merged them into df (and df_clean)
    # Ideally we should have kept counts per split.
    # Since we have df_clean (filtered), let's see if we can recover split distribution or if we should have counted before.
    # But filtering removes resumes. Let's count based on 'df_clean' if we can trace back, OR just standard raw split counts.
    # The split keys were printed at the start. Let's assume standard HF splits if we can't easily attribute rows back.
    # Actually, calculating on the raw dataframe 'df' (before filtering) is probably what "Official split proportions" means.
    # But wait, df didn't have a 'split' column added.
    # Let's reload just the lengths quickly or rely on the fact that we printed them implicitly if we added logging.
    # To be precise, let's just count from the 'dataset' object we loaded.
    
    total_raw = 0
    split_counts = {}
    for sk in dataset.keys():
        n = len(dataset[sk])
        split_counts[sk] = n
        total_raw += n
    
    for sk, count in split_counts.items():
        pct = (count / total_raw) * 100
        print(f"  {sk}: {count:,} ({pct:.1f}%)")

    # 5. Career Length Statistics
    print("\n[Career Length Statistics]")
    # Group by _id and count rows
    career_lengths = df_clean.groupby('_id').size()
    
    min_len = career_lengths.min()
    max_len = career_lengths.max()
    mean_len = career_lengths.mean()
    median_len = career_lengths.median()
    std_len = career_lengths.std()
    
    print(f"  Range: {min_len} - {max_len}")
    print(f"  Mean: {mean_len:.2f}")
    print(f"  Median: {median_len:.2f}")
    print(f"  Standard Deviation: {std_len:.2f}")
    
    # Percentiles
    percentiles = [10, 25, 50, 75, 90, 95]
    perc_values = np.percentile(career_lengths, percentiles)
    print("  Percentiles:")
    for p, v in zip(percentiles, perc_values):
        print(f"    {p}th: {v:.1f}")

    # Skewness
    skewness = career_lengths.skew()
    print(f"  Skewness: {skewness:.2f}")
    
    # 6. Taxonomy Coverage (Unique ESCO Concept URIs in Dataset / Total ESCO Concept URIs)
    print("\n[Taxonomy Coverage]")
    # We have CONTENT_TO_ESCO which maps (title, desc) -> ESCO ID
    # We need to apply this to the cleaned dataset rows.
    # We can reuse the logic: create keys from df_clean, lookup ESCO ID, count unique ESCO IDs found.
    
    # Optimization: deduplicate (title, desc) pairs first
    unique_pairs = df_clean[['new_job_title_en_cp', 'new_job_description_en_cp']].drop_duplicates()
    
    # Mapping
    found_esco_ids = set()
    for _, row in unique_pairs.iterrows():
        t = str(row['new_job_title_en_cp']) # Already cleaned in df_clean? Yes.
        d = str(row['new_job_description_en_cp']) # Already cleaned (prefix removed).
        
        # NOTE: CONTENT_TO_ESCO keys were built from Master CSV which has raw_title/raw_desc.
        # Our cleaning steps in 254-260 aim to reproduce that.
        # But wait, Master CSV 'raw_description' might NOT have the prefix removed if the prefix is IN the raw data?
        # Let's assume our cleaning matches the keys we built.
        # If not, coverage will be low. But we saw high ISCO mapping, so keys likely match.
        
        k = (t, d)
        eid = CONTENT_TO_ESCO.get(k)
        if eid:
            found_esco_ids.add(eid)
            
    num_unique_esco_in_dataset = len(found_esco_ids)
    
    # Total unique ESCO IDs (Occupations)
    if 'occupations' in locals() and not occupations.empty:
        total_esco_occupations = occupations['conceptUri'].nunique()
        coverage_pct = (num_unique_esco_in_dataset / total_esco_occupations) * 100
        print(f"  Unique ESCO Concepts in Dataset: {num_unique_esco_in_dataset:,}")
        print(f"  Total ESCO Occupations (Taxonomy): {total_esco_occupations:,}")
        print(f"  Coverage: {coverage_pct:.2f}%")
    else:
        print("  Occupations DataFrame not available for coverage calculation.")

    # 7. Frequency distribution of occupation labels
    print("\n[Label Frequency Distribution]")
    # Count frequency of each unique label (Title)
    # Or should it be unique (Title, Description) pair? "Occupation labels" usually implies the text label.
    # Let's do Title.
    label_counts = df_clean['new_job_title_en_cp'].value_counts()
    
    num_labels = len(label_counts)
    singletons = (label_counts == 1).sum()
    singleton_pct = (singletons / num_labels) * 100
    
    print(f"  Total Unique Labels (Titles): {num_labels:,}")
    print(f"  Singletons (appear only once): {singletons:,} ({singleton_pct:.2f}%)")
    
    # Rare labels (< 5 occurrences)
    rare = (label_counts < 5).sum()
    rare_pct = (rare / num_labels) * 100
    print(f"  Rare Labels (< 5 occurrences): {rare:,} ({rare_pct:.2f}%)")
    
    print("  Top 10 Most Frequent Labels:")
    print(label_counts.head(10))

    # --- Generate Figure 4.1 ---
    # We need career_lengths (Series) and dataset_isco_first_digits (List -> convert to Counter)
    # dataset_isco_first_digits was computed earlier.
    # We constructed 'counts' from it in step 198: counts = Counter(dataset_isco_first_digits)
    # But 'counts' might be out of scope if defined inside if block? 
    # Actually lines 198-200 defined 'counts' inside "if dataset_isco_first_digits:".
    # Let's ensure we have the counts.
    
    from collections import Counter
    if 'dataset_isco_first_digits' in locals() and dataset_isco_first_digits:
        isco_counts_counter = Counter(dataset_isco_first_digits)
    else:
        isco_counts_counter = Counter()

    generate_figure_4_1(career_lengths, isco_counts_counter)

if __name__ == "__main__":
    main()
