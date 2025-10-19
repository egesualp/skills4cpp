import os
import pandas as pd
from pathlib import Path

def check_title_pairs_coverage():
    """
    Checks the coverage of ESCO occupations in the title_pairs dataset.

    This function reads all CSV files from the 'data/title_pairs' directory,
    compares the ESCO IDs found against the master list of ESCO occupations,
    and generates a coverage report. The report is saved to '~/reports/coverage_report.txt'.
    """
    # Define paths
    project_root = Path(__file__).resolve().parents[1]
    occupations_file = project_root / 'data' / 'occupations_en.csv'
    title_pairs_dir = project_root / 'data' / 'title_pairs'
    report_dir = project_root / 'reports'
    report_file = report_dir / 'coverage_report.txt'

    # Create reports directory if it doesn't exist
    report_dir.mkdir(parents=True, exist_ok=True)

    # Load ESCO occupations data
    try:
        esco_df = pd.read_csv(occupations_file)
    except FileNotFoundError:
        print(f"Error: ESCO occupations file not found at {occupations_file}")
        return

    all_esco_ids = set(esco_df['conceptUri'])
    esco_id_to_title = dict(zip(esco_df['conceptUri'], esco_df['preferredLabel']))

    # Find all CSV files in the title_pairs directory
    title_pairs_files = list(title_pairs_dir.glob('*.csv'))
    if not title_pairs_files:
        print(f"No CSV files found in {title_pairs_dir}")
        return

    # --- Overall Coverage Calculation ---
    all_covered_esco_ids = set()
    file_coverage_data = {}

    for file_path in title_pairs_files:
        try:
            df = pd.read_csv(file_path)
            if 'esco_id' in df.columns:
                file_esco_ids = set(df['esco_id'].dropna().unique())
                all_covered_esco_ids.update(file_esco_ids)
                file_coverage_data[file_path.name] = file_esco_ids
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    # Calculate overall coverage
    overall_missing_ids = all_esco_ids - all_covered_esco_ids
    overall_coverage_percentage = (len(all_covered_esco_ids) / len(all_esco_ids)) * 100 if all_esco_ids else 0

    # --- Generate the Report ---
    with open(report_file, 'w') as f:
        f.write("ESCO Occupations Overall Coverage Report\n")
        f.write("========================================\n\n")
        f.write(f"Total number of ESCO occupations: {len(all_esco_ids)}\n")
        f.write(f"Number of unique ESCO occupations covered in title_pairs: {len(all_covered_esco_ids)}\n")
        f.write(f"Overall Coverage: {overall_coverage_percentage:.2f}%\n\n")

        if overall_missing_ids:
            f.write("Overall Missing ESCO Occupations (from all files combined):\n")
            f.write("----------------------------------------------------------\n")
            f.write(f"{'ESCO ID':<70} {'ESCO Title'}\n")
            f.write(f"{'-'*70} {'-'*30}\n")
            missing_occupations = sorted(
                [(id, esco_id_to_title.get(id, 'N/A')) for id in overall_missing_ids],
                key=lambda x: x[1]
            )
            for esco_id, esco_title in missing_occupations:
                f.write(f"{esco_id:<70} {esco_title}\n")
        else:
            f.write("All ESCO occupations are covered across all files.\n")
        
        f.write("\n\n" + "="*80 + "\n\n")

        # --- Per-File Coverage Details ---
        f.write("Per-File Coverage Details\n")
        f.write("=========================\n\n")

        for file_name, covered_ids in sorted(file_coverage_data.items()):
            f.write(f"--- File: {file_name} ---\n")
            
            file_coverage_percentage = (len(covered_ids) / len(all_esco_ids)) * 100 if all_esco_ids else 0
            f.write(f"  - Unique ESCO occupations covered: {len(covered_ids)}\n")
            f.write(f"  - Coverage: {file_coverage_percentage:.2f}%\n")
            
            missing_for_file = all_esco_ids - covered_ids
            
            if missing_for_file:
                f.write("  - Missing ESCO Occupations for this file:\n")
                # To avoid clutter, we can list a few or just the count.
                # Here, we'll list them similar to the overall report.
                # For a very long list, you might just want the count:
                f.write(f"    - Total missing: {len(missing_for_file)}\n\n")

                # Uncomment the block below to list all missing IDs for each file.
                # f.write(f"    {'ESCO ID':<70} {'ESCO Title'}\n")
                # f.write(f"    {'-'*70} {'-'*30}\n")
                # missing_occupations_file = sorted(
                #     [(id, esco_id_to_title.get(id, 'N/A')) for id in missing_for_file],
                #     key=lambda x: x[1]
                # )
                # for esco_id, esco_title in missing_occupations_file:
                #      f.write(f"    {esco_id:<70} {esco_title}\n")
            else:
                f.write("  - All ESCO occupations are covered in this file.\n")
            
            f.write("\n")

    print(f"Coverage report saved to {report_file}")

if __name__ == '__main__':
    check_title_pairs_coverage()
