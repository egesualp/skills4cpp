import pandas as pd
import os

def main():
    # Paths
    base_dir = '/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc'
    test_pairs_path = os.path.join(base_dir, 'decorte_test_pairs.csv')
    augmented_path = '/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/processed/augmentation/augmented_decorte_occupations_with_desc_2.csv'
    relations_path = '/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/esco_datasets/occupationSkillRelations_en.csv'
    output_path = os.path.join(base_dir, 'category_test_split.csv')

    print(f"Reading test pairs from {test_pairs_path}...")
    test_pairs_df = pd.read_csv(test_pairs_path)
    
    print(f"Reading augmented data from {augmented_path}...")
    augmented_df = pd.read_csv(augmented_path)
    
    print(f"Reading skill relations from {relations_path}...")
    relations_df = pd.read_csv(relations_path)

    # 1. Map test pairs to augmented data to get job_id and skill_brief
    # We use raw_title and raw_description as keys to ensure correct mapping
    print("Mapping test pairs to augmented data...")
    test_with_augmented = test_pairs_df.merge(
        augmented_df[['raw_title', 'raw_description', 'job_id', 'skill_brief']],
        on=['raw_title', 'raw_description'],
        how='inner'
    )
    
    print(f"Found {len(test_with_augmented)} matching augmented samples.")

    # 2. Map with related skills
    # esco_id in test_pairs corresponds to occupationUri in relations_df
    print("Mapping to related skills...")
    final_df = test_with_augmented.merge(
        relations_df[['occupationUri', 'skillUri']],
        left_on='esco_id',
        right_on='occupationUri',
        how='inner'
    )

    # 3. Final selection and renaming
    # job_id, job_text (skill_brief), skillUri
    result = final_df[['job_id', 'skill_brief', 'skillUri']]

    print(f"Saving to {output_path}...")
    result.to_csv(output_path, index=False)
    print(f"Done! Created test split with {len(result)} rows and {result['job_id'].nunique()} unique jobs.")

if __name__ == "__main__":
    main()

