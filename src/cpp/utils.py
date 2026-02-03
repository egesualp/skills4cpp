from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import re

SEP_TOKEN = "<SEP>"  # Separator token, used to separate sentences in a document pair. This can be model specific.
DATA_PATH = Path("./data/")

# Master dataset paths for job_id mapping (maps job title + description to unique job_id)
MASTER_DATASET_PATHS = {
    'decorte': Path("/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/decorte_master_3.csv"),
    'karrierewege_occ': Path("/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/karrierewege_plus_occ_master.csv"),
    'karrierewege_cp': Path("/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/data/title_pairs_desc/karrierewege_plus_cp_master_3.csv"),
    # Placeholder for other datasets - add paths as needed
    'decorte_esco': None,  # Uses ESCO titles, not free text
    'karrierewege': None,   # Uses ESCO titles, not free text
    'karrierewege_100k': None,  # Uses ESCO titles, not free text
}


def load_job_id_lookup(master_csv_path: Path) -> Dict[Tuple[str, str], str]:
    """
    Load a lookup dictionary mapping (raw_title, raw_description) to job_id.
    
    Args:
        master_csv_path: Path to the master CSV file containing job_id mappings
        
    Returns:
        Dictionary mapping (raw_title.lower(), raw_description.lower()) -> job_id (as string)
    """
    df = pd.read_csv(master_csv_path)
    lookup = {}
    for _, row in df.iterrows():
        raw_desc = row['raw_description']
        if pd.isna(raw_desc):
            desc_str = ""
        else:
            desc_str = str(raw_desc).strip().lower()
            
        key = (str(row['raw_title']).strip().lower(), desc_str)
        lookup[key] = str(row['job_id'])
    print(f"  > Loaded job_id lookup with {len(lookup)} entries from {master_csv_path}")
    return lookup


def get_job_id(title: str, description: str, lookup: Dict[Tuple[str, str], str]) -> Optional[str]:
    """
    Look up job_id for a given title and description.
    
    Args:
        title: Raw job title
        description: Raw job description
        lookup: Dictionary from load_job_id_lookup
        
    Returns:
        job_id as string, or None if not found
    """
    if description is None:
        desc_str = ""
    else:
        desc_str = str(description).strip().lower()
        
    key = (str(title).strip().lower(), desc_str)
    return lookup.get(key, None)



def replace_esco_titles(example, i):
    """
    Replaces specific ESCO job titles with alternative titles for consistency.

    Args:
        example (dict): A dictionary representing a dataset row.
        i (int): The index of the ESCO title column.

    Returns:
        dict: Updated dictionary with the replaced ESCO title and URI.
    """
    replacements_title = {
        'ICT security engineer': 'cyber incident responder',
        'ict security engineer': 'cyber incident responder',
        'care at home worker': 'care home worker',
        'residential care home worker': 'care home worker',
        'ICT security manager': 'cybersecurity risk manager',
        'ict security manager': 'cybersecurity risk manager',
        'care at hmoe worker': 'care home worker',
        'handyman': 'handyperson',
        'corporate banking manager': 'corporate banking adviser',
    }

    original_title = example[f'ESCO_title_{i}']
    if not pd.isna(original_title):
        processed_title = original_title.strip().lower()
        final_title = replacements_title.get(processed_title, processed_title)
    else:
        final_title = original_title

    example[f'ESCO_title_{i}'] = final_title

    replacements_uri = {
        'http://data.europa.eu/esco/occupation/81309031-dad2-4a7a-bde6-7f6e518f89ff': 
        'http://data.europa.eu/esco/occupation/f4525ed8-54eb-4a3b-90db-55cc01b0d9fd'
    }
    
    example[f'ESCO_uri_{i}'] = replacements_uri.get(example[f'ESCO_uri_{i}'], example[f'ESCO_uri_{i}'])
    
    return example


def subspans(lst):
    """
    Generates all subspans of a list with a minimum length of 2.

    Args:
        lst (List[str]): List of elements.

    Yields:
        Generator[List[str], None, None]: Subspan of the input list.
    """
    for i in range(2, len(lst) + 1):
        for j in range(len(lst) - i + 1):
            yield lst[j:j + i]


def clean_title_remove_dots(title):
    """Remove three dots if they appear at the start of the job title."""
    if isinstance(title, str) and title.startswith("..."):
        if title != "...":
            return title[3:].lstrip()
    return title


def clean_title_fill(title, description):
    """If title is '...' and desc contains a 'job_title: ...' pattern, set title from there."""
    if title == '...' and isinstance(description, str):
        # Search for the pattern '<something>: <rest of desc>' at the start
        # Try extracting the prefix before the first colon-space
        m = re.match(r"([^:]+):", description.strip())
        if m:
            extracted_title = m.group(1).strip()
            if extracted_title and extracted_title != "...":
                return extracted_title
    return title


def clean_description_remove_prefix(title, description):
    if isinstance(description, str) and isinstance(title, str):
        prefix = f"{title}: "
        if description.startswith(prefix):
            return description[len(prefix):]
    return description


def load_prepare_karrierewege(minus_last, consider_all_subspans_of_len_at_least_2=False, language='en'):
    """
    Loads and processes the Karrierewege dataset for training.

    Args:
        minus_last (bool): If True, removes the last experience in subspans.
        consider_all_subspans_of_len_at_least_2 (bool, optional): If True, considers all subspans with at least 2 elements. Defaults to False.
        language (str, optional): Specifies the dataset language variant. Defaults to 'en'.

    Returns:
        tuple: (train_pairs, train_job_ids, val_pairs, val_job_ids, test_pairs, test_job_ids)
               - *_pairs: List of (doc_1, doc_2) tuples
               - *_job_ids: List of lists, each inner list contains job_ids for jobs in doc_1
    """
    
    # Load job_id lookup for free-text variants
    job_id_lookup = None
    if language == 'en_free':
        master_path = MASTER_DATASET_PATHS.get('karrierewege_occ')
        if master_path and master_path.exists():
            print(f"Loading job_id lookup for karrierewege_occ dataset...")
            job_id_lookup = load_job_id_lookup(master_path)
        else:
            print(f"Warning: Master dataset not found at {master_path}. job_ids will be None.")
    elif language == 'en_free_cp':
        master_path = MASTER_DATASET_PATHS.get('karrierewege_cp')
        if master_path and master_path.exists():
            print(f"Loading job_id lookup for karrierewege_cp dataset...")
            job_id_lookup = load_job_id_lookup(master_path)
        else:
            print(f"Warning: Master dataset not found at {master_path}. job_ids will be None.")
    # For ESCO-based variants (en, esco_100k), job_id_lookup remains None



    def create_pairs_from_dataset(_dataset):
        document_pairs = []
        all_job_ids = []  # List of lists: job_ids for each doc_1
        
        # Additional formatting
        _dataset_df = _dataset.to_pandas()
        if language == 'en_free_cp':
            _dataset_df.loc[:, 'new_job_title_en_cp'] = _dataset_df.loc[:, 'new_job_title_en_cp'].apply(clean_title_remove_dots)
            
            def apply_fill_missing(row):
                return clean_title_fill(row['new_job_title_en_cp'], row['new_job_description_en_cp'])
            _dataset_df.loc[:,'new_job_title_en_cp'] = _dataset_df.apply(apply_fill_missing, axis=1)
            
            def apply_remove_prefix(row):
                return clean_description_remove_prefix(row['new_job_title_en_cp'], row['new_job_description_en_cp'])
            _dataset_df.loc[:, 'new_job_description_en_cp'] = _dataset_df.apply(apply_remove_prefix, axis=1)
            c_ids_no_title = _dataset_df.query('new_job_title_en_cp == "..."')._id.unique()
            _dataset_df = _dataset_df.query('_id not in @c_ids_no_title')

        grouped = _dataset_df.groupby('_id')
        print('len grouped', len(grouped))

        # Iterate over the dataset it is da df and each row has a title, description ...
        for _id, group in tqdm(grouped):
            #sort by 'experience_order' ascending
            group = group.sort_values('experience_order')
            #differ by language, the same for German or other ESCO language variants possible
            if language == 'en' or language == 'esco_100k':
                titles = group['preferredLabel_en'].tolist()
                descriptions = group['description_en'].tolist()
            elif language == 'en_free':
                titles = group['new_job_title_en_occ'].tolist()
                descriptions = group['new_job_description_en_occ'].tolist()
                titles_esco = group['preferredLabel_en'].tolist()
                descriptions_esco = group['description_en'].tolist()
            elif language == 'en_free_cp':
                titles = group['new_job_title_en_cp'].tolist()
                descriptions = group['new_job_description_en_cp'].tolist()
                titles_esco = group['preferredLabel_en'].tolist()
                descriptions_esco = group['description_en'].tolist()
            number_of_experiences = len(group)

            all_experience_indexes = list(range(number_of_experiences))

                
            # Create document pair
            if language == 'en' or language == 'de' or language == 'esco_100k':
                if consider_all_subspans_of_len_at_least_2 and number_of_experiences > 1:
                    _title_subspans = list(subspans(titles))
                    _description_subspans = list(subspans(descriptions))
                    _experience_indexes_subspans = list(subspans(all_experience_indexes))

                else:
                    _title_subspans = [titles]
                    _description_subspans = [descriptions]
                    _experience_indexes_subspans = [all_experience_indexes]

                
                for _titles, _descriptions, _experience_indexes in zip(_title_subspans, _description_subspans, _experience_indexes_subspans):

                    if minus_last:
                        span_discount = 1
                    else:
                        span_discount = 0

                    _num_experiences_subspan = len(_titles)
                    # doc_2: the title and description from the last experience in the subspan
                    doc_2 = f"esco role: {titles[_experience_indexes[-1]]} \n description: {descriptions[_experience_indexes[-1]]}"

                    # doc_1: current career history subspan
                    doc_1 = SEP_TOKEN.join(
                        [
                            f"role: {_titles[i]} \n description: {_descriptions[i]}"
                            for i in range(_num_experiences_subspan-span_discount)
                        ]
                    )
    
                    # For ESCO-based variants, job_ids are empty (no master dataset)
                    doc_1_job_ids = []
          
                    # Add document pair and job_ids to lists
                    document_pairs.append((doc_1, doc_2))
                    all_job_ids.append(doc_1_job_ids)
            
            elif language=='en_free' or language == 'de_free' or language == 'en_free_cp':
                if consider_all_subspans_of_len_at_least_2 and number_of_experiences > 1:
                    _title_subspans = list(subspans(titles))
                    _description_subspans = list(subspans(descriptions))
                    _ESCO_title_subspans = list(subspans(titles_esco))
                    _ESCO_uri_subspans = list(subspans(descriptions_esco))
                    _experience_indexes_subspans = list(subspans(all_experience_indexes))
                else:
                    _title_subspans = [titles]
                    _description_subspans = [descriptions]
                    _ESCO_title_subspans = [titles_esco]
                    _ESCO_uri_subspans = [descriptions_esco]
                    _experience_indexes_subspans = [all_experience_indexes]
                
                for _titles, _descriptions, _titles_esco, _descriptions_esco, _experience_indexes in zip(_title_subspans, _description_subspans, _ESCO_title_subspans, _ESCO_uri_subspans, _experience_indexes_subspans):

                    _num_experiences_subspan = len(_titles)
                    if minus_last:
                        span_discount = 1
                    else:
                        span_discount = 0

                    doc_2 = f"esco role: {titles_esco[_experience_indexes[-1]]} \n description: {descriptions_esco[_experience_indexes[-1]]}"

                    # doc_1: current career history subspan
                    doc_1 = SEP_TOKEN.join(
                        [
                            f"role: {_titles[i]} \n description: {_descriptions[i]}"
                            for i in range(_num_experiences_subspan-span_discount)
                        ]
                    )
                    
                    # Collect job_ids for jobs in doc_1 (before concatenation)
                    doc_1_job_ids = []
                    if job_id_lookup is not None:
                        for i in range(_num_experiences_subspan - span_discount):
                            job_id = get_job_id(_titles[i], _descriptions[i], job_id_lookup)
                            if job_id is not None:
                                doc_1_job_ids.append(job_id)
                            # If job_id not found, we skip it (won't have skills for this job)
                          
                    # Add document pair and job_ids to lists
                    document_pairs.append((doc_1, doc_2))
                    all_job_ids.append(doc_1_job_ids)
                    
        return document_pairs, all_job_ids
    
  
    # Load the dataset
    if language == 'en_free' or language == 'de_free' or language == 'esco_100k' or language == 'en_free_cp' or language == 'de_free_cp':
        dataset = load_dataset("ElenaSenger/Karrierewege_plus")
    elif language == 'en':
        dataset = load_dataset("ElenaSenger/Karrierewege")

    train_pairs, train_job_ids = create_pairs_from_dataset(dataset["train"])
    val_pairs, val_job_ids = create_pairs_from_dataset(dataset["validation"])
    test_pairs, test_job_ids = create_pairs_from_dataset(dataset["test"])


    return train_pairs, train_job_ids, val_pairs, val_job_ids, test_pairs, test_job_ids


def load_prepare_decorte(minus_last, consider_all_subspans_of_len_at_least_2=False, verbose=False, max_len=16):
    """
    Loads and processes the Decorte dataset for training.

    Args:
        minus_last (bool): If True, removes the last experience in subspans.
        consider_all_subspans_of_len_at_least_2 (bool, optional): If True, considers all subspans with at least 2 elements. Defaults to False.
        verbose (bool, optional): If True, prints additional information. Defaults to False.
        max_len (int, optional): Maximum length of subspans. Defaults to 16.

    Returns:
        tuple: (train_pairs, train_job_ids, val_pairs, val_job_ids, test_pairs, test_job_ids)
               - *_pairs: List of (doc_1, doc_2) tuples
               - *_job_ids: List of lists, each inner list contains job_ids for jobs in doc_1
    """

    # Load job_id lookup from master CSV
    master_path = MASTER_DATASET_PATHS.get('decorte')
    job_id_lookup = None
    if master_path and master_path.exists():
        print(f"Loading job_id lookup for decorte dataset...")
        job_id_lookup = load_job_id_lookup(master_path)
    else:
        print(f"Warning: Master dataset not found at {master_path}. job_ids will be None.")

    # Load the dataset
    dataset = load_dataset("jensjorisdecorte/anonymous-working-histories")

    # Apply replacements to all columns in the dataset beginning with ESCO_title
    for i in range(16):
        dataset['train'] = dataset['train'].map(lambda example: replace_esco_titles(example, i))
        dataset['validation'] = dataset['validation'].map(lambda example: replace_esco_titles(example, i))
        dataset['test'] = dataset['test'].map(lambda example: replace_esco_titles(example, i))

    # Load descriptions for ESCO occupations
    ESCO_occupations = pd.read_csv(DATA_PATH / "occupations_en.csv")


    # Create dictionary for ESCO occupations
    ESCO_occupations_dict = ESCO_occupations.set_index("conceptUri")[
        "description"
    ].to_dict()

    # Add to ESCO_occupations_dict keys which are the names of the occupations, and as value the description of the occupation
    ESCO_occupations_dict.update(
        ESCO_occupations.set_index("preferredLabel")["description"].to_dict()
    )

    # For every occupation, go through the altLabels and add them to the dictionary
    for index, row in ESCO_occupations.iterrows():
        # If there are no altLabels, skip
        if pd.isna(row["altLabels"]):
            continue
        for alt_label in row["altLabels"].split("\n"):
            ESCO_occupations_dict[alt_label] = row["description"]

    def create_pairs_from_dataset(_dataset):
        document_pairs = []
        all_job_ids = []  # List of lists: job_ids for each doc_1
        
        # Iterate over the dataset
        for example in tqdm(_dataset):

            titles = [
                example[f"title_{i}"] for i in range(example["number_of_experiences"])
            ]
            descriptions = [
                example[f"description_{i}"]
                for i in range(example["number_of_experiences"])
            ]
            ESCO_uris = [
                example[f"ESCO_uri_{i}"]
                for i in range(example["number_of_experiences"])
            ]
            ESCO_titles = [
                example[f"ESCO_title_{i}"]
                for i in range(example["number_of_experiences"])
            ]

            if verbose:
                # Inspection
                for i in range(example["number_of_experiences"]):
                    print(f"Title: {example[f'title_{i}']}")
                    print(f"Description: {example[f'description_{i}']}")
                    print(f"ESCO URI: {example[f'ESCO_uri_{i}']}")
                    print(f"ESCO Title: {example[f'ESCO_title_{i}']}")
                    print()


            def free_text_experience(_experience_title, _experience_description):
                return f"role: {_experience_title} \n description: {_experience_description}"

            def ESCO_experience(_ESCO_title, _ESCO_uri):
                try:
                    return f"esco role: {_ESCO_title} \n description: {ESCO_occupations_dict[_ESCO_uri]}"
                except KeyError:
                    return f"esco role: {_ESCO_title} \n description: {ESCO_occupations_dict[_ESCO_title]}"
                
            all_experience_indexes = list(range(example["number_of_experiences"]))

            #ESCO_titles withouth additional spaces
            ESCO_titles = [title.strip() for title in ESCO_titles]

            if consider_all_subspans_of_len_at_least_2 and example["number_of_experiences"] > 1:
                _title_subspans = list(subspans(titles))
                _description_subspans = list(subspans(descriptions))
                _ESCO_title_subspans = list(subspans(ESCO_titles))
                _ESCO_uri_subspans = list(subspans(ESCO_uris))
                _experience_indexes_subspans = list(subspans(all_experience_indexes))
                # keep only the last jobs in length max_len
                if len(_title_subspans) > max_len:
                    _title_subspans = _title_subspans[-max_len:]
                    _description_subspans = _description_subspans[-max_len:]
                    _ESCO_title_subspans = _ESCO_title_subspans[-max_len:]
                    _ESCO_uri_subspans = _ESCO_uri_subspans[-max_len:]
                    _experience_indexes_subspans = _experience_indexes_subspans[-max_len:]
            else:
                _title_subspans = [titles]
                _description_subspans = [descriptions]
                _ESCO_title_subspans = [ESCO_titles]
                _ESCO_uri_subspans = [ESCO_uris]
                _experience_indexes_subspans = [all_experience_indexes]
            
            for _titles, _descriptions, _ESCO_titles, _ESCO_uris, _experience_indexes in zip(_title_subspans, _description_subspans, _ESCO_title_subspans, _ESCO_uri_subspans, _experience_indexes_subspans):

                _num_experiences_subspan = len(_titles)
                if minus_last:
                    span_discount = 1
                else:
                    span_discount = 0
                

                # As doc_2 the esco role and description of the last job in the career history
                doc_2 = ESCO_experience(
                    ESCO_titles[_experience_indexes[-1]],
                    ESCO_uris[_experience_indexes[-1]],
                )


                # As doc_2 set the next ESCO experience in the career history
                # As doc_1 set the current career history subspan
                doc_1 = SEP_TOKEN.join(
                    [
                        free_text_experience(_titles[i], _descriptions[i])
                        for i in range(_num_experiences_subspan-span_discount)
                    ]
                )

                # Collect job_ids for jobs in doc_1 (before concatenation)
                doc_1_job_ids = []
                if job_id_lookup is not None:
                    for i in range(_num_experiences_subspan - span_discount):
                        job_id = get_job_id(_titles[i], _descriptions[i], job_id_lookup)
                        # Always append, even if None, to maintain alignment with doc_1 segments
                        doc_1_job_ids.append(job_id)

                # Add document pair and job_ids to lists
                document_pairs.append((doc_1, doc_2))
                all_job_ids.append(doc_1_job_ids)

        return document_pairs, all_job_ids

    train_pairs, train_job_ids = create_pairs_from_dataset(dataset["train"])
    val_pairs, val_job_ids = create_pairs_from_dataset(dataset["validation"])
    test_pairs, test_job_ids = create_pairs_from_dataset(dataset["test"])

    return train_pairs, train_job_ids, val_pairs, val_job_ids, test_pairs, test_job_ids

def load_prepare_decorte_esco(minus_last, consider_all_subspans_of_len_at_least_2=False, verbose=False, max_len = 16):
    """
    Loads and processes the Decorte ESCO dataset for training.

    Args:
        minus_last (bool): If True, removes the last experience in subspans.
        consider_all_subspans_of_len_at_least_2 (bool, optional): If True, considers all subspans with at least 2 elements. Defaults to False.
        verbose (bool, optional): If True, prints additional information. Defaults to False.
        max_len (int, optional): Maximum length of subspans. Defaults to 16.

    Returns:
        tuple: (train_pairs, train_job_ids, val_pairs, val_job_ids, test_pairs, test_job_ids)
               - *_pairs: List of (doc_1, doc_2) tuples
               - *_job_ids: List of lists (empty lists for ESCO-based datasets)
    """


    # Load the dataset
    dataset = load_dataset("jensjorisdecorte/anonymous-working-histories")


    # Apply replacements to all columns in the dataset beginning with ESCO_title
    for i in range(16):
        dataset['train'] = dataset['train'].map(lambda example: replace_esco_titles(example, i))
        dataset['validation'] = dataset['validation'].map(lambda example: replace_esco_titles(example, i))
        dataset['test'] = dataset['test'].map(lambda example: replace_esco_titles(example, i))

    # Load descriptions for ESCO occupations
    ESCO_occupations = pd.read_csv(DATA_PATH / "occupations_en.csv")



    # Create dictionary for ESCO occupations
    ESCO_occupations_dict = ESCO_occupations.set_index("conceptUri")[
        "description"
    ].to_dict()

    # Add to ESCO_occupations_dict keys which are the names of the occupations, and as value the description of the occupation
    ESCO_occupations_dict.update(
        ESCO_occupations.set_index("preferredLabel")["description"].to_dict()
    )

    # For every occupation, go through the altLabels and add them to the dictionary
    for index, row in ESCO_occupations.iterrows():
        # If there are no altLabels, skip
        if pd.isna(row["altLabels"]):
            continue
        for alt_label in row["altLabels"].split("\n"):
            ESCO_occupations_dict[alt_label] = row["description"]

    def create_pairs_from_dataset(_dataset):
        document_pairs = []
        all_job_ids = []  # Empty lists for ESCO-based datasets (no master dataset)
        
        # Iterate over the dataset
        for example in tqdm(_dataset):

            titles = [
                example[f"title_{i}"] for i in range(example["number_of_experiences"])
            ]
            descriptions = [
                example[f"description_{i}"]
                for i in range(example["number_of_experiences"])
            ]
            ESCO_uris = [
                example[f"ESCO_uri_{i}"]
                for i in range(example["number_of_experiences"])
            ]
            ESCO_titles = [
                example[f"ESCO_title_{i}"]
                for i in range(example["number_of_experiences"])
            ]

            if verbose:
                # Inspection
                for i in range(example["number_of_experiences"]):
                    print(f"Title: {example[f'title_{i}']}")
                    print(f"Description: {example[f'description_{i}']}")
                    print(f"ESCO URI: {example[f'ESCO_uri_{i}']}")
                    print(f"ESCO Title: {example[f'ESCO_title_{i}']}")
                    print()


            def free_text_experience(_experience_title, _experience_description):
                return f"role: {_experience_title} \n description: {_experience_description}"

            def ESCO_experience(_ESCO_title, _ESCO_uri):
                try:
                    return f"esco role: {_ESCO_title} \n description: {ESCO_occupations_dict[_ESCO_uri]}"
                except KeyError:
                    return f"esco role: {_ESCO_title} \n description: {ESCO_occupations_dict[_ESCO_title]}"
                
            all_experience_indexes = list(range(example["number_of_experiences"]))

            if consider_all_subspans_of_len_at_least_2 and example["number_of_experiences"] > 1:
                _title_subspans = list(subspans(titles))
                _description_subspans = list(subspans(descriptions))
                _ESCO_title_subspans = list(subspans(ESCO_titles))
                _ESCO_uri_subspans = list(subspans(ESCO_uris))
                _experience_indexes_subspans = list(subspans(all_experience_indexes))
                # keep only the last jobs in length max_len
                if len(_title_subspans) > max_len:
                    _title_subspans = _title_subspans[-max_len:]
                    _description_subspans = _description_subspans[-max_len:]
                    _ESCO_title_subspans = _ESCO_title_subspans[-max_len:]
                    _ESCO_uri_subspans = _ESCO_uri_subspans[-max_len:]
                    _experience_indexes_subspans = _experience_indexes_subspans[-max_len:]
            else:
                _title_subspans = [titles]
                _description_subspans = [descriptions]
                _ESCO_title_subspans = [ESCO_titles]
                _ESCO_uri_subspans = [ESCO_uris]
                _experience_indexes_subspans = [all_experience_indexes]
            
            for _titles, _descriptions, _ESCO_titles, _ESCO_uris, _experience_indexes in zip(_title_subspans, _description_subspans, _ESCO_title_subspans, _ESCO_uri_subspans, _experience_indexes_subspans):

                if minus_last:
                    span_discount = 1
                else:
                    span_discount = 0

                _num_experiences_subspan = len(_titles)
                
                # Create document pair

                # As doc_2 set the next ESCO experience in the career history
                # As doc_1 set the current career history subspan
                doc_1 = SEP_TOKEN.join(
                    [
                        ESCO_experience(ESCO_titles[i], ESCO_uris[i])
                        for i in range(_num_experiences_subspan-span_discount)
                    ]
                )


                # As doc_2 the esco role and description of the last job in the career history
                doc_2 = ESCO_experience(
                    ESCO_titles[_experience_indexes[-1]],
                    ESCO_uris[_experience_indexes[-1]],
                )

                # Add document pair and empty job_ids to lists
                document_pairs.append((doc_1, doc_2))
                all_job_ids.append([])  # Empty list for ESCO-based datasets

        return document_pairs, all_job_ids

    train_pairs, train_job_ids = create_pairs_from_dataset(dataset["train"])
    val_pairs, val_job_ids = create_pairs_from_dataset(dataset["validation"])
    test_pairs, test_job_ids = create_pairs_from_dataset(dataset["test"])

    return train_pairs, train_job_ids, val_pairs, val_job_ids, test_pairs, test_job_ids


        

