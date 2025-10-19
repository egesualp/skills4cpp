# elenasenger/karrierewege/src/utils.py (modified)

from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from pathlib import Path

SEP_TOKEN = "<SEP>"  # Separator token, used to separate sentences in a document pair. This can be model specific.
DATA_PATH = Path(__file__).parent.parent / "data"



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


def load_prepare_karrierewege(minus_last, consider_all_subspans_of_len_at_least_2=False, language='en', max_rows: int = None):
    """
    Loads and processes the Karrierewege dataset for training.

    Args:
        minus_last (bool): If True, removes the last experience in subspans.
        consider_all_subspans_of_len_at_least_2 (bool, optional): If True, considers all subspans with at least 2 elements. Defaults to False.
        language (str, optional): Specifies the dataset language variant. Defaults to 'en'.
        max_rows (int, optional): The maximum number of rows to load from each split. Defaults to None (load all).

    Returns:
        tuple: (train_pairs, val_pairs, test_pairs) - Prepared document pairs.
    """

    def create_pairs_from_dataset(_dataset):
        document_pairs = []
        _dataset_df = _dataset.to_pandas()
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
    
          
                    # Add document pair to list
                    document_pairs.append((doc_1, doc_2))
            
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
                          
                    # Add document pair to list
                    document_pairs.append((doc_1,doc_2))
                    
        return document_pairs
    
  
    # Load the dataset
    split_slice = f"[:{max_rows}]" if max_rows is not None else ""
    if language == 'en_free' or language == 'de_free' or language == 'esco_100k' or language == 'en_free_cp' or language == 'de_free_cp':
        dataset = load_dataset("ElenaSenger/Karrierewege_plus", split={s: s + split_slice for s in ["train", "validation", "test"]})
    elif language == 'en':
        dataset = load_dataset("ElenaSenger/Karrierewege", split={s: s + split_slice for s in ["train", "validation", "test"]})

    train_pairs = create_pairs_from_dataset(dataset["train"])
    val_pairs = create_pairs_from_dataset(dataset["validation"])
    test_pairs = create_pairs_from_dataset(dataset["test"])


    return train_pairs, val_pairs, test_pairs


def load_prepare_decorte(minus_last, consider_all_subspans_of_len_at_least_2=False, verbose=False, max_len=16, max_rows: int = None):
    """
    Loads and processes the Decorte dataset for training.

    Args:
        minus_last (bool): If True, removes the last experience in subspans.
        consider_all_subspans_of_len_at_least_2 (bool, optional): If True, considers all subspans with at least 2 elements. Defaults to False.
        verbose (bool, optional): If True, prints additional information. Defaults to False.
        max_len (int, optional): Maximum length of subspans. Defaults to 16.
        max_rows (int, optional): The maximum number of rows to load from each split. Defaults to None (load all).

    Returns:
        tuple: (train_pairs, val_pairs, test_pairs) - Prepared document pairs.
    """


    # Load the dataset
    split_slice = f"[:{max_rows}]" if max_rows is not None else ""
    dataset = load_dataset("jensjorisdecorte/anonymous-working-histories", split={s: s + split_slice for s in ["train", "validation", "test"]})

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


                # Add document pair to list
                document_pairs.append((doc_1,doc_2))

        return document_pairs

    train_pairs = create_pairs_from_dataset(dataset["train"])
    val_pairs = create_pairs_from_dataset(dataset["validation"])
    test_pairs = create_pairs_from_dataset(dataset["test"])

    return train_pairs, val_pairs, test_pairs

def load_prepare_decorte_esco(minus_last, consider_all_subspans_of_len_at_least_2=False, verbose=False, max_len = 16, max_rows: int = None):
    """
    Loads and processes the Decorte ESCO dataset for training.

    Args:
        minus_last (bool): If True, removes the last experience in subspans.
        consider_all_subspans_of_len_at_least_2 (bool, optional): If True, considers all subspans with at least 2 elements. Defaults to False.
        verbose (bool, optional): If True, prints additional information. Defaults to False.
        max_len (int, optional): Maximum length of subspans. Defaults to 16.
        max_rows (int, optional): The maximum number of rows to load from each split. Defaults to None (load all).

    Returns:
        tuple: (train_pairs, val_pairs, test_pairs) - Prepared document pairs.
    """


    # Load the dataset
    split_slice = f"[:{max_rows}]" if max_rows is not None else ""
    dataset = load_dataset("jensjorisdecorte/anonymous-working-histories", split={s: s + split_slice for s in ["train", "validation", "test"]})


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

                # Add document pair to list
                document_pairs.append((doc_1,doc_2))

        return document_pairs

    train_pairs = create_pairs_from_dataset(dataset["train"])
    val_pairs = create_pairs_from_dataset(dataset["validation"])
    test_pairs = create_pairs_from_dataset(dataset["test"])

    return train_pairs, val_pairs, test_pairs

# do i really need this?
def load_raw_to_esco_pairs(dataset_name, max_rows: int = None, kw_source: str = 'all'):
    """
    Loads a dataset and extracts unique pairs of (raw job title, ESCO job title).

    Args:
        dataset_name (str): The name of the dataset to load.
                            Supported: 'decorte', 'karrierewege_plus'.
        max_rows (int, optional): Max rows to load from each split. Defaults to None.
        kw_source (str, optional): For 'karrierewege_plus', specifies the source of raw titles.
                                   Can be 'occ', 'cp', or 'all'. Defaults to 'all'.

    Returns:
        tuple: (train_pairs, val_pairs, test_pairs) containing unique (raw, ESCO title, ESCO URI) triplets.
    """

    split_slice = f"[:{max_rows}]" if max_rows is not None else ""

    if dataset_name == 'decorte':
        dataset = load_dataset("jensjorisdecorte/anonymous-working-histories", split={s: s + split_slice for s in ["train", "validation", "test"]})

        def create_triplets_from_decorte(_dataset):
            triplets = []
            for example in tqdm(_dataset):
                for i in range(example["number_of_experiences"]):
                    raw_title = example.get(f"title_{i}")
                    esco_title = example.get(f"ESCO_title_{i}")
                    esco_uri = example.get(f"ESCO_uri_{i}")
                    if raw_title and esco_title and esco_uri and pd.notna(raw_title) and pd.notna(esco_title) and pd.notna(esco_uri):
                        triplets.append((raw_title.strip(), esco_title.strip(), esco_uri.strip()))
            return list(set(triplets))  # Return unique triplets

        train_pairs = create_triplets_from_decorte(dataset['train'])
        val_pairs = create_triplets_from_decorte(dataset['validation'])
        test_pairs = create_triplets_from_decorte(dataset['test'])

        return train_pairs, val_pairs, test_pairs

    elif dataset_name == 'karrierewege_plus':
        dataset = load_dataset("ElenaSenger/Karrierewege_plus", split={s: s + split_slice for s in ["train", "validation", "test"]})
        
        esco_occupations = pd.read_csv(DATA_PATH / "occupations_en.csv")
        esco_label_to_uri = esco_occupations.set_index('preferredLabel')['conceptUri'].to_dict()

        def create_triplets_from_kw(_dataset):
            df = _dataset.to_pandas()
            all_triplets = []

            if kw_source in ['occ', 'all']:
                pairs_occ = df[['new_job_title_en_occ', 'preferredLabel_en']].dropna()
                for _, row in pairs_occ.iterrows():
                    esco_title = row['preferredLabel_en'].strip()
                    concept_uri = esco_label_to_uri.get(row['preferredLabel_en'])
                    if concept_uri:
                        all_triplets.append((row['new_job_title_en_occ'].strip(), esco_title, concept_uri))


            if kw_source in ['cp', 'all']:
                pairs_cp = df[['new_job_title_en_cp', 'preferredLabel_en']].dropna()
                for _, row in pairs_cp.iterrows():
                    esco_title = row['preferredLabel_en'].strip()
                    concept_uri = esco_label_to_uri.get(row['preferredLabel_en'])
                    if concept_uri:
                        all_triplets.append((row['new_job_title_en_cp'].strip(), esco_title, concept_uri))
            
            if kw_source not in ['occ', 'cp', 'all']:
                raise ValueError("For 'karrierewege_plus', kw_source must be 'occ', 'cp', or 'all'.")

            return list(set(all_triplets))  # Return unique triplets

        train_pairs = create_triplets_from_kw(dataset['train'])
        val_pairs = create_triplets_from_kw(dataset['validation'])
        test_pairs = create_triplets_from_kw(dataset['test'])

        return train_pairs, val_pairs, test_pairs

    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}. Supported are 'decorte', 'karrierewege_plus'.")


def load_esco_titles(path: str, lowercase: bool = False) -> tuple[list[str], list[str]]:
    """Loads ESCO titles and their IDs from a CSV file."""
    df = pd.read_csv(path)
    # drop rows with missing titles
    df = df.dropna(subset=['preferredLabel', 'conceptUri'])
    ids = df['conceptUri'].tolist()
    titles = df['preferredLabel'].tolist()
    if lowercase:
        ids = [str(i).lower() for i in ids]
        titles = [str(t).lower() for t in titles]
    return ids, titles


def load_pairs(path: str | list[str], lowercase_raw: bool = False, lowercase_esco: bool = False) -> list[dict]:
    """Loads job title pairs from one or more CSV files."""
    if isinstance(path, str):
        df = pd.read_csv(path)
    else:
        df = pd.concat([pd.read_csv(p) for p in path], ignore_index=True)
        
    # drop rows with missing titles
    cols_to_check = ['raw_title', 'esco_id']
    if 'esco_title' in df.columns:
        cols_to_check.append('esco_title')
    df = df.dropna(subset=cols_to_check)

    pairs = []
    for _, row in df.iterrows():
        job_title = str(row['raw_title'])
        esco_id = str(row['esco_id'])

        if lowercase_raw:
            job_title = job_title.lower()

        if 'esco_title' in df.columns:
            esco_title = str(row['esco_title'])
            if lowercase_esco:
                esco_title = esco_title.lower()
                esco_id = esco_id.lower()
            
            pairs.append({
                "job_title": job_title,
                "esco_title": esco_title,
                "esco_id": esco_id,
            })
        else:
            if lowercase_esco:
                esco_id = esco_id.lower()

            pairs.append({
                "job_title": job_title,
                "esco_id": esco_id,
            })
    return pairs


def load_talent_clef_training_data():
    """
    Loads and processes the Talent CLEF Task A training data.

    Returns:
        pandas.DataFrame: A DataFrame with 'id' and 'job_title' columns.
    """
    file_path = DATA_PATH / "talent_clef/TaskA/training/english/taskA_training_en.tsv"
    df = pd.read_csv(file_path, sep='\t', header=None, names=['family_id', 'id', 'jobtitle_1', 'jobtitle_2'])

    df1 = df[['id', 'jobtitle_1']].copy()
    df1.rename(columns={'jobtitle_1': 'job_title'}, inplace=True)

    df2 = df[['id', 'jobtitle_2']].copy()
    df2.rename(columns={'jobtitle_2': 'job_title'}, inplace=True)

    result_df = pd.concat([df1, df2], ignore_index=True)
    return result_df