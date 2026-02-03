from cpp.utils import load_prepare_decorte, load_prepare_karrierewege, load_prepare_decorte_esco
import cpp.utils as utils
import re
from typing import List, Tuple, Optional

class Data:
    """
    A class to load and process data for various datasets.

    This class supports multiple data sources and formats, including `decorte`, `decorte_esco`, 
    and `karrierewege`, and provides methods for extracting specific titles, filtering data, 
    and retrieving dataset splits for training, validation, and testing.

    Attributes:
        DATA_TYPE (str): The type of dataset to be loaded.
        DOC_1_PROMPT (str, optional): An optional prompt for document 1.
        DOC_2_PROMPT (str, optional): An optional prompt for document 2.
        ONLY_TITLES (bool): Flag to indicate whether only titles should be extracted.
        train_pairs (list): Training data pairs.
        val_pairs (list): Validation data pairs.
        test_pairs (list): Test data pairs.
        train_job_ids (list): Job IDs for each doc_1 in train_pairs.
        val_job_ids (list): Job IDs for each doc_1 in val_pairs.
        test_job_ids (list): Job IDs for each doc_1 in test_pairs.
        labels (list): Unique labels in the dataset.
    """

    def __init__(self, DATA_TYPE, DOC_1_PROMPT=None, DOC_2_PROMPT=None, ONLY_TITLES=False, LOAD_CLEAN_TEST=None, consider_subspans=True):
        """
        Initializes the Data class by loading the appropriate dataset based on the specified type.

        Args:
            DATA_TYPE (str): The dataset type to load.
            DOC_1_PROMPT (str, optional): Prompt for document 1 (default: None).
            DOC_2_PROMPT (str, optional): Prompt for document 2 (default: None).
            ONLY_TITLES (bool): If True, extracts only job titles (default: False).
            consider_subspans (bool): If True, consider all subspans of length at least 2 (default: True).
        """
        self.DATA_TYPE = DATA_TYPE
        self.DOC_1_PROMPT = DOC_1_PROMPT
        self.DOC_2_PROMPT = DOC_2_PROMPT
        self.ONLY_TITLES = ONLY_TITLES
        self.train_pairs = None
        self.val_pairs = None
        self.test_pairs = None
        self.test_pairs_clean = None
        self.train_job_ids = None
        self.val_job_ids = None
        self.test_job_ids = None
        self.test_job_ids_clean = None
        self.labels = None
        self.LOAD_CLEAN_TEST = LOAD_CLEAN_TEST
        self.consider_subspans = consider_subspans
        self.__load_data()

    def __load_data(self):
        """
        Loads data based on the specified `DATA_TYPE`.

        Depending on the dataset type, this method calls the appropriate `load_prepare_*` function 
        to load and preprocess the dataset. It also extracts unique labels from the dataset.
        
        The load functions now return:
        (train_pairs, train_job_ids, val_pairs, val_job_ids, test_pairs, test_job_ids)
        """
        if self.DATA_TYPE == 'decorte':
            (self.train_pairs, self.train_job_ids, 
             self.val_pairs, self.val_job_ids, 
             self.test_pairs, self.test_job_ids) = load_prepare_decorte(
                consider_all_subspans_of_len_at_least_2=self.consider_subspans, minus_last=False, 
            )
            if self.LOAD_CLEAN_TEST:
                (_, _, _, _, 
                 self.test_pairs_clean, self.test_job_ids_clean) = load_prepare_decorte(
                    consider_all_subspans_of_len_at_least_2=False, minus_last=False
                )
        elif self.DATA_TYPE == 'decorte_esco':
            (self.train_pairs, self.train_job_ids, 
             self.val_pairs, self.val_job_ids, 
             self.test_pairs, self.test_job_ids) = load_prepare_decorte_esco(
                consider_all_subspans_of_len_at_least_2=self.consider_subspans, minus_last=False, 
            )
            if self.LOAD_CLEAN_TEST:
                (_, _, _, _, 
                 self.test_pairs_clean, self.test_job_ids_clean) = load_prepare_decorte_esco(
                    consider_all_subspans_of_len_at_least_2=False, minus_last=False
                )
        elif self.DATA_TYPE == 'karrierewege':
            (self.train_pairs, self.train_job_ids, 
             self.val_pairs, self.val_job_ids, 
             self.test_pairs, self.test_job_ids) = load_prepare_karrierewege(
                consider_all_subspans_of_len_at_least_2=self.consider_subspans, minus_last=False, language='en'
            )
            if self.LOAD_CLEAN_TEST:
                (_, _, _, _, 
                 self.test_pairs_clean, self.test_job_ids_clean) = load_prepare_karrierewege(
                    consider_all_subspans_of_len_at_least_2=False, minus_last=False, language='en'
                )
        elif self.DATA_TYPE == 'karrierewege_occ':
            (self.train_pairs, self.train_job_ids, 
             self.val_pairs, self.val_job_ids, 
             self.test_pairs, self.test_job_ids) = load_prepare_karrierewege(
                consider_all_subspans_of_len_at_least_2=self.consider_subspans, minus_last=False, language='en_free'
            )
            if self.LOAD_CLEAN_TEST:
                (_, _, _, _, 
                 self.test_pairs_clean, self.test_job_ids_clean) = load_prepare_karrierewege(
                    consider_all_subspans_of_len_at_least_2=False, minus_last=False, language='en_free'
                )
        elif self.DATA_TYPE == 'karrierewege_100k':
            (self.train_pairs, self.train_job_ids, 
             self.val_pairs, self.val_job_ids, 
             self.test_pairs, self.test_job_ids) = load_prepare_karrierewege(
                consider_all_subspans_of_len_at_least_2=self.consider_subspans, minus_last=False, language='esco_100k'
            )
            if self.LOAD_CLEAN_TEST:
                (_, _, _, _, 
                 self.test_pairs_clean, self.test_job_ids_clean) = load_prepare_karrierewege(
                    consider_all_subspans_of_len_at_least_2=False, minus_last=False, language='esco_100k'
                )
        elif self.DATA_TYPE == 'karrierewege_cp':
            (self.train_pairs, self.train_job_ids, 
             self.val_pairs, self.val_job_ids, 
             self.test_pairs, self.test_job_ids) = load_prepare_karrierewege(
                consider_all_subspans_of_len_at_least_2=self.consider_subspans, minus_last=False, language='en_free_cp'
            )
            if self.LOAD_CLEAN_TEST:
                (_, _, _, _, 
                 self.test_pairs_clean, self.test_job_ids_clean) = load_prepare_karrierewege(
                    consider_all_subspans_of_len_at_least_2=False, minus_last=False, language='en_free_cp'
                )

        # Extract unique labels from the dataset
        self.labels = list(set([pair[1] for pair in self.train_pairs + self.val_pairs + self.test_pairs]))

    @staticmethod
    def __minus_last(data_pairs):
        """
        Removes the last segment of document 1 in each data pair.

        This method splits `doc1` by the separator token and removes the last part to create a modified dataset.

        Args:
            data_pairs (list of tuples): List of (doc1, doc2) pairs.

        Returns:
            list of tuples: Modified list where the last segment of `doc1` has been removed.
        """
        new_data_pairs = []
        for doc1, doc2 in data_pairs:
            segments = doc1.split(utils.SEP_TOKEN)
            if len(segments) > 1:  # Only modify if there are multiple segments
                new_doc1 = utils.SEP_TOKEN.join(segments[:-1])
                new_data_pairs.append((new_doc1, doc2))
        return new_data_pairs
    
    @staticmethod
    def _extract_titles(list_of_tuples):
        """
        Extracts job titles from document pairs.

        This method searches for job roles in `doc1` and `doc2` using regex patterns 
        and returns a new list of extracted job title pairs.

        Args:
            list_of_tuples (list of tuples): List of (doc1, doc2) pairs.

        Returns:
            list of tuples: List of (titles from doc1, title from doc2).
        """
        sequences = [re.findall(r"role: (.*?)\n", element[0]) for element in list_of_tuples]
        sequences = [utils.SEP_TOKEN.join(element) for element in sequences]  # Join role lists into strings
        
        targets = [re.findall(r"esco role: (.*?)\n", element[1]) for element in list_of_tuples]
        targets = [element[0] for element in targets]  # Convert list of lists to a flat list
        
        return list(zip(sequences, targets))  # Return as pairs
    
    def get_data(self, stage, include_clean_test=False):
        """
        Retrieves dataset splits based on the given stage.

        This method returns data in different formats depending on the stage:
        - `embedding_finetuning`: Returns full pairs or only titles based on `ONLY_TITLES`.
        - `transformation_finetuning` or `evaluation`: Applies `__minus_last` filtering.

        Args:
            stage (str): The stage of training or evaluation.
            include_clean_test (bool): If True, returns a 4th element: clean_test_data.

        Returns:
            tuple: (train_data, val_data, test_data, [test_data_clean]) depending on the selected stage.

        Raises:
            ValueError: If the stage is invalid.
        """
        # Define the processing logic based on stage
        if stage == 'embedding_finetuning':
            if self.ONLY_TITLES:
                process_fn = self._extract_titles
            else:
                process_fn = lambda x: x
                
        elif stage in ['transformation_finetuning', 'evaluation']:
            if self.ONLY_TITLES:
                process_fn = lambda x: self.__minus_last(self._extract_titles(x))
            else:
                process_fn = self.__minus_last
        else:
            raise ValueError(f"Invalid stage: {stage}")

        # Process standard splits
        train = process_fn(self.train_pairs)
        val = process_fn(self.val_pairs)
        test = process_fn(self.test_pairs)

        # Handle clean test set
        if include_clean_test:
            if self.test_pairs_clean is None:
                # If explicitly requested but not loaded, we can either error or warn. 
                # Error is safer to avoid silent failures in experiments.
                raise ValueError("Clean test pairs not loaded. Initialize Data with LOAD_CLEAN_TEST=True")
            
            test_clean = process_fn(self.test_pairs_clean)
            return train, val, test, test_clean
            
        return train, val, test
    
    @staticmethod
    def __minus_last_with_job_ids(data_pairs: List[Tuple[str, str]], 
                                   job_ids_list: List[List[str]]) -> Tuple[List[Tuple[str, str]], List[List[str]]]:
        """
        Removes the last segment of document 1 in each data pair, and corresponding job_id.

        Args:
            data_pairs: List of (doc1, doc2) pairs.
            job_ids_list: List of job_id lists, one per data pair.

        Returns:
            Tuple of (new_data_pairs, new_job_ids_list) with last segment removed.
        """
        new_data_pairs = []
        new_job_ids = []
        for (doc1, doc2), job_ids in zip(data_pairs, job_ids_list):
            segments = doc1.split(utils.SEP_TOKEN)
            if len(segments) > 1:  # Only modify if there are multiple segments
                new_doc1 = utils.SEP_TOKEN.join(segments[:-1])
                new_data_pairs.append((new_doc1, doc2))
                # Remove last job_id as well (if exists)
                new_job_ids.append(job_ids[:-1] if job_ids else [])
        return new_data_pairs, new_job_ids
    
    @staticmethod
    def _extract_titles_with_job_ids(list_of_tuples: List[Tuple[str, str]], 
                                      job_ids_list: List[List[str]]) -> Tuple[List[Tuple[str, str]], List[List[str]]]:
        """
        Extracts job titles from document pairs while keeping job_ids aligned.

        Args:
            list_of_tuples: List of (doc1, doc2) pairs.
            job_ids_list: List of job_id lists, one per pair.

        Returns:
            Tuple of (extracted_pairs, job_ids_list) where extracted_pairs has titles only.
        """
        sequences = [re.findall(r"role: (.*?)\n", element[0]) for element in list_of_tuples]
        sequences = [utils.SEP_TOKEN.join(element) for element in sequences]
        
        targets = [re.findall(r"esco role: (.*?)\n", element[1]) for element in list_of_tuples]
        targets = [element[0] if element else "" for element in targets]
        
        extracted_pairs = list(zip(sequences, targets))
        # job_ids remain unchanged - they're still aligned with the samples
        return extracted_pairs, job_ids_list
    
    def get_data_with_job_ids(self, stage: str, include_clean_test: bool = False):
        """
        Retrieves dataset splits with corresponding job_ids based on the given stage.

        This method is similar to get_data but also returns job_ids for skill mapping.
        Supports both full documents and ONLY_TITLES mode for ablation studies.
        
        Args:
            stage (str): The stage of training or evaluation.
                - 'embedding_finetuning': Returns full pairs and job_ids.
                - 'transformation_finetuning' or 'evaluation': Applies minus_last filtering.
            include_clean_test (bool): If True, returns clean_test_data and job_ids.

        Returns:
            If include_clean_test is False:
                Tuple of ((train_pairs, train_job_ids), (val_pairs, val_job_ids), (test_pairs, test_job_ids))
            If include_clean_test is True:
                Tuple of ((train_pairs, train_job_ids), (val_pairs, val_job_ids), 
                          (test_pairs, test_job_ids), (test_clean_pairs, test_clean_job_ids))

        Raises:
            ValueError: If the stage is invalid.
        """
        if stage == 'embedding_finetuning':
            # Start with raw pairs and job_ids
            train_pairs, train_job_ids = self.train_pairs, self.train_job_ids
            val_pairs, val_job_ids = self.val_pairs, self.val_job_ids
            test_pairs, test_job_ids = self.test_pairs, self.test_job_ids
            
            # Apply title extraction if ONLY_TITLES is True
            if self.ONLY_TITLES:
                train_pairs, train_job_ids = self._extract_titles_with_job_ids(train_pairs, train_job_ids)
                val_pairs, val_job_ids = self._extract_titles_with_job_ids(val_pairs, val_job_ids)
                test_pairs, test_job_ids = self._extract_titles_with_job_ids(test_pairs, test_job_ids)
            
            train = (train_pairs, train_job_ids)
            val = (val_pairs, val_job_ids)
            test = (test_pairs, test_job_ids)
            
        elif stage in ['transformation_finetuning', 'evaluation']:
            if self.ONLY_TITLES:
                # Extract titles first, then apply minus_last
                train_pairs, train_job_ids = self._extract_titles_with_job_ids(self.train_pairs, self.train_job_ids)
                val_pairs, val_job_ids = self._extract_titles_with_job_ids(self.val_pairs, self.val_job_ids)
                test_pairs, test_job_ids = self._extract_titles_with_job_ids(self.test_pairs, self.test_job_ids)
                
                train_pairs, train_job_ids = self.__minus_last_with_job_ids(train_pairs, train_job_ids)
                val_pairs, val_job_ids = self.__minus_last_with_job_ids(val_pairs, val_job_ids)
                test_pairs, test_job_ids = self.__minus_last_with_job_ids(test_pairs, test_job_ids)
            else:
                # Apply minus_last to both pairs and job_ids
                train_pairs, train_job_ids = self.__minus_last_with_job_ids(
                    self.train_pairs, self.train_job_ids)
                val_pairs, val_job_ids = self.__minus_last_with_job_ids(
                    self.val_pairs, self.val_job_ids)
                test_pairs, test_job_ids = self.__minus_last_with_job_ids(
                    self.test_pairs, self.test_job_ids)
            
            train = (train_pairs, train_job_ids)
            val = (val_pairs, val_job_ids)
            test = (test_pairs, test_job_ids)
        else:
            raise ValueError(f"Invalid stage: {stage}")

        # Handle clean test set
        if include_clean_test:
            if self.test_pairs_clean is None:
                raise ValueError("Clean test pairs not loaded. Initialize Data with LOAD_CLEAN_TEST=True")
            
            if stage == 'embedding_finetuning':
                if self.ONLY_TITLES:
                    test_clean_pairs, test_clean_job_ids = self._extract_titles_with_job_ids(
                        self.test_pairs_clean, self.test_job_ids_clean)
                    test_clean = (test_clean_pairs, test_clean_job_ids)
                else:
                    test_clean = (self.test_pairs_clean, self.test_job_ids_clean)
            else:
                if self.ONLY_TITLES:
                    test_clean_pairs, test_clean_job_ids = self._extract_titles_with_job_ids(
                        self.test_pairs_clean, self.test_job_ids_clean)
                    test_clean_pairs, test_clean_job_ids = self.__minus_last_with_job_ids(
                        test_clean_pairs, test_clean_job_ids)
                else:
                    test_clean_pairs, test_clean_job_ids = self.__minus_last_with_job_ids(
                        self.test_pairs_clean, self.test_job_ids_clean)
                test_clean = (test_clean_pairs, test_clean_job_ids)
            
            return train, val, test, test_clean
            
        return train, val, test
