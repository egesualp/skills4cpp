# elenasenger/karrierewege/src/data_classes.py (modified)
from utils import load_prepare_decorte, load_prepare_karrierewege, load_prepare_decorte_esco
import utils
import re

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
        labels (list): Unique labels in the dataset.
    """

    def __init__(self, DATA_TYPE, DOC_1_PROMPT=None, DOC_2_PROMPT=None, ONLY_TITLES=False, max_rows: int = None):
        """
        Initializes the Data class by loading the appropriate dataset based on the specified type.

        Args:
            DATA_TYPE (str): The dataset type to load.
            DOC_1_PROMPT (str, optional): Prompt for document 1 (default: None).
            DOC_2_PROMPT (str, optional): Prompt for document 2 (default: None).
            ONLY_TITLES (bool): If True, extracts only job titles (default: False).
            max_rows (int, optional): The maximum number of rows to load from each split. Defaults to None (load all).
        """
        self.DATA_TYPE = DATA_TYPE
        self.DOC_1_PROMPT = DOC_1_PROMPT
        self.DOC_2_PROMPT = DOC_2_PROMPT
        self.ONLY_TITLES = ONLY_TITLES
        self.max_rows = max_rows
        self.train_pairs = None
        self.val_pairs = None
        self.test_pairs = None
        self.labels = None
        self.__load_data()

    def __load_data(self):
        """
        Loads data based on the specified `DATA_TYPE`.

        Depending on the dataset type, this method calls the appropriate `load_prepare_*` function 
        to load and preprocess the dataset. It also extracts unique labels from the dataset.
        """
        if self.DATA_TYPE == 'decorte':
            self.train_pairs, self.val_pairs, self.test_pairs = load_prepare_decorte(
                consider_all_subspans_of_len_at_least_2=True, minus_last=False, max_rows=self.max_rows
            )
        elif self.DATA_TYPE == 'decorte_esco':
            self.train_pairs, self.val_pairs, self.test_pairs = load_prepare_decorte_esco(
                consider_all_subspans_of_len_at_least_2=True, minus_last=False, max_rows=self.max_rows
            )
        elif self.DATA_TYPE == 'karrierewege':
            self.train_pairs, self.val_pairs, self.test_pairs = load_prepare_karrierewege(
                consider_all_subspans_of_len_at_least_2=True, minus_last=False, language='en', max_rows=self.max_rows
            )
        elif self.DATA_TYPE == 'karrierewege_occ':
            self.train_pairs, self.val_pairs, self.test_pairs = load_prepare_karrierewege(
                consider_all_subspans_of_len_at_least_2=True, minus_last=False, language='en_free', max_rows=self.max_rows
            )
        elif self.DATA_TYPE == 'karrierewege_100k':
            self.train_pairs, self.val_pairs, self.test_pairs = load_prepare_karrierewege(
                consider_all_subspans_of_len_at_least_2=True, minus_last=False, language='esco_100k', max_rows=self.max_rows
            )
        elif self.DATA_TYPE == 'karrierewege_cp':
            self.train_pairs, self.val_pairs, self.test_pairs = load_prepare_karrierewege(
                consider_all_subspans_of_len_at_least_2=True, minus_last=False, language='en_free_cp', max_rows=self.max_rows
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
        
        targets_list = [re.findall(r"esco role: (.*?)\n", element[1]) for element in list_of_tuples]
        targets = [element[0] if element else "" for element in targets_list]  # Convert list of lists to a flat list
        
        return list(zip(sequences, targets))  # Return as pairs
    
    def get_data(self, stage):
        """
        Retrieves dataset splits based on the given stage.

        This method returns data in different formats depending on the stage:
        - `embedding_finetuning`: Returns full pairs or only titles based on `ONLY_TITLES`.
        - `transformation_finetuning` or `evaluation`: Applies `__minus_last` filtering.

        Args:
            stage (str): The stage of training or evaluation.

        Returns:
            tuple: (train_data, val_data, test_data) depending on the selected stage.

        Raises:
            ValueError: If the stage is invalid.
        """
        if stage == 'embedding_finetuning':
            if self.ONLY_TITLES:
                return self._extract_titles(self.train_pairs), self._extract_titles(self.val_pairs), self._extract_titles(self.test_pairs)
            else:
                return self.train_pairs, self.val_pairs, self.test_pairs
        elif stage in ['transformation_finetuning', 'evaluation']:
            if self.ONLY_TITLES:
                return self.__minus_last(self._extract_titles(self.train_pairs)), self.__minus_last(self._extract_titles(self.val_pairs)), self.__minus_last(self._extract_titles(self.test_pairs))
            else:
                return self.__minus_last(self.train_pairs), self.__minus_last(self.val_pairs), self.__minus_last(self.test_pairs)
        else:
            raise ValueError(f"Invalid stage: {stage}")