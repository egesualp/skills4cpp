import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils import (
    load_esco_titles,
    load_pairs,
    replace_esco_titles, 
    subspans, 
    load_prepare_decorte, 
    load_prepare_decorte_esco, 
    load_prepare_karrierewege, 
    load_raw_to_esco_pairs, 
    SEP_TOKEN
)


class TestNewUtils:
    """Test suite for new utility functions in utils.py."""

    @pytest.fixture
    def mock_esco_csv(self, tmp_path):
        """Create a mock ESCO CSV file."""
        csv_content = (
            "conceptUri,preferredLabel\n"
            "uri1,title1\n"
            "uri2,title2\n"
            "uri3,title3"
        )
        csv_file = tmp_path / "esco.csv"
        csv_file.write_text(csv_content)
        return str(csv_file)

    @pytest.fixture
    def mock_pairs_csv(self, tmp_path):
        """Create a mock pairs CSV file."""
        csv_content = (
            "raw_title,esco_id\n"
            "job1,uri1\n"
            "job2,uri2\n"
            "job3,uri3"
        )
        csv_file = tmp_path / "pairs.csv"
        csv_file.write_text(csv_content)
        return str(csv_file)

    def test_load_esco_titles(self, mock_esco_csv):
        """Test the load_esco_titles function."""
        ids, titles = load_esco_titles(mock_esco_csv)
        assert ids == ["uri1", "uri2", "uri3"]
        assert titles == ["title1", "title2", "title3"]

    def test_load_pairs(self, mock_pairs_csv):
        """Test the load_pairs function."""
        pairs = load_pairs(mock_pairs_csv)
        expected = [
            {"job_title": "job1", "esco_id": "uri1"},
            {"job_title": "job2", "esco_id": "uri2"},
            {"job_title": "job3", "esco_id": "uri3"},
        ]
        assert pairs == expected


class TestUtils:
    """Test suite for utility functions in utils.py."""

    def test_replace_esco_titles(self):
        """Test the replace_esco_titles function."""
        # Test a title that should be replaced
        example1 = {'ESCO_title_0': 'ICT security engineer', 'ESCO_uri_0': 'some_uri'}
        result1 = replace_esco_titles(example1, 0)
        assert result1['ESCO_title_0'] == 'cyber incident responder'

        # Test a title that should not be replaced
        example2 = {'ESCO_title_0': 'Software Developer', 'ESCO_uri_0': 'some_uri'}
        result2 = replace_esco_titles(example2, 0)
        assert result2['ESCO_title_0'] == 'software developer' # it lowercases it

        # Test a URI that should be replaced
        example3 = {'ESCO_title_0': 'Some Title', 'ESCO_uri_0': 'http://data.europa.eu/esco/occupation/81309031-dad2-4a7a-bde6-7f6e518f89ff'}
        result3 = replace_esco_titles(example3, 0)
        assert result3['ESCO_uri_0'] == 'http://data.europa.eu/esco/occupation/f4525ed8-54eb-4a3b-90db-55cc01b0d9fd'

        # Test with NaN title
        example4 = {'ESCO_title_0': pd.NA, 'ESCO_uri_0': 'some_uri'}
        result4 = replace_esco_titles(example4, 0)
        assert pd.isna(result4['ESCO_title_0'])

    def test_subspans(self):
        """Test the subspans generator function."""
        lst = [1, 2, 3, 4]
        result = list(subspans(lst))
        expected = [
            [1, 2], [2, 3], [3, 4],  # length 2
            [1, 2, 3], [2, 3, 4],    # length 3
            [1, 2, 3, 4]             # length 4
        ]
        assert result == expected

        # Test with a list of length 1 (should be empty)
        lst2 = [1]
        result2 = list(subspans(lst2))
        assert result2 == []

    @patch('utils.pd.read_csv')
    @patch('utils.load_dataset')
    def test_load_prepare_decorte(self, mock_load_dataset, mock_read_csv):
        """Test the load_prepare_decorte function with mocked data."""
        # Mock dataset
        mock_data = {
            'number_of_experiences': [2],
            'title_0': ['Software Intern'], 'description_0': ['Wrote code.'],
            'ESCO_uri_0': ['uri1'], 'ESCO_title_0': ['Software Developer'],
            'title_1': ['Software Engineer'], 'description_1': ['Developed software.'],
            'ESCO_uri_1': ['uri2'], 'ESCO_title_1': ['Senior Software Developer'],
        }
        
        # Mock dataset object that can be iterated and mapped
        mock_dataset_obj = MagicMock()
        mock_dataset_obj.__iter__.return_value = pd.DataFrame(mock_data).to_dict('records')
        mock_dataset_obj.map.return_value = mock_dataset_obj

        mock_dataset = {
            'train': mock_dataset_obj,
            'validation': mock_dataset_obj,
            'test': mock_dataset_obj,
        }
        mock_load_dataset.return_value = mock_dataset

        # Mock ESCO occupations DataFrame
        mock_esco_df = pd.DataFrame({
            'conceptUri': ['uri1', 'uri2'],
            'description': ['Description for URI1', 'Description for URI2'],
            'preferredLabel': ['Software Developer', 'Senior Software Developer'],
            'altLabels': [None, None]
        })
        mock_read_csv.return_value = mock_esco_df

        train_pairs, val_pairs, test_pairs = load_prepare_decorte(minus_last=False, consider_all_subspans_of_len_at_least_2=True)
        
        expected_doc1 = (
            f"role: Software Intern \n description: Wrote code.{SEP_TOKEN}"
            f"role: Software Engineer \n description: Developed software."
        )
        expected_doc2 = "esco role: Senior Software Developer \n description: Description for URI2"
        
        assert len(train_pairs) == 1
        assert train_pairs[0][0] == expected_doc1
        assert train_pairs[0][1] == expected_doc2
        assert val_pairs == train_pairs
        assert test_pairs == train_pairs


    @patch('utils.pd.read_csv')
    @patch('utils.load_dataset')
    def test_load_prepare_decorte_esco(self, mock_load_dataset, mock_read_csv):
        """Test the load_prepare_decorte_esco function with mocked data."""
        mock_data = {
            'number_of_experiences': [2],
            'title_0': ['Intern'], 'description_0': ['Code.'],
            'ESCO_uri_0': ['uri1'], 'ESCO_title_0': ['Software Developer'],
            'title_1': ['Engineer'], 'description_1': ['Software.'],
            'ESCO_uri_1': ['uri2'], 'ESCO_title_1': ['Senior Developer'],
        }
        mock_dataset_obj = MagicMock()
        mock_dataset_obj.__iter__.return_value = pd.DataFrame(mock_data).to_dict('records')
        mock_dataset_obj.map.return_value = mock_dataset_obj

        mock_dataset = {
            'train': mock_dataset_obj,
            'validation': mock_dataset_obj,
            'test': mock_dataset_obj,
        }
        mock_load_dataset.return_value = mock_dataset

        mock_esco_df = pd.DataFrame({
            'conceptUri': ['uri1', 'uri2'],
            'description': ['Desc1', 'Desc2'],
            'preferredLabel': ['Software Developer', 'Senior Developer'],
            'altLabels': [None, None]
        })
        mock_read_csv.return_value = mock_esco_df

        train_pairs, val_pairs, test_pairs = load_prepare_decorte_esco(minus_last=False, consider_all_subspans_of_len_at_least_2=True)
        
        expected_doc1 = (
            f"esco role: Software Developer \n description: Desc1{SEP_TOKEN}"
            f"esco role: Senior Developer \n description: Desc2"
        )
        expected_doc2 = "esco role: Senior Developer \n description: Desc2"

        assert len(train_pairs) == 1
        assert train_pairs[0][0] == expected_doc1
        assert train_pairs[0][1] == expected_doc2
        assert val_pairs == train_pairs
        assert test_pairs == train_pairs

    @patch('utils.load_dataset')
    def test_load_prepare_karrierewege(self, mock_load_dataset):
        """Test the load_prepare_karrierewege function with mocked data."""
        # Mock dataset convertible to pandas DataFrame
        mock_data = {
            '_id': ['cv1', 'cv1'],
            'experience_order': [0, 1],
            'preferredLabel_en': ['Junior Developer', 'Senior Developer'],
            'description_en': ['Coding', 'Leading projects']
        }
        mock_df = pd.DataFrame(mock_data)
        
        # Mock dataset object that has a to_pandas() method
        mock_dataset_obj = MagicMock()
        mock_dataset_obj.to_pandas.return_value = mock_df
        
        mock_dataset = {
            'train': mock_dataset_obj,
            'validation': mock_dataset_obj,
            'test': mock_dataset_obj,
        }
        mock_load_dataset.return_value = mock_dataset

        train_pairs, val_pairs, test_pairs = load_prepare_karrierewege(minus_last=False, consider_all_subspans_of_len_at_least_2=True, language='en')

        expected_doc1 = (
            f"role: Junior Developer \n description: Coding{SEP_TOKEN}"
            f"role: Senior Developer \n description: Leading projects"
        )
        expected_doc2 = "esco role: Senior Developer \n description: Leading projects"

        assert len(train_pairs) == 1
        assert train_pairs[0][0] == expected_doc1
        assert train_pairs[0][1] == expected_doc2
        assert val_pairs == train_pairs
        assert test_pairs == train_pairs

    @patch('utils.load_dataset')
    def test_load_raw_to_esco_pairs_decorte(self, mock_load_dataset):
        """Test load_raw_to_esco_pairs with the 'decorte' dataset."""
        mock_data = [
            {
                'number_of_experiences': 2,
                'title_0': 'Software Intern', 'ESCO_title_0': 'Software Developer', 'ESCO_uri_0': 'uri1',
                'title_1': 'Developer', 'ESCO_title_1': 'Software Developer', 'ESCO_uri_1': 'uri1',
            },
            {
                'number_of_experiences': 1,
                'title_0': 'Data Analyst', 'ESCO_title_0': 'Data Scientist', 'ESCO_uri_0': 'uri2',
            }
        ]
        mock_dataset_obj = MagicMock()
        mock_dataset_obj.__iter__.return_value = mock_data
        
        mock_dataset = {
            'train': mock_dataset_obj,
            'validation': mock_dataset_obj,
            'test': mock_dataset_obj,
        }
        mock_load_dataset.return_value = mock_dataset

        train_pairs, val_pairs, test_pairs = load_raw_to_esco_pairs('decorte')

        expected_pairs = {
            ('Software Intern', 'Software Developer'),
            ('Developer', 'Software Developer'),
            ('Data Analyst', 'Data Scientist')
        }
        assert set(train_pairs) == expected_pairs
        assert set(val_pairs) == expected_pairs
        assert set(test_pairs) == expected_pairs

    @patch('utils.load_dataset')
    def test_load_raw_to_esco_pairs_karrierewege(self, mock_load_dataset):
        """Test load_raw_to_esco_pairs with the 'karrierewege_plus' dataset."""
        mock_data = {
            'new_job_title_en_occ': ['Raw Title 1', 'Raw Title 2'],
            'preferredLabel_en': ['esco_1', 'esco_2'],
            'new_job_title_en_cp': ['Raw Title 3', 'Raw Title 1'],
        }
        mock_df = pd.DataFrame(mock_data)
        # Manually create the pairs as the function does to ensure correct test data
        pairs_occ = mock_df[['new_job_title_en_occ', 'preferredLabel_en']].dropna().values.tolist()
        pairs_cp = mock_df[['new_job_title_en_cp', 'preferredLabel_en']].dropna().values.tolist()
        all_pairs = set(map(tuple, pairs_occ + pairs_cp))


        mock_dataset_obj = MagicMock()
        mock_dataset_obj.to_pandas.return_value = mock_df
        
        mock_dataset = {
            'train': mock_dataset_obj,
            'validation': mock_dataset_obj,
            'test': mock_dataset_obj,
        }
        mock_load_dataset.return_value = mock_dataset
        
        train_pairs, val_pairs, test_pairs = load_raw_to_esco_pairs('karrierewege_plus')

        # The function returns a list, convert to set for comparison
        assert set(train_pairs) == all_pairs
        assert set(val_pairs) == all_pairs
        assert set(test_pairs) == all_pairs

    def test_load_raw_to_esco_pairs_unsupported(self):
        """Test load_raw_to_esco_pairs with an unsupported dataset name."""
        with pytest.raises(ValueError):
            load_raw_to_esco_pairs('unsupported_dataset')


class TestLoadRawToEscoPairsKarrierewege:
    """Test suite for load_raw_to_esco_pairs with the 'karrierewege_plus' dataset."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.mock_data = {
            'new_job_title_en_occ': ['occ_title_1', 'occ_title_2', 'shared_title', None, None, None],
            'new_job_title_en_cp': [None, None, None, 'cp_title_1', 'cp_title_2', 'shared_title'],
            'preferredLabel_en': ['esco_occ_1', 'esco_occ_2', 'esco_shared', 'esco_cp_1', 'esco_cp_2', 'esco_shared_2']
        }
        self.mock_df = pd.DataFrame(self.mock_data)
        
        # Mock dataset object that has a to_pandas() method
        mock_dataset_obj = MagicMock()
        mock_dataset_obj.to_pandas.return_value = self.mock_df
        
        self.mock_dataset = {
            'train': mock_dataset_obj,
            'validation': mock_dataset_obj,
            'test': mock_dataset_obj,
        }

        with patch('utils.load_dataset') as mock_load_dataset:
            mock_load_dataset.return_value = self.mock_dataset
            yield

    def test_kw_source_all(self):
        """Test with kw_source='all'."""
        train_pairs, _, _ = load_raw_to_esco_pairs('karrierewege_plus', kw_source='all')
        
        # Manually create expected pairs from mock_df
        pairs_occ = self.mock_df[['new_job_title_en_occ', 'preferredLabel_en']].dropna()
        pairs_cp = self.mock_df[['new_job_title_en_cp', 'preferredLabel_en']].dropna()
        
        expected_pairs = set()
        for _, row in pairs_occ.iterrows():
            expected_pairs.add((row['new_job_title_en_occ'], row['preferredLabel_en']))
        for _, row in pairs_cp.iterrows():
            expected_pairs.add((row['new_job_title_en_cp'], row['preferredLabel_en']))
            
        assert set(train_pairs) == expected_pairs

    def test_kw_source_occ(self):
        """Test with kw_source='occ'."""
        train_pairs, _, _ = load_raw_to_esco_pairs('karrierewege_plus', kw_source='occ')
        
        pairs_occ = self.mock_df[['new_job_title_en_occ', 'preferredLabel_en']].dropna()
        expected_pairs = set()
        for _, row in pairs_occ.iterrows():
            expected_pairs.add((row['new_job_title_en_occ'], row['preferredLabel_en']))

        assert set(train_pairs) == expected_pairs

    def test_kw_source_cp(self):
        """Test with kw_source='cp'."""
        train_pairs, _, _ = load_raw_to_esco_pairs('karrierewege_plus', kw_source='cp')
        
        pairs_cp = self.mock_df[['new_job_title_en_cp', 'preferredLabel_en']].dropna()
        expected_pairs = set()
        for _, row in pairs_cp.iterrows():
            expected_pairs.add((row['new_job_title_en_cp'], row['preferredLabel_en']))

        assert set(train_pairs) == expected_pairs

    def test_kw_source_invalid(self):
        """Test with an invalid kw_source."""
        with pytest.raises(ValueError, match="For 'karrierewege_plus', kw_source must be 'occ', 'cp', or 'all'."):
            load_raw_to_esco_pairs('karrierewege_plus', kw_source='invalid_source')
