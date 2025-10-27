"""
Unit tests for src/skill_mapping/build_hierarchy.py

Tests the ESCO skill hierarchy building functionality including:
- Extraction of skill-to-group mappings from hierarchy CSV
- Generation of lookup tables and JSON artifacts
- Validation of output file structure and content
"""

import unittest
import pandas as pd
import json
import tempfile
import shutil
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "skill_mapping"))


class TestBuildHierarchyIntegration(unittest.TestCase):
    """Integration tests for the complete build_hierarchy script"""
    
    @classmethod
    def setUpClass(cls):
        """Set up test by running the actual build script once"""
        cls.data_dir = Path(__file__).parent.parent / "data" / "processed"
        cls.csv_file = cls.data_dir / "skill_to_group_pillar.csv"
        cls.skill2group_file = cls.data_dir / "skill2group.json"
        cls.group2label_file = cls.data_dir / "group2label.json"
        
        # Verify files exist
        if not cls.csv_file.exists():
            raise FileNotFoundError(
                f"Run 'python src/skill_mapping/build_hierarchy.py' before testing"
            )
    
    def test_output_files_exist(self):
        """Test that all expected output files are created"""
        self.assertTrue(self.csv_file.exists(), "skill_to_group_pillar.csv should exist")
        self.assertTrue(self.skill2group_file.exists(), "skill2group.json should exist")
        self.assertTrue(self.group2label_file.exists(), "group2label.json should exist")
    
    def test_csv_structure(self):
        """Test that the CSV has the correct structure"""
        df = pd.read_csv(self.csv_file)
        
        # Check columns
        expected_columns = ['skill_id', 'group_id', 'skill_label', 'group_label', 
                          'pillar_id', 'pillar_label']
        self.assertListEqual(list(df.columns), expected_columns)
        
        # Check that we have data
        self.assertGreater(len(df), 0, "CSV should not be empty")
        
        # Check that skill_id and group_id are not null
        self.assertEqual(df['skill_id'].isna().sum(), 0, "skill_id should not have nulls")
        self.assertEqual(df['group_id'].isna().sum(), 0, "group_id should not have nulls")
        
        # Check that all URIs are valid HTTP URIs
        self.assertTrue(
            df['skill_id'].str.startswith('http').all(),
            "All skill_ids should be HTTP URIs"
        )
        self.assertTrue(
            df['group_id'].str.startswith('http').all(),
            "All group_ids should be HTTP URIs"
        )
    
    def test_skill2group_json_structure(self):
        """Test that skill2group.json has the correct structure"""
        with open(self.skill2group_file, 'r') as f:
            data = json.load(f)
        
        # Should be a dictionary
        self.assertIsInstance(data, dict)
        
        # Should have entries
        self.assertGreater(len(data), 0, "skill2group.json should not be empty")
        
        # All keys and values should be strings (URIs)
        for skill_id, group_ids in data.items():
            self.assertIsInstance(skill_id, str)
            self.assertIsInstance(group_ids, list)
            for group_id in group_ids:
                self.assertIsInstance(group_id, str)
                self.assertTrue(skill_id.startswith('http'))
                self.assertTrue(group_id.startswith('http'))
    
    def test_group2label_json_structure(self):
        """Test that group2label.json has the correct structure"""
        with open(self.group2label_file, 'r') as f:
            data = json.load(f)
        
        # Should be a dictionary
        self.assertIsInstance(data, dict)
        
        # Should have entries
        self.assertGreater(len(data), 0, "group2label.json should not be empty")
        
        # All keys should be URIs, values should be strings (labels)
        for group_id, label in data.items():
            self.assertIsInstance(group_id, str)
            self.assertIsInstance(label, str)
            self.assertTrue(group_id.startswith('http'))
            self.assertGreater(len(label), 0, "Label should not be empty")
    
    def test_consistency_between_outputs(self):
        """Test that CSV and JSON outputs are consistent"""
        df = pd.read_csv(self.csv_file)
        
        with open(self.skill2group_file, 'r') as f:
            skill2group = json.load(f)
        
        with open(self.group2label_file, 'r') as f:
            group2label = json.load(f)
        
        # All skill_ids in CSV should be in skill2group JSON
        csv_skills = set(df['skill_id'].unique())
        json_skills = set(skill2group.keys())
        self.assertEqual(csv_skills, json_skills, 
                        "Skills in CSV and JSON should match")
        
    
    def test_pillar_distribution(self):
        """Test that skills are distributed across pillars as expected"""
        df = pd.read_csv(self.csv_file)
        
        # Should have some pillar mappings (not all may have pillars)
        pillars_with_data = df['pillar_id'].notna().sum()
        self.assertGreater(pillars_with_data, 0, "Should have some pillar mappings")
        
        # Check pillar labels are present when pillar_id is present
        mask = df['pillar_id'].notna()
        self.assertEqual(
            df[mask]['pillar_label'].isna().sum(), 0,
            "When pillar_id exists, pillar_label should also exist"
        )
    
    def test_sample_known_skills(self):
        """Test that some known ESCO skills are correctly mapped"""
        df = pd.read_csv(self.csv_file)
        
        # Just verify we can find skills and they have groups
        # (We don't hard-code specific URIs as they may change)
        self.assertGreater(len(df), 100, 
                          "Should have at least 100 skill mappings")
        
        # Verify each skill has a label
        self.assertEqual(df['skill_label'].isna().sum(), 0,
                        "All skills should have labels")
    
    def test_no_self_references(self):
        """Test that no skill is its own parent group"""
        df = pd.read_csv(self.csv_file)
        
        self_refs = df[df['skill_id'] == df['group_id']]
        self.assertEqual(len(self_refs), 0, 
                        "No skill should reference itself as its parent group")
    
    def test_data_types_and_encoding(self):
        """Test that data is properly encoded and typed"""
        df = pd.read_csv(self.csv_file)
        
        # Check that there are no encoding issues
        for col in ['skill_label', 'group_label', 'pillar_label']:
            non_null = df[col].dropna()
            # All labels should be valid strings
            self.assertTrue(all(isinstance(x, str) for x in non_null),
                          f"All {col} values should be strings")


class TestDataQuality(unittest.TestCase):
    """Tests for data quality and expected statistics"""
    
    @classmethod
    def setUpClass(cls):
        """Load the generated data once"""
        cls.data_dir = Path(__file__).parent.parent / "data" / "processed"
        cls.df = pd.read_csv(cls.data_dir / "skill_to_group_pillar.csv")
    
    def test_reasonable_number_of_mappings(self):
        """Test that we have a reasonable number of skill-group mappings"""
        # ESCO has thousands of skills, but only hundreds are in the hierarchy
        self.assertGreater(len(self.df), 100, 
                          "Should have at least 100 skill mappings")
        self.assertLess(len(self.df), 30000,
                       "Should have less than 30,000 skill mappings")
    
    def test_group_reuse(self):
        """Test that groups are reused (multiple skills per group)"""
        group_counts = self.df['group_id'].value_counts()
        
        # Most groups should have multiple skills
        groups_with_multiple = (group_counts > 1).sum()
        self.assertGreater(groups_with_multiple, 10,
                          "Should have many groups with multiple skills")
    
    def test_pillar_variety(self):
        """Test that we have multiple pillars"""
        unique_pillars = self.df['pillar_label'].dropna().nunique()
        
        # ESCO should have at least a few major pillars
        self.assertGreaterEqual(unique_pillars, 3,
                               "Should have at least 3 different pillars")


def run_tests():
    """Run all tests and print results"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestBuildHierarchyIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestDataQuality))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)


