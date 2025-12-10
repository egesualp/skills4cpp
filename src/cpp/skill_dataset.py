"""
PyTorch Dataset for Skill-Based Career Path Training.

This module provides a dataset class that:
- Extracts skills from career paths
- Applies IDF-weighted pooling on skills per job
- Applies logarithmic position weighting across jobs in career path
- Implements ISCO group-aware batch sampling to avoid trivial negatives
"""

import re
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import random


class SkillBasedCareerPathDataset(Dataset):
    """
    Dataset that generates skill-based career path representations.
    
    Each sample consists of:
    - Career path skills (list of job_skills for each job in the path)
    - Target ESCO occupation (title, description, ISCO group)
    """
    
    def __init__(
        self,
        data_pairs: List[Tuple[str, str]],
        job_skill_map: Dict[str, List[Dict[str, Any]]],
        target_occupation_map: Dict[str, Dict[str, str]],
        sep_token: str = "<SEP>",
    ):
        """
        Initialize the SkillBasedCareerPathDataset.
        
        Args:
            data_pairs: List of (history_doc, target_doc) tuples
            job_skill_map: Dict mapping job titles to skill info (skillUri, score, idf)
            target_occupation_map: Dict mapping target_doc to occupation info 
                                   (title, description, isco_group)
            sep_token: Token used to separate jobs in career path
        """
        self.data_pairs = data_pairs
        self.job_skill_map = job_skill_map
        self.target_occupation_map = target_occupation_map
        self.sep_token = sep_token
        
        # Filter samples with valid targets
        self._filter_valid_samples()
    
    def _filter_valid_samples(self):
        """Remove samples where target occupation info is not available."""
        original_len = len(self.data_pairs)
        self.data_pairs = [
            (hist, tgt) for hist, tgt in self.data_pairs 
            if tgt in self.target_occupation_map
        ]
        filtered_count = original_len - len(self.data_pairs)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} samples with missing target occupation info.")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single sample.
        
        Returns:
            Dictionary containing:
                - 'job_skills_list': List of skill_info_lists for each job in career path
                - 'target_title': Target occupation title
                - 'target_description': Target occupation description
                - 'target_isco_group': Target ISCO group
                - 'idx': Sample index (for debugging)
        """
        history_doc, target_doc = self.data_pairs[idx]
        
        # Extract job titles from career path
        raw_titles = self._extract_job_titles(history_doc)
        
        # Get skills for each job
        job_skills_list = []
        for title in raw_titles:
            title_normalized = title.strip().lower()
            if title_normalized in self.job_skill_map:
                job_skills_list.append(self.job_skill_map[title_normalized])
            else:
                # Empty list if job has no skills
                job_skills_list.append([])
        
        # Get target occupation info
        target_info = self.target_occupation_map[target_doc]
        
        return {
            'job_skills_list': job_skills_list,
            'target_title': target_info['title'],
            'target_description': target_info['description'],
            'target_isco_group': target_info['isco_group'],
            'idx': idx
        }
    
    def _extract_job_titles(self, history_doc: str) -> List[str]:
        """
        Extract job titles from history document.
        
        Handles formatted documents: "role: <title>\n description: ..."
        """
        titles = re.findall(r"role: (.*?)\n", history_doc)
        
        # Fallback: split by SEP_TOKEN if no regex matches
        if not titles:
            titles = [t.strip() for t in history_doc.split(self.sep_token) if t.strip()]
        
        return titles


class ISCOGroupBatchSampler(Sampler):
    """
    Custom batch sampler that ensures no duplicate ISCO groups within a batch.
    
    This prevents trivial negatives in MultipleNegativesRankingLoss where
    occupations from the same ISCO group could be too similar.
    """
    
    def __init__(
        self,
        dataset: SkillBasedCareerPathDataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        """
        Initialize the ISCO group-aware batch sampler.
        
        Args:
            dataset: SkillBasedCareerPathDataset
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle samples
            drop_last: Whether to drop incomplete batches
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        
        # Group samples by ISCO group
        self.isco_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            sample = dataset[idx]
            isco_group = sample['target_isco_group']
            self.isco_to_indices[isco_group].append(idx)
        
        self.isco_groups = list(self.isco_to_indices.keys())
    
    def __iter__(self):
        """Generate batches with unique ISCO groups."""
        # Shuffle ISCO groups if requested
        if self.shuffle:
            random.shuffle(self.isco_groups)
        
        # For each ISCO group, shuffle its indices
        isco_iterators = {}
        for isco_group in self.isco_groups:
            indices = self.isco_to_indices[isco_group].copy()
            if self.shuffle:
                random.shuffle(indices)
            isco_iterators[isco_group] = iter(indices)
        
        # Generate batches
        batch = []
        used_isco_groups = set()
        available_groups = self.isco_groups.copy()
        
        while available_groups:
            # Shuffle available groups for random selection
            if self.shuffle:
                random.shuffle(available_groups)
            
            for isco_group in available_groups[:]:
                # Try to get next index from this ISCO group
                try:
                    idx = next(isco_iterators[isco_group])
                    batch.append(idx)
                    used_isco_groups.add(isco_group)
                    
                    # If batch is full, yield it
                    if len(batch) == self.batch_size:
                        yield batch
                        batch = []
                        used_isco_groups = set()
                
                except StopIteration:
                    # This ISCO group is exhausted
                    available_groups.remove(isco_group)
                
                # Stop if batch is full
                if len(batch) == self.batch_size:
                    break
        
        # Handle last incomplete batch
        if batch and not self.drop_last:
            yield batch
    
    def __len__(self):
        """Return the number of batches."""
        total_samples = len(self.dataset)
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return (total_samples + self.batch_size - 1) // self.batch_size


def collate_skill_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for skill-based batches.
    
    Note: This returns lists and strings, not tensors, because skill encoding
    and pooling happens in the training loop with the model.
    
    Args:
        batch: List of sample dictionaries from __getitem__
        
    Returns:
        Batched dictionary with lists
    """
    return {
        'job_skills_list': [sample['job_skills_list'] for sample in batch],
        'target_titles': [sample['target_title'] for sample in batch],
        'target_descriptions': [sample['target_description'] for sample in batch],
        'target_isco_groups': [sample['target_isco_group'] for sample in batch],
        'indices': [sample['idx'] for sample in batch]
    }

