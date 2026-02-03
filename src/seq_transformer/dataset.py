import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class CareerSequenceDataset(Dataset):
    """
    Dataset for sequential career path modeling.
    Loads job experiences for each user, maps them to embeddings, 
    and returns pairs of (history_sequence, target_job).
    
    Generates subsequences for training: for a path [J1, J2, J3, J4], it yields:
    ([J1], J2)
    ([J1, J2], J3)
    ([J1, J2, J3], J4)
    """
    def __init__(
        self, 
        job_embeddings_path: str,
        data_type: str,
        skill_embeddings_path: str = None,
        occupations_path: str = "data/occupations_en.csv",
        split: str = "train",
        combine_method: str = "concat",
        max_len: int = None,
        min_seq_len: int = 2, # Minimum length including target (so 1 context + 1 target)
        use_all_subspans: bool = False
    ):
        """
        Args:
            job_embeddings_path: Path to .pt file containing job embeddings.
            data_type: Type of dataset to load (decorte, decorte_esco, karrierewege, etc.)
            skill_embeddings_path: Path to .pt file containing skill embeddings (optional).
            occupations_path: Path to occupations CSV for mapping titles to URIs.
            split: Dataset split to load.
            combine_method: 'concat' or 'sum' to combine job and skill vectors.
            max_len: Maximum sequence length for the history context.
            min_seq_len: Minimum total sequence length to consider a path valid.
            use_all_subspans: If True, generates all contiguous subsequences as samples (data augmentation).
        """
        self.combine_method = combine_method
        self.max_len = max_len
        self.min_seq_len = min_seq_len
        self.data_type = data_type
        self.use_all_subspans = use_all_subspans
        
        # Load mappings and embeddings
        self.label_to_uri = self._load_occupations_mapping(occupations_path)
        self.job_embeddings = self._load_embeddings(job_embeddings_path)
        
        if skill_embeddings_path:
            self.skill_embeddings = self._load_embeddings(skill_embeddings_path)
            self.use_skills = True
        else:
            self.skill_embeddings = {}
            self.use_skills = False
        
        # Load and process data into list of full paths
        self.raw_sequences = self._load_and_process_data(split)
        
        # Generate training samples (subsequences)
        self.samples = self._generate_samples(self.raw_sequences)
        logger.info(f"Generated {len(self.samples)} training samples from {len(self.raw_sequences)} unique career paths")

    def _load_occupations_mapping(self, path: str) -> Dict[str, str]:
        logger.info(f"Loading occupations mapping from {path}")
        df = pd.read_csv(path)
        # Create mapping from preferredLabel to conceptUri
        # We lowercase for better matching
        mapping = {}
        for _, row in df.iterrows():
            if pd.notna(row['preferredLabel']) and pd.notna(row['conceptUri']):
                mapping[row['preferredLabel'].strip().lower()] = row['conceptUri']
        return mapping

    def _load_embeddings(self, path: str) -> Dict[str, torch.Tensor]:
        logger.info(f"Loading embeddings from {path}")
        try:
            data = torch.load(path, map_location='cpu')
            if isinstance(data, dict) and 'embeddings' in data and 'ids' in data:
                # Format from extract_features.py
                embeddings = data['embeddings']
                ids = data['ids']
                return dict(zip(ids, embeddings))
            elif isinstance(data, dict):
                 # Assume dictionary mapping ID -> Tensor directly
                return data
            else:
                raise ValueError(f"Unknown embedding format in {path}")
        except FileNotFoundError:
            logger.warning(f"Embeddings file not found at {path}. Ignoring (will result in empty sequences if keys missing).")
            return {}
        except Exception as e:
            logger.error(f"Error loading embeddings from {path}: {e}")
            raise

    def _load_and_process_data(self, split: str) -> List[List[str]]:
        logger.info(f"Loading dataset for data_type={self.data_type}, split={split}")
        
        if self.data_type in ['decorte', 'decorte_esco']:
            dataset_name = "jensjorisdecorte/anonymous-working-histories"
            dataset = load_dataset(dataset_name, split=split)
            return self._process_decorte(dataset)
            
        elif self.data_type in ['karrierewege', 'karrierewege_occ', 'karrierewege_100k', 'karrierewege_cp']:
            if self.data_type == 'karrierewege':
                 dataset_name = "ElenaSenger/Karrierewege"
            else:
                 dataset_name = "ElenaSenger/Karrierewege_plus"
            
            dataset = load_dataset(dataset_name, split=split)
            return self._process_karrierewege(dataset)
        
        else:
             raise ValueError(f"Unsupported data_type: {self.data_type}")

    def _process_decorte(self, dataset) -> List[List[str]]:
        sequences = []
        for example in dataset:
            num_experiences = example.get("number_of_experiences", 0)
            uris = []
            
            for i in range(num_experiences):
                uri = example.get(f"ESCO_uri_{i}")
                
                if uri and pd.notna(uri):
                    # Check if uri exists in job embeddings.
                    # If using skills, must also exist in skill embeddings.
                    valid = (uri in self.job_embeddings)
                    if self.use_skills:
                        valid = valid and (uri in self.skill_embeddings)
                        
                    if valid:
                         uris.append(uri)
                    else:
                         title = example.get(f"ESCO_title_{i}")
                         if title:
                             clean_title = title.strip().lower()
                             mapped_uri = self.label_to_uri.get(clean_title)
                             
                             valid_mapped = mapped_uri and (mapped_uri in self.job_embeddings)
                             if self.use_skills:
                                 valid_mapped = valid_mapped and (mapped_uri in self.skill_embeddings)

                             if valid_mapped:
                                 uris.append(mapped_uri)
            
            if len(uris) >= self.min_seq_len:
                sequences.append(uris)
                
        return sequences

    def _process_karrierewege(self, dataset) -> List[List[str]]:
        df = dataset.to_pandas()
        if '_id' not in df.columns:
             raise ValueError("Karrierewege dataset missing '_id' column")
             
        grouped = df.groupby('_id')
        sequences = []
        
        for _, group in grouped:
            if 'experience_order' in group.columns:
                group = group.sort_values('experience_order')
            
            # Using preferredLabel_en (ESCO) for consistent embedding lookup
            titles = group['preferredLabel_en'].tolist()

            valid_uris = []
            for title in titles:
                if pd.isna(title):
                    continue
                clean_title = title.strip().lower()
                uri = self.label_to_uri.get(clean_title)
                
                valid = uri and (uri in self.job_embeddings)
                if self.use_skills:
                    valid = valid and (uri in self.skill_embeddings)
                
                if valid:
                    valid_uris.append(uri)
            
            if len(valid_uris) >= self.min_seq_len:
                sequences.append(valid_uris)
        
        return sequences

    def _generate_samples(self, sequences: List[List[str]]) -> List[Tuple[List[str], str]]:
        """
        Generates (history, target) pairs from full career sequences.
        For a sequence [A, B, C, D]:
        - ([A], B)
        - ([A, B], C)
        - ([A, B, C], D)
        
        If use_all_subspans is True, it also generates:
        - ([B], C)
        - ([B, C], D)
        - ([C], D)
        """
        samples = []
        for seq in sequences:
            # Iterate over all possible lengths
            for length in range(self.min_seq_len, len(seq) + 1):
                # Determine possible start indices
                if self.use_all_subspans:
                    # All valid start indices for this length
                    start_indices = range(len(seq) - length + 1)
                else:
                    # Only start from the beginning
                    start_indices = [0]
                
                for start_idx in start_indices:
                    subspan = seq[start_idx : start_idx + length]
                    
                    history = subspan[:-1]
                    target = subspan[-1]
                    
                    # Apply max_len to history if set
                    if self.max_len and len(history) > self.max_len:
                        history = history[-self.max_len:]
                    
                    samples.append((history, target))
        return samples

    def _get_combined_vector(self, uri: str) -> torch.Tensor:
        v_job = self.job_embeddings[uri]
        
        if not self.use_skills:
            return v_job
            
        v_skill = self.skill_embeddings[uri]
        
        if self.combine_method == 'concat':
            return torch.cat([v_job, v_skill], dim=-1)
        elif self.combine_method == 'sum':
            return v_job + v_skill
        else:
            raise ValueError(f"Unknown combine_method: {self.combine_method}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        history_uris, target_uri = self.samples[idx]
        
        # Get history vectors
        history_vectors = [self._get_combined_vector(uri) for uri in history_uris]
        history_tensor = torch.stack(history_vectors)
        
        # Get target vector
        target_tensor = self._get_combined_vector(target_uri)
        
        return history_tensor, target_tensor

def collate_fn(batch):
    """
    Collate function for padding sequences.
    Batch is list of (history_tensor, target_tensor) tuples.
    Returns:
        padded_history: (batch_size, max_seq_len, dim)
        targets: (batch_size, dim)
        lengths: (batch_size) - lengths of history sequences
    """
    from torch.nn.utils.rnn import pad_sequence
    
    histories = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    lengths = torch.tensor([len(h) for h in histories])
    
    padded_history = pad_sequence(histories, batch_first=True, padding_value=0.0)
    stacked_targets = torch.stack(targets)
    
    return padded_history, stacked_targets, lengths
