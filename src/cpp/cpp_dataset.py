"""
PyTorch Dataset for Career Path Prediction with on-the-fly embedding generation.

This module provides a Dataset class that generates embeddings dynamically during training,
avoiding the need to pre-compute and store large embedding files.
"""

import re
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Any, Optional


class CareerPathDataset(Dataset):
    """
    PyTorch Dataset for Career Path Prediction that generates embeddings on-the-fly.
    
    Supports:
    - Multiple pooling strategies (mean, weighted_mean, weighted_idf)
    - Skill text variations (name only vs name + description)
    - Multiple feature modalities (text, skill_text, structured)
    """
    
    def __init__(
        self,
        data_pairs: List[Tuple[str, str]],
        encoder,
        Y_target_dict: Dict[str, np.ndarray],
        job_skill_map: Dict[str, List[Dict[str, Any]]],
        esco_skill_text_map: Dict[str, Dict[str, str]],
        skill_properties_map: Dict[str, List[str]],
        all_vocabs: Dict[str, Dict[str, int]],
        use_skill_description: bool = False,
        pooling_strategy: str = "mean",
        alpha: float = 1.0,
        beta: float = 1.0,
        include_text: bool = True,
        include_skill_text: bool = True,
        include_structured: bool = True,
        encoder_skill = None,
        pre_h_text: Optional[np.ndarray] = None,
        pre_h_skill_text: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
        pin_embeddings_to_gpu: bool = False,
    ):
        """
        Initialize the CareerPathDataset.

        Args:
            data_pairs: List of (history_doc, target_doc) tuples
            encoder: SentenceTransformer encoder model for text history
            Y_target_dict: Dict mapping target job strings to their embeddings
            job_skill_map: Dict mapping job titles to skill info (skillUri, score, idf)
            esco_skill_text_map: Dict mapping skillUri to skill name and description
            skill_properties_map: Dict mapping skillUri to structured feature strings
            all_vocabs: Dict of vocabularies for structured features
            use_skill_description: Whether to include skill descriptions in text
            pooling_strategy: One of ["mean", "weighted_mean", "weighted_idf"]
            alpha: Exponent for confidence score (used with weighted_idf)
            beta: Exponent for IDF score (used with weighted_idf)
            include_text: Whether to generate text features
            include_skill_text: Whether to generate skill text features
            include_structured: Whether to generate structured features
            encoder_skill: Optional separate encoder for skills (if None, use encoder)
            pre_h_text: Optional pre-computed text history embeddings [n_samples, embed_dim]
            pre_h_skill_text: Optional pre-computed skill text embeddings [n_samples, embed_dim]
            device: Device to pin embeddings to (for GPU-resident embeddings)
            pin_embeddings_to_gpu: If True, move pre-computed embeddings to GPU and keep them there
        """
        self.data_pairs = data_pairs
        self.encoder = encoder
        self.encoder_skill = encoder_skill if encoder_skill is not None else encoder
        self.Y_target_dict = Y_target_dict
        self.job_skill_map = job_skill_map
        self.esco_skill_text_map = esco_skill_text_map
        self.skill_properties_map = skill_properties_map
        self.all_vocabs = all_vocabs
        self.device = device
        self.pin_embeddings_to_gpu = pin_embeddings_to_gpu

        # Configuration
        self.use_skill_description = use_skill_description
        self.pooling_strategy = pooling_strategy
        self.alpha = alpha
        self.beta = beta
        self.include_text = include_text
        self.include_skill_text = include_skill_text
        self.include_structured = include_structured

        # Pre-computed embeddings - optimized for GPU or shared memory
        self._setup_embeddings(pre_h_text, pre_h_skill_text)
        
        # Pre-compute dimensions
        self.embed_dim = encoder.get_sentence_embedding_dimension()
        # Skill embedding dim: infer from precomputed embeddings or encoder
        if pre_h_skill_text is not None:
            self.skill_embed_dim = pre_h_skill_text.shape[1]
        elif encoder_skill is not None:
            self.skill_embed_dim = encoder_skill.get_sentence_embedding_dimension()
        else:
            self.skill_embed_dim = self.embed_dim
        
        self.structured_dims = {
            key: len(vocab) for key, vocab in all_vocabs.items()
        }
        
        # Pre-compute zero vectors for padding
        self.zero_vec_text = np.zeros(self.embed_dim, dtype=np.float32)
        self.zero_vec_skill = np.zeros(self.skill_embed_dim, dtype=np.float32)
        self.zero_vecs_structured = {
            key: np.zeros(dim, dtype=np.float32) 
            for key, dim in self.structured_dims.items()
        }
        
        # Filter out samples with missing targets
        self._filter_valid_samples()
    
    def _setup_embeddings(self, pre_h_text, pre_h_skill_text):
        """
        Setup pre-computed embeddings with optimizations:
        - GPU pinning: Move embeddings to GPU to avoid CPU->GPU transfers
        - Shared memory: Use torch shared memory for multi-process DataLoader
        """
        if self.pin_embeddings_to_gpu and self.device is not None and self.device.type == 'cuda':
            # Solution 3: Move embeddings to GPU and keep them there
            print(f"📌 Pinning embeddings to GPU ({self.device})...")
            
            if pre_h_text is not None:
                self.pre_h_text = torch.from_numpy(pre_h_text).float().to(self.device)
                print(f"  ✓ Text embeddings on GPU: {self.pre_h_text.shape} ({self.pre_h_text.element_size() * self.pre_h_text.nelement() / 1024**3:.2f} GB)")
            else:
                self.pre_h_text = None
            
            if pre_h_skill_text is not None:
                self.pre_h_skill_text = torch.from_numpy(pre_h_skill_text).float().to(self.device)
                print(f"  ✓ Skill embeddings on GPU: {self.pre_h_skill_text.shape} ({self.pre_h_skill_text.element_size() * self.pre_h_skill_text.nelement() / 1024**3:.2f} GB)")
            else:
                self.pre_h_skill_text = None
        else:
            # Solution 4: Use shared memory for multi-process DataLoader
            if pre_h_text is not None:
                self.pre_h_text = torch.from_numpy(pre_h_text).float().share_memory_()
                print(f"  ✓ Text embeddings in shared memory: {self.pre_h_text.shape}")
            else:
                self.pre_h_text = None
            
            if pre_h_skill_text is not None:
                self.pre_h_skill_text = torch.from_numpy(pre_h_skill_text).float().share_memory_()
                print(f"  ✓ Skill embeddings in shared memory: {self.pre_h_skill_text.shape}")
            else:
                self.pre_h_skill_text = None
    
    def _filter_valid_samples(self):
        """Remove samples where target embedding is not available."""
        original_len = len(self.data_pairs)
        self.data_pairs = [
            (hist, tgt) for hist, tgt in self.data_pairs 
            if tgt in self.Y_target_dict
        ]
        filtered_count = original_len - len(self.data_pairs)
        if filtered_count > 0:
            print(f"Filtered out {filtered_count} samples with missing target embeddings.")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data_pairs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Generate features for a single sample on-the-fly.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Dictionary containing:
                - 'h_text': Text embedding of career history (if enabled)
                - 'h_skill_text': Pooled skill text embedding (if enabled)
                - 'h_structured_*': Multi-hot structured features (if enabled)
                - 'y': Target job embedding
        """
        history_doc, target_doc = self.data_pairs[idx]
        
        features = {}
        
        # --- 1. Generate h_text (Text History) ---
        if self.include_text:
            if self.pre_h_text is not None:
                # Embeddings are already torch tensors (GPU or shared memory)
                features['h_text'] = self.pre_h_text[idx]
            else:
                h_text = self.encoder.encode(history_doc, convert_to_numpy=True)
                features['h_text'] = torch.from_numpy(h_text).float()

        # --- 2. Get target y (Next ESCO Job) ---
        y_vector = self.Y_target_dict[target_doc]
        features['y'] = torch.from_numpy(y_vector).float()

        # --- 3. Generate h_skill_text ---
        # PRIORITY: Use pre-computed embeddings if available (computed with job_ids in v3)
        if self.include_skill_text:
            if self.pre_h_skill_text is not None:
                # Pre-computed embeddings take priority (e.g., from build_last_job_skill_embeddings)
                features['h_skill_text'] = self.pre_h_skill_text[idx]
            else:
                # Fallback: Extract skills from history using title lookup
                # Handle both formatted documents (e.g., "role: cook\n description: ...")
                # and plain titles (e.g., "cook " or "cook <SEP>head chef ")
                raw_titles_in_history = re.findall(r"role: (.*?)\n", history_doc)
                
                # If no matches found, assume it's plain title(s), possibly with <SEP> separator
                if not raw_titles_in_history:
                    from cpp import utils
                    raw_titles_in_history = [t.strip() for t in history_doc.split(utils.SEP_TOKEN) if t.strip()]
                
                skill_info_list = []
                for title in raw_titles_in_history:
                    title_normalized = title.strip().lower()
                    if title_normalized in self.job_skill_map:
                        skill_info_list.extend(self.job_skill_map[title_normalized])
                
                if skill_info_list:
                    h_skill_text = self._generate_skill_text_embedding(skill_info_list)
                    features['h_skill_text'] = torch.from_numpy(h_skill_text).float()
                else:
                    # No skills found - use zero vector
                    zero_skill = torch.from_numpy(self.zero_vec_skill).float()
                    if self.pin_embeddings_to_gpu and self.device is not None:
                        zero_skill = zero_skill.to(self.device)
                    features['h_skill_text'] = zero_skill

        # --- 4. Extract skills for structured features ---
        # This section still uses title lookup for structured features
        skill_info_list_for_struct = []
        if self.include_structured:
            raw_titles_in_history = re.findall(r"role: (.*?)\n", history_doc)
            if not raw_titles_in_history:
                from cpp import utils
                raw_titles_in_history = [t.strip() for t in history_doc.split(utils.SEP_TOKEN) if t.strip()]
            
            for title in raw_titles_in_history:
                title_normalized = title.strip().lower()
                if title_normalized in self.job_skill_map:
                    skill_info_list_for_struct.extend(self.job_skill_map[title_normalized])
        
        # --- 5. Generate h_structured ---
        if self.include_structured:
            if skill_info_list_for_struct:
                structured_vectors = self._generate_structured_features(skill_info_list_for_struct)
                for key, vec in structured_vectors.items():
                    vec_tensor = torch.from_numpy(vec).float()
                    if self.pin_embeddings_to_gpu and self.device is not None:
                        vec_tensor = vec_tensor.to(self.device)
                    features[f'h_structured_{key}'] = vec_tensor
            else:
                # No skills found for structured features - use zero vectors
                for key in self.all_vocabs.keys():
                    zero_struct = torch.from_numpy(self.zero_vecs_structured[key]).float()
                    if self.pin_embeddings_to_gpu and self.device is not None:
                        zero_struct = zero_struct.to(self.device)
                    features[f'h_structured_{key}'] = zero_struct
        
        return features
    
    def _generate_skill_text_embedding(self, skill_info_list: List[Dict]) -> np.ndarray:
        """
        Generate pooled skill text embedding from a list of skills.
        
        Args:
            skill_info_list: List of dicts with 'skillUri', 'score', and optionally 'idf'
            
        Returns:
            Pooled embedding vector
        """
        strings_to_embed = []
        weights_for_pooling = []
        
        for skill_info in skill_info_list:
            skill_uri = skill_info['skillUri']
            
            if skill_uri in self.esco_skill_text_map:
                skill_text = self.esco_skill_text_map[skill_uri]
                
                # Format skill text
                if self.use_skill_description:
                    text = f"role: {skill_text['name']} \n description: {skill_text['desc']}"
                else:
                    text = skill_text['name']
                strings_to_embed.append(text)
                
                # Calculate weight for pooling
                if self.pooling_strategy == "mean":
                    weights_for_pooling.append(1.0)
                elif self.pooling_strategy == "weighted_mean":
                    weights_for_pooling.append(skill_info['score'])
                elif self.pooling_strategy == "weighted_idf":
                    c_i = skill_info['score']
                    idf_i = skill_info.get('idf', 0)
                    weight = (c_i ** self.alpha) * (idf_i ** self.beta)
                    weights_for_pooling.append(weight)
        
        # Pool embeddings
        if not strings_to_embed:
            return self.zero_vec_skill
        
        # Use skill-specific encoder if available
        skill_embeddings = self.encoder_skill.encode(strings_to_embed, convert_to_numpy=True)
        weights = np.array(weights_for_pooling, dtype=np.float32)
        
        # Use np.average for numerically stable weighted average
        if self.pooling_strategy == "mean" or np.sum(weights) == 0:
            h_skill_text = np.mean(skill_embeddings, axis=0)
        else:
            h_skill_text = np.average(skill_embeddings, axis=0, weights=weights)
        
        return h_skill_text.astype(np.float32)
    
    def _generate_structured_features(self, skill_info_list: List[Dict]) -> Dict[str, np.ndarray]:
        """
        Generate multi-hot structured feature vectors from a list of skills.
        
        Args:
            skill_info_list: List of dicts with 'skillUri'
            
        Returns:
            Dict of structured feature vectors
        """
        structured_vectors = {
            key: np.zeros(len(vocab), dtype=np.float32) 
            for key, vocab in self.all_vocabs.items()
        }
        
        for skill_info in skill_info_list:
            skill_uri = skill_info['skillUri']
            
            if skill_uri in self.skill_properties_map:
                features = self.skill_properties_map[skill_uri]
                for feature_string in features:
                    # Check if this feature exists in the vocab
                    if 'structured' in structured_vectors and \
                       feature_string in self.all_vocabs.get('structured', {}):
                        idx = self.all_vocabs['structured'][feature_string]
                        structured_vectors['structured'][idx] = 1.0
        
        return structured_vectors


def collate_career_path_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for DataLoader to properly stack feature dictionaries.
    
    Args:
        batch: List of feature dictionaries from __getitem__
        
    Returns:
        Batched feature dictionary with stacked tensors
    """
    if len(batch) == 0:
        return {}
    
    # Get all keys from the first sample
    keys = batch[0].keys()

    # Determine target device (use device of first tensor we find)
    target_device = None
    for sample in batch:
        for key, value in sample.items():
            if value.is_cuda:
                target_device = value.device
                break
        if target_device is not None:
            break
    
    # Stack tensors for each key
    batched = {}
    for key in keys:
        tensors = [sample[key] for sample in batch]
        # Move all tensors to target device if one was GPU
        if target_device is not None:
            tensors = [t.to(target_device) if not t.is_cuda else t for t in tensors]
        batched[key] = torch.stack(tensors)

    
    return batched

