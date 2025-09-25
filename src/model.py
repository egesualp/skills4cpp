"""
Model Layer Architecture for Skills4Cpp

This module provides a pluggable model layer architecture for semantic matching between job descriptions
and ESCO (European Skills, Competences, Qualifications and Occupations) skills. The architecture is designed
to support three main model types:

1. **BiEncoder**: Uses dual encoders to separately encode job descriptions and ESCO skills, then computes
   similarity scores between the encoded representations. This approach is efficient for large-scale
   retrieval tasks.

2. **LinearMap**: Employs a linear mapping layer to transform job embeddings into the ESCO embedding space,
   enabling direct comparison with ESCO skill embeddings. This method learns a cross-modal mapping function.

3. **HybridBiEncoder**: Combines the strengths of both BiEncoder and LinearMap approaches by using a hybrid
   architecture that can leverage both separate encoding and learned transformations.

The BaseModel abstract class defines the core interface that all model implementations must follow, ensuring
consistent behavior across different model architectures while allowing for specialized implementations.

Key Features:
- Device-aware operations (supports any valid torch.device)
- Batch processing for efficient inference
- L2 normalization for consistent embedding spaces
- Cosine similarity scoring
- Comprehensive type hints and documentation
- Pluggable architecture for easy extension

Usage:
    The BaseModel interface provides methods for encoding job descriptions and ESCO skills, computing
    similarity scores, and saving/loading models. Concrete implementations should inherit from BaseModel
    and implement the abstract methods according to their specific architecture.
"""

from abc import ABC, abstractmethod
from typing import List, Union
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import pickle
import os
import json
import warnings
from dataclasses import dataclass

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    warnings.warn("sentence-transformers not available. BiEncoder will not work without it.")
    SentenceTransformer = None


def l2norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    L2 normalize a tensor along the last dimension.

    This function performs L2 normalization on the input tensor, which is commonly used
    in embedding spaces to ensure consistent magnitude across different embeddings.
    The epsilon value provides numerical stability by preventing division by zero.

    Args:
        x (torch.Tensor): Input tensor to normalize.
        eps (float): Small epsilon value for numerical stability (default: 1e-12).

    Returns:
        torch.Tensor: L2-normalized tensor with the same shape as input.

    Example:
        >>> x = torch.tensor([[3.0, 4.0], [0.0, 5.0]])
        >>> normalized = l2norm(x)
        >>> print(normalized)
        tensor([[0.6000, 0.8000],
                [0.0000, 1.0000]])
    """
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    return x / (norm + eps)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy array.

    This utility function converts a PyTorch tensor to a numpy array with proper
    dtype handling. The conversion preserves the tensor's data while moving it
    to CPU memory if necessary.

    Args:
        x (torch.Tensor): Input PyTorch tensor to convert.

    Returns:
        np.ndarray: Numpy array with float32 dtype containing the tensor data.

    Example:
        >>> x = torch.tensor([1.0, 2.0, 3.0])
        >>> arr = to_numpy(x)
        >>> print(arr.dtype)
        float32
    """
    return x.detach().cpu().numpy().astype(np.float32)


def cosine_sim_mat(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity matrix between two sets of embeddings.

    This function computes the cosine similarity matrix between two numpy arrays
    of embeddings. For correct results, both input arrays MUST be L2-normalized
    (unit length vectors). This ensures that the cosine similarity equals the
    dot product and produces meaningful similarity scores between 0 and 1.

    Args:
        a (np.ndarray): First set of embeddings, shape (N, D) where N is number of samples.
                       Must be L2-normalized (unit length vectors).
        b (np.ndarray): Second set of embeddings, shape (M, D) where M is number of samples.
                       Must be L2-normalized (unit length vectors).

    Returns:
        np.ndarray: Cosine similarity matrix of shape (N, M) with values in range [-1, 1].

    Raises:
        ValueError: If input arrays have incompatible dimensions for matrix multiplication.

    Note:
        Use the l2norm utility function to normalize embeddings before calling this function,
        or use BaseModel.score() with normalize=True for automatic normalization.

    Example:
        >>> a = np.array([[1.0, 0.0], [0.0, 1.0]])  # L2-normalized embeddings
        >>> b = np.array([[0.7071, 0.7071], [1.0, 0.0]])  # L2-normalized embeddings
        >>> sim_matrix = cosine_sim_mat(a, b)
        >>> print(sim_matrix.shape)
        (2, 2)
    """
    if a.shape[1] != b.shape[1]:
        raise ValueError(f"Incompatible dimensions: a.shape[1]={a.shape[1]}, b.shape[1]={b.shape[1]}")

    # Cosine similarity = dot product for L2-normalized vectors
    return np.dot(a, b.T)


@dataclass
class ModelConfig:
    """
    Configuration class for BiEncoder models.

    This dataclass manages all configuration parameters for BiEncoder models,
    including model identifiers, projection dimensions, and operational settings.

    Attributes:
        hf_id (str): HuggingFace model identifier (e.g., "intfloat/e5-base").
        proj_dim (int): Projection dimension for embedding heads (default: 256).
        asymmetric (bool): Whether to use separate heads for job/ESCO encoding (default: False).
        normalize_output (bool): Whether to L2-normalize output embeddings (default: True).
        device (str): Device specification for model operations (default: "cuda").

    Example:
        >>> config = ModelConfig(
        ...     hf_id="intfloat/e5-base",
        ...     proj_dim=256,
        ...     asymmetric=False,
        ...     normalize_output=True,
        ...     device="cuda"
        ... )
        >>> print(config.hf_id)
        'intfloat/e5-base'
    """
    hf_id: str
    proj_dim: int = 256
    asymmetric: bool = False
    normalize_output: bool = True
    device: str = "cuda"


class BaseModel(ABC):
    """
    Abstract base class for all model implementations in the Skills4Cpp framework.

    This class defines the core interface that all model implementations must follow,
    providing a consistent API for encoding job descriptions and ESCO skills, computing
    similarity scores, and managing model persistence. The abstract methods ensure
    that all concrete implementations provide the necessary functionality while allowing
    for specialized architectures.

    Attributes:
        device (torch.device): Device to run the model on (any valid torch.device).
        model_name (str): Name/identifier of the specific model implementation.

    Key Methods:
        - encode_job: Encode job descriptions into embedding vectors
        - encode_esco: Encode ESCO skills into embedding vectors
        - score: Compute similarity scores between job and ESCO embeddings
        - save: Persist model to disk
        - load: Load model from disk or HuggingFace model hub
    """

    def __init__(self, device: Union[str, torch.device] = "cuda"):
        """
        Initialize the BaseModel with specified device.

        Args:
            device (Union[str, torch.device]): Device to run the model on. Can be any valid
                                               torch.device specification including 'cpu', 'cuda',
                                               'cuda:0', 'cuda:1', etc. If a CUDA device is
                                               specified but CUDA is not available, falls back to 'cpu'.

        Raises:
            RuntimeError: If torch.device() fails to parse the device specification.
        """
        try:
            # Try to create the torch device first
            requested_device = torch.device(device)

            # Check CUDA availability for CUDA devices
            if requested_device.type == "cuda" and not torch.cuda.is_available():
                print(f"Warning: CUDA not available, falling back to CPU")
                self.device = torch.device("cpu")
            else:
                self.device = requested_device
        except RuntimeError as e:
            # torch.device() will raise RuntimeError for invalid device strings
            raise RuntimeError(f"Invalid device specification: {device}. Error: {e}")

        self.model_name = self.__class__.__name__

    @abstractmethod
    def encode_job(self, texts: List[str], batch_size: int = 128, normalize: bool = True) -> np.ndarray:
        """
        Encode job descriptions into embedding vectors.

        This abstract method must be implemented by all concrete model classes to
        provide job description encoding functionality. The method should handle
        batch processing for efficiency and optionally apply L2 normalization.

        Args:
            texts (List[str]): List of job description texts to encode.
            batch_size (int): Batch size for processing (default: 128).
            normalize (bool): Whether to apply L2 normalization (default: True).

        Returns:
            np.ndarray: Array of job embeddings with shape (len(texts), embedding_dim).

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def encode_esco(self, texts: List[str], batch_size: int = 128, normalize: bool = True) -> np.ndarray:
        """
        Encode ESCO skills into embedding vectors.

        This abstract method must be implemented by all concrete model classes to
        provide ESCO skill encoding functionality. The method should handle batch
        processing for efficiency and optionally apply L2 normalization.

        Args:
            texts (List[str]): List of ESCO skill texts to encode.
            batch_size (int): Batch size for processing (default: 128).
            normalize (bool): Whether to apply L2 normalization (default: True).

        Returns:
            np.ndarray: Array of ESCO embeddings with shape (len(texts), embedding_dim).

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        pass

    def score(self, job_embs: np.ndarray, esco_embs: np.ndarray, normalize: bool = False) -> np.ndarray:
        """
        Compute similarity scores between job and ESCO embeddings.

        This method computes cosine similarity scores between job embeddings and ESCO
        embeddings using the cosine_sim_mat utility function. It serves as the standard
        scoring mechanism across all model implementations.

        Args:
            job_embs (np.ndarray): Job embeddings of shape (num_jobs, embedding_dim).
            esco_embs (np.ndarray): ESCO embeddings of shape (num_skills, embedding_dim).
            normalize (bool): Whether to apply L2 normalization to embeddings before
                             computing similarity. Default is False to maintain backward
                             compatibility. When True, both job and ESCO embeddings are
                             normalized to unit length. (default: False)

        Returns:
            np.ndarray: Similarity matrix of shape (num_jobs, num_skills).

        Note:
            For accurate cosine similarity scores, embeddings should be L2-normalized
            (unit length). Set normalize=True if your embeddings are not pre-normalized,
            or normalize them manually using the l2norm utility function.

        Example:
            >>> job_embs = np.random.rand(10, 384)  # 10 jobs, 384-dim embeddings
            >>> esco_embs = np.random.rand(100, 384)  # 100 skills, 384-dim embeddings
            >>> scores = model.score(job_embs, esco_embs, normalize=True)
            >>> print(scores.shape)
            (10, 100)
        """
        if normalize:
            job_embs = l2norm(torch.from_numpy(job_embs)).numpy()
            esco_embs = l2norm(torch.from_numpy(esco_embs)).numpy()

        return cosine_sim_mat(job_embs, esco_embs)

    @abstractmethod
    def save(self, out_dir: str) -> None:
        """
        Save model to disk.

        This abstract method must be implemented by all concrete model classes to
        provide model persistence functionality. The method should save all necessary
        components including model weights, configuration, and metadata.

        Args:
            out_dir (str): Directory path where the model should be saved.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path_or_hf_id: str, **kwargs) -> "BaseModel":
        """
        Load model from disk or HuggingFace model hub.

        This abstract classmethod must be implemented by all concrete model classes to
        provide model loading functionality. The method should support loading from
        local disk paths as well as HuggingFace model identifiers.

        Args:
            path_or_hf_id (str): Either a local path to saved model or HuggingFace model ID.
            **kwargs: Additional keyword arguments for model-specific loading parameters.

        Returns:
            BaseModel: Loaded model instance ready for inference.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
            FileNotFoundError: If the specified path does not exist.
            ValueError: If the model file is corrupted or incompatible.
        """
        pass


class BiEncoder(BaseModel):
    """
    BiEncoder model implementation using SentenceTransformers with configurable projection heads.

    This class implements a bi-encoder architecture that uses separate encoders for job descriptions
    and ESCO skills. The model leverages SentenceTransformers for text encoding and applies
    configurable projection heads (1-2 layer MLPs with GELU activation) to map embeddings to
    the desired dimensionality. The projection heads can be shared (symmetric) or separate
    (asymmetric) for job/ESCO encoding.

    Key Features:
        - SentenceTransformer integration for high-quality text embeddings
        - Configurable projection heads with 1-layer Linear(d_model→proj_dim) + GELU
        - Support for both symmetric (shared) and asymmetric (separate) projection heads
        - Automatic L2 normalization for consistent embedding spaces
        - Comprehensive save/load system supporting both local and HuggingFace model sources
        - Device-aware operations with automatic fallback to CPU if CUDA unavailable

    Attributes:
        config (ModelConfig): Configuration object containing model parameters.
        encoder (SentenceTransformer): The underlying SentenceTransformer model.
        job_proj (torch.nn.Linear): Projection head for job descriptions.
        esco_proj (torch.nn.Linear): Projection head for ESCO skills (separate if asymmetric).
        d_model (int): Dimensionality of the base SentenceTransformer embeddings.

    Example:
        >>> config = ModelConfig(hf_id="intfloat/e5-base", proj_dim=256, asymmetric=False)
        >>> model = BiEncoder(config)
        >>> job_texts = ["Software engineer with Python experience"]
        >>> esco_texts = ["Programming in Python"]
        >>> job_embs = model.encode_job(job_texts)
        >>> esco_embs = model.encode_esco(esco_texts)
        >>> scores = model.score(job_embs, esco_embs)
        >>> print(scores.shape)
        (1, 1)
    """

    def __init__(self, config: ModelConfig):
        """
        Initialize the BiEncoder with the given configuration.

        Args:
            config (ModelConfig): Configuration object containing model parameters.

        Raises:
            RuntimeError: If sentence-transformers is not available or model loading fails.
            ValueError: If configuration parameters are invalid.
        """
        super().__init__(device=config.device)

        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is required for BiEncoder. Install with: pip install sentence-transformers")

        self.config = config

        try:
            # Initialize SentenceTransformer
            self.encoder = SentenceTransformer(config.hf_id, device=self.device)
            self.d_model = self.encoder.get_sentence_embedding_dimension()
        except Exception as e:
            raise RuntimeError(f"Failed to load SentenceTransformer model '{config.hf_id}': {e}")

        # Create projection heads
        self.job_proj = self._create_projection_head()
        self.esco_proj = self.job_proj if not config.asymmetric else self._create_projection_head()

        # Move projection heads to device
        self.job_proj = self.job_proj.to(self.device)
        self.esco_proj = self.esco_proj.to(self.device)

    def _create_projection_head(self) -> nn.Linear:
        """
        Create a projection head for dimensionality reduction.

        Creates a 1-layer MLP with Linear transformation followed by GELU activation
        to map from the base embedding dimension to the projection dimension.

        Returns:
            torch.nn.Linear: Projection head module (d_model -> proj_dim).
        """
        return nn.Sequential(
            nn.Linear(self.d_model, self.config.proj_dim),
            nn.GELU()
        ).to(self.device)

    def _encode(self, texts: List[str], proj_head: nn.Module, normalize: bool = True) -> np.ndarray:
        """
        Encode texts using SentenceTransformer and apply projection head.

        This method handles the core encoding pipeline: text tokenization, SentenceTransformer
        encoding, projection head application, and optional L2 normalization.

        Args:
            texts (List[str]): List of texts to encode.
            proj_head (torch.nn.Module): Projection head to apply after encoding.
            normalize (bool): Whether to apply L2 normalization (default: True).

        Returns:
            np.ndarray: Encoded embeddings with shape (len(texts), proj_dim).

        Raises:
            RuntimeError: If encoding process fails.
        """
        try:
            # Get embeddings from SentenceTransformer
            embeddings = self.encoder.encode(
                texts,
                batch_size=128,
                convert_to_tensor=True,
                device=self.device,
                show_progress_bar=False
            )

            # Apply projection head
            projected = proj_head(embeddings)

            # Apply L2 normalization if requested
            if normalize or self.config.normalize_output:
                projected = l2norm(projected)

            return to_numpy(projected)

        except Exception as e:
            raise RuntimeError(f"Encoding failed: {e}")

    def encode_job(self, texts: List[str], batch_size: int = 128, normalize: bool = True) -> np.ndarray:
        """
        Encode job descriptions into embedding vectors.

        Args:
            texts (List[str]): List of job description texts to encode.
            batch_size (int): Batch size for processing (ignored, kept for API consistency).
            normalize (bool): Whether to apply L2 normalization (default: True).

        Returns:
            np.ndarray: Array of job embeddings with shape (len(texts), proj_dim).
        """
        return self._encode(texts, self.job_proj, normalize)

    def encode_esco(self, texts: List[str], batch_size: int = 128, normalize: bool = True) -> np.ndarray:
        """
        Encode ESCO skills into embedding vectors.

        Args:
            texts (List[str]): List of ESCO skill texts to encode.
            batch_size (int): Batch size for processing (ignored, kept for API consistency).
            normalize (bool): Whether to apply L2 normalization (default: True).

        Returns:
            np.ndarray: Array of ESCO embeddings with shape (len(texts), proj_dim).
        """
        return self._encode(texts, self.esco_proj, normalize)

    def save(self, out_dir: str) -> None:
        """
        Save the BiEncoder model to disk.

        Saves the SentenceTransformer encoder to a subdirectory, projection heads to a .pt file,
        and configuration to a JSON file. This allows for complete model reconstruction.

        Args:
            out_dir (str): Directory path where the model should be saved.

        Raises:
            RuntimeError: If saving process fails.
        """
        try:
            out_path = Path(out_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            # Save SentenceTransformer encoder to subdirectory
            encoder_dir = out_path / "encoder"
            self.encoder.save(str(encoder_dir))

            # Save projection heads
            heads_path = out_path / "projection_heads.pt"
            torch.save({
                'job_proj_state_dict': self.job_proj.state_dict(),
                'esco_proj_state_dict': self.esco_proj.state_dict(),
                'd_model': self.d_model,
                'config': self.config
            }, heads_path)

            # Save configuration as JSON
            config_path = out_path / "config.json"
            with open(config_path, 'w') as f:
                json.dump({
                    'hf_id': self.config.hf_id,
                    'proj_dim': self.config.proj_dim,
                    'asymmetric': self.config.asymmetric,
                    'normalize_output': self.config.normalize_output,
                    'device': str(self.device)
                }, f, indent=2)

        except Exception as e:
            raise RuntimeError(f"Failed to save model to {out_dir}: {e}")

    @classmethod
    def load(cls, path_or_hf_id: str, **kwargs) -> "BiEncoder":
        """
        Load BiEncoder model from disk or HuggingFace model hub.

        This method supports loading from:
        1. Local directory containing saved model files
        2. HuggingFace model identifier (creates new model with config)

        Args:
            path_or_hf_id (str): Either a local path to saved model or HuggingFace model ID.
            **kwargs: Additional keyword arguments for model configuration.

        Returns:
            BiEncoder: Loaded model instance ready for inference.

        Raises:
            FileNotFoundError: If the specified local path does not exist.
            RuntimeError: If loading process fails or model is corrupted.
            ValueError: If configuration parameters are invalid.
        """
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is required for BiEncoder. Install with: pip install sentence-transformers")

        # Check if it's a local path or HuggingFace ID
        if Path(path_or_hf_id).exists():
            # Load from local directory
            model_path = Path(path_or_hf_id)

            # Load configuration
            config_path = model_path / "config.json"
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            with open(config_path, 'r') as f:
                config_dict = json.load(f)

            config = ModelConfig(**config_dict)

            # Load encoder
            encoder_dir = model_path / "encoder"
            if not encoder_dir.exists():
                raise FileNotFoundError(f"Encoder directory not found: {encoder_dir}")

            try:
                encoder = SentenceTransformer(str(encoder_dir))
            except Exception as e:
                raise RuntimeError(f"Failed to load encoder from {encoder_dir}: {e}")

            # Load projection heads
            heads_path = model_path / "projection_heads.pt"
            if not heads_path.exists():
                raise FileNotFoundError(f"Projection heads file not found: {heads_path}")

            checkpoint = torch.load(heads_path, map_location='cpu')
            job_proj_state = checkpoint['job_proj_state_dict']
            esco_proj_state = checkpoint['esco_proj_state_dict']
            d_model = checkpoint['d_model']

            # Create model instance
            model = cls.__new__(cls)
            model.config = config
            model.encoder = encoder
            model.d_model = d_model

            # Create projection heads
            model.job_proj = cls._create_projection_head_static(model)
            model.esco_proj = model.job_proj if not config.asymmetric else cls._create_projection_head_static(model)

            # Load state dictionaries
            model.job_proj.load_state_dict(job_proj_state)
            model.esco_proj.load_state_dict(esco_proj_state)

            # Set device
            model.device = torch.device(config.device)

            # Move to device
            model.job_proj = model.job_proj.to(model.device)
            model.esco_proj = model.esco_proj.to(model.device)

            return model

        else:
            # Load from HuggingFace model hub
            # Create configuration with provided kwargs
            config_dict = {
                'hf_id': path_or_hf_id,
                'proj_dim': kwargs.get('proj_dim', 256),
                'asymmetric': kwargs.get('asymmetric', False),
                'normalize_output': kwargs.get('normalize_output', True),
                'device': kwargs.get('device', 'cuda')
            }

            config = ModelConfig(**config_dict)
            return cls(config)

    @staticmethod
    def _create_projection_head_static(model_instance) -> nn.Linear:
        """
        Static method to create projection head for loading (avoids self reference).

        Args:
            model_instance: Model instance with d_model and config attributes.

        Returns:
            torch.nn.Linear: Projection head module.
        """
        return nn.Sequential(
            nn.Linear(model_instance.d_model, model_instance.config.proj_dim),
            nn.GELU()
        )
