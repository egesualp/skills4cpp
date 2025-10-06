from abc import ABC, abstractmethod
import tempfile
import yaml
from src.config import ModelConfig, load_config, dump_config, Config, DataConfig, InferConfig, ArtifactsConfig
import json
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


class BaseModel(nn.Module, ABC):
    """Abstract base class for all models."""

    def __init__(self, model_config: ModelConfig, device: str):
        super().__init__()
        self.model_config = model_config
        self.device = device

    @abstractmethod
    def encode_job(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Encode job descriptions."""
        raise NotImplementedError

    @abstractmethod
    def encode_esco(self, texts: list[str], batch_size: int) -> np.ndarray:
        """Encode ESCO skills/occupations."""
        raise NotImplementedError

    def score(self, job_embs: np.ndarray, esco_embs: np.ndarray) -> np.ndarray:
        """Compute cosine similarity matrix between job and esco embeddings."""
        return job_embs @ esco_embs.T

    @abstractmethod
    def save(self, out_dir: str) -> None:
        """Save model artifacts to a directory."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, path: str, device: str) -> "BaseModel":
        """Load a model from a directory."""
        raise NotImplementedError


class BiEncoder(BaseModel):
    """
    Bi-encoder model using sentence-transformers with an optional projection head.
    """

    def __init__(self, model_config: ModelConfig, device: str):
        super().__init__(model_config, device)

        self.st_model = SentenceTransformer(model_config.hf_id, device=device)
        self.embedding_dim = self.st_model.get_sentence_embedding_dimension()

        self.job_proj: Optional[nn.Module] = None
        self.esco_proj: Optional[nn.Module] = None

        if model_config.proj_dim:
            self.job_proj = nn.Linear(self.embedding_dim, model_config.proj_dim)
            if model_config.asymmetric:
                self.esco_proj = nn.Linear(self.embedding_dim, model_config.proj_dim)
            else:
                self.esco_proj = self.job_proj
        
        if self.job_proj:
            self.job_proj.to(device)
        if self.esco_proj and model_config.asymmetric:
            self.esco_proj.to(device)


    def _encode(
        self, texts: list[str], batch_size: int, normalize: bool = True, show_progress_bar: bool = False
    ) -> torch.Tensor:
        """Internal encoding function."""
        return self.st_model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=True,
            device=self.device,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress_bar,
        )

    def _apply_projection(self, embeddings: torch.Tensor, proj_layer: Optional[nn.Module]) -> torch.Tensor:
        """Apply projection if it exists."""
        if proj_layer:
            # We clone the tensor to prevent potential in-place modification errors
            # during backpropagation if the model is being trained. PyTorch's
            # autograd engine needs the original tensor for gradient calculations,
            # and modifying it in-place can corrupt the computational graph.
            embeddings = F.gelu(proj_layer(embeddings.clone()))
        return embeddings

    def _process_embeddings(self, embeddings: torch.Tensor) -> np.ndarray:
        """Convert embeddings to numpy array."""
        return embeddings.detach().cpu().numpy().astype(np.float32)

    def encode_job(
        self, texts: list[str], batch_size: int = 128, normalize: bool = True, show_progress_bar: bool = False
    ) -> np.ndarray:
        embeddings = self._encode(texts, batch_size, normalize, show_progress_bar)
        embeddings = self._apply_projection(embeddings, self.job_proj)
        return self._process_embeddings(embeddings)

    def encode_esco(
        self, texts: list[str], batch_size: int = 128, normalize: bool = True, show_progress_bar: bool = False
    ) -> np.ndarray:
        embeddings = self._encode(texts, batch_size, normalize, show_progress_bar)
        embeddings = self._apply_projection(embeddings, self.esco_proj)
        return self._process_embeddings(embeddings)

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.st_model.save(os.path.join(out_dir, "st_model"))
        if self.job_proj:
            torch.save(self.job_proj.state_dict(), os.path.join(out_dir, "job_proj.pt"))
        if self.esco_proj and self.model_config.asymmetric:
            torch.save(self.esco_proj.state_dict(), os.path.join(out_dir, "esco_proj.pt"))
        
        # Save model config for loading
        with open(os.path.join(out_dir, "model_config.json"), "w") as f:
            json.dump(self.model_config.__dict__, f)

    @classmethod
    def load(cls, path: str, device: str) -> "BiEncoder":
        with open(os.path.join(path, "model_config.json"), "r") as f:
            config_dict = json.load(f)
        
        config_dict['hf_id'] = os.path.join(path, "st_model")
        model_config = ModelConfig(**config_dict)
        
        model = cls(model_config, device)
        
        if model.job_proj:
            model.job_proj.load_state_dict(torch.load(os.path.join(path, "job_proj.pt"), map_location=device))
        if model.esco_proj and model.model_config.asymmetric:
            model.esco_proj.load_state_dict(torch.load(os.path.join(path, "esco_proj.pt"), map_location=device))
            
        return model


if __name__ == "__main__":

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy data files first to satisfy DataConfig validation
        with open(os.path.join(tmpdir, "dummy_pairs.jsonl"), "w") as f:
            f.write("{}\\n")
        with open(os.path.join(tmpdir, "dummy_esco.jsonl"), "w") as f:
            f.write("{}\\n")

        # Change CWD to tmpdir to make relative paths in config work
        original_cwd = os.getcwd()
        os.chdir(tmpdir)

        # 1. Create a dummy config object for the smoke test
        # This ensures the module is self-contained and runnable for tests.
        # For actual experiments, you would load a YAML file from disk.
        model_config = ModelConfig(
            hf_id="sentence-transformers/all-MiniLM-L6-v2",
            proj_dim=128,
            asymmetric=True,
            normalize_output=True,
        )
        dummy_config = Config(
            seed=42,
            device="cpu",
            model=model_config,
            data=DataConfig(
                pairs_path="dummy_pairs.jsonl",
                esco_titles_path="dummy_esco.jsonl",
            ),
            infer=InferConfig(batch_size=32, topk=5),
            artifacts=ArtifactsConfig(run_dir="runs/smoke_test"),
        )

        # Use dump_config to create the config file
        config_path = dump_config(dummy_config, tmpdir)
        print(f"Dummy config file created at: {config_path}")

        # 2. Load config using the config manager
        config = load_config("config.yaml")

        # 3. Instantiate the model with the loaded config
        model = BiEncoder(config.model, config.device)
        print("Model instantiated successfully from config.")

        # 4. Test encoding and scoring
        job_titles = ["Software Engineer", "Data Scientist"]
        esco_skills = ["Develops software", "Analyzes data"]
        
        job_embs = model.encode_job(job_titles)
        esco_embs = model.encode_esco(esco_skills)
        scores = model.score(job_embs, esco_embs)
        
        print(f"Job embeddings shape: {job_embs.shape}")
        print(f"ESCO embeddings shape: {esco_embs.shape}")
        print(f"Scores shape: {scores.shape}")

        # 5. Test save and load
        model_save_dir = os.path.join(tmpdir, "saved_model")
        model.save(model_save_dir)
        print(f"Model saved to {model_save_dir}")

        loaded_model = BiEncoder.load(model_save_dir, config.device)
        print("Model loaded successfully.")

        # Verify loaded model works
        loaded_scores = loaded_model.score(job_embs, esco_embs)
        assert np.allclose(scores, loaded_scores)
        print("Save/load test passed!")

        # Restore original CWD
        os.chdir(original_cwd)
