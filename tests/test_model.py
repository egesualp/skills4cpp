import os
import shutil
import tempfile
import pytest
import torch
import numpy as np

from src.config import Config, ModelConfig, DataConfig, InferConfig, ArtifactsConfig
from src.model import BiEncoder


@pytest.fixture(scope="module")
def temp_dir():
    """Create a temporary directory for test artifacts."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@pytest.fixture(scope="module")
def dummy_data_files(temp_dir):
    """Create dummy data files required by DataConfig."""
    pairs_path = os.path.join(temp_dir, "pairs.jsonl")
    esco_path = os.path.join(temp_dir, "esco.jsonl")
    with open(pairs_path, "w") as f:
        f.write("{}\\n")
    with open(esco_path, "w") as f:
        f.write("{}\\n")
    return pairs_path, esco_path


@pytest.fixture
def base_config(dummy_data_files):
    """A base config that can be modified by other fixtures."""
    pairs_path, esco_path = dummy_data_files
    return Config(
        seed=42,
        device="cpu",
        model=ModelConfig(
            hf_id="sentence-transformers/all-MiniLM-L6-v2",
            proj_dim=None,
            asymmetric=False,
            normalize_output=True,
        ),
        data=DataConfig(pairs_path=pairs_path, esco_titles_path=esco_path),
        infer=InferConfig(batch_size=32, topk=5),
        artifacts=ArtifactsConfig(run_dir="runs/test"),
    )


def test_initialization(base_config):
    """Test that the BiEncoder model can be initialized correctly."""
    model = BiEncoder(base_config.model, base_config.device)
    assert model is not None
    assert model.job_proj is None, "Projection should be None when proj_dim is not set"


def test_encoding_no_projection(base_config):
    """Test encoding shapes without a projection head."""
    model = BiEncoder(base_config.model, base_config.device)
    job_titles = ["Software Engineer", "Data Scientist"]
    esco_skills = ["Develops software", "Analyzes data"]

    job_embs = model.encode_job(job_titles)
    esco_embs = model.encode_esco(esco_skills)

    assert job_embs.shape == (2, 384), "Job embedding shape is incorrect"
    assert esco_embs.shape == (2, 384), "ESCO embedding shape is incorrect"
    assert job_embs.dtype == np.float32
    assert esco_embs.dtype == np.float32


def test_encoding_with_symmetric_projection(base_config):
    """Test encoding shapes with a symmetric projection head."""
    base_config.model.proj_dim = 128
    model = BiEncoder(base_config.model, base_config.device)

    assert model.job_proj is not None, "Job projection layer should exist"
    assert model.esco_proj is model.job_proj, "ESCO projection should be the same as job projection"

    job_titles = ["Software Engineer"]
    job_embs = model.encode_job(job_titles)
    assert job_embs.shape == (1, 128), "Projected job embedding shape is incorrect"


def test_encoding_with_asymmetric_projection(base_config):
    """Test encoding shapes with an asymmetric projection head."""
    base_config.model.proj_dim = 64
    base_config.model.asymmetric = True
    model = BiEncoder(base_config.model, base_config.device)

    assert model.job_proj is not None
    assert model.esco_proj is not None
    assert model.esco_proj is not model.job_proj, "Projection layers should be different"

    job_titles = ["Software Engineer"]
    esco_skills = ["Develops software"]
    job_embs = model.encode_job(job_titles)
    esco_embs = model.encode_esco(esco_skills)

    assert job_embs.shape == (1, 64)
    assert esco_embs.shape == (1, 64)


def test_scoring(base_config):
    """Test that the score matrix has the correct shape."""
    model = BiEncoder(base_config.model, base_config.device)
    job_embs = np.random.rand(3, 384).astype(np.float32)
    esco_embs = np.random.rand(5, 384).astype(np.float32)

    scores = model.score(job_embs, esco_embs)
    assert scores.shape == (3, 5), "Score matrix shape is incorrect"


def test_save_and_load(base_config, temp_dir):
    """Test that saving and loading a model preserves its functionality."""
    base_config.model.proj_dim = 128
    base_config.model.asymmetric = True
    model = BiEncoder(base_config.model, base_config.device)
    model.eval()  # Set to evaluation mode

    # Define save directory
    save_path = os.path.join(temp_dir, "saved_model")

    # Save the model
    model.save(save_path)

    # Check that all artifacts were created
    assert os.path.exists(os.path.join(save_path, "st_model"))
    assert os.path.exists(os.path.join(save_path, "job_proj.pt"))
    assert os.path.exists(os.path.join(save_path, "esco_proj.pt"))
    assert os.path.exists(os.path.join(save_path, "model_config.json"))

    # Load the model
    loaded_model = BiEncoder.load(save_path, base_config.device)
    loaded_model.eval()

    # Test that the loaded model produces the same outputs
    job_titles = ["Senior Python Developer", "UX Designer"]
    
    with torch.no_grad():
        original_embs = model.encode_job(job_titles)
        loaded_embs = loaded_model.encode_job(job_titles)

    assert np.allclose(original_embs, loaded_embs, atol=1e-6), "Loaded model produces different embeddings"

    # Test that scores are also identical
    esco_skills = ["Writes Python code", "Designs user interfaces"]
    original_scores = model.score(original_embs, model.encode_esco(esco_skills))
    loaded_scores = loaded_model.score(loaded_embs, loaded_model.encode_esco(esco_skills))

    assert np.allclose(original_scores, loaded_scores, atol=1e-6), "Loaded model produces different scores"


def test_normalization_flag(base_config):
    """Test that the normalize_output flag works correctly."""
    # Test with normalization ON
    base_config.model.proj_dim = None # No projection for simplicity
    model_norm = BiEncoder(base_config.model, base_config.device)
    
    job_titles = ["Data Scientist"]
    job_embs_norm = model_norm.encode_job(job_titles, normalize=True)
    
    # Check that the L2 norm is close to 1
    norm = np.linalg.norm(job_embs_norm, axis=1)
    assert np.allclose(norm, 1.0), "Embeddings should be normalized"

    # Test with normalization OFF
    model_no_norm = BiEncoder(base_config.model, base_config.device)

    job_embs_no_norm = model_no_norm.encode_job(job_titles, normalize=False)

    # Check that the L2 norm is NOT 1
    norm_unnormalized = np.linalg.norm(job_embs_no_norm, axis=1)
    assert not np.allclose(norm_unnormalized, 1.0), "Embeddings should not be normalized"
