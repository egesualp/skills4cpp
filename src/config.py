"""Configuration management with YAML loading and validation."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml
from box import Box


@dataclass
class ModelConfig:
    """Model configuration."""
    hf_id: str
    proj_dim: Optional[int]
    asymmetric: bool
    normalize_output: bool


@dataclass
class DataConfig:
    """Data configuration with path validation."""
    pairs_path: str | list[str]
    esco_titles_path: str

    def __post_init__(self):
        """Validate that data paths exist."""
        if isinstance(self.pairs_path, list):
            for p in self.pairs_path:
                self._validate_path(p, "pairs_path")
        else:
            self._validate_path(self.pairs_path, "pairs_path")
        self._validate_path(self.esco_titles_path, "esco_titles_path")

    def _validate_path(self, path: str, field_name: str):
        """Validate that a path exists and raise helpful error if not."""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Configuration error: {field_name}='{path}' does not exist. "
                f"Please ensure the file exists or update the path in your config file."
            )


@dataclass
class InferConfig:
    """Inference configuration."""
    batch_size: int
    topk: int


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    batch_size: int
    topk: int
    cache_dir: str  # where to save index/embeddings/metrics
    use_faiss: bool = True
    save_predictions: bool = True
    save_embeddings: bool = False


@dataclass
class ArtifactsConfig:
    """Artifacts configuration."""
    run_dir: str


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""
    enabled: bool = False
    project: Optional[str] = None
    entity: Optional[str] = None
    name: Optional[str] = None


@dataclass
class Config:
    """Main configuration object."""
    seed: int
    device: str
    model: ModelConfig
    data: DataConfig
    infer: InferConfig
    eval: EvalConfig
    artifacts: ArtifactsConfig
    wandb: WandbConfig


def load_config(path: str) -> Box:
    """
    Load configuration from a YAML file.

    Args:
        path: The path to the YAML configuration file.

    Returns:
        A Box object containing the loaded configuration.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(config_path, "r", encoding="utf-8") as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file: {e}")

    return Box(config_dict)


def dump_config(config: Config, out_dir: str) -> str:
    """
    Dump configuration object to YAML file in the specified directory.

    Args:
        config: Config object to dump
        out_dir: Directory to save the config file in

    Returns:
        Path to the created config file

    Raises:
        OSError: If output directory cannot be created or written to
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Create filename with timestamp
    config_file = out_path / "config.yaml"

    # Convert config to dictionary
    config_dict = {
        'seed': config.seed,
        'device': config.device,
        'model': {
            'hf_id': config.model.hf_id,
            'proj_dim': config.model.proj_dim,
            'asymmetric': config.model.asymmetric,
            'normalize_output': config.model.normalize_output
        },
        'data': {
            'pairs_path': config.data.pairs_path,
            'esco_titles_path': config.data.esco_titles_path
        },
        'infer': {
            'batch_size': config.infer.batch_size,
            'topk': config.infer.topk
        },
        'eval': {
            'batch_size': config.eval.batch_size,
            'topk': config.eval.topk,
            'use_faiss': config.eval.use_faiss,
            'cache_dir': config.eval.cache_dir,
            'save_predictions': config.eval.save_predictions,
            'save_embeddings': config.eval.save_embeddings,
        },
        'artifacts': {
            'run_dir': config.artifacts.run_dir
        },
        'wandb': {
            'enabled': config.wandb.enabled,
            'project': config.wandb.project,
            'entity': config.wandb.entity,
            'name': config.wandb.name,
        }
    }

    # Write to YAML file
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    return str(config_file)
