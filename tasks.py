import os
import pandas as pd
from invoke import Context, task

from src.utils import load_raw_to_esco_pairs

WINDOWS = os.name == "nt"
PROJECT_NAME = "skills4cpp"
PYTHON_VERSION = "3.11"

# Setup commands
@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )

@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task(requirements)
def dev_requirements(ctx: Context) -> None:
    """Install development requirements."""
    ctx.run('pip install -e .["dev"]', echo=True, pty=not WINDOWS)

# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"python src/{PROJECT_NAME}/data.py data/raw data/processed", echo=True, pty=not WINDOWS)

@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)

@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m -i", echo=True, pty=not WINDOWS)

@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )

# Documentation commands
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)


@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)


@task
def prepare_title_pairs(ctx: Context) -> None:
    """
    Loads raw job titles and their corresponding ESCO URIs from the DECORTE
    and Karrierewege+ datasets, then saves them as CSV files.
    """
    output_dir = "data/title_pairs"
    os.makedirs(output_dir, exist_ok=True)

    def save_pairs(pairs, output_path):
        """Saves a list of pairs to a CSV file."""
        df = pd.DataFrame(pairs, columns=['raw_title', 'esco_id'])
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} unique pairs to {output_path}")

    # --- Process DECORTE Dataset ---
    print("Processing DECORTE dataset...")
    decorte_train, decorte_val, decorte_test = load_raw_to_esco_pairs('decorte')

    save_pairs(decorte_train, os.path.join(output_dir, "decorte_train_pairs.csv"))
    save_pairs(decorte_val, os.path.join(output_dir, "decorte_val_pairs.csv"))
    save_pairs(decorte_test, os.path.join(output_dir, "decorte_test_pairs.csv"))
    print("-" * 20)

    # --- Process Karrierewege+ Dataset ---
    print("Processing Karrierewege+ dataset...")
    kw_source = "occ"
    kw_train, kw_val, kw_test = load_raw_to_esco_pairs('karrierewege_plus', kw_source=kw_source)

    save_pairs(kw_train, os.path.join(output_dir, "karrierewege_plus_occ_train_pairs.csv"))
    save_pairs(kw_val, os.path.join(output_dir, "karrierewege_plus_occ_val_pairs.csv"))
    save_pairs(kw_test, os.path.join(output_dir, "karrierewege_plus_occ_test_pairs.csv"))
    print("-" * 20)

    kw_source = "cp"
    kw_train, kw_val, kw_test = load_raw_to_esco_pairs('karrierewege_plus', kw_source=kw_source)
    save_pairs(kw_train, os.path.join(output_dir, "karrierewege_plus_cp_train_pairs.csv"))
    save_pairs(kw_val, os.path.join(output_dir, "karrierewege_plus_cp_val_pairs.csv"))
    save_pairs(kw_test, os.path.join(output_dir, "karrierewege_plus_cp_test_pairs.csv"))
    print("-" * 20)


@task
def sanity_check_pairs(ctx: Context, file_path: str = "data/title_pairs/decorte_train_pairs.csv") -> None:
    """
    Performs a sanity check on a given CSV file of title pairs.

    Args:
        file_path (str): The path to the CSV file to check.
    """
    print(f"--- Running Sanity Check on: {file_path} ---")

    # 1. Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at '{file_path}'")
        print("Please run 'invoke prepare_title_pairs' first to generate the data.")
        return

    # 2. Check if the file can be loaded as a DataFrame
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error: Failed to read or parse the CSV file. Details: {e}")
        return

    # 3. Check for expected columns
    expected_columns = ['raw_title', 'esco_id']
    if not all(col in df.columns for col in expected_columns):
        print(f"Error: Missing expected columns. Found: {list(df.columns)}, Expected: {expected_columns}")
        return

    # 4. Check if the DataFrame is empty
    if df.empty:
        print("Warning: The file is empty. No data pairs found.")
        return

    # 5. Print summary and head
    print(f"✅ Sanity check passed!")
    print(f"Total pairs found: {len(df)}")
    print("Top 5 pairs:")
    print(df.head())
    print("-" * 20)
