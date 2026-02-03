import argparse
import json
from pathlib import Path

import pandas as pd
import typer
from loguru import logger

from src.JobToESCO.data import build_job_text


def main(
    job_data_path: Path = typer.Option(
        ..., "--job-data-path", "-i", help="Path to the job dataset (e.g., decorte_test.csv)"
    ),
    out_path: Path = typer.Option(
        ..., "--out-path", "-o", help="Path to save the mapping JSON file"
    ),
    text_fields: str = typer.Option("title+desc", help="Text fields to build job text from"),
    is_structured: bool = typer.Option(False, help="Whether to use structured text for job text"),
):
    """
    Generates a JSON mapping from job_id (e.g., 'job_0') to the original job text.
    """
    logger.info(f"Loading job data from: {job_data_path}")
    if not job_data_path.is_file():
        logger.error(f"Job data file not found at: {job_data_path}")
        raise typer.Exit(code=1)

    job_df = pd.read_csv(job_data_path)

    # Recreate the job_id from the index
    job_df_with_idx = job_df.reset_index().rename(columns={"index": "original_job_idx"})
    job_df_with_idx["job_id"] = job_df_with_idx["original_job_idx"].apply(
        lambda x: f"job_{x}"
    )

    job_id_to_text_map = {}
    logger.info("Building job_id to job_text mapping...")
    for _, row in job_df_with_idx.iterrows():
        job_id = row["job_id"]
        job_text = build_job_text(
            row, text_fields=text_fields, is_structured=is_structured
        )
        job_id_to_text_map[job_id] = job_text

    logger.info(f"Saving mapping for {len(job_id_to_text_map)} jobs to: {out_path}")
    with open(out_path, "w") as f:
        json.dump(job_id_to_text_map, f, indent=2)

    logger.success("Mapping file created successfully!")


if __name__ == "__main__":
    typer.run(main)
























