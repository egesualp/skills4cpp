import argparse
import logging
import os
from pathlib import Path
from typing import List, Union, Dict

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from jinja2 import Template
from huggingface_hub import snapshot_download

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_input_text(
    df: pd.DataFrame, 
    columns: List[str] = None,
    exclude_columns: List[str] = None,
    template_str: str = None,
    print_sample: bool = True
) -> List[str]:
    """
    Combines specified columns into a single text string for each row using Jinja2 templates.
    
    Args:
        df: Input DataFrame.
        columns: List of column names to use in the template. If None, uses all columns.
        exclude_columns: List of column names to exclude (only used if columns is None).
        template_str: Jinja2 template string. If None, uses a simple space-separated format.
                     Template receives each column as a variable (e.g., {{ column_name }}).
        print_sample: Whether to print a sample of the raw text output.
        
    Returns:
        List of combined text strings.
        
    Example template strings:
        - "{{ col1 }} {{ col2 }}" (simple concatenation)
        - "Title: {{ title }}\nDescription: {{ description }}" (structured)
        - "{{ skill }} ({{ level }})" (with parentheses)
    """
    # Determine which columns to use
    if columns is None:
        # Use all columns, optionally excluding some
        columns = list(df.columns)
        if exclude_columns:
            columns = [col for col in columns if col not in exclude_columns]
            logger.info(f"Using all columns except: {exclude_columns}")
    else:
        # Verify specified columns exist
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found in DataFrame: {missing_cols}")
    
    logger.info(f"Selected columns for text generation: {columns}")
    
    # Default template: space-separated values
    if template_str is None:
        template_str = " ".join(f"{{{{ {col} }}}}" for col in columns)
    
    # Compile Jinja2 template
    template = Template(template_str)
    
    # Fill NaN with empty string to avoid "nan" in text
    df_filled = df[columns].fillna("")
    
    # Combine columns using template
    combined_text = []
    for _, row in df_filled.iterrows():
        # Create context dict for template
        context = {col: str(row[col]) if row[col] != "" else "" for col in columns}
        rendered = template.render(**context)
        combined_text.append(rendered)
    
    # Print sample if requested
    if print_sample and combined_text:
        logger.info("=" * 80)
        logger.info("SAMPLE RAW TEXT OUTPUT:")
        logger.info("-" * 80)
        logger.info(f"{combined_text[0]}")
        logger.info("=" * 80)
    
    return combined_text

def extract_features(
    input_path: str,
    output_path: str,
    model_name: str,
    input_columns: List[str] = None,
    exclude_columns: List[str] = None,
    batch_size: int = 32,
    device: str = None,
    checkpoint_subfolder: str = None,
    id_column: str = None,
    template: str = None
):
    """
    Extracts features using a Sentence Transformer model and saves them to disk.
    
    Args:
        input_columns: List of columns to use. If None, uses all columns.
        exclude_columns: List of columns to exclude (only used if input_columns is None).
        template: Optional Jinja2 template string for formatting input text.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Load Data
    logger.info(f"Loading data from {input_path}")
    try:
        if input_path.endswith('.csv'):
            df = pd.read_csv(input_path)
        elif input_path.endswith('.parquet'):
            df = pd.read_parquet(input_path)
        else:
            # Fallback to csv
            df = pd.read_csv(input_path)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise

    logger.info(f"Data loaded: {len(df)} rows")

    # Prepare Text
    logger.info(f"Preparing input text from columns: {input_columns if input_columns else 'all columns'}")
    if template:
        logger.info(f"Using custom Jinja2 template: {template}")
    texts = prepare_input_text(
        df, 
        columns=input_columns, 
        exclude_columns=exclude_columns,
        template_str=template, 
        print_sample=True
    )

    # Load Model
    logger.info(f"Loading model: {model_name}")
    try:
        if 'bge' in model_name or checkpoint_subfolder is not None:
            # 1. Define the repo and the specific subfolder you want

            # 2. Download *only* the contents of that subfolder
            # This downloads the files to your local cache and returns the path to the main snapshot
            snapshot_path = snapshot_download(
                repo_id=model_name,
                allow_patterns=[f"{checkpoint_subfolder}/*"]  # This downloads only the checkpoint files
            )

            # 3. Create the full local path to the model files
            # The files are inside the checkpoint folder within the snapshot
            model_path = os.path.join(snapshot_path, checkpoint_subfolder)

            # 4. Now, load the model from the *local path*
            print(f"Loading model from local path: {model_path}")
            model = SentenceTransformer(model_path)
        else:
            model = SentenceTransformer(model_name, device=device)
            if 'BERT' in model_name:
                model = SentenceTransformer(modules=[model[0], model[1]], device=device)
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise

    # Encode
    logger.info(f"Starting encoding with batch size {batch_size}")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_tensor=True,
        device=device
    )

    # Prepare output dictionary
    output_dict = {
        'embeddings': embeddings.cpu() if device != 'cpu' else embeddings
    }
    
    if id_column:
        if id_column in df.columns:
            output_dict['ids'] = df[id_column].tolist()
        else:
            logger.warning(f"ID column '{id_column}' not found in data. Skipping IDs.")

    # Save
    logger.info(f"Saving features to {output_path}")
    try:
        torch.save(output_dict, output_path)
        logger.info("Successfully saved features.")
    except Exception as e:
        logger.error(f"Failed to save features: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Extract features using Sentence Transformer")
    
    parser.add_argument(
        "--input_path", 
        type=str, 
        required=True, 
        help="Path to input CSV file"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True, 
        help="Path to save the output .pt file"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="pj-mathematician/JobSkillBGE-large-en-v1.5",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--input_columns", 
        nargs='+', 
        default=None,
        help="List of columns to use as input text. If not specified, uses all columns."
    )
    parser.add_argument(
        "--exclude_columns",
        nargs='+',
        default=None,
        help="List of columns to exclude (only used if --input_columns is not specified)"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32, 
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu", 
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--id_column",
        type=str,
        default="conceptUri",
        help="Column to use as identifier (optional)"
    )
    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Jinja2 template string for formatting input text (e.g., '{{ col1 }} - {{ col2 }}')"
    )

    parser.add_argument(
        "--checkpoint_subfolder",
        type=str,
        default=None,
        help='Checkpoint of the model.'
    )

    args = parser.parse_args()

    extract_features(
        input_path=args.input_path,
        output_path=args.output_path,
        model_name=args.model_name,
        input_columns=args.input_columns,
        exclude_columns=args.exclude_columns,
        batch_size=args.batch_size,
        device=args.device,
        id_column=args.id_column,
        template=args.template,
        checkpoint_subfolder=args.checkpoint_subfolder
    )

if __name__ == "__main__":
    main()

