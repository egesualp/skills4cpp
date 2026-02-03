# LLM Data Augmentation

This directory contains scripts for augmenting datasets using Large Language Models (specifically OpenAI's GPT models).

## `augment_data.py`

This is the main script for batch processing checks or text generation using the OpenAI API. It is designed to be **flexible and configuration-driven**, allowing you to switch between different augmentation tasks by modifying the configuration section at the top of the file.

### Logic
The script performs the following steps:
1.  **Loads Data**: Reads an input CSV/Parquet/JSON file into a Pandas DataFrame.
2.  **Renders Prompts**: Uses Jinja2 templates to create a specific prompt for each row in the dataset (e.g., "Given job title {{ title }}, describe skills...").
3.  **Async API Calls**: Sends requests to the OpenAI API asynchronously (using `asyncio`) to process multiple rows in parallel for speed.
4.  **Saves Progress**: periodically saves the output during execution to prevent data loss.

### Usage

To run a specific augmentation task (e.g., generating skill descriptions for ESCO occupations, or processing the Decorte dataset), you simply **edit the Configuration section** in `augment_data.py`.

1.  **Open** `src/llm_augmentation/augment_data.py`.
2.  **Locate** the `--- CONFIGURATION START ---` block.
3.  **Uncomment** the block corresponding to your target dataset (and comment out others).
    *   **Input/Output**: `INPUT_PATH`, `OUTPUT_DIR`, `OUTPUT_FILENAME`.
    *   **Prompt**: `PROMPT_TEMPLATE` (Jinja2 format).
    *   **Columns**: `OUTPUT_COLUMN_KEY` (where to save result), `KEY_COLUMN` (unique ID).
4.  **Run** the script:
    ```bash
    export OPENAI_API_KEY="your-key-here"
    python src/llm_augmentation/augment_data.py
    ```

### Configuration Examples

The file currently contains presets for:
*   **ESCO Occupations**: Generating descriptions from titles/synonyms.
*   **Decorte Dataset**: Generating skill briefs from raw job titles and descriptions.
*   **Skills**: Generating job roles that require specific skills.
*   **TalentCLEF**: Validation data augmentation.

### Key Parameters
*   `MAX_CONCURRENT_REQUESTS`: Controls how many API calls are made in parallel (default: 20).
*   `LIMIT_ROWS`: Set to an integer (e.g., `10`) to test on a small subset before running the full dataset.
