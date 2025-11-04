# Sentence Transformer Model Testing

This directory contains scripts to test Hugging Face sentence transformer models iteratively, with robust error handling and detailed logging.

## Files

- `test_sentence_transformers.py` - Main testing script with comprehensive test suite
- `example_usage.py` - Examples of different ways to use the tester
- `example_model_list.txt` - Sample model list file format

## Features

- **Robust Error Handling**: Catches and logs errors, continues with next model
- **Comprehensive Testing**: Tests model loading, encoding, similarity computation
- **Detailed Logging**: Logs to both file and console with timestamps
- **Performance Metrics**: Tracks load time, encoding time, embedding dimensions
- **Results Export**: Saves detailed results to JSON file
- **Memory Management**: Cleans up models and GPU memory between tests
- **Flexible Input**: Support for hardcoded lists or file-based model lists

## Quick Start

### 1. Basic Usage (Default Models)

```bash
python test_sentence_transformers.py
```

This will test a default set of popular sentence transformer models.

### 2. Custom Model List

Edit the `default_models` list in `test_sentence_transformers.py` or use the programmatic approach:

```python
from test_sentence_transformers import SentenceTransformerTester

# Your custom models
models = [
    "all-MiniLM-L6-v2",
    "paraphrase-MiniLM-L6-v2",
    "your-custom-model-name"
]

tester = SentenceTransformerTester()
tester.test_models(models)
```

### 3. Models from File

Create a text file with one model name per line:

```
# my_models.txt
all-MiniLM-L6-v2
paraphrase-MiniLM-L6-v2
all-mpnet-base-v2
```

Then load and test:

```python
from test_sentence_transformers import SentenceTransformerTester, load_models_from_file

models = load_models_from_file("my_models.txt")
tester = SentenceTransformerTester()
tester.test_models(models)
```

## Test Details

Each model undergoes the following tests:

1. **Model Loading**: Attempts to load the model from Hugging Face
2. **Basic Encoding**: Encodes a set of test sentences
3. **Embedding Validation**: Checks embedding shape and data type
4. **Similarity Computation**: Tests cosine similarity between embeddings
5. **Single Sentence Encoding**: Tests encoding of individual sentences

## Output Files

- `model_test_results.log` - Detailed log file with timestamps
- `test_results.json` - Structured results in JSON format

### JSON Results Structure

```json
{
  "test_timestamp": "2024-01-01T12:00:00",
  "successful_models": [
    {
      "model_name": "all-MiniLM-L6-v2",
      "success": true,
      "load_time": 2.34,
      "encoding_time": 0.12,
      "model_info": {
        "embedding_dimension": 384,
        "max_seq_length": 256
      },
      "test_details": {
        "embedding_shape": [4, 384],
        "sample_similarity": 0.1234
      }
    }
  ],
  "failed_models": [
    {
      "model_name": "non-existent-model",
      "success": false,
      "error": "Model not found",
      "traceback": "..."
    }
  ],
  "test_summary": {
    "total_models_tested": 5,
    "successful_models": 4,
    "failed_models": 1,
    "success_rate": 0.8,
    "average_load_time": 2.1,
    "average_encoding_time": 0.15
  }
}
```

## Requirements

- Python 3.7+
- sentence-transformers
- torch
- numpy

Install dependencies:
```bash
pip install sentence-transformers torch numpy
```

## Advanced Usage

### Custom Test Configuration

```python
tester = SentenceTransformerTester(
    log_file="my_test.log",
    results_file="my_results.json"
)

# Test with specific device
tester.test_models(models, device="cuda")  # or "cpu"
```

### Single Model Testing

```python
tester = SentenceTransformerTester()
result = tester.test_model("all-MiniLM-L6-v2")

if result["success"]:
    print(f"✅ Success! Embedding dim: {result['model_info']['embedding_dimension']}")
else:
    print(f"❌ Failed: {result['error']}")
```

## Error Handling

The script handles various error scenarios:

- **Model not found**: Logs error and continues
- **Memory issues**: Clears GPU cache between models
- **Import errors**: Graceful failure with informative messages
- **Encoding failures**: Captures and logs detailed error information

## Performance Notes

- GPU will be used automatically if available
- Models are cleaned from memory between tests
- Large models may take longer to load and test
- Consider testing in smaller batches for very large model lists

## Example Output

```
2024-01-01 12:00:00 - INFO - Testing model: all-MiniLM-L6-v2
2024-01-01 12:00:02 - INFO - ✓ Model loaded successfully in 2.34s
2024-01-01 12:00:02 - INFO - ✓ Encoding completed in 0.12s
2024-01-01 12:00:02 - INFO - ✓ Embeddings shape: (4, 384)
2024-01-01 12:00:02 - INFO - ✓ Similarity computation successful: 0.1234
2024-01-01 12:00:02 - INFO - ✓ Single sentence encoding successful
2024-01-01 12:00:02 - INFO - ✅ All tests passed for all-MiniLM-L6-v2

==================================================
TEST SUMMARY
==================================================
Total models tested: 5
Successful: 4
Failed: 1
Success rate: 80.00%
Average load time: 2.10s
Average encoding time: 0.15s
```
