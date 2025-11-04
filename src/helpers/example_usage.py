#!/usr/bin/env python3
"""
Example usage of the SentenceTransformerTester.
Shows different ways to test models.
"""

from test_sentence_transformers import SentenceTransformerTester, load_models_from_file
import torch

def test_custom_model_list():
    """Test models from a custom list."""
    # Your custom model list
    custom_models = [
        "all-MiniLM-L6-v2",
        "paraphrase-MiniLM-L6-v2",
        "all-mpnet-base-v2"
    ]
    
    # Initialize tester with custom file names
    tester = SentenceTransformerTester(
        log_file="custom_test_results.log",
        results_file="custom_test_results.json"
    )
    
    # Run tests
    tester.test_models(custom_models)

def test_models_from_file():
    """Test models loaded from a file."""
    # Load models from file
    models = load_models_from_file("example_model_list.txt")
    
    if not models:
        print("No models found in file!")
        return
    
    # Initialize tester
    tester = SentenceTransformerTester(
        log_file="file_based_test_results.log", 
        results_file="file_based_test_results.json"
    )
    
    # Run tests
    tester.test_models(models)

def test_single_model():
    """Test a single model."""
    tester = SentenceTransformerTester()
    
    # Test just one model
    result = tester.test_model("all-MiniLM-L6-v2")
    
    if result["success"]:
        print(f"✅ Model test successful!")
        print(f"Load time: {result['load_time']:.2f}s")
        print(f"Encoding time: {result['encoding_time']:.2f}s")
        print(f"Embedding dimension: {result['model_info']['embedding_dimension']}")
    else:
        print(f"❌ Model test failed: {result['error']}")

if __name__ == "__main__":
    print("Choose test method:")
    print("1. Test custom model list")
    print("2. Test models from file")
    print("3. Test single model")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        test_custom_model_list()
    elif choice == "2":
        test_models_from_file()
    elif choice == "3":
        test_single_model()
    else:
        print("Invalid choice. Running default test...")
        test_custom_model_list()
