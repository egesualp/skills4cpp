#!/usr/bin/env python3
"""
Script to test Hugging Face sentence transformer models iteratively.
Tests model loading, encoding functionality, and basic operations.
Logs errors and continues with the next model if one fails.
"""

import logging
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import traceback
import sys
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import torch
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install required packages: pip install sentence-transformers torch numpy")
    sys.exit(1)


class SentenceTransformerTester:
    """Test suite for sentence transformer models."""
    
    def __init__(self, log_file: str = "model_test_results.log", results_file: str = "test_results.json"):
        """Initialize the tester with logging configuration."""
        self.log_file = log_file
        self.results_file = results_file
        self.results = {
            "test_timestamp": datetime.now().isoformat(),
            "successful_models": [],
            "failed_models": [],
            "test_summary": {}
        }
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Test sentences for encoding
        self.test_sentences = [
            "This is a test sentence for encoding.",
            "Machine learning models process natural language text.",
            "Python programming is used for data science.",
            "Artificial intelligence transforms various industries."
        ]
        
    def test_model(self, model_name: str, device: str = None) -> Dict[str, Any]:
        """
        Test a single sentence transformer model.
        
        Args:
            model_name: Name of the Hugging Face model to test
            device: Device to load model on ('cpu', 'cuda', etc.)
            
        Returns:
            Dictionary containing test results
        """
        test_result = {
            "model_name": model_name,
            "success": False,
            "error": None,
            "load_time": None,
            "encoding_time": None,
            "model_info": {},
            "test_details": {}
        }
        
        self.logger.info(f"Testing model: {model_name}")
        
        try:
            # Test 1: Model loading
            start_time = time.time()
            model = SentenceTransformer(model_name, device=device)
            load_time = time.time() - start_time
            test_result["load_time"] = load_time
            
            self.logger.info(f"✓ Model loaded successfully in {load_time:.2f}s")
            
            # Get model info
            test_result["model_info"] = {
                "max_seq_length": getattr(model, 'max_seq_length', 'Unknown'),
                "device": str(model.device) if hasattr(model, 'device') else 'Unknown',
                "embedding_dimension": None  # Will be filled after encoding
            }
            
            # Test 2: Basic encoding
            start_time = time.time()
            embeddings = model.encode(self.test_sentences)
            encoding_time = time.time() - start_time
            test_result["encoding_time"] = encoding_time
            
            self.logger.info(f"✓ Encoding completed in {encoding_time:.2f}s")
            
            # Test 3: Verify embedding properties
            if isinstance(embeddings, np.ndarray):
                embedding_shape = embeddings.shape
                test_result["model_info"]["embedding_dimension"] = embedding_shape[1]
                test_result["test_details"]["embedding_shape"] = embedding_shape
                test_result["test_details"]["embedding_dtype"] = str(embeddings.dtype)
                
                self.logger.info(f"✓ Embeddings shape: {embedding_shape}")
            else:
                raise ValueError(f"Expected numpy array, got {type(embeddings)}")
            
            # Test 4: Similarity computation
            if len(embeddings) >= 2:
                similarity = np.dot(embeddings[0], embeddings[1]) / (
                    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
                )
                test_result["test_details"]["sample_similarity"] = float(similarity)
                self.logger.info(f"✓ Similarity computation successful: {similarity:.4f}")
            
            # Test 5: Single sentence encoding
            single_embedding = model.encode("Single test sentence")
            if isinstance(single_embedding, np.ndarray) and single_embedding.shape[0] == embedding_shape[1]:
                test_result["test_details"]["single_encoding_shape"] = single_embedding.shape
                self.logger.info(f"✓ Single sentence encoding successful")
            else:
                raise ValueError(f"Single encoding failed, unexpected shape: {single_embedding.shape}")
            
            test_result["success"] = True
            self.logger.info(f"✅ All tests passed for {model_name}")
            
        except Exception as e:
            error_msg = f"Error testing {model_name}: {str(e)}"
            test_result["error"] = error_msg
            test_result["traceback"] = traceback.format_exc()
            self.logger.error(error_msg)
            self.logger.debug(f"Full traceback:\n{traceback.format_exc()}")
        
        finally:
            # Clean up model from memory
            try:
                if 'model' in locals():
                    del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
        
        return test_result
    
    def test_models(self, model_list: List[str], device: str = None) -> None:
        """
        Test multiple sentence transformer models.
        
        Args:
            model_list: List of Hugging Face model names to test
            device: Device to load models on
        """
        self.logger.info(f"Starting tests for {len(model_list)} models")
        self.logger.info(f"Device: {device or 'auto'}")
        
        for i, model_name in enumerate(model_list, 1):
            self.logger.info(f"\n--- Testing model {i}/{len(model_list)}: {model_name} ---")
            
            result = self.test_model(model_name, device=device)
            
            if result["success"]:
                self.results["successful_models"].append(result)
                self.logger.info(f"✅ {model_name}: SUCCESS")
            else:
                self.results["failed_models"].append(result)
                self.logger.error(f"❌ {model_name}: FAILED")
        
        # Generate summary
        self._generate_summary()
        self._save_results()
    
    def _generate_summary(self) -> None:
        """Generate test summary."""
        total_models = len(self.results["successful_models"]) + len(self.results["failed_models"])
        successful_count = len(self.results["successful_models"])
        failed_count = len(self.results["failed_models"])
        
        self.results["test_summary"] = {
            "total_models_tested": total_models,
            "successful_models": successful_count,
            "failed_models": failed_count,
            "success_rate": successful_count / total_models if total_models > 0 else 0,
            "average_load_time": self._calculate_average_load_time(),
            "average_encoding_time": self._calculate_average_encoding_time()
        }
        
        self.logger.info("\n" + "="*50)
        self.logger.info("TEST SUMMARY")
        self.logger.info("="*50)
        self.logger.info(f"Total models tested: {total_models}")
        self.logger.info(f"Successful: {successful_count}")
        self.logger.info(f"Failed: {failed_count}")
        self.logger.info(f"Success rate: {self.results['test_summary']['success_rate']:.2%}")
        
        if successful_count > 0:
            self.logger.info(f"Average load time: {self.results['test_summary']['average_load_time']:.2f}s")
            self.logger.info(f"Average encoding time: {self.results['test_summary']['average_encoding_time']:.2f}s")
        
        if failed_count > 0:
            self.logger.info("\nFailed models:")
            for failed_model in self.results["failed_models"]:
                self.logger.info(f"  - {failed_model['model_name']}: {failed_model['error']}")
    
    def _calculate_average_load_time(self) -> float:
        """Calculate average load time for successful models."""
        load_times = [m["load_time"] for m in self.results["successful_models"] if m["load_time"]]
        return sum(load_times) / len(load_times) if load_times else 0
    
    def _calculate_average_encoding_time(self) -> float:
        """Calculate average encoding time for successful models."""
        encoding_times = [m["encoding_time"] for m in self.results["successful_models"] if m["encoding_time"]]
        return sum(encoding_times) / len(encoding_times) if encoding_times else 0
    
    def _save_results(self) -> None:
        """Save results to JSON file."""
        try:
            with open(self.results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            self.logger.info(f"\nResults saved to: {self.results_file}")
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")


def main():
    """Main function to run the tests."""
    # Example list of popular sentence transformer models
    # You can modify this list with your desired models
    default_models = [
        "all-MiniLM-L6-v2",
        "all-MiniLM-L12-v2", 
        "all-mpnet-base-v2",
        "paraphrase-MiniLM-L6-v2",
        "paraphrase-albert-small-v2",
        "distiluse-base-multilingual-cased",
        "sentence-transformers/all-roberta-large-v1"
    ]
    
    # You can also load models from a file
    # Example: models = load_models_from_file("model_list.txt")
    
    # Determine device
    device = None
    if torch.cuda.is_available():
        device = "cuda"
        print(f"CUDA available, using GPU")
    else:
        device = "cpu"
        print(f"CUDA not available, using CPU")
    
    # Initialize tester
    tester = SentenceTransformerTester()
    
    # Run tests
    print(f"Testing {len(default_models)} sentence transformer models...")
    tester.test_models(default_models, device=device)
    
    print(f"\nTesting complete! Check {tester.log_file} and {tester.results_file} for detailed results.")


def load_models_from_file(file_path: str) -> List[str]:
    """
    Load model names from a text file (one model per line).
    
    Args:
        file_path: Path to file containing model names
        
    Returns:
        List of model names
    """
    try:
        with open(file_path, 'r') as f:
            models = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        return models
    except FileNotFoundError:
        print(f"Model list file not found: {file_path}")
        return []


if __name__ == "__main__":
    main()
