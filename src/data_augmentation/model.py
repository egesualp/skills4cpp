import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    BitsAndBytesConfig,
    pipeline
)
from typing import List, Dict, Union

# --- CONFIGURATION ---
# Use an open model that doesn't require gated access and fits in Mac memory
LLM_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"  # ~4GB RAM, no gated access
# Alternative options:
# "google/gemma-2-2b-it"  # ~2GB RAM, requires HF token but easier access
# "microsoft/Phi-3.5-mini-instruct"  # ~4GB RAM, no gated access
# Gated models (require Meta approval): 
# "meta-llama/Llama-3.2-1B-Instruct", "meta-llama/Meta-Llama-3-8B-Instruct" 
# Use a 4-bit quantization config (essential for running on local Mac/limited VRAM)
# 4-bit quantization reduces VRAM usage for the 8B model from ~16GB (FP16) to ~4-6GB
QUANT_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4", 
    bnb_4bit_compute_dtype=torch.float16
)

# --- LLM CLIENT CLASS ---

class LLMClient:
    """
    Client for loading a quantized HuggingFace model and performing batched inference.
    """
    def __init__(self, model_id: str = LLM_MODEL_ID, quant_config: BitsAndBytesConfig = QUANT_CONFIG):
        """
        Initializes tokenizer and model with quantization.
        """
        print(f"Loading tokenizer: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token for batching
        
        # Determine device: 'mps' for Mac, 'cuda' for cluster GPU, 'cpu' as fallback
        if torch.cuda.is_available():
            device = "cuda"
            device_map = 'auto'
            use_quantization = True  # bitsandbytes works with CUDA
        elif torch.backends.mps.is_available():
            device = "mps"
            device_map = "mps"
            use_quantization = False  # bitsandbytes doesn't support MPS
        else:
            device = "cpu"
            device_map = "cpu"
            use_quantization = False  # bitsandbytes on CPU is slow, skip it
            
        print(f"Detected device: {device_map} | Loading model...")
        if use_quantization:
            print("Using 4-bit quantization (bitsandbytes)")
        else:
            print("Loading without quantization (bitsandbytes not available for this device)")

        try:
            # Load the model with or without quantization based on device
            if use_quantization:
                # CUDA: Use 4-bit quantization
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=quant_config,
                    device_map=device_map,
                    dtype=torch.float16,  # Use dtype (torch_dtype is deprecated)
                    trust_remote_code=True  # Required for Phi models
                )
            else:
                # MPS or CPU: Load without quantization, use appropriate dtype
                if device == "mps":
                    # MPS works best with float16
                    model_dtype = torch.float16
                else:
                    # CPU can use bfloat16 or float32
                    model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map=device_map,
                    dtype=model_dtype,  # Use dtype (torch_dtype is deprecated)
                    trust_remote_code=True,  # Required for Phi models
                    low_cpu_mem_usage=True,  # More memory efficient loading
                    use_safetensors=True  # Use safetensors format if available
                )
            
            # Create a HuggingFace pipeline for efficient batched generation
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if device == "cuda" else device
            )
            print("Model loaded successfully.")
            
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load model {model_id} on {device_map}. {e}")
            if device == "mps":
                print("For Mac M-series: bitsandbytes is not supported. The model will load without quantization.")
                print("This requires sufficient RAM (16GB+ recommended for 8B models).")
            else:
                print("This likely requires installing `accelerate` and `bitsandbytes`.")
            self.generator = None

    def generate_batched_descriptions(self, prompts: List[str], batch_size: int = 16) -> List[str]:
        """
        Generates LLM responses for a list of prompts efficiently.
        """
        if not self.generator:
            return ["LLM Error: Generator not available"] * len(prompts)

        # Common generation arguments for structured, concise output
        gen_kwargs = {
            "max_new_tokens": 128,          # Max length for the concise description (e.g., 2-3 sentences)
            "do_sample": False,             # Disable sampling for deterministic augmentation (important for reproducibility)
            "return_full_text": False,      # Only return the generated text
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        
        # Generator handles the chunking and batched inference for efficiency
        results = self.generator(
            prompts,
            batch_size=batch_size,
            **gen_kwargs
        )
        
        # Extract the text from the complex pipeline output structure
        generated_texts = [r[0]['generated_text'].strip() for r in results]
        return generated_texts

# --- EXECUTION CHECK (Optional, for debugging model loading) ---
if __name__ == '__main__':
    print("--- Testing Model Load ---")
    client = LLMClient()
    
    if client.generator:
        test_prompt = client.tokenizer.apply_chat_template([
            {"role": "system", "content": "You are an expert HR assistant. Generate a highly concise job description."},
            {"role": "user", "content": "Job Title: Senior Data Architect. Generate description in 2 sentences."}
        ], tokenize=False, add_generation_prompt=True)
        
        print("\nSending test prompt to model...")
        test_output = client.generate_batched_descriptions([test_prompt], batch_size=1)
        print("Model Output:", test_output[0])