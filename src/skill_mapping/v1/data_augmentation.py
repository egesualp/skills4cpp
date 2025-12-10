import pandas as pd
from tqdm import tqdm
import argparse
import time
import os
import logging
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# ---
# Setup robust logging
# ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ---
# Prompts from the pjmathematician paper (Section 2.2.1)
# ---

# Prompt for generating 'skill_brief' from a job title
JOB_PROMPT_TEMPLATE = """Given a job role (and its synonyms), briefly (1-2 sentences) describe the skills needed for that job.

Job Role: "{title}"
Synonyms: "{synonyms}"
"""

# Prompt for generating 'job_brief' from a skill
SKILL_PROMPT_TEMPLATE = """Given a skill (and its synonyms), briefly (1-2 sentences) describe the job roles that require that skill.

Skill: "{title}"
Synonyms: "{synonyms}"
"""

# System prompt for the Qwen Instruct model
SYSTEM_PROMPT = "You are a helpful assistant, an expert in Human Resources. Please be concise, professional, and factual."


@torch.no_grad()
def generate_batch(model, tokenizer, prompt_list: list[str], batch_size: int) -> list[str]:
    """
    Generates responses for a batch of prompts.
    """
    final_responses = []
    
    for i in tqdm(range(0, len(prompt_list), batch_size), desc="Generating Batches"):
        batch_prompts = prompt_list[i : i + batch_size]
        
        # 1. Create the chat template for each prompt in the batch
        batch_messages = []
        for prompt in batch_prompts:
            batch_messages.append([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ])

        # 2. Apply the template and tokenize
        text_list = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) 
            for m in batch_messages
        ]
        
        model_inputs = tokenizer(
            text_list, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512 # Truncate long prompts (e.g., skills with many aliases)
        ).to(model.device)

        # 3. Generate
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=150, # "1-2 sentences"
            do_sample=False,    # Use greedy decoding for deterministic, factual output
            pad_token_id=tokenizer.eos_token_id # Suppress warnings
        )
        
        # 4. Decode and slice out the prompt
        # This is the robust way to handle batching
        input_token_lengths = [len(x) for x in model_inputs.input_ids]
        generated_ids = [
            generated_ids[j][input_token_lengths[j]:] for j in range(len(generated_ids))
        ]
        
        responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        final_responses.extend([r.strip() for r in responses])
        
    return final_responses

def main():
    parser = argparse.ArgumentParser(description="Local LLM Augmentation Script (pjmathematician)")
    
    parser.add_argument("--task", type=str, required=True, choices=['job', 'skill'])
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--title_col", type=str, required=True)
    parser.add_argument("--synonym_col", type=str, default=None)
    parser.add_argument("--output_col", type=str, required=True)
    
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-32B-Instruct")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for model.generate(). Adjust based on your A100 VRAM.")
    parser.add_argument("--quantize", type=str, default=None, choices=['4bit', '8bit'],
                        help="Optional: Use 4-bit or 8-bit quantization if VRAM is an issue.")

    args = parser.parse_args()
    
    # --- 1. Load Model & Tokenizer ---
    logger.info(f"Loading model: {args.model_name}. This will take time and VRAM...")
    
    quantization_config = None
    if args.quantize == '4bit':
        logger.info("Applying 4-bit quantization (NF4)...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    elif args.quantize == '8bit':
        logger.info("Applying 8-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype="auto",
            device_map="auto", # This is the key for your multi-GPU A100s
            quantization_config=quantization_config,
            trust_remote_code=True # Qwen models often require this
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token # Set pad token for batching
        
    except Exception as e:
        logger.error(f"Failed to load model. Do you have 'trust_remote_code=True'?")
        logger.error(f"Error: {e}")
        if "bitsandbytes" in str(e):
             logger.error("Quantization failed. Try installing 'bitsandbytes': pip install bitsandbytes")
        return

    logger.success("Model loaded successfully.")
    
    # --- 2. Load and Prepare Data ---
    try:
        df = pd.read_csv(args.input_file)
    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input_file}")
        return
        
    logger.info(f"Preparing prompts for task: '{args.task}'...")
    
    if args.task == 'job':
        prompt_template = JOB_PROMPT_TEMPLATE
        df_unique = df.drop_duplicates(subset=[args.title_col])
        titles = df_unique[args.title_col].dropna().tolist()
        synonyms_list = [""] * len(titles) # Job files don't have synonyms
        title_map_key = args.title_col
    else: # 'skill'
        prompt_template = SKILL_PROMPT_TEMPLATE
        df_unique = df.drop_duplicates(subset=[args.title_col])
        titles = df_unique[args.title_col].dropna().tolist()
        if args.synonym_col and args.synonym_col in df_unique.columns:
            synonyms_list = df_unique[args.synonym_col].fillna("").str.replace('\n', ', ').tolist()
        else:
            logger.warning(f"Synonym column '{args.synonym_col}' not found. Using empty synonyms.")
            synonyms_list = [""] * len(titles)
        title_map_key = args.title_col

    # --- 3. Generate All Prompts ---
    prompt_list = []
    for title, synonyms in zip(titles, synonyms_list):
        prompt_list.append(
            prompt_template.format(title=title, synonyms=synonyms)
        )
    
    logger.info(f"Generated {len(prompt_list)} unique prompts.")
    
    # --- 4. Run Batch Inference ---
    start_time = time.time()
    generated_responses = generate_batch(model, tokenizer, prompt_list, args.batch_size)
    end_time = time.time()
    
    duration = end_time - start_time
    responses_per_sec = len(prompt_list) / duration
    
    logger.success(f"Generation complete in {duration:.2f} seconds ({responses_per_sec:.2f} prompts/sec).")
    
    # --- 5. Map and Save ---
    if len(titles) != len(generated_responses):
        logger.error(f"Mismatch in items! Got {len(titles)} titles but {len(generated_responses)} responses.")
        return

    # Create the lookup map
    response_map = {title: response for title, response in zip(titles, generated_responses)}
    
    # Map back to the original (non-unique) dataframe
    df[args.output_col] = df[title_map_key].map(response_map)
    
    df.to_csv(args.output_file, index=False)
    logger.success(f"Successfully saved augmented data to {args.output_file}")

if __name__ == "__main":
    # ---
    # Dependencies for this script:
    # pip install torch pandas tqdm transformers accelerate bitsandbytes
    # ---
    main()