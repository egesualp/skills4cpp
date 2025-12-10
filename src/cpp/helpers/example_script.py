# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import defaultdict
import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging

import pandas as pd
import importlib
from packaging import version
from packaging.version import parse

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer,
    EarlyStoppingCallback,
    TrainerCallback

)
from datasets import load_dataset, Dataset
#from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Evaluation
from evaluate import load
sari = load("sari")
bleu = load("bleu")
rouge = load("rouge")
bert_score = load("bertscore")
import deepspeed

from typing import List, Optional
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
import torch._dynamo
import glob



if torch.cuda.is_available():   
    torch.backends.cuda.matmul.allow_tf32 = True

#from accelerate.utils import write_basic_config, DeepSpeedPlugin

logging.basicConfig(
    filename="train.log",  # ✅ Save logs to a file
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())  # ✅ Ensures logs appear in console too

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

import sys
print("Command-line arguments:", sys.argv)  # ✅ Shows exactly what `accelerate launch` is passing

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="EleutherAI/pythia-12b"
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    use_fast_tokenizer: Optional[bool] = field(
        default=True,
        metadata={"help": "Use a fast Rust-based tokenizer if it is supported for a given model. If a fast tokenizer is not available for a given model, a normal Python-based tokenizer is returned instead."}
    )


@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf]"}
    )
    
    compute_bleu: Optional[bool] = field(default=False)
    compute_sari: Optional[bool] = field(default=False)
    compute_rouge: Optional[bool] = field(default=False)
    compute_bert_score: Optional[bool] = field(default=False)

@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):

    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='adamw_torch', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=16, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=1, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=10000, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'})
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=40, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})
    torch_compile: bool = field(default=True, metadata={"help": "Enable torch.compile, see docs for details"})
    torch_compile_backend: str = field(default="inductor", metadata={"help": "Choose backend for torch.compile"})
    task: str = field(default='simp', metadata={"help": 'The task basically.'})
    deepspeed: str = field(default=None, metadata={"help": "Path to DeepSpeed config file"})

@dataclass
class GenerationArguments:
    # For more hyperparameters check:
    # https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens : Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)
    max_new_tokens: Optional[int] = field(default=256)
    #max_length: Optional[int] = field(default=None)


def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
    )

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    non_special_tokens = None,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict) + tokenizer.add_tokens(non_special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg
    print(f"Resized tokenizer and embedding to {len(tokenizer)} tokens.")

@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict

def extract_unnatural_instructions_data(examples, extract_reformulations=False):
    out = {
        'input': [],
        'output': [],
    }
    for example_instances in examples['instances']:
        for instance in example_instances:
            out['input'].append(instance['instruction_with_input'])
            out['output'].append(instance['output'])
    if extract_reformulations:
        for example_reformulations in examples['reformulations']:
            if example_reformulations is not None:
                for instance in example_reformulations:
                    out['input'].append(instance['instruction_with_input'])
                    out['output'].append(instance['output'])
    return out

ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

def clean_text(text):
    return text.replace('-RRB-', ')').replace('-LRB-', '(').replace('-RSB-', ']').replace('-LSB-', '[')

def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {'input': prompt_format.format(**example)}

def local_dataset(dataset_name, args):
    if dataset_name.endswith('.json') or dataset_name.endswith('.jsonl'):
        full_dataset = dict()
        if args.do_train:
            train_dataset = Dataset.from_json(path_or_paths=dataset_name, field='train')
            full_dataset['train'] = train_dataset
        if (args.do_eval and not args.do_predict):
            eval_dataset = Dataset.from_json(path_or_paths=dataset_name, field='eval')
            full_dataset['eval'] = eval_dataset
        if args.do_predict:
            test_dataset = Dataset.from_json(path_or_paths=dataset_name, field='test')
            if args.task == 'simp':
                print("Don't worry, filter is done!")
                test_dataset = test_dataset.filter(lambda example: example["src_info"]["prompt_name"] == "simplification_1")
            full_dataset['test'] = test_dataset
            full_dataset['eval'] = test_dataset
    elif dataset_name.endswith('.csv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith('.tsv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter='\t'))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    #split_dataset = full_dataset.train_test_split(test_size=0.1)
    return full_dataset

def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }

    Available datasets to be selected with `dataset` argument:
        - alpaca, 52002 examples
        - alpaca cleaned, 51942 examples
        - chip2 (OIG), 210289 examples
        - self-instruct, 82612 examples
        - hh-rlhf (Anthropic), 160800 examples
        - longform, 23.7k examples
        - oasst1 (OpenAssistant) primary message tree only, 9,846 examples

    Coming soon:
        - unnatural instructions core, 66010 examples
        - unnatural instructions full, 240670 examples
        - alpaca-gpt4, 52002 examples
        - unnatural-instructions-gpt4, 9000 examples
        - supernatural-instructions, 69624 examples (same as paper with 100 ex/task more can be used)
        - flan (FLAN v2), up to 20M examples available
        - vicuna

    """
    def load_data(dataset_name):
        if dataset_name == 'alpaca':
            return load_dataset("tatsu-lab/alpaca")
        elif dataset_name == 'alpaca-clean':
            return load_dataset("yahma/alpaca-cleaned")
        elif dataset_name == 'chip2':
            return load_dataset("laion/OIG", data_files='unified_chip2.jsonl')
        elif dataset_name == 'hh-rlhf':
            return load_dataset("Anthropic/hh-rlhf")
        elif dataset_name == 'longform':
            return load_dataset("akoksal/LongForm")
        elif dataset_name == 'oasst1':
            return load_dataset("timdettmers/openassistant-guanaco")
        elif dataset_name == "OpenAssistant/oasst_top1_2023-08-25":
            return load_dataset("OpenAssistant/oasst_top1_2023-08-25")
        elif dataset_name == 'vicuna':
            raise NotImplementedError("Vicuna data was not released.")
        else:
            if os.path.exists(dataset_name):
                try:
                    args.dataset_format = args.dataset_format if args.dataset_format else "input-output"
                    full_dataset = local_dataset(dataset_name, args)
                    return full_dataset
                except:
                    raise ValueError(f"Error loading dataset from {dataset_name}")
            else:
                raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

    def format_dataset(dataset, dataset_format):
        if (
            dataset_format == 'alpaca' or dataset_format == 'alpaca-clean' or
            (dataset_format is None and args.dataset in ['alpaca', 'alpaca-clean'])
        ):
            dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
        elif dataset_format == 'chip2' or (dataset_format is None and args.dataset == 'chip2'):
            dataset = dataset.map(lambda x: {
                'input': x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''),
                'output': x['text'].split('\n<bot>: ')[1],
            })
        elif dataset_format == 'self-instruct' or (dataset_format is None and args.dataset == 'self-instruct'):
            for old, new in [["prompt", "input"], ["completion", "output"]]:
                dataset = dataset.rename_column(old, new)
        elif dataset_format == 'hh-rlhf' or (dataset_format is None and args.dataset == 'hh-rlhf'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['chosen']
            })
        elif dataset_format == 'oasst1' or (dataset_format is None and args.dataset == 'oasst1'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['text'],
            })
        elif dataset_format == 'prompt':
            if isinstance(dataset, dict):  # If dataset is a DatasetDict (multiple splits)
                for split in dataset.keys():
                    dataset[split] = dataset[split].map(lambda x: {
                'input': f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n{clean_text(x['input'])}\n\n### Response:",
                'output': clean_text(x['output'])
            })

            else:
                dataset = dataset.map(lambda x: {
                    'input': f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n{clean_text(x['input'])}\n\n### Response:",
                    'output': clean_text(x['output'])
                })
        
        elif dataset_format == 'input-output':
            # leave as is
            pass
        # Check if dataset is a DatasetDict (dict-like object with splits)
        if isinstance(dataset, dict):  # DatasetDict or similar
            for split in dataset.keys():
                if split != 'test':
                    dataset[split] = dataset[split].remove_columns(
                        [col for col in dataset[split].column_names if col not in ['input', 'output', 'labels']]
                    )
        else:  # Single Dataset
            if split != 'test':
                dataset = dataset.remove_columns(
                    [col for col in dataset.column_names if col not in ['input', 'output', 'labels']]
                )
        return dataset

     # Load dataset.
    dataset = load_data(args.dataset)
    dataset = format_dataset(dataset, args.dataset_format)
 
    # Split train/eval, reduce size
    if args.do_eval:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        else:
            print('Splitting train dataset in train and validation according to `eval_dataset_size`')
            dataset = dataset["train"].train_test_split(
                test_size=args.eval_dataset_size, shuffle=True, seed=42
            )
            eval_dataset = dataset['test']
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if args.group_by_length:
            eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
    if args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
    if args.do_predict:
        if 'test' in dataset:
            test_dataset = dataset['test']
            # temporary adjsutment
            test_dataset = test_dataset
        if args.max_test_samples is not None and len(test_dataset) > args.max_test_samples:
            test_dataset = test_dataset.select(range(args.max_test_samples))

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    )
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=test_dataset if args.do_predict else None,
        data_collator=data_collator
    )

import os
from os.path import isdir, exists, join

def get_last_checkpoint(checkpoint_dir):
    """Finds the latest checkpoint in a directory."""
    if not isdir(checkpoint_dir):
        return None, False  # First training, no checkpoint dir exists.

    is_completed = exists(join(checkpoint_dir, 'completed'))
    if is_completed:
        return None, True  # Training already finished.

    # Extract valid checkpoint numbers
    checkpoint_steps = [
        int(f.replace("checkpoint-", "")) for f in os.listdir(checkpoint_dir)
        if f.startswith("checkpoint-") and f.replace("checkpoint-", "").isdigit()
    ]

    if not checkpoint_steps:
        return None, is_completed  # Training started but no checkpoints yet.

    # Get the latest checkpoint
    last_checkpoint = join(checkpoint_dir, f"checkpoint-{max(checkpoint_steps)}")
    print(f"Found a previous checkpoint at: {last_checkpoint}")
    return last_checkpoint, is_completed

def get_accelerate_model(args, checkpoint_dir):
    device_map = None if True else "auto" # opt

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        if True:  # Ensure DeepSpeed doesn't use a device_map (opt)
            device_map = {"": local_rank}

    print(f'loading base model {args.model_name_or_path}...')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16,
        trust_remote_code=args.trust_remote_code,
        #low_cpu_mem_usage=False,  # ❌ Disable (DeepSpeed Zero-3 incompatibility)
        #device_map=device_map  # ❌ Remove (Handled by DeepSpeed)
        )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        padding_side="right",
        use_fast=args.use_fast_tokenizer,  # Allows toggling
        trust_remote_code=args.trust_remote_code,
    )
    special_tokens_dict = dict()
    if tokenizer._pad_token is None:
        special_tokens_dict['pad_token'] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

    # Resize tokenizer only if new special tokens are added
    if special_tokens_dict:
        logger.info(f'Adding special_tokens_dict for {special_tokens_dict}')
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=special_tokens_dict,
            tokenizer=tokenizer,
            model=model
        )
        #tokenizer.save_pretrained(args.model_name_or_path)  # ✅ Save updated tokenizer

    return model, tokenizer

def save_args(args, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert args to a dictionary
    args_dict = vars(args)

    # Convert `GenerationConfig` if present
    if isinstance(args_dict.get("generation_config"), transformers.GenerationConfig):
        args_dict["generation_config"] = args_dict["generation_config"].to_dict()

    # Convert `PartialState` and other non-serializable objects to strings
    for key, value in args_dict.items():
        if not isinstance(value, (str, int, float, bool, list, dict, type(None))):
            args_dict[key] = str(value)

    with open(f"{output_dir}/run_config.json", "w") as f:
        json.dump(args_dict, f, indent=4)

    print(f"Arguments saved to {output_dir}/run_config.json")

def log_model_distribution(model):
    """Logs model parameter distribution across GPUs."""
    device_count = torch.cuda.device_count()
    logger.info(f"Number of GPUs Available: {device_count}")

    for name, param in model.named_parameters():
        logger.info(f"Param: {name} -> Device: {param.device}")
import json

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def evaluation_loop(self, *args, **kwargs):
        torch.cuda.empty_cache()  # Free unused GPU memory before evaluation
        with torch.no_grad():  # Ensure no gradients are computed
            return super().evaluation_loop(*args, **kwargs)
    
    def get_train_dataloader(self):
        # Use DistributedSampler if training with multiple GPUs
        sampler = DistributedSampler(self.train_dataset) if self.args.world_size > 1 else None
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,  # Explicitly disable shuffling
            sampler=sampler,
            collate_fn=self.data_collator  # Use your data collator
        )

# Custom callback to delete optimizer states after saving
class CleanupCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        """
        Deletes optimizer checkpoint files from DeepSpeed checkpoints 
        inside `checkpoint-*/global_step*` folders.
        """
        # Get all DeepSpeed checkpoint folders
        checkpoint_dirs = sorted(glob.glob(os.path.join(args.output_dir, "checkpoint-*/global_step*")), 
                                 key=os.path.getctime, reverse=True)

        if not checkpoint_dirs:
            print("No DeepSpeed checkpoints found for cleanup.")
            return control

        latest_checkpoint = checkpoint_dirs[0]  # Get the latest saved checkpoint

        # Build file paths safely
        optimizer_files = [
            os.path.join(latest_checkpoint, "optimizer.pt"),
            *glob.glob(os.path.join(latest_checkpoint, "*_optim_states.pt")),
        ]

        deleted_count = 0
        for file in optimizer_files:
            try:
                os.remove(file)
                print(f"🗑 Deleted: {file}")
                deleted_count += 1
            except Exception as e:
                print(f"⚠️ Error deleting {file}: {e}")

        # Print summary
        if deleted_count > 0:
            print(f"Deleted {deleted_count} optimizer-related files from: {latest_checkpoint}")

        return control
    
def train():
    torch.cuda.empty_cache()  # Clear old GPU allocations
    deepspeed.init_distributed()
    torch._dynamo.config.optimize_ddp = False
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    # Load DeepSpeed config manually
    #training_args.deepspeed = create_ds_config(training_args)  # ✅ Auto-generate DeepSpeed config    
    #ds_plugin = DeepSpeedPlugin(hf_ds_config=training_args.deepspeed)
        
    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        logger.info("Detected that training was already completed!")

    model, tokenizer = get_accelerate_model(args, checkpoint_dir)

    model.config.use_cache = False
    logger.info("Loaded model")
    set_seed(args.seed)
    logger.info(f"Deepspeed config: {args.deepspeed}")

    data_module = make_data_module(tokenizer=tokenizer, args=args)
    task = args.task

    # Saving all arguments
    save_args(args, training_args.output_dir)

    trainer = CustomSeq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k:v for k,v in data_module.items() if k != 'predict_dataset'},
        callbacks=[CleanupCallback()]

        #compute_metrics=lambda p: compute_metrics(
        #    eval_preds=p.predictions,
        #    eval_labels = p.label_ids,
        #    tokenizer=tokenizer,
        #    task=task  # Dynamically fetched from the config
        #)
    )
    logger.info("Trainer is created!")
    logger.info(f"Number of GPUs Trainer sees: {trainer.args.n_gpu}")

    # Verifying the datatypes and parameter counts before training.
    print_trainable_parameters(args, model)
    # Ensure model is unwrapped if DeepSpeed is used
    if hasattr(model, "module"):  
        model = model.module  

    # Check if parameters exist
    dtypes = {}
    for name, p in model.named_parameters():
        dtype = p.dtype
        dtypes[dtype] = dtypes.get(dtype, 0) + p.numel()

    total_params = sum(dtypes.values()) or 1  # Prevents division by zero
    if total_params == 1:
        logger.warning("⚠️ No parameters found. Check if DeepSpeed Zero-3 is offloading the model.")

    for dtype, count in dtypes.items():
        logger.info(f"{dtype}: {count} ({count/total_params:.2%})")

    all_metrics = {"run_name": args.run_name}

    # Training
    if args.do_train:
        logger.info("*** Train ***")
        print(f'device_count = {torch.cuda.device_count()}')
        logger.info(f"Training dataset size: {len(data_module['train_dataset']) if data_module['train_dataset'] else 0}")
        logger.info(f"Evaluation dataset size: {len(data_module['eval_dataset']) if data_module['eval_dataset'] else 0}")
        # Prepare model & optimizer with Accelerator
        #accelerator = Accelerator(deepspeed_plugin=ds_plugin)
        #model, trainer.optimizer = accelerator.prepare(
        #    model, trainer.optimizer
        #)
        

        #trainer.training_step = custom_training_step.__get__(trainer)
        
        ## Log device after model distribution
        #logger.info("Logging model device placement AFTER Accelerator.prepare():")
        #log_model_distribution(model)
        
        # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
        # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)

    # Prediction
    if args.do_predict:
        logger.info("*** Predict ***")
        logger.info(f"Test dataset size: {len(data_module['predict_dataset']) if data_module['predict_dataset'] else 0}")
        if args.do_eval:
            metrics = trainer.evaluate(metric_key_prefix="eval")
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
            all_metrics.update(metrics)

        prediction_output = trainer.predict(
            test_dataset=data_module['predict_dataset'],
            metric_key_prefix="predict",
            eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            pad_token_id=tokenizer.eos_token_id
            )
        prediction_metrics = prediction_output.metrics
        predictions = prediction_output.predictions
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        
        # ✅ Extract predictions without input text more robustly
        #predictions_wo_input = []
        #for i, example in enumerate(data_module["predict_dataset"]):
        #    input_text = example["input"]
#
        #    # 🔹 Strip the input part based on token count instead of replace()
        #    input_tokens = tokenizer(input_text, return_tensors="pt")["input_ids"].shape[1]  # Get token length of input
        #    pred_tokens = tokenizer(predictions[i], return_tensors="pt")["input_ids"]  # Tokenize prediction
#
        #    # 🔹 Extract only the newly generated tokens
        #    generated_tokens = pred_tokens[:, input_tokens:].squeeze(0)  # Remove input tokens from prediction
        #    clean_prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
#
        #    predictions_wo_input.append(clean_prediction)

        # new method for prompting
        predictions_wo_input = []
        for i, example in enumerate(data_module["predict_dataset"]):
            input_text = example["input"]

            # 🔹 Split on "### Response:" to extract the generated response
            if "### Response:" in predictions[i]:
                clean_prediction = predictions[i].split("### Response:", 1)[-1].strip()
            else:
                clean_prediction = predictions[i].strip()  # Fallback: if "### Response:" is missing

            predictions_wo_input.append(clean_prediction)


        # ✅ Save predictions to JSONL
        with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as fout:
            for i, example in enumerate(data_module["predict_dataset"]):
                example["prediction_with_input"] = predictions[i].strip()
                example["prediction"] = predictions_wo_input[i]
                fout.write(json.dumps(example) + "\n")

        # ✅ Get references from the dataset
        if task.lower() == 'simp': 
            references = [
                [entry["simple_sentence"]] for entry in data_module["predict_dataset"]["src_info"]
            ]
        elif task.lower() == 'empd':
            references = [
                [entry["output_text"]] for entry in data_module["predict_dataset"]["src_info"]
            ]
        elif task.lower() == 'inqqg':
            references = [
                [entry["question"] for entry in data_module["predict_dataset"]["src_info"]]
            ]

        if args.compute_bleu:
            bleu_score = bleu.compute(predictions = predictions_wo_input, references=references)
            all_metrics.update({'bleu': bleu_score['bleu']})
        if args.compute_rouge:
            rouge_score = rouge.compute(predictions=predictions_wo_input, references=references)
            all_metrics.update(
                {'rouge1': rouge_score['rouge1'],
                'rouge2': rouge_score['rouge2'],
                'rougeL': rouge_score['rougeL']}
                )
        if args.compute_bert_score:
            bscore = bert_score.compute(predictions=predictions_wo_input, references=references, lang='en')
            all_metrics.update(
                {"bertscore_f1": np.mean(bscore['f1']),
                    "bertscore_P": np.mean(bscore['precision']),
                    "bertscore_R": np.mean(bscore['recall'])}
                )
        if args.compute_sari:
            sources = [entry["normal_sentence"] for entry in data_module["predict_dataset"]["src_info"]]
            sari_score = sari.compute(sources=sources, predictions=predictions_wo_input, references=references)
            all_metrics.update({'sari': sari_score['sari']})

        trainer.log_metrics("predict", prediction_metrics)
        trainer.save_metrics("predict", prediction_metrics)
        all_metrics.update(prediction_metrics)
        print(all_metrics)

    if (args.do_train or args.do_eval or args.do_predict):
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))

if __name__ == "__main__":
    train()