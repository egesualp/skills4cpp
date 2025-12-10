import os
# Specify GPU if needed
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 

from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from datasets import Dataset
import random
import argparse
from config_utils import load_train_config
from data_classes import Data


class CustomTrainer(SentenceTransformerTrainer):
    """
    Custom trainer class that extends SentenceTransformerTrainer.

    This class overrides the `log` method to include the number of data points
    seen in the logs during training.
    """

    def log(self, logs: dict):
        """
        Logs training progress, including the number of data points processed.

        Args:
            logs (dict): Dictionary containing log information.
        """
        logs["data_points_seen"] = (
            self.state.global_step * self.args.per_device_train_batch_size
        )
        super().log(logs)


def fine_tune(train_pairs, valid_pairs, model, MODEL_OUTPUT_PATH):
    """
    Fine-tune a Sentence Transformer model on the provided training data.

    Args:
        train_pairs (list): List of (anchor, positive) training pairs.
        valid_pairs (list): List of (anchor, positive) validation pairs.
        model (SentenceTransformer): Pretrained Sentence Transformer model.
        MODEL_OUTPUT_PATH (str): Path to save the fine-tuned model.

    Returns:
        SentenceTransformer: The fine-tuned model.
    """
    print("Creating the training dataset...")
    train_dataset = Dataset.from_dict(
        {
            "anchor": [a for a, _ in train_pairs],
            "positive": [p for _, p in train_pairs],
        }
    )

    print("Creating the validation dataset...")
    eval_dataset = Dataset.from_dict(
        {
            "anchor": [a for a, p in valid_pairs],
            "positive": [p for a, p in valid_pairs],
        }
    )

    print("Defining the loss function...")
    loss = losses.MultipleNegativesRankingLoss(model, scale=20.0)

    _eval_save_n_steps_interval = int(
        config["finetuning"]["epoch_eval_frac"]
        * len(train_dataset)
        / config["finetuning"]["batch_size"]
    )
    print(f"Eval and save steps interval: {_eval_save_n_steps_interval}")

    print("Setting training arguments...")
    args = SentenceTransformerTrainingArguments(
        output_dir=MODEL_OUTPUT_PATH,
        num_train_epochs=config["finetuning"]["epochs"],
        per_device_train_batch_size=config["finetuning"]["batch_size"],
        per_device_eval_batch_size=config["finetuning"]["batch_size"],
        learning_rate=config["finetuning"]["learning_rate"],
        fp16=True,
        bf16=False,
        max_grad_norm=1.0,
        eval_strategy="steps",
        eval_steps=_eval_save_n_steps_interval,
        save_strategy="steps",
        save_steps=_eval_save_n_steps_interval,
        save_total_limit=2,
        logging_steps=_eval_save_n_steps_interval,
        run_name="sts",
        report_to="tensorboard",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    args = args.set_lr_scheduler(name="linear", warmup_ratio=0.05)
    args = args.set_training(
        learning_rate=config["finetuning"]["learning_rate"],
        batch_size=config["finetuning"]["batch_size"],
        weight_decay=0,
        num_epochs=config["finetuning"]["epochs"],
        max_steps=-1,
        gradient_accumulation_steps=1,
        seed=42,
        gradient_checkpointing=False,
    )

    print("Creating evaluator...")
    dev_evaluator = construct_evaluator(valid_pairs)

    print("Creating the trainer...")
    trainer = CustomTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        args=args,
        evaluator=dev_evaluator,
    )

    trainer.train()

    return model


def construct_evaluator(valid_pairs):
    """
    Constructs an evaluator for measuring sentence embedding similarity.

    Args:
        valid_pairs (list): List of (anchor, positive) validation pairs.

    Returns:
        EmbeddingSimilarityEvaluator: Evaluator for embedding similarity.
    """
    print("Constructing evaluator...")
    valid_input_pairs = []
    for a, p in valid_pairs:
        valid_input_pairs.append(InputExample(texts=[a, p], label=1))
        r_a, r_p = random.choice(valid_pairs)
        valid_input_pairs.append(InputExample(texts=[a, r_p], label=0))
    
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        valid_input_pairs, write_csv=False, show_progress_bar=False
    )
    
    return evaluator


def main(config):
    """
    Main function to load data, initialize the model, and fine-tune it.

    Args:
        config (dict): Configuration dictionary containing settings for data loading,
                       model training, and output paths.
    """
    # Load Data
    print("Loading data and model...")
    data = Data(
        config["data"]["data_type"],
        config["data"]["doc_1_prompt"],
        config["data"]["doc_2_prompt"],
        config["data"]["only_titles"],
    )

    train_pairs, val_pairs, _ = data.get_data(stage="embedding_finetuning")

    # Load Model
    print("Loading Sentence Transformer model...")
    model = SentenceTransformer(config["model"]["embedding_model_finetuning"])

    # Fine-tune Model
    print("Fine-tuning the model...")
    model = fine_tune(
        train_pairs, val_pairs, model, config["output"]["path_embedding_model"]
    )

    # Save Best Model
    print(f"Saving the best model to: {config['output']['path_embedding_model']}")
    model.save(config["output"]["path_embedding_model"] + "/best-model")

    print("Model fine-tuning and saving completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fintetune_config_name", type=str, help="Name of the finetuning configuration file.")
    args = parser.parse_args()
    
    # Load configuration
    config = load_train_config(args.fintetune_config_name)

    # Run the main function
    main(config)
