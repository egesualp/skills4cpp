"""
Skill-Based Sentence Transformer Finetuning Script (v3, simplified).

This version:
- Reuses the skill-based pooling pipeline (IDF per skill, positional pooling over jobs)
- Uses MultipleNegativesRankingLoss (MNRL) like the original base model
- Performs step-based evaluation on **validation loss** (cheaper than MRR)
- Is configured via command-line arguments (no external config file)
"""

import argparse
import os
import sys
import time
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

from sentence_transformers import SentenceTransformer

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from src.cpp.data_classes import Data
from src.cpp.utils import SEP_TOKEN
from src.cpp.skill_dataset import (
    SkillBasedCareerPathDataset,
    ISCOGroupBatchSampler,
    collate_skill_batch,
)
from src.cpp.skill_pooling import (
    load_skill_mappings,
    load_skill_descriptions,
    load_occupation_isco_groups,
    calculate_idf_scores,
    create_target_occupation_map,
    process_career_path_batch,
    precompute_skill_embeddings,
    process_career_path_batch_train,
    cap_skills_per_job_lexicographic,
)


# ============================================================================
# LOSS UTILS
# ============================================================================

def compute_mnrl_loss(
    model: SentenceTransformer,
    career_embeds_tensor: torch.Tensor,
    target_texts: List[str],
    device: torch.device,
    mixed_precision: bool,
) -> torch.Tensor:
    """
    Compute MultipleNegativesRankingLoss between fixed career embeddings
    and target texts encoded by the model.
    """
    device_type = device.type

    if mixed_precision:
        # Use torch.amp.autocast for target encoding
        with torch.amp.autocast(device_type=device_type, enabled=(device_type == "cuda")):
            target_features = model.tokenize(target_texts)
            target_features = {k: v.to(device) for k, v in target_features.items()}
            target_embeds = model(target_features)["sentence_embedding"]

            # Normalize
            career_norm = F.normalize(career_embeds_tensor, p=2, dim=1)
            target_norm = F.normalize(target_embeds, p=2, dim=1)

            # Similarity matrix scaled (as in standard MNRL)
            sim_matrix = career_norm @ target_norm.t() * 20.0
            labels = torch.arange(len(career_embeds_tensor), device=device)
            loss = F.cross_entropy(sim_matrix, labels)
    else:
        target_features = model.tokenize(target_texts)
        target_features = {k: v.to(device) for k, v in target_features.items()}
        target_embeds = model(target_features)["sentence_embedding"]

        career_norm = F.normalize(career_embeds_tensor, p=2, dim=1)
        target_norm = F.normalize(target_embeds, p=2, dim=1)

        sim_matrix = career_norm @ target_norm.t() * 20.0
        labels = torch.arange(len(career_embeds_tensor), device=device)
        loss = F.cross_entropy(sim_matrix, labels)

    return loss


@torch.no_grad()
def evaluate_validation_loss(
    model: SentenceTransformer,
    val_loader: DataLoader,
    skill_desc_map: Dict[str, Dict[str, str]],
    args,
    device: torch.device,
    precomputed_skill_embeddings: Optional[Dict[str, Any]] = None,
    max_val_batches: Optional[int] = None,
) -> float:
    """
    Evaluate average MNRL loss on the validation set.

    This is cheaper than computing full ranking metrics and can be run
    more frequently (e.g., every X% of training steps).
    """
    model.eval()

    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(
        tqdm(val_loader, desc="Validating", leave=False)
    ):
        if max_val_batches is not None and batch_idx >= max_val_batches:
            break

        career_embeds, target_texts = process_career_path_batch_train(
            batch,
            skill_desc_map,
            model,
            args.alpha_decay,
            args.use_skill_description,
            device,
            precomputed_skill_embeddings=precomputed_skill_embeddings,
        )

        # Convert list of tensors to a single tensor
        career_tensor = torch.stack(career_embeds, dim=0)

        loss = compute_mnrl_loss(
            model,
            career_tensor,
            target_texts,
            device,
            mixed_precision=args.mixed_precision,
        )

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches if n_batches > 0 else 0.0


# ============================================================================
# TRAINING LOOP
# ============================================================================

def train_model_v3(
    model: SentenceTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    skill_desc_map: Dict[str, Dict[str, str]],
    args,
    device: torch.device,
    precomputed_skill_embeddings: Optional[Dict[str, Any]] = None,
) -> float:
    """
    Simplified training loop:
    - MNRL between skill-pooled career embeddings and target texts
    - Step-based evaluation on validation loss (every epoch_eval_frac * steps)
    - Saves best model based on lowest validation loss
    """
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    # Mixed precision scaler
    scaler = None
    device_type = device.type
    if args.mixed_precision and device_type == "cuda":
        from torch.amp import GradScaler

        scaler = GradScaler(device=device_type)
        print("Mixed precision training enabled (FP16).")

    num_training_steps_per_epoch = len(train_loader)
    eval_steps_interval = max(
        1, int(args.epoch_eval_frac * num_training_steps_per_epoch)
    )
    print(
        f"Eval interval: every {eval_steps_interval} training steps "
        f"(~{args.epoch_eval_frac*100:.1f}% of an epoch)."
    )

    global_step = 0
    best_val_loss = float("inf")

    if WANDB_AVAILABLE and args.use_wandb:
        wandb.watch(model, log="all", log_freq=eval_steps_interval)

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        model.train()

        epoch_loss = 0.0
        num_batches = 0
        accumulation_counter = 0

        epoch_start_time = time.time()

        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}"):
            global_step += 1

            career_embeds, target_texts = process_career_path_batch_train(
                batch,
                skill_desc_map,
                model,
                args.alpha_decay,
                args.use_skill_description,
                device,
                precomputed_skill_embeddings=precomputed_skill_embeddings,
            )

            # Convert list of tensors to a single tensor
            career_tensor = torch.stack(career_embeds, dim=0)

            if accumulation_counter == 0:
                optimizer.zero_grad()

            if args.mixed_precision and scaler is not None:
                with torch.autocast(device_type=device_type, enabled=(device_type == "cuda")):
                    loss = compute_mnrl_loss(
                        model,
                        career_tensor,
                        target_texts,
                        device,
                        mixed_precision=False,  # already under autocast
                    )
                loss = loss / args.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                loss = compute_mnrl_loss(
                    model,
                    career_tensor,
                    target_texts,
                    device,
                    mixed_precision=False,
                )
                loss = loss / args.gradient_accumulation_steps
                loss.backward()

            epoch_loss += loss.item()
            accumulation_counter += 1
            num_batches += 1

            # Optimizer step
            if accumulation_counter >= args.gradient_accumulation_steps:
                if args.mixed_precision and scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                accumulation_counter = 0

            # Step-based evaluation
            if global_step % eval_steps_interval == 0:
                val_loss = evaluate_validation_loss(
                    model,
                    val_loader,
                    skill_desc_map,
                    args,
                    device,
                    precomputed_skill_embeddings=precomputed_skill_embeddings,
                    max_val_batches=args.max_val_batches,
                )

                print(
                    f"[Step {global_step}] "
                    f"Train loss (avg so far): {epoch_loss / num_batches:.4f} | "
                    f"Val loss: {val_loss:.4f}"
                )

                if WANDB_AVAILABLE and args.use_wandb:
                    wandb.log(
                        {
                            "train_loss": epoch_loss / num_batches,
                            "val_loss": val_loss,
                            "global_step": global_step,
                            "epoch": epoch + (num_batches / num_training_steps_per_epoch),
                        }
                    )

                # Save best model based on validation loss
                if args.save_model and val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_path = os.path.join(args.output_dir, "best_model")
                    os.makedirs(args.output_dir, exist_ok=True)
                    model.save(save_path)
                    print(f"  ✓ New best model saved to {save_path} (val_loss={val_loss:.4f})")

        epoch_time = time.time() - epoch_start_time
        print(
            f"Epoch {epoch + 1} finished in {epoch_time/60:.1f} min. "
            f"Avg train loss: {epoch_loss / num_batches:.4f}"
        )

    return best_val_loss


# ============================================================================
# EVALUATION UTILS (RANKING METRICS)
# ============================================================================


def compute_all_target_embeddings(
    model: SentenceTransformer,
    all_target_texts: List[str],
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Encode all unique target occupation texts with the given model.
    """
    print(
        f"  > Encoding {len(all_target_texts)} unique target occupations "
        f"with batch_size={batch_size}..."
    )
    embeddings = model.encode(
        all_target_texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=batch_size,
        device=device,
    )
    print(f"  > Target embeddings shape: {embeddings.shape}")
    return embeddings


def evaluate_ranking(
    model: SentenceTransformer,
    dataloader: DataLoader,
    skill_desc_map: Dict[str, Dict[str, str]],
    all_target_embeddings: np.ndarray,
    all_target_texts: List[str],
    args,
    device: torch.device,
    precomputed_skill_embeddings: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Evaluate ranking performance (MRR, precision/recall@5 and @10) on a dataset.

    Compares pooled career-path embeddings against target occupation embeddings.
    """
    model.eval()

    pred_embeddings: List[np.ndarray] = []
    true_target_texts: List[str] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            career_embeds, target_texts = process_career_path_batch(
                batch,
                skill_desc_map,
                model,
                args.alpha_decay,
                args.use_skill_description,
                device,
                precomputed_skill_embeddings=precomputed_skill_embeddings,
            )

            if precomputed_skill_embeddings is not None:
                # torch tensors on-device → numpy on CPU
                pred_embeddings.extend(
                    [emb.detach().cpu().numpy() for emb in career_embeds]
                )
            else:
                pred_embeddings.extend(career_embeds)

            true_target_texts.extend(target_texts)

    if not pred_embeddings:
        print("  ! No predictions produced during evaluation.")
        return {
            "MRR": 0.0,
            "P@5": 0.0,
            "R@5": 0.0,
            "P@10": 0.0,
            "R@10": 0.0,
        }

    pred_embeddings_arr = np.array(pred_embeddings)

    # Cosine similarities between predicted career embeddings and all targets
    sim_matrix = cosine_similarity(pred_embeddings_arr, all_target_embeddings)
    sorted_indices = np.argsort(sim_matrix, axis=1)[:, ::-1]  # descending

    # Map target text → index
    target_text_to_idx = {text: idx for idx, text in enumerate(all_target_texts)}

    reciprocal_ranks: List[float] = []
    precision_at_5 = 0.0
    recall_at_5 = 0.0
    precision_at_10 = 0.0
    recall_at_10 = 0.0

    for i, true_text in enumerate(true_target_texts):
        if true_text not in target_text_to_idx:
            reciprocal_ranks.append(0.0)
            continue

        true_idx = target_text_to_idx[true_text]
        rank_list = list(sorted_indices[i])

        if true_idx in rank_list:
            rank = rank_list.index(true_idx) + 1  # 1-based
            reciprocal_ranks.append(1.0 / rank)

            # Top-5
            if rank <= 5:
                recall_at_5 += 1.0
                precision_at_5 += 1.0 / 5.0

            # Top-10
            if rank <= 10:
                recall_at_10 += 1.0
                precision_at_10 += 1.0 / 10.0
        else:
            reciprocal_ranks.append(0.0)

    n_samples = len(true_target_texts)
    if n_samples == 0:
        return {
            "MRR": 0.0,
            "P@5": 0.0,
            "R@5": 0.0,
            "P@10": 0.0,
            "R@10": 0.0,
        }

    metrics = {
        "MRR": float(np.mean(reciprocal_ranks)),
        "P@5": precision_at_5 / n_samples,
        "R@5": recall_at_5 / n_samples,
        "P@10": precision_at_10 / n_samples,
        "R@10": recall_at_10 / n_samples,
    }
    return metrics


# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Skill-Based Sentence Transformer Finetuning (v3, simplified)"
    )

    # Data paths
    parser.add_argument(
        "--data_type",
        type=str,
        default="karrierewege_100k",
        help="Dataset type (default: karrierewege_100k)",
    )
    parser.add_argument(
        "--job_title_skills_csv",
        type=str,
        default="results/karrierewege_esco_100k_esco_ground_truth/job_title_skills_master.csv",
        help="Path to job title skills mapping CSV",
    )
    parser.add_argument(
        "--skills_csv",
        type=str,
        default="data/esco_datasets/skills_en.csv",
        help="Path to ESCO skills CSV",
    )
    parser.add_argument(
        "--occupations_csv",
        type=str,
        default="data/esco_datasets/occupations_en.csv",
        help="Path to ESCO occupations CSV",
    )

    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        default="ElenaSenger/career-path-representation-mpnet-karrierewege",
        help="Base sentence transformer model",
    )

    # Training hyperparameters
    parser.add_argument(
        "--alpha_decay",
        type=float,
        default=0.5,
        help="Logarithmic decay parameter for job position weighting (0 for mean pooling)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Training batch size per step",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=256,
        help="Evaluation batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=2,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--epoch_eval_frac",
        type=float,
        default=0.01,
        help="Fraction of an epoch between evaluations (e.g., 0.01 = every 1% of steps)",
    )
    parser.add_argument(
        "--use_skill_description",
        action="store_true",
        help="Include skill descriptions in encoding",
    )
    parser.add_argument(
        "--max_skills_per_job",
        type=int,
        default=None,
        help=(
            "Optional cap on number of skills per job. "
            "If set, keeps the lexicographically smallest K skills per job "
            "according to skillUri before training."
        ),
    )

    # GPU / optimization
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        help="Enable mixed precision training (FP16)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients (simulates larger batch size)",
    )
    parser.add_argument(
        "--precompute_skill_embeddings",
        action="store_true",
        help="Precompute and freeze skill text embeddings (much faster training)",
    )
    parser.add_argument(
        "--max_val_batches",
        type=int,
        default=None,
        help="Optional cap on number of validation batches per evaluation (for speed).",
    )

    # Output and logging
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/cpp_skills_v3",
        help="Output directory for models",
    )
    parser.add_argument(
        "--save_model",
        action="store_true",
        help="Save the best model during training",
    )
    parser.add_argument(
        "--evaluate_base_model",
        action="store_true",
        help="Also evaluate the base (un-finetuned) model on the test set",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="cpp-skills-v3",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity name",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="skill_based_training_v3",
        help="Run name for logging",
    )

    # Device / dataloader
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of DataLoader workers",
    )

    return parser.parse_args()


# ============================================================================
# MAIN
# ============================================================================

def main():
    args = parse_args()

    print("Configuration:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("")

    device = torch.device(args.device)

    # --- Step 1: Load data ---
    print("[1/7] Loading career path data...")
    data = Data(DATA_TYPE=args.data_type, ONLY_TITLES=False)
    train_pairs, val_pairs, test_pairs = data.get_data(stage="embedding_finetuning")
    print(
        f"  ✓ Train: {len(train_pairs)}, Val: {len(val_pairs)}, "
        f"Test: {len(test_pairs)}"
    )

    # --- Step 2: Load skill mappings and descriptions ---
    print("[2/7] Loading skill mappings and descriptions...")
    job_skill_map = load_skill_mappings(args.job_title_skills_csv)
    skill_desc_map = load_skill_descriptions(args.skills_csv)
    isco_map = load_occupation_isco_groups(args.occupations_csv)
    print("")

    # --- Step 3: Calculate IDF scores ---
    print("[3/7] Calculating IDF scores...")
    job_skill_map = calculate_idf_scores(job_skill_map)
    print("")

    # --- Optional: Cap number of skills per job (to control memory) ---
    if args.max_skills_per_job is not None and args.max_skills_per_job > 0:
        print("[3.5/7] Capping skills per job using IDF + description length + lexicographic label...")
        job_skill_map = cap_skills_per_job_lexicographic(
            job_skill_map,
            max_skills_per_job=args.max_skills_per_job,
            skill_desc_map=skill_desc_map,
        )
        print("")

    # --- Step 4: Create target occupation map ---
    print("[4/7] Creating target occupation mappings...")
    all_pairs = train_pairs + val_pairs + test_pairs
    target_occupation_map = create_target_occupation_map(all_pairs, isco_map)
    print("")

    # --- Step 5: Load model ---
    print("[5/7] Loading sentence transformer model...")
    model = SentenceTransformer(args.model_name)
    model.to(device)
    print(f"  ✓ Model loaded: {args.model_name}")
    print(f"  ✓ Embedding dimension: {model.get_sentence_embedding_dimension()}\n")

    # Optional: initialize wandb
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=args.run_name,
        )
        print(f"🚀 W&B logging enabled for run: {args.run_name}\n")

    # Optionally precompute skill embeddings
    precomputed_skill_embeddings = None
    if args.precompute_skill_embeddings:
        print("[5.5/7] Precomputing skill text embeddings (one per skillUri)...")
        precomputed_skill_embeddings = precompute_skill_embeddings(
            job_skill_map=job_skill_map,
            skill_desc_map=skill_desc_map,
            encoder=model,
            use_skill_description=args.use_skill_description,
            device=device,
            batch_size=1024,
        )
        print(
            f"  ✓ Precomputed {len(precomputed_skill_embeddings)} unique skill "
            f"embeddings\n"
        )

    # --- Step 6: Create datasets and dataloaders ---
    print("[6/7] Creating datasets and dataloaders...")

    train_dataset = SkillBasedCareerPathDataset(
        data_pairs=train_pairs,
        job_skill_map=job_skill_map,
        target_occupation_map=target_occupation_map,
        sep_token=SEP_TOKEN,
    )
    val_dataset = SkillBasedCareerPathDataset(
        data_pairs=val_pairs,
        job_skill_map=job_skill_map,
        target_occupation_map=target_occupation_map,
        sep_token=SEP_TOKEN,
    )

    train_sampler = ISCOGroupBatchSampler(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    use_cuda = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_skill_batch,
        pin_memory=use_cuda,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_skill_batch,
        pin_memory=use_cuda,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    print(f"  ✓ Train batches: {len(train_loader)}")
    print(f"  ✓ Val batches:   {len(val_loader)}\n")

    # --- Step 7: Train model ---
    print("[7/7] Training model (v3)...")
    best_val_loss = train_model_v3(
        model,
        train_loader,
        val_loader,
        skill_desc_map,
        args,
        device,
        precomputed_skill_embeddings=precomputed_skill_embeddings,
    )

    print(f"\n✓ Training complete. Best Val Loss: {best_val_loss:.4f}")

    if WANDB_AVAILABLE and args.use_wandb:
        wandb.log({"best_val_loss": best_val_loss})
        wandb.finish()

    # --- Step 8: Final evaluation on test split (ranking metrics) ---
    print("\n[8/8] Evaluating models on test split (MRR, P@K, R@K)...")

    # Build test dataset/loader
    test_dataset = SkillBasedCareerPathDataset(
        data_pairs=test_pairs,
        job_skill_map=job_skill_map,
        target_occupation_map=target_occupation_map,
        sep_token=SEP_TOKEN,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_skill_batch,
        pin_memory=use_cuda,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    # Build the set of unique target occupation texts
    all_target_texts = [
        f"role: {info['title']} \n description: {info['description']}"
        for info in target_occupation_map.values()
    ]

    # 1) Evaluate best / final fine-tuned model
    eval_model = model
    if args.save_model:
        best_model_path = os.path.join(args.output_dir, "best_model")
        if os.path.exists(best_model_path):
            print(f"  > Loading best model from {best_model_path} for test evaluation...")
            eval_model = SentenceTransformer(best_model_path)
            eval_model.to(device)
        else:
            print(
                f"  ! Expected best model at {best_model_path} not found. "
                "Using in-memory model weights for evaluation."
            )

    print("\n  >> Test evaluation: fine-tuned model")
    target_embeds_ft = compute_all_target_embeddings(
        eval_model,
        all_target_texts,
        device,
        batch_size=256,
    )
    ft_metrics = evaluate_ranking(
        eval_model,
        test_loader,
        skill_desc_map,
        target_embeds_ft,
        all_target_texts,
        args,
        device,
        precomputed_skill_embeddings=precomputed_skill_embeddings,
    )
    for k, v in ft_metrics.items():
        print(f"    {k}: {v:.4f}")

    # 2) Optionally evaluate the base (un-finetuned) model
    if args.evaluate_base_model:
        print("\n  >> Test evaluation: base model (un-finetuned)")
        base_model = SentenceTransformer(args.model_name)
        base_model.to(device)

        target_embeds_base = compute_all_target_embeddings(
            base_model,
            all_target_texts,
            device,
            batch_size=256,
        )
        base_metrics = evaluate_ranking(
            base_model,
            test_loader,
            skill_desc_map,
            target_embeds_base,
            all_target_texts,
            args,
            device,
            precomputed_skill_embeddings=precomputed_skill_embeddings,
        )
        for k, v in base_metrics.items():
            print(f"    {k}: {v:.4f}")


if __name__ == "__main__":
    main()


