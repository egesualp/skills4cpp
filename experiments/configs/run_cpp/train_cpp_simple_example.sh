#!/bin/bash
# Example usage of train_cpp_simple.py

# ============================================================================
# Example 1: Quick start with default config
# ============================================================================
python src/cpp/train_cpp_simple.py \
    --config configs/cpp_simple_baseline.yaml \
    --use_wandb \
    --run_name baseline_run

# ============================================================================
# Example 2: Custom hyperparameters without config file
# ============================================================================
python src/cpp/train_cpp_simple.py \
    --data_type decorte \
    --encoder ElenaSenger/career-path-representation-mpnet-decorte \
    --hidden_dim 768 \
    --n_layers 3 \
    --dropout 0.2 \
    --max_epochs 100 \
    --patience 10 \
    --lr 0.0001 \
    --batch_size 512 \
    --use_wandb \
    --wandb_project my-cpp-experiments \
    --run_name deep_network_experiment \
    --save_model

# ============================================================================
# Example 3: Override specific config values
# ============================================================================
python src/cpp/train_cpp_simple.py \
    --config configs/cpp_simple_baseline.yaml \
    --max_epochs 150 \
    --patience 15 \
    --hidden_dim 1024 \
    --run_name large_model_long_training

# ============================================================================
# Example 4: With job descriptions (not just titles)
# ============================================================================
python src/cpp/train_cpp_simple.py \
    --config configs/cpp_simple_baseline.yaml \
    --use_text_description \
    --use_skill_description \
    --run_name with_descriptions

# ============================================================================
# Example 5: Different dataset
# ============================================================================
python src/cpp/train_cpp_simple.py \
    --config configs/cpp_simple_baseline.yaml \
    --data_type karrierewege_100k \
    --run_name karrierewege_experiment

# ============================================================================
# Example 6: Quick test run (fast iteration)
# ============================================================================
python src/cpp/train_cpp_simple.py \
    --data_type decorte \
    --encoder sentence-transformers/all-mpnet-base-v2 \
    --max_epochs 5 \
    --patience 2 \
    --batch_size 256 \
    --run_name quick_test \
    --force_recompute  # Force recompute if encoder changed

# ============================================================================
# Comparison with train_cpp_enhanced_v2.py
# ============================================================================

# OLD WAY (train_cpp_enhanced_v2.py - complex):
# python src/cpp/train_cpp_enhanced_v2.py \
#     --data_type decorte \
#     --encoder_text ElenaSenger/career-path-representation-mpnet-decorte \
#     --use_text_history \
#     --use_skill_text \
#     --use_structured \
#     --optuna \
#     --n_trials 50 \
#     --max_epochs 10 \
#     --patience 2 \
#     --optuna_patience 3 \
#     --val_sample_ratio 0.1 \
#     --train_sample_ratio 1.0 \
#     --batch_size 512 \
#     --eval_batch_size 4092 \
#     --use_wandb \
#     --wandb_project cpp-enhanced \
#     --run_name my_experiment \
#     --output_dir results/cpp \
#     # ... many more arguments

# NEW WAY (train_cpp_simple.py - simple):
# python src/cpp/train_cpp_simple.py \
#     --config configs/cpp_simple_baseline.yaml \
#     --run_name my_experiment

# That's it! ✨



