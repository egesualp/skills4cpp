"""
Wrapper script that adds profiling to train_cpp_enhanced.py without modifying it.

Usage:
    python src/cpp/train_with_profiling.py --quick-profile [... normal training args ...]
    
This will run your training with timing information for each major step.
"""

import sys
import time
import argparse
from loguru import logger

# Monkey-patch timing into the training functions
def add_profiling_to_training():
    """Add timing instrumentation to training functions."""
    import src.cpp.train_cpp_enhanced as train_module
    import src.cpp.data_loaders as data_module
    from functools import wraps
    
    # Wrap functions with timing
    def time_wrapper(func, name):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger.info(f"⏱️  Starting: {name}")
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info(f"✅ Completed: {name} in {elapsed:.2f}s ({elapsed/60:.2f} min)")
            return result
        return wrapper
    
    # Patch key functions
    logger.info("🔧 Adding profiling instrumentation...")
    
    # Data loading functions
    data_module.precompute_target_embeddings = time_wrapper(
        data_module.precompute_target_embeddings, 
        "Pre-compute Target Embeddings"
    )
    data_module.precompute_input_embeddings = time_wrapper(
        data_module.precompute_input_embeddings,
        "Pre-compute Input Embeddings"
    )
    data_module.load_all_vocabs = time_wrapper(
        data_module.load_all_vocabs,
        "Load Vocabularies"
    )
    data_module.load_job_and_skill_data = time_wrapper(
        data_module.load_job_and_skill_data,
        "Load Job and Skill Data"
    )
    
    # Training functions
    original_train_epoch = train_module.train_epoch
    original_evaluate = train_module.evaluate
    
    def train_epoch_timed(*args, **kwargs):
        start = time.perf_counter()
        result = original_train_epoch(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"  ⏱️  Epoch training time: {elapsed:.2f}s")
        return result
    
    def evaluate_timed(*args, **kwargs):
        start = time.perf_counter()
        result = original_evaluate(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"  ⏱️  Evaluation time: {elapsed:.2f}s")
        return result
    
    train_module.train_epoch = train_epoch_timed
    train_module.evaluate = evaluate_timed
    
    # Patch objective to add trial timing
    original_objective = train_module.objective
    
    def objective_timed(trial, *args, **kwargs):
        logger.info(f"\n{'='*80}")
        logger.info(f"⏱️  Starting Trial {trial.number}")
        logger.info(f"{'='*80}")
        start = time.perf_counter()
        result = original_objective(trial, *args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{'='*80}")
        logger.info(f"✅ Trial {trial.number} completed in {elapsed:.2f}s ({elapsed/60:.2f} min)")
        logger.info(f"{'='*80}\n")
        return result
    
    train_module.objective = objective_timed
    
    logger.info("✅ Profiling instrumentation added\n")


def main():
    # Check if --quick-profile flag is present
    if '--quick-profile' in sys.argv:
        sys.argv.remove('--quick-profile')
        add_profiling_to_training()
    
    # Import and run main training
    from src.cpp.train_cpp_enhanced import main as train_main
    
    logger.info("="*80)
    logger.info("TRAINING WITH PROFILING ENABLED")
    logger.info("="*80)
    logger.info("Timing information will be displayed for each major step\n")
    
    overall_start = time.perf_counter()
    
    try:
        train_main()
    finally:
        overall_elapsed = time.perf_counter() - overall_start
        logger.info("\n" + "="*80)
        logger.info(f"TOTAL EXECUTION TIME: {overall_elapsed:.2f}s ({overall_elapsed/60:.2f} min, {overall_elapsed/3600:.2f} hours)")
        logger.info("="*80)


if __name__ == "__main__":
    main()














