"""
Profiling utilities for training bottleneck analysis.

This script helps identify performance bottlenecks in the training pipeline by:
1. Timing each major component
2. Profiling data loading
3. Profiling model forward/backward passes
4. Memory profiling
5. GPU utilization monitoring
"""

import time
import contextlib
import functools
import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List
from loguru import logger


class PerformanceProfiler:
    """Context manager and decorator for timing code blocks."""
    
    def __init__(self, name: str = "Operation", enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.start_time = None
        self.elapsed = None
        
    def __enter__(self):
        if self.enabled:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        if self.enabled:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.elapsed = time.perf_counter() - self.start_time
            logger.info(f"⏱️  {self.name}: {self.elapsed:.3f}s")
            
    def __call__(self, func):
        """Use as decorator."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with PerformanceProfiler(f"{func.__name__}", self.enabled):
                return func(*args, **kwargs)
        return wrapper


class TimingStats:
    """Accumulate timing statistics across multiple calls."""
    
    def __init__(self):
        self.times = defaultdict(list)
        
    def record(self, name: str, duration: float):
        self.times[name].append(duration)
        
    def report(self):
        """Print summary statistics."""
        logger.info("\n" + "=" * 80)
        logger.info("TIMING STATISTICS")
        logger.info("=" * 80)
        
        for name, durations in sorted(self.times.items()):
            mean = np.mean(durations)
            std = np.std(durations)
            total = np.sum(durations)
            count = len(durations)
            logger.info(f"{name:40s} | Count: {count:4d} | Total: {total:8.2f}s | Mean: {mean:7.3f}s ± {std:6.3f}s")
        
        logger.info("=" * 80 + "\n")


class DataLoaderProfiler:
    """Profile DataLoader performance to identify bottlenecks."""
    
    @staticmethod
    def profile_dataloader(dataloader, num_batches: int = 10, device: str = 'cuda'):
        """
        Profile DataLoader to measure:
        - Time to fetch batch
        - Time to move to device
        - Batch processing overhead
        """
        logger.info(f"\n📊 Profiling DataLoader (first {num_batches} batches)...")
        
        fetch_times = []
        transfer_times = []
        
        start = time.perf_counter()
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
                
            fetch_time = time.perf_counter() - start
            fetch_times.append(fetch_time)
            
            # Time device transfer
            transfer_start = time.perf_counter()
            batch = {k: v.to(device) for k, v in batch.items()}
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            transfer_time = time.perf_counter() - transfer_start
            transfer_times.append(transfer_time)
            
            start = time.perf_counter()
        
        logger.info(f"  Fetch time:    {np.mean(fetch_times):.4f}s ± {np.std(fetch_times):.4f}s")
        logger.info(f"  Transfer time: {np.mean(transfer_times):.4f}s ± {np.std(transfer_times):.4f}s")
        logger.info(f"  Total per batch: {np.mean(fetch_times) + np.mean(transfer_times):.4f}s")
        
        return {
            'fetch_mean': np.mean(fetch_times),
            'transfer_mean': np.mean(transfer_times),
        }


class ModelProfiler:
    """Profile model forward and backward passes."""
    
    @staticmethod
    def profile_model_step(model, batch, optimizer, criterion, device, num_iters: int = 10):
        """
        Profile a single training step breakdown:
        - Forward pass
        - Loss computation
        - Backward pass
        - Optimizer step
        """
        logger.info(f"\n🔬 Profiling Model Training Step ({num_iters} iterations)...")
        
        forward_times = []
        loss_times = []
        backward_times = []
        optimizer_times = []
        
        model.train()
        
        for i in range(num_iters):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            y_pred = model(batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_times.append(time.perf_counter() - start)
            
            # Loss computation
            start = time.perf_counter()
            target = torch.ones(y_pred.size(0)).to(device)
            loss = criterion(y_pred, batch['y'], target)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            loss_times.append(time.perf_counter() - start)
            
            # Backward pass
            start = time.perf_counter()
            optimizer.zero_grad()
            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            backward_times.append(time.perf_counter() - start)
            
            # Optimizer step
            start = time.perf_counter()
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            optimizer_times.append(time.perf_counter() - start)
        
        logger.info(f"  Forward pass:   {np.mean(forward_times)*1000:.2f}ms ± {np.std(forward_times)*1000:.2f}ms")
        logger.info(f"  Loss compute:   {np.mean(loss_times)*1000:.2f}ms ± {np.std(loss_times)*1000:.2f}ms")
        logger.info(f"  Backward pass:  {np.mean(backward_times)*1000:.2f}ms ± {np.std(backward_times)*1000:.2f}ms")
        logger.info(f"  Optimizer step: {np.mean(optimizer_times)*1000:.2f}ms ± {np.std(optimizer_times)*1000:.2f}ms")
        logger.info(f"  Total per step: {(np.mean(forward_times) + np.mean(loss_times) + np.mean(backward_times) + np.mean(optimizer_times))*1000:.2f}ms")
        
        return {
            'forward': np.mean(forward_times),
            'loss': np.mean(loss_times),
            'backward': np.mean(backward_times),
            'optimizer': np.mean(optimizer_times),
        }


class MemoryProfiler:
    """Profile GPU memory usage."""
    
    @staticmethod
    def log_memory_stats(prefix: str = ""):
        """Log current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            logger.info(f"  {prefix}GPU Memory - Allocated: {allocated:.2f}GB | Reserved: {reserved:.2f}GB | Peak: {max_allocated:.2f}GB")
    
    @staticmethod
    def reset_peak_memory():
        """Reset peak memory stats."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


def profile_embedding_computation(encoder, texts: List[str], batch_sizes: List[int] = [8, 16, 32, 64, 128]):
    """
    Profile different batch sizes for embedding computation.
    This helps identify optimal batch size for pre-computing embeddings.
    """
    logger.info("\n🧪 Profiling Embedding Computation...")
    
    # Sample texts
    sample_texts = texts[:500] if len(texts) > 500 else texts
    
    results = {}
    for batch_size in batch_sizes:
        times = []
        for i in range(0, len(sample_texts), batch_size):
            batch = sample_texts[i:i+batch_size]
            
            start = time.perf_counter()
            _ = encoder.encode(batch, show_progress_bar=False, convert_to_numpy=False)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            times.append(elapsed)
        
        avg_time = np.mean(times)
        throughput = batch_size / avg_time
        results[batch_size] = {'time': avg_time, 'throughput': throughput}
        logger.info(f"  Batch size {batch_size:3d}: {avg_time:.4f}s/batch, {throughput:.1f} texts/s")
    
    return results


def compare_num_workers(dataset, batch_size: int, num_workers_list: List[int] = [0, 2, 4, 8]):
    """
    Compare DataLoader performance with different num_workers settings.
    """
    from torch.utils.data import DataLoader
    from src.cpp.cpp_dataset import collate_career_path_batch
    
    logger.info("\n👷 Testing DataLoader num_workers...")
    
    results = {}
    for num_workers in num_workers_list:
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_career_path_batch,
            pin_memory=True
        )
        
        times = []
        start = time.perf_counter()
        for i, batch in enumerate(loader):
            if i >= 20:  # Test first 20 batches
                break
            times.append(time.perf_counter() - start)
            start = time.perf_counter()
        
        avg_time = np.mean(times[1:])  # Skip first batch (warmup)
        results[num_workers] = avg_time
        logger.info(f"  num_workers={num_workers}: {avg_time:.4f}s/batch")
    
    return results


def estimate_total_time(train_size: int, batch_size: int, time_per_batch: float, 
                       num_epochs: int, num_trials: int = 1):
    """
    Estimate total training time based on profiling results.
    """
    batches_per_epoch = (train_size + batch_size - 1) // batch_size
    time_per_epoch = batches_per_epoch * time_per_batch
    total_time = time_per_epoch * num_epochs * num_trials
    
    logger.info("\n⏰ Time Estimation:")
    logger.info(f"  Batches per epoch: {batches_per_epoch}")
    logger.info(f"  Time per epoch: {time_per_epoch/60:.2f} minutes")
    logger.info(f"  Total time ({num_trials} trials × {num_epochs} epochs): {total_time/3600:.2f} hours")


# PyTorch Profiler integration
@contextlib.contextmanager
def pytorch_profiler(output_dir: str = "profiler_output", enabled: bool = True):
    """
    Use PyTorch's built-in profiler for detailed CPU/GPU analysis.
    View results with: tensorboard --logdir=profiler_output
    """
    if not enabled:
        yield None
        return
        
    logger.info(f"🔍 Starting PyTorch Profiler (output: {output_dir})")
    
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        yield prof


if __name__ == "__main__":
    logger.info("Profiling utilities loaded. Import and use in your training script.")














