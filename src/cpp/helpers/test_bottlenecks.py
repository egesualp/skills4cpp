"""
Test script to identify bottlenecks in the training pipeline.

Usage:
    python src/cpp/test_bottlenecks.py --quick        # Quick test
    python src/cpp/test_bottlenecks.py --full         # Full profiling
    python src/cpp/test_bottlenecks.py --component data_loading  # Test specific component
"""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from loguru import logger
import numpy as np

# Import original modules
from src.cpp.data_classes import Data
from src.cpp.utils import SEP_TOKEN
from src.cpp.cpp_dataset import CareerPathDataset, collate_career_path_batch
from src.cpp.data_loaders import (
    load_all_vocabs,
    load_job_and_skill_data,
    precompute_target_embeddings,
    precompute_input_embeddings
)
from src.cpp.train_cpp_enhanced import MultiModalCPPModel, SimpleConcatModel

# Import profiling utilities
from src.cpp.profile_training import (
    PerformanceProfiler,
    DataLoaderProfiler,
    ModelProfiler,
    MemoryProfiler,
    TimingStats,
    compare_num_workers,
    estimate_total_time
)


def test_data_loading_bottleneck(args):
    """Test data loading and pre-processing bottlenecks."""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: DATA LOADING BOTTLENECK")
    logger.info("="*80)
    
    stats = TimingStats()
    
    # Step 1: Encoder loading
    with PerformanceProfiler("Load encoder"):
        encoder_text = SentenceTransformer(args.encoder_text)
        encoder_skill = encoder_text
        text_dim = encoder_text.get_sentence_embedding_dimension()
        skill_text_dim = text_dim
    
    # Step 2: Load vocabs
    with PerformanceProfiler("Load vocabularies") as timer:
        all_vocabs = load_all_vocabs(args.vocab_dir)
        structured_dim = sum(len(vocab) for vocab in all_vocabs.values())
    stats.record("Load vocabularies", timer.elapsed)
    
    # Step 3: Load skill mappings
    with PerformanceProfiler("Load skill mappings") as timer:
        job_skill_map, esco_skill_text_map, skill_properties_map = load_job_and_skill_data(
            master_skill_file=args.master_skill_file,
            esco_skills_file=args.esco_skills_file,
            skill_properties_file=args.skill_properties_file,
            pooling_strategy=args.pooling_strategy,
            alpha=args.alpha,
            beta=args.beta
        )
    stats.record("Load skill mappings", timer.elapsed)
    
    # Step 4: Load data pairs
    with PerformanceProfiler("Load career path data") as timer:
        data = Data(DATA_TYPE=args.data_type, ONLY_TITLES=True)
        train_pairs, val_pairs, test_pairs = data.get_data(stage='transformation_finetuning')
    stats.record("Load career path data", timer.elapsed)
    
    # Use subset for quick testing
    if args.quick:
        train_pairs = train_pairs[:1000]
        val_pairs = val_pairs[:200]
    
    logger.info(f"Dataset sizes - Train: {len(train_pairs)}, Val: {len(val_pairs)}")
    
    # Step 5: Pre-compute target embeddings
    with PerformanceProfiler("Pre-compute target embeddings") as timer:
        actual_labels = list(set([pair[1] for pair in train_pairs + val_pairs]))
        Y_target_dict = precompute_target_embeddings(encoder_text, actual_labels, show_progress=True)
        Y_target_all = np.array(list(Y_target_dict.values()))
        output_dim = Y_target_all.shape[1]
    stats.record("Pre-compute target embeddings", timer.elapsed)
    
    # Step 6: Pre-compute input embeddings (THIS IS OFTEN A BOTTLENECK!)
    with PerformanceProfiler("Pre-compute input embeddings (train)") as timer:
        train_pairs, train_h_text, train_h_skill = precompute_input_embeddings(
            train_pairs, Y_target_dict, encoder_text, encoder_skill,
            job_skill_map, esco_skill_text_map,
            use_skill_description=False,
            pooling_strategy=args.pooling_strategy, alpha=args.alpha, beta=args.beta,
            use_text_history=True, use_skill_text=True
        )
    stats.record("Pre-compute input embeddings (train)", timer.elapsed)
    
    with PerformanceProfiler("Pre-compute input embeddings (val)") as timer:
        val_pairs, val_h_text, val_h_skill = precompute_input_embeddings(
            val_pairs, Y_target_dict, encoder_text, encoder_skill,
            job_skill_map, esco_skill_text_map,
            use_skill_description=False,
            pooling_strategy=args.pooling_strategy, alpha=args.alpha, beta=args.beta,
            use_text_history=True, use_skill_text=True
        )
    stats.record("Pre-compute input embeddings (val)", timer.elapsed)
    
    stats.report()
    
    return {
        'train_pairs': train_pairs,
        'val_pairs': val_pairs,
        'train_h_text': train_h_text,
        'train_h_skill': train_h_skill,
        'val_h_text': val_h_text,
        'val_h_skill': val_h_skill,
        'encoder_text': encoder_text,
        'encoder_skill': encoder_skill,
        'Y_target_dict': Y_target_dict,
        'Y_target_all': Y_target_all,
        'job_skill_map': job_skill_map,
        'esco_skill_text_map': esco_skill_text_map,
        'skill_properties_map': skill_properties_map,
        'all_vocabs': all_vocabs,
        'text_dim': text_dim,
        'skill_text_dim': skill_text_dim,
        'structured_dim': structured_dim,
        'output_dim': output_dim
    }


def test_dataloader_bottleneck(data_dict, args):
    """Test DataLoader performance with different configurations."""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: DATALOADER BOTTLENECK")
    logger.info("="*80)
    
    # Create dataset
    train_dataset = CareerPathDataset(
        data_pairs=data_dict['train_pairs'],
        encoder=data_dict['encoder_text'],
        Y_target_dict=data_dict['Y_target_dict'],
        job_skill_map=data_dict['job_skill_map'],
        esco_skill_text_map=data_dict['esco_skill_text_map'],
        skill_properties_map=data_dict['skill_properties_map'],
        all_vocabs=data_dict['all_vocabs'],
        use_skill_description=False,
        pooling_strategy=args.pooling_strategy,
        alpha=args.alpha,
        beta=args.beta,
        encoder_skill=data_dict['encoder_skill'],
        include_text=True,
        include_skill_text=True,
        include_structured=True,
        pre_h_text=data_dict['train_h_text'],
        pre_h_skill_text=data_dict['train_h_skill'],
    )
    
    logger.info(f"Dataset size: {len(train_dataset)}")
    
    # Test different num_workers
    num_workers_list = [0, 2, 4, 8] if not args.quick else [0, 4]
    worker_results = compare_num_workers(train_dataset, args.batch_size, num_workers_list)
    
    # Find best num_workers
    best_workers = min(worker_results.items(), key=lambda x: x[1])
    logger.info(f"\n✅ Best num_workers: {best_workers[0]} ({best_workers[1]:.4f}s/batch)")
    
    # Profile DataLoader in detail
    best_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=best_workers[0],
        collate_fn=collate_career_path_batch,
        pin_memory=True
    )
    
    dataloader_stats = DataLoaderProfiler.profile_dataloader(best_loader, num_batches=20, device=args.device)
    
    return best_loader, best_workers[0], dataloader_stats


def test_model_bottleneck(data_dict, train_loader, args):
    """Test model forward/backward pass bottlenecks."""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: MODEL TRAINING BOTTLENECK")
    logger.info("="*80)
    
    device = torch.device(args.device)
    
    # Create model
    if args.use_advanced:
        model = MultiModalCPPModel(
            text_dim=data_dict['text_dim'],
            skill_text_dim=data_dict['skill_text_dim'],
            structured_dim=data_dict['structured_dim'],
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            dropout=args.dropout,
            output_dim=data_dict['output_dim'],
            use_modality_weights=False,
            use_text=True,
            use_skill=True,
            use_struct=True
        ).to(device)
    else:
        input_dim = data_dict['text_dim'] + data_dict['skill_text_dim'] + data_dict['structured_dim']
        model = SimpleConcatModel(
            input_dim=input_dim,
            output_dim=data_dict['output_dim'],
            n_layers=args.n_layers,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            use_text=True,
            use_skill=True,
            use_struct=True
        ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CosineEmbeddingLoss()
    
    # Get a sample batch
    batch = next(iter(train_loader))
    
    # Profile model step
    MemoryProfiler.log_memory_stats("Before training: ")
    model_stats = ModelProfiler.profile_model_step(
        model, batch, optimizer, criterion, device, num_iters=20
    )
    MemoryProfiler.log_memory_stats("After training: ")
    
    # Estimate throughput
    total_step_time = sum(model_stats.values())
    samples_per_second = args.batch_size / total_step_time
    logger.info(f"\n📊 Training throughput: {samples_per_second:.1f} samples/second")
    
    return model_stats


def test_full_epoch_bottleneck(data_dict, train_loader, args):
    """Test a full training epoch to identify bottlenecks."""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: FULL EPOCH BOTTLENECK")
    logger.info("="*80)
    
    device = torch.device(args.device)
    
    # Create model
    if args.use_advanced:
        model = MultiModalCPPModel(
            text_dim=data_dict['text_dim'],
            skill_text_dim=data_dict['skill_text_dim'],
            structured_dim=data_dict['structured_dim'],
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            dropout=args.dropout,
            output_dim=data_dict['output_dim'],
            use_modality_weights=False,
            use_text=True,
            use_skill=True,
            use_struct=True
        ).to(device)
    else:
        input_dim = data_dict['text_dim'] + data_dict['skill_text_dim'] + data_dict['structured_dim']
        model = SimpleConcatModel(
            input_dim=input_dim,
            output_dim=data_dict['output_dim'],
            n_layers=args.n_layers,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            use_text=True,
            use_skill=True,
            use_struct=True
        ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CosineEmbeddingLoss()
    
    # Time a full epoch
    model.train()
    
    batch_times = []
    data_load_times = []
    train_step_times = []
    
    start_data = time.perf_counter()
    
    max_batches = 50 if args.quick else len(train_loader)
    
    for i, batch in enumerate(train_loader):
        if i >= max_batches:
            break
            
        # Time data loading
        data_load_time = time.perf_counter() - start_data
        data_load_times.append(data_load_time)
        
        # Time training step
        start_train = time.perf_counter()
        
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        y_pred = model(batch)
        target = torch.ones(y_pred.size(0)).to(device)
        loss = criterion(y_pred, batch['y'], target)
        loss.backward()
        optimizer.step()
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        train_step_time = time.perf_counter() - start_train
        train_step_times.append(train_step_time)
        
        batch_time = data_load_time + train_step_time
        batch_times.append(batch_time)
        
        if i % 10 == 0:
            logger.info(f"  Batch {i:3d}: Data={data_load_time*1000:6.2f}ms, Train={train_step_time*1000:6.2f}ms, Total={batch_time*1000:6.2f}ms")
        
        start_data = time.perf_counter()
    
    # Summary
    avg_data_load = np.mean(data_load_times[1:])  # Skip first (warmup)
    avg_train_step = np.mean(train_step_times[1:])
    avg_batch = np.mean(batch_times[1:])
    
    logger.info(f"\n📊 Epoch Statistics:")
    logger.info(f"  Avg data loading:   {avg_data_load*1000:.2f}ms ({avg_data_load/avg_batch*100:.1f}% of batch time)")
    logger.info(f"  Avg training step:  {avg_train_step*1000:.2f}ms ({avg_train_step/avg_batch*100:.1f}% of batch time)")
    logger.info(f"  Avg batch total:    {avg_batch*1000:.2f}ms")
    logger.info(f"  Throughput:         {args.batch_size/avg_batch:.1f} samples/s")
    
    # Estimate total time
    estimate_total_time(
        train_size=len(data_dict['train_pairs']),
        batch_size=args.batch_size,
        time_per_batch=avg_batch,
        num_epochs=10,
        num_trials=args.n_trials
    )
    
    return {
        'avg_data_load': avg_data_load,
        'avg_train_step': avg_train_step,
        'avg_batch': avg_batch
    }


def main():
    parser = argparse.ArgumentParser(description="Profile training bottlenecks")
    
    # Test mode
    parser.add_argument("--quick", action='store_true', help="Quick test with subset of data")
    parser.add_argument("--full", action='store_true', help="Full profiling")
    parser.add_argument("--component", type=str, choices=['data_loading', 'dataloader', 'model', 'epoch', 'all'],
                       default='all', help="Which component to test")
    
    # Data paths (same as train_cpp_enhanced.py)
    parser.add_argument("--data_type", type=str, default="decorte")
    parser.add_argument("--master_skill_file", type=str,
                       default="/dss/dsshome1/02/ra95kix2/thesis/skills4cpp/results/decorte_jobbert_v2_baseline/job_title_skills_master.csv")
    parser.add_argument("--esco_skills_file", type=str, default="data/esco_datasets/skills_en.csv")
    parser.add_argument("--vocab_dir", type=str, default="data/processed/master_datasets_2/")
    parser.add_argument("--skill_properties_file", type=str, default="data/processed/master_datasets_2/skill_properties_map.json")
    
    # Encoder
    parser.add_argument("--encoder_text", type=str, default="ElenaSenger/career-path-representation-mpnet-decorte")
    parser.add_argument("--pooling_strategy", type=str, default="weighted_idf")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    
    # Model
    parser.add_argument("--use_advanced", action='store_true')
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-5)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("TRAINING BOTTLENECK PROFILING")
    logger.info("="*80)
    logger.info(f"Mode: {'Quick' if args.quick else 'Full'}")
    logger.info(f"Component: {args.component}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("="*80)
    
    results = {}
    
    # Test 1: Data loading
    if args.component in ['data_loading', 'all']:
        data_dict = test_data_loading_bottleneck(args)
        results['data_loading'] = data_dict
    else:
        # Load minimal data for other tests
        data_dict = test_data_loading_bottleneck(args)
        results['data_loading'] = data_dict
    
    # Test 2: DataLoader
    if args.component in ['dataloader', 'all']:
        train_loader, best_workers, dataloader_stats = test_dataloader_bottleneck(data_dict, args)
        results['dataloader'] = {'best_workers': best_workers, 'stats': dataloader_stats}
    else:
        # Create default loader for other tests
        train_dataset = CareerPathDataset(
            data_pairs=data_dict['train_pairs'],
            encoder=data_dict['encoder_text'],
            Y_target_dict=data_dict['Y_target_dict'],
            job_skill_map=data_dict['job_skill_map'],
            esco_skill_text_map=data_dict['esco_skill_text_map'],
            skill_properties_map=data_dict['skill_properties_map'],
            all_vocabs=data_dict['all_vocabs'],
            use_skill_description=False,
            pooling_strategy=args.pooling_strategy,
            alpha=args.alpha,
            beta=args.beta,
            encoder_skill=data_dict['encoder_skill'],
            include_text=True,
            include_skill_text=True,
            include_structured=True,
            pre_h_text=data_dict['train_h_text'],
            pre_h_skill_text=data_dict['train_h_skill'],
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=4, collate_fn=collate_career_path_batch, pin_memory=True
        )
    
    # Test 3: Model
    if args.component in ['model', 'all']:
        model_stats = test_model_bottleneck(data_dict, train_loader, args)
        results['model'] = model_stats
    
    # Test 4: Full epoch
    if args.component in ['epoch', 'all']:
        epoch_stats = test_full_epoch_bottleneck(data_dict, train_loader, args)
        results['epoch'] = epoch_stats
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("PROFILING SUMMARY")
    logger.info("="*80)
    logger.info("\n🔍 Key Findings:")
    
    if 'dataloader' in results:
        logger.info(f"  • Best num_workers: {results['dataloader']['best_workers']}")
    
    if 'epoch' in results:
        epoch = results['epoch']
        data_pct = epoch['avg_data_load'] / epoch['avg_batch'] * 100
        train_pct = epoch['avg_train_step'] / epoch['avg_batch'] * 100
        
        logger.info(f"  • Data loading overhead: {data_pct:.1f}% of batch time")
        logger.info(f"  • Training computation: {train_pct:.1f}% of batch time")
        
        if data_pct > 30:
            logger.info("\n⚠️  WARNING: Data loading is a significant bottleneck!")
            logger.info("   Consider:")
            logger.info("   - Increasing num_workers")
            logger.info("   - Pre-loading more data into memory")
            logger.info("   - Using faster storage (SSD)")
        
        if train_pct > 70:
            logger.info("\n✅ Training is compute-bound (good for GPU utilization)")
            logger.info("   Consider:")
            logger.info("   - Increasing batch size to maximize GPU usage")
            logger.info("   - Using mixed precision training (torch.cuda.amp)")
    
    logger.info("\n" + "="*80)


if __name__ == "__main__":
    main()














