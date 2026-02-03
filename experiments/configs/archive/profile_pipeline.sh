#!/bin/bash
# Convenience script for profiling the training pipeline

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

cd "$PROJECT_ROOT"

echo "=================================="
echo "Training Pipeline Profiler"
echo "=================================="
echo ""

# Parse command
COMMAND=${1:-help}

case $COMMAND in
    quick)
        echo "Running quick bottleneck test..."
        python src/cpp/test_bottlenecks.py --quick
        ;;
    
    full)
        echo "Running full bottleneck analysis..."
        python src/cpp/test_bottlenecks.py --full
        ;;
    
    data)
        echo "Testing data loading bottleneck..."
        python src/cpp/test_bottlenecks.py --component data_loading
        ;;
    
    loader)
        echo "Testing DataLoader bottleneck..."
        python src/cpp/test_bottlenecks.py --component dataloader
        ;;
    
    model)
        echo "Testing model training bottleneck..."
        python src/cpp/test_bottlenecks.py --component model
        ;;
    
    epoch)
        echo "Testing full epoch bottleneck..."
        python src/cpp/test_bottlenecks.py --component epoch
        ;;
    
    train)
        echo "Running training with profiling instrumentation..."
        shift  # Remove 'train' from args
        python src/cpp/train_with_profiling.py --quick-profile "$@"
        ;;
    
    gpu)
        echo "Monitoring GPU usage (Ctrl+C to stop)..."
        echo "Run your training in another terminal!"
        watch -n 0.5 nvidia-smi
        ;;
    
    compare-workers)
        echo "Comparing different num_workers settings..."
        for workers in 0 2 4 8; do
            echo ""
            echo "Testing num_workers=$workers..."
            python src/cpp/test_bottlenecks.py --component dataloader --quick 2>&1 | grep "num_workers=$workers"
        done
        ;;
    
    estimate)
        echo "Estimating training time..."
        python src/cpp/test_bottlenecks.py --component epoch --quick 2>&1 | grep -A 5 "Time Estimation"
        ;;
    
    help|*)
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  quick              - Quick bottleneck test (5 min)"
        echo "  full               - Full bottleneck analysis (15-30 min)"
        echo "  data               - Test data loading only"
        echo "  loader             - Test DataLoader configuration"
        echo "  model              - Test model training speed"
        echo "  epoch              - Test full epoch breakdown"
        echo "  train [args]       - Run training with profiling"
        echo "  gpu                - Monitor GPU usage in real-time"
        echo "  compare-workers    - Compare num_workers settings"
        echo "  estimate           - Estimate total training time"
        echo "  help               - Show this help"
        echo ""
        echo "Examples:"
        echo "  $0 quick                                    # Quick test"
        echo "  $0 train --n_trials 10 --max_epochs 5      # Profile training"
        echo "  $0 epoch                                    # Test epoch speed"
        echo ""
        ;;
esac

