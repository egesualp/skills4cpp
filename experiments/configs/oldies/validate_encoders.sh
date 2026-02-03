#!/bin/bash
#SBATCH --job-name=validate_encoders        # Name of the job
#SBATCH --qos=mcml
#SBATCH --output=experiments/logs/validate_encoders_%j.log    # Log output (%j expands to job ID)
#SBATCH --error=experiments/logs/validate_encoders_%j.err     # Error output
#SBATCH --partition=mcml-dgx-a100-40x8  # Specify the partition to use
#SBATCH --gres=gpu:1                     # Number of GPUs to use
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --time=00:30:00                   # Time limit

# Script to validate encoder performance on different datasets
# This helps understand why decorte encoder performs better on decorte_esco data

echo "=========================================="
echo "Encoder Validation Experiments"
echo "=========================================="
echo ""

source /dss/dsshome1/02/ra95kix2/miniconda3/bin/activate thesis

# Create output directory
RESULTS_DIR="results/encoder_validation"
mkdir -p "$RESULTS_DIR"

# Test 1: decorte data with decorte encoder (EXPECTED MATCH)
echo "Test 1: decorte data + decorte encoder"
echo "------------------------------------------"
python -m src.cpp.validate_encoder \
    --data_type decorte \
    --encoder ElenaSenger/career-path-representation-mpnet-decorte \
    --only_titles \
    --epochs 10 \
    --lr 1e-4 \
    --batch_size 64 \
    | tee "$RESULTS_DIR/decorte_data_decorte_encoder.log"

echo ""
echo ""

# Test 2: decorte_esco data with decorte_esco encoder (EXPECTED MATCH)
echo "Test 2: decorte_esco data + decorte_esco encoder"
echo "------------------------------------------"
python -m src.cpp.validate_encoder \
    --data_type decorte_esco \
    --encoder ElenaSenger/career-path-representation-mpnet-decorte-esco \
    --only_titles \
    --epochs 10 \
    --lr 1e-4 \
    --batch_size 64 \
    | tee "$RESULTS_DIR/decorte_esco_data_decorte_esco_encoder.log"

echo ""
echo ""

# Test 3: decorte_esco data with decorte encoder (UNEXPECTED - BUT WORKS BETTER?)
echo "Test 3: decorte_esco data + decorte encoder (CROSS-TEST)"
echo "------------------------------------------"
python -m src.cpp.validate_encoder \
    --data_type decorte_esco \
    --encoder ElenaSenger/career-path-representation-mpnet-decorte \
    --only_titles \
    --epochs 10 \
    --lr 1e-4 \
    --batch_size 64 \
    | tee "$RESULTS_DIR/decorte_esco_data_decorte_encoder.log"

echo ""
echo ""

# Test 4: decorte data with decorte_esco encoder (CROSS-TEST)
echo "Test 4: decorte data + decorte_esco encoder (CROSS-TEST)"
echo "------------------------------------------"
python -m src.cpp.validate_encoder \
    --data_type decorte \
    --encoder ElenaSenger/career-path-representation-mpnet-decorte-esco \
    --only_titles \
    --epochs 10 \
    --lr 1e-4 \
    --batch_size 64 \
    | tee "$RESULTS_DIR/decorte_data_decorte_esco_encoder.log"

echo ""
echo ""
echo "=========================================="
echo "All tests completed!"
echo "Results saved to: $RESULTS_DIR"
echo "=========================================="
echo ""
echo "Summary:"
grep -A 2 "Test MRR" "$RESULTS_DIR"/*.log


