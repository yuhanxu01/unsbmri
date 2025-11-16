#!/bin/bash
# ==============================================================================
# Test All Experiments
# ==============================================================================
#
# This script runs inference on all trained models
#
# Usage:
#   bash test_all_experiments.sh
#
# Or test specific experiments:
#   bash test_all_experiments.sh exp1 exp3 exp5
#
# ==============================================================================

set -e

# Configuration
DATAROOT="./datasets/YOUR_DATASET"  # CHANGE THIS
RESULTS_DIR="./results"
EPOCH="latest"  # or specific epoch number

# Experiments to test
if [ $# -eq 0 ]; then
    EXPERIMENTS=("exp1_baseline" "exp2_latter_steps" "exp3_nila_adaptive" "exp4_combined" "exp5_full" "exp6_with_denoise_aug")
else
    EXPERIMENTS=("$@")
fi

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Testing All Experiments${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

mkdir -p $RESULTS_DIR

for exp_name in "${EXPERIMENTS[@]}"; do
    echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} Testing ${exp_name}..."

    python test.py \
        --name $exp_name \
        --dataroot $DATAROOT \
        --model sb \
        --dataset_mode mri_unaligned \
        --mri_representation real_imag \
        --mri_normalize_per_case \
        --mri_normalize_method percentile_95 \
        --input_nc 2 \
        --output_nc 2 \
        --num_timesteps 20 \
        --netG resnet_9blocks_cond \
        --epoch $EPOCH \
        --results_dir $RESULTS_DIR/$exp_name \
        --num_test 50 \
        --gpu_ids 0

    echo -e "${GREEN}  Results saved to: ${RESULTS_DIR}/${exp_name}${NC}"
    echo ""
done

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}All experiments tested!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Results directory: $RESULTS_DIR"
echo ""
echo "Next step: Run evaluation"
echo "  python evaluate_experiments.py --results_dir $RESULTS_DIR"
