#!/bin/bash
# ==============================================================================
# Parallel Training Script for All Noise-Adaptive Experiments
# ==============================================================================
#
# This script runs all 6 experiments in parallel on different GPUs
# Make sure you have enough GPUs and modify GPU assignments as needed
#
# Usage:
#   bash run_all_experiments.sh
#
# Or run specific experiments:
#   bash run_all_experiments.sh exp1 exp3 exp5
#
# ==============================================================================

set -e  # Exit on error

# Configuration
DATAROOT="./datasets/YOUR_DATASET"  # CHANGE THIS
GPUS=(0 1 2 3 0 1)  # GPU assignments for each experiment

# Experiments to run (can be overridden by command line)
if [ $# -eq 0 ]; then
    EXPERIMENTS=("exp1" "exp2" "exp3" "exp4" "exp5" "exp6")
else
    EXPERIMENTS=("$@")
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting Noise-Adaptive Experiments${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Create logs directory
mkdir -p logs

# Function to run experiment
run_experiment() {
    local exp_name=$1
    local gpu_id=$2
    local config_file="configs/${exp_name}_baseline.sh"

    if [[ $exp_name == "exp2" ]]; then
        config_file="configs/exp2_latter_steps.sh"
    elif [[ $exp_name == "exp3" ]]; then
        config_file="configs/exp3_nila_adaptive.sh"
    elif [[ $exp_name == "exp4" ]]; then
        config_file="configs/exp4_combined.sh"
    elif [[ $exp_name == "exp5" ]]; then
        config_file="configs/exp5_full.sh"
    elif [[ $exp_name == "exp6" ]]; then
        config_file="configs/exp6_with_denoise_aug.sh"
    fi

    local log_file="logs/${exp_name}_$(date +%Y%m%d_%H%M%S).log"

    echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} Starting ${YELLOW}${exp_name}${NC} on GPU ${gpu_id}"
    echo -e "  Config: ${config_file}"
    echo -e "  Log: ${log_file}"

    # Run experiment in background
    CUDA_VISIBLE_DEVICES=$gpu_id bash $config_file \
        --dataroot $DATAROOT \
        --gpu_ids 0 \
        > $log_file 2>&1 &

    # Store PID
    echo $! >> logs/pids.txt
}

# Clean up previous PIDs
rm -f logs/pids.txt

# Launch all experiments
echo ""
echo -e "${BLUE}Launching experiments...${NC}"
echo ""

idx=0
for exp in "${EXPERIMENTS[@]}"; do
    gpu_id=${GPUS[$idx]}
    run_experiment $exp $gpu_id
    idx=$((idx + 1))
    sleep 2  # Stagger start times
done

echo ""
echo -e "${GREEN}All experiments launched!${NC}"
echo ""
echo -e "${BLUE}Monitor progress:${NC}"
echo "  - Check logs in logs/ directory"
echo "  - Use: tail -f logs/exp*.log"
echo "  - View tensorboard: tensorboard --logdir checkpoints/"
echo "  - WandB: https://wandb.ai/YOUR_PROJECT/mri-noise-adaptive-experiments"
echo ""
echo -e "${BLUE}Running processes:${NC}"
cat logs/pids.txt | while read pid; do
    if ps -p $pid > /dev/null 2>&1; then
        echo "  - PID $pid: Running"
    else
        echo "  - PID $pid: Not found"
    fi
done
echo ""

# Optional: Wait for all to complete
if [[ "${WAIT_FOR_ALL:-false}" == "true" ]]; then
    echo -e "${YELLOW}Waiting for all experiments to complete...${NC}"
    while read pid; do
        wait $pid
    done < logs/pids.txt
    echo -e "${GREEN}All experiments completed!${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}To stop all experiments:${NC}"
echo "  bash stop_experiments.sh"
echo -e "${BLUE}========================================${NC}"
