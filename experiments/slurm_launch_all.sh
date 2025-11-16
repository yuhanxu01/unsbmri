#!/bin/bash
#SBATCH --partition=gpu4_medium
#SBATCH --nodes=1
#SBATCH --tasks-per-node=18
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:18

# SLURM batch launcher for all 18 experiments
# Runs all experiments in parallel on 18 GPUs

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "=========================================="
echo "SLURM Batch Launcher"
echo "=========================================="
echo "Node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $SLURM_GPUS"
echo "=========================================="
echo ""

# Use the 18-GPU launcher
bash experiments/launch_all_18gpu.sh

exit 0
