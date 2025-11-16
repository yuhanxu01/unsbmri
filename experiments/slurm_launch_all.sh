#!/bin/bash
#SBATCH --partition=gpu4_medium
#SBATCH --nodes=1
#SBATCH --tasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:8

# SLURM batch launcher for all 8 experiments
# All paired experiments use 10% data

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

echo "=========================================="
echo "SLURM Batch Launcher (8 Experiments)"
echo "=========================================="
echo "Node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: 8"
echo "All paired experiments: 10% data"
echo "=========================================="
echo ""

# Use the 8-GPU launcher
bash experiments/launch_all_8gpu.sh

exit 0
