#!/bin/bash
#SBATCH --partition=gpu4_medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# SLURM script for single paired training experiment
# Usage: sbatch slurm_train_paired.sh <strategy_name> <strategy_type>
# Example: sbatch slurm_train_paired.sh schemeA sb_gt_transport

STRATEGY_NAME=${1:-"schemeA"}
STRATEGY_TYPE=${2:-"sb_gt_transport"}

# Change to the working directory
BASE_DIR="/gpfs/scratch/rl5285/unsb_mri/unsbmri_2stage"
cd "$BASE_DIR" || { echo "ERROR: Failed to cd to $BASE_DIR"; exit 1; }

echo "=========================================="
echo "SLURM Paired Training"
echo "=========================================="
echo "Node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Strategy: $STRATEGY_NAME ($STRATEGY_TYPE)"
echo "Working Dir: $(pwd)"
echo "=========================================="
echo ""

# Configuration
export DATAROOT="/gpfs/scratch/rl5285/unsb_mri/datasets/fastmri_knee"
export PYTHON_BIN="/gpfs/scratch/rl5285/miniconda3/envs/UNSB/bin/python3.8"

# Experiment setup
export EXPERIMENT_NAME="baseline"
export STAGE_NAME="$STRATEGY_NAME"

# Training parameters
export N_EPOCHS=100
export N_EPOCHS_DECAY=100
export BATCH_SIZE=1

# Paired training configuration
export PAIRED_STAGE="--paired_stage"
export PAIRED_STRATEGY="$STRATEGY_TYPE"
export PAIRED_SUBSET_RATIO=0.1
export PAIRED_SUBSET_SEED=42
export COMPUTE_METRICS="--compute_paired_metrics"

# Load from pre-trained unpaired model
export CONTINUE_TRAIN="--continue_train"
export PRETRAINED_NAME="unpaired"
export LOAD_EPOCH="latest"
export EPOCH_COUNT=401  # Continue from epoch 401

echo "Configuration:"
echo "  Pretrained: checkpoints/unpaired"
echo "  Output: checkpoints/baseline_${STRATEGY_NAME}"
echo "  Paired data: 10%"
echo "  Epochs: 100+100 (continue from 401)"
echo ""

# Run training
bash run_train.sh

exit_code=$?

echo ""
echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "Training completed: $STRATEGY_NAME"
else
    echo "Training failed: $STRATEGY_NAME"
fi
echo "=========================================="

exit $exit_code
