#!/bin/bash
#SBATCH --partition=gpu4_medium
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1

# Full Paired Training Experiment
# Train with all paired data from the beginning with L1 loss supervision

# Configuration
DATAROOT=/gpfs/scratch/rl5285/unsb_mri/datasets/fastmri_knee
PYTHON_BIN=/gpfs/scratch/rl5285/miniconda3/envs/UNSB/bin/python3.8

# Experiment name
EXPERIMENT_NAME=PDtoPDFS_mag_paired_full

# Training parameters
BATCH_SIZE=1
N_EPOCHS=200
N_EPOCHS_DECAY=200

# Loss weights
LAMBDA_SB=1.0
LAMBDA_NCE=1.0
LAMBDA_L1=1.0

echo "=========================================="
echo "Full Paired Training Experiment"
echo "=========================================="
echo "Experiment name: ${EXPERIMENT_NAME}"
echo "Data: Full dataset, paired mode"
echo "Epochs: ${N_EPOCHS} + ${N_EPOCHS_DECAY}"
echo "L1 loss weight: ${LAMBDA_L1}"
echo "=========================================="
echo ""

$PYTHON_BIN train.py \
  --dataroot $DATAROOT \
  --name $EXPERIMENT_NAME \
  --dataset_mode mri_unaligned \
  --mri_representation magnitude \
  --input_nc 1 \
  --output_nc 1 \
  --batch_size $BATCH_SIZE \
  --n_epochs $N_EPOCHS \
  --n_epochs_decay $N_EPOCHS_DECAY \
  --lambda_SB $LAMBDA_SB \
  --lambda_NCE $LAMBDA_NCE \
  --lambda_L1 $LAMBDA_L1 \
  --mri_normalize_per_slice \
  --mode sb \
  --paired_stage \
  --paired_subset_ratio 1.0 \
  --compute_paired_metrics

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "TRAINING COMPLETED SUCCESSFULLY"
    echo "=========================================="
    echo "Checkpoint saved to: ./checkpoints/${EXPERIMENT_NAME}/"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "TRAINING FAILED"
    echo "=========================================="
    exit 1
fi
