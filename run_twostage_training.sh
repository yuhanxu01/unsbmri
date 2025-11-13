#!/bin/bash
#SBATCH --partition=gpu4_medium
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1

# Two-Stage Training Experiment
# Stage 1: Unpaired training on full dataset
# Stage 2: Paired training on 30% subset with L1 loss, continuing from Stage 1 checkpoint

# Configuration
DATAROOT=/gpfs/scratch/rl5285/unsb_mri/datasets/fastmri_knee
PYTHON_BIN=/gpfs/scratch/rl5285/miniconda3/envs/UNSB/bin/python3.8

# Experiment names
STAGE1_NAME=PDtoPDFS_mag_stage1_unpaired
STAGE2_NAME=PDtoPDFS_mag_stage2_paired_30pct

# Training parameters
BATCH_SIZE=1
N_EPOCHS_STAGE1=200
N_EPOCHS_DECAY_STAGE1=200
N_EPOCHS_STAGE2=100
N_EPOCHS_DECAY_STAGE2=100

# Loss weights
LAMBDA_SB=1.0
LAMBDA_NCE=1.0
LAMBDA_L1=1.0

echo "=========================================="
echo "Two-Stage Training Experiment"
echo "=========================================="
echo "Stage 1: Unpaired training (${N_EPOCHS_STAGE1} + ${N_EPOCHS_DECAY_STAGE1} epochs)"
echo "Stage 2: Paired training with 30% data (${N_EPOCHS_STAGE2} + ${N_EPOCHS_DECAY_STAGE2} epochs)"
echo "=========================================="
echo ""

#############################################
# Stage 1: Unpaired Training
#############################################
echo "=========================================="
echo "STAGE 1: Starting Unpaired Training"
echo "=========================================="
echo "Experiment name: ${STAGE1_NAME}"
echo "Data: Full dataset, unpaired mode"
echo "Epochs: ${N_EPOCHS_STAGE1} + ${N_EPOCHS_DECAY_STAGE1}"
echo ""

$PYTHON_BIN train.py \
  --dataroot $DATAROOT \
  --name $STAGE1_NAME \
  --dataset_mode mri_unaligned \
  --mri_representation magnitude \
  --input_nc 1 \
  --output_nc 1 \
  --batch_size $BATCH_SIZE \
  --n_epochs $N_EPOCHS_STAGE1 \
  --n_epochs_decay $N_EPOCHS_DECAY_STAGE1 \
  --lambda_SB $LAMBDA_SB \
  --lambda_NCE $LAMBDA_NCE \
  --mri_normalize_per_slice \
  --mode sb

# Check if Stage 1 completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "STAGE 1: Completed Successfully"
    echo "=========================================="
    echo ""
else
    echo ""
    echo "=========================================="
    echo "STAGE 1: FAILED"
    echo "=========================================="
    exit 1
fi

#############################################
# Stage 2: Paired Training (30% subset)
#############################################
echo "=========================================="
echo "STAGE 2: Starting Paired Training (30% subset)"
echo "=========================================="
echo "Experiment name: ${STAGE2_NAME}"
echo "Data: 30% paired subset"
echo "Epochs: ${N_EPOCHS_STAGE2} + ${N_EPOCHS_DECAY_STAGE2}"
echo "Loading from: ${STAGE1_NAME}/latest"
echo "L1 loss weight: ${LAMBDA_L1}"
echo ""

# Calculate the starting epoch for Stage 2
# Stage 2 continues from where Stage 1 left off
STAGE2_START_EPOCH=$((N_EPOCHS_STAGE1 + N_EPOCHS_DECAY_STAGE1 + 1))

$PYTHON_BIN train.py \
  --dataroot $DATAROOT \
  --name $STAGE2_NAME \
  --dataset_mode mri_unaligned \
  --mri_representation magnitude \
  --input_nc 1 \
  --output_nc 1 \
  --batch_size $BATCH_SIZE \
  --n_epochs $N_EPOCHS_STAGE2 \
  --n_epochs_decay $N_EPOCHS_DECAY_STAGE2 \
  --lambda_SB $LAMBDA_SB \
  --lambda_NCE $LAMBDA_NCE \
  --lambda_L1 $LAMBDA_L1 \
  --mri_normalize_per_slice \
  --mode sb \
  --paired_stage \
  --paired_subset_ratio 0.3 \
  --paired_subset_seed 42 \
  --compute_paired_metrics \
  --continue_train \
  --pretrained_name $STAGE1_NAME \
  --epoch latest \
  --epoch_count $STAGE2_START_EPOCH

# Check if Stage 2 completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "STAGE 2: Completed Successfully"
    echo "=========================================="
    echo ""
    echo "=========================================="
    echo "TWO-STAGE TRAINING COMPLETE"
    echo "=========================================="
    echo "Stage 1 checkpoint: ./checkpoints/${STAGE1_NAME}/"
    echo "Stage 2 checkpoint: ./checkpoints/${STAGE2_NAME}/"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "STAGE 2: FAILED"
    echo "=========================================="
    exit 1
fi
