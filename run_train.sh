#!/bin/bash
# Modular training script for MRI contrast transfer experiments
# Usage: Set environment variables to configure, then run this script
# Example: EXPERIMENT_NAME=test PAIRED_STRATEGY=sb_gt_transport ./run_train.sh

# Default configuration
DATAROOT=${DATAROOT:-"/gpfs/scratch/rl5285/unsb_mri/datasets/fastmri_knee"}
PYTHON_BIN=${PYTHON_BIN:-"/gpfs/scratch/rl5285/miniconda3/envs/UNSB/bin/python3.8"}

# Experiment naming
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"experiment"}
STAGE_NAME=${STAGE_NAME:-""}  # For multi-stage training

# Basic training parameters
BATCH_SIZE=${BATCH_SIZE:-1}
N_EPOCHS=${N_EPOCHS:-200}
N_EPOCHS_DECAY=${N_EPOCHS_DECAY:-200}
EPOCH_COUNT=${EPOCH_COUNT:-1}

# Loss weights
LAMBDA_SB=${LAMBDA_SB:-1.0}
LAMBDA_NCE=${LAMBDA_NCE:-1.0}
LAMBDA_L1=${LAMBDA_L1:-0.0}
LAMBDA_PERCEPTUAL=${LAMBDA_PERCEPTUAL:-0.0}

# Paired training configuration
PAIRED_STAGE=${PAIRED_STAGE:-""}  # Set to "--paired_stage" to enable
PAIRED_STRATEGY=${PAIRED_STRATEGY:-"none"}  # none, sb_gt_transport, l1_loss, regularization, etc.
PAIRED_SUBSET_RATIO=${PAIRED_SUBSET_RATIO:-1.0}
PAIRED_SUBSET_SEED=${PAIRED_SUBSET_SEED:-42}
COMPUTE_METRICS=${COMPUTE_METRICS:-""}  # Set to "--compute_paired_metrics" to enable

# Resume training
CONTINUE_TRAIN=${CONTINUE_TRAIN:-""}  # Set to "--continue_train" to enable
PRETRAINED_NAME=${PRETRAINED_NAME:-""}
LOAD_EPOCH=${LOAD_EPOCH:-"latest"}

# Data configuration
MRI_REPRESENTATION=${MRI_REPRESENTATION:-"magnitude"}
INPUT_NC=${INPUT_NC:-1}
OUTPUT_NC=${OUTPUT_NC:-1}
NORMALIZE_MODE=${NORMALIZE_MODE:-"--mri_normalize_per_slice"}

# Full experiment name
if [ -n "$STAGE_NAME" ]; then
    FULL_NAME="${EXPERIMENT_NAME}_${STAGE_NAME}"
else
    FULL_NAME="${EXPERIMENT_NAME}"
fi

echo "======================================"
echo "MRI Contrast Transfer Training"
echo "======================================"
echo "Experiment: $FULL_NAME"
echo "Strategy: $PAIRED_STRATEGY"
echo "Epochs: $N_EPOCHS + $N_EPOCHS_DECAY"
echo "======================================"
echo ""

# Build command
CMD="$PYTHON_BIN train.py \
  --dataroot $DATAROOT \
  --name $FULL_NAME \
  --dataset_mode mri_unaligned \
  --mri_representation $MRI_REPRESENTATION \
  --input_nc $INPUT_NC \
  --output_nc $OUTPUT_NC \
  --batch_size $BATCH_SIZE \
  --n_epochs $N_EPOCHS \
  --n_epochs_decay $N_EPOCHS_DECAY \
  --epoch_count $EPOCH_COUNT \
  --lambda_SB $LAMBDA_SB \
  --lambda_NCE $LAMBDA_NCE \
  --lambda_L1 $LAMBDA_L1 \
  --lambda_perceptual $LAMBDA_PERCEPTUAL \
  --paired_strategy $PAIRED_STRATEGY \
  --paired_subset_ratio $PAIRED_SUBSET_RATIO \
  --paired_subset_seed $PAIRED_SUBSET_SEED \
  $NORMALIZE_MODE \
  --mode sb \
  $PAIRED_STAGE \
  $COMPUTE_METRICS \
  $CONTINUE_TRAIN"

# Add pretrained model if specified
if [ -n "$PRETRAINED_NAME" ]; then
    CMD="$CMD --pretrained_name $PRETRAINED_NAME --epoch $LOAD_EPOCH"
fi

echo "Running: $CMD"
echo ""

# Execute
eval $CMD

# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "Training completed successfully"
    echo "======================================"
else
    echo ""
    echo "======================================"
    echo "Training failed"
    echo "======================================"
    exit 1
fi
