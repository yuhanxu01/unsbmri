#!/bin/bash

# Paired testing script for CUT model on MRI knee data
# This script tests the CUT model with paired data and computes evaluation metrics
# (SSIM, PSNR, NRMSE)

# Set default values
DATAROOT=${DATAROOT:-"./datasets/knee_mri"}
NAME=${NAME:-"cut_knee_experiment"}
EPOCH=${EPOCH:-"latest"}
MRI_REPR=${MRI_REPR:-"real_imag"}
GPU_IDS=${GPU_IDS:-"0"}
NUM_TEST=${NUM_TEST:-10000}

echo "======================================"
echo "Testing CUT Model with Paired Data"
echo "======================================"
echo "Data root: $DATAROOT"
echo "Experiment name: $NAME"
echo "Epoch: $EPOCH"
echo "MRI representation: $MRI_REPR"
echo "======================================"
echo ""

python test_paired.py \
  --dataroot "$DATAROOT" \
  --name "$NAME" \
  --model cut \
  --dataset_mode mri_unaligned \
  --mri_representation "$MRI_REPR" \
  --mri_normalize_per_slice \
  --epoch "$EPOCH" \
  --gpu_ids "$GPU_IDS" \
  --num_test "$NUM_TEST" \
  --phase test \
  --eval \
  "$@"

echo ""
echo "======================================"
echo "Testing completed!"
echo "Check results in ./results/${NAME}/"
echo "======================================"
