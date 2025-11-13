#!/bin/bash

# Paired testing script for UNSB model
# Tests 10 cases with 12 middle slices per case = 120 total slices
# Generates visualization images and computes SSIM, PSNR, NRMSE metrics

# Set default values
DATAROOT=${DATAROOT:-"./datasets/knee_mri"}
NAME=${NAME:-"unsb_experiment"}
EPOCH=${EPOCH:-"latest"}
MRI_REPR=${MRI_REPR:-"real_imag"}
GPU_IDS=${GPU_IDS:-"0"}

echo "======================================"
echo "Testing UNSB Model with Paired Data"
echo "10 Cases × 12 Middle Slices = 120 Total"
echo "======================================"
echo "Data root: $DATAROOT"
echo "Experiment name: $NAME"
echo "Epoch: $EPOCH"
echo "MRI representation: $MRI_REPR"
echo "======================================"
echo ""

python test_unsb_paired.py \
  --dataroot "$DATAROOT" \
  --name "$NAME" \
  --model sb \
  --dataset_mode mri_unaligned \
  --mri_representation "$MRI_REPR" \
  --mri_normalize_per_slice \
  --epoch "$EPOCH" \
  --gpu_ids "$GPU_IDS" \
  --phase test \
  --eval \
  "$@"

echo ""
echo "======================================"
echo "Testing completed!"
echo "Check results in ./results/${NAME}/"
echo "======================================"
