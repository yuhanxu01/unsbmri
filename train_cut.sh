#!/bin/bash

# Training script for CUT model on MRI knee data
# This script trains the CUT (Contrastive Unpaired Translation) model
# on h5 knee MRI data for contrast transfer

# Set default values
DATAROOT=${DATAROOT:-"./datasets/knee_mri"}
NAME=${NAME:-"cut_knee_experiment"}
WANDB_PROJECT=${WANDB_PROJECT:-"cut-mri"}
BATCH_SIZE=${BATCH_SIZE:-4}
N_EPOCHS=${N_EPOCHS:-200}
N_EPOCHS_DECAY=${N_EPOCHS_DECAY:-200}
MRI_REPR=${MRI_REPR:-"real_imag"}
GPU_IDS=${GPU_IDS:-"0"}

echo "======================================"
echo "Training CUT Model on MRI Data"
echo "======================================"
echo "Data root: $DATAROOT"
echo "Experiment name: $NAME"
echo "MRI representation: $MRI_REPR"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $N_EPOCHS + $N_EPOCHS_DECAY (decay)"
echo "======================================"
echo ""

python train.py \
  --dataroot "$DATAROOT" \
  --name "$NAME" \
  --model cut \
  --CUT_mode CUT \
  --dataset_mode mri_unaligned \
  --mri_representation "$MRI_REPR" \
  --mri_normalize_per_slice \
  --wandb_project "$WANDB_PROJECT" \
  --batch_size "$BATCH_SIZE" \
  --n_epochs "$N_EPOCHS" \
  --n_epochs_decay "$N_EPOCHS_DECAY" \
  --gpu_ids "$GPU_IDS" \
  --netG resnet_9blocks \
  --netD basic \
  --lambda_GAN 1.0 \
  --lambda_NCE 1.0 \
  --nce_idt \
  --display_freq 400 \
  --print_freq 100 \
  --save_latest_freq 5000 \
  --save_epoch_freq 20 \
  --lr 0.0002 \
  --beta1 0.5 \
  "$@"

echo ""
echo "======================================"
echo "Training completed!"
echo "======================================"
