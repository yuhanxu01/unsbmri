#!/bin/bash
# Training script for I2SB (Image-to-Image Schrödinger Bridge) model
# This script is configured for paired MRI reconstruction

# ============================================================================
# Environment Configuration
# ============================================================================
PYTHON_BIN="/home/user/anaconda3/envs/pytorch/bin/python"
DATAROOT="/home/user/data/mri_paired"  # Update this path

# ============================================================================
# Experiment Configuration
# ============================================================================
NAME="i2sb_mri_paired"
MODEL="i2sb"  # Use the new I2SB model
GPU_IDS="0"

# ============================================================================
# Data Configuration
# ============================================================================
DATASET_MODE="mri_unaligned"
DIRECTION="AtoB"
INPUT_NC=1
OUTPUT_NC=1
LOAD_SIZE=256
CROP_SIZE=256
BATCH_SIZE=4

# MRI-specific data settings
MRI_REPRESENTATION="magnitude"  # or "real_imag"
MRI_NORMALIZE="--mri_normalize_per_slice"  # or "--mri_normalize_per_case"
MRI_SLICE_PREFIX="slices_"

# ============================================================================
# I2SB Model Configuration
# ============================================================================
# Network architecture
NETG="resnet_9blocks_cond"  # Conditional ResNet with time embedding
NGF=64
NORMG="instance"
NO_DROPOUT=""  # Use "" to enable dropout, "--no_dropout" to disable

# Diffusion parameters
I2SB_NUM_TIMESTEPS=1000  # Number of diffusion timesteps for training
I2SB_BETA_SCHEDULE="linear"  # Options: linear, cosine, quadratic
I2SB_BETA_START=0.0001
I2SB_BETA_END=0.02
I2SB_OBJECTIVE="pred_noise"  # Options: pred_noise, pred_x0, pred_v

# Sampling parameters
I2SB_SAMPLING_TIMESTEPS=250  # Fewer steps for faster sampling
I2SB_DDIM_ETA=0.0  # 0 = deterministic DDIM, 1 = stochastic DDPM

# ============================================================================
# Loss Configuration
# ============================================================================
LAMBDA_DIFFUSION=1.0  # Main diffusion loss weight
LAMBDA_SIMPLE=1.0     # Simple MSE loss weight
LAMBDA_L1=0.1         # L1 loss on x0 prediction
LAMBDA_PERCEPTUAL=0.0  # Perceptual loss (requires LPIPS)
LAMBDA_VLB=0.0        # Variational lower bound

# GAN loss (optional, for quality improvement)
USE_GAN="--use_gan"  # Use "" to disable GAN
LAMBDA_GAN=0.1

# ============================================================================
# Training Configuration
# ============================================================================
N_EPOCHS=100
N_EPOCHS_DECAY=100
LR=0.0002
BETA1=0.5
BETA2=0.999
LR_POLICY="linear"

# ============================================================================
# Display and Logging Configuration
# ============================================================================
DISPLAY_FREQ=100
PRINT_FREQ=100
SAVE_LATEST_FREQ=5000
SAVE_EPOCH_FREQ=5

# Weights & Biases logging
WANDB_PROJECT="mri-i2sb"
WANDB_NAME="${NAME}"

# ============================================================================
# Additional Options
# ============================================================================
NUM_THREADS=4
INIT_TYPE="normal"
INIT_GAIN=0.02

# Paired data settings (I2SB uses paired data exclusively)
PAIRED_STAGE="--paired_stage"
COMPUTE_METRICS="--compute_paired_metrics"

# ============================================================================
# Execute Training
# ============================================================================
$PYTHON_BIN train.py \
  --dataroot $DATAROOT \
  --name $NAME \
  --model $MODEL \
  --gpu_ids $GPU_IDS \
  --dataset_mode $DATASET_MODE \
  --direction $DIRECTION \
  --input_nc $INPUT_NC \
  --output_nc $OUTPUT_NC \
  --load_size $LOAD_SIZE \
  --crop_size $CROP_SIZE \
  --batch_size $BATCH_SIZE \
  --netG $NETG \
  --ngf $NGF \
  --normG $NORMG \
  $NO_DROPOUT \
  --i2sb_num_timesteps $I2SB_NUM_TIMESTEPS \
  --i2sb_beta_schedule $I2SB_BETA_SCHEDULE \
  --i2sb_beta_start $I2SB_BETA_START \
  --i2sb_beta_end $I2SB_BETA_END \
  --i2sb_objective $I2SB_OBJECTIVE \
  --i2sb_sampling_timesteps $I2SB_SAMPLING_TIMESTEPS \
  --i2sb_ddim_sampling_eta $I2SB_DDIM_ETA \
  --lambda_diffusion $LAMBDA_DIFFUSION \
  --lambda_simple $LAMBDA_SIMPLE \
  --lambda_l1 $LAMBDA_L1 \
  --lambda_perceptual $LAMBDA_PERCEPTUAL \
  --lambda_vlb $LAMBDA_VLB \
  $USE_GAN \
  --lambda_gan $LAMBDA_GAN \
  --n_epochs $N_EPOCHS \
  --n_epochs_decay $N_EPOCHS_DECAY \
  --lr $LR \
  --beta1 $BETA1 \
  --beta2 $BETA2 \
  --lr_policy $LR_POLICY \
  --display_freq $DISPLAY_FREQ \
  --print_freq $PRINT_FREQ \
  --save_latest_freq $SAVE_LATEST_FREQ \
  --save_epoch_freq $SAVE_EPOCH_FREQ \
  --num_threads $NUM_THREADS \
  --init_type $INIT_TYPE \
  --init_gain $INIT_GAIN \
  --mri_representation $MRI_REPRESENTATION \
  --mri_slice_prefix $MRI_SLICE_PREFIX \
  $MRI_NORMALIZE \
  $PAIRED_STAGE \
  $COMPUTE_METRICS \
  --wandb_project_name $WANDB_PROJECT \
  --wandb_run_name $WANDB_NAME \
  --isTrain \
  --continue_train
