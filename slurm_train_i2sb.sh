#!/bin/bash
#SBATCH --job-name=i2sb_mri
#SBATCH --output=logs/i2sb_%j.out
#SBATCH --error=logs/i2sb_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu

# ============================================================================
# SLURM Training Script for I2SB Model
# For paired MRI reconstruction using Schrödinger Bridge
# ============================================================================

# Print job information
echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Load necessary modules (adjust based on your cluster)
module load cuda/11.8
module load cudnn/8.6.0
module load python/3.9

# Activate conda environment
source activate pytorch

# Set working directory
cd /home/user/unsbmri || exit

# Create logs directory if it doesn't exist
mkdir -p logs

# ============================================================================
# Experiment Configuration
# ============================================================================
NAME="i2sb_mri_paired_${SLURM_JOB_ID}"
MODEL="i2sb"
GPU_IDS="0"

# Data paths (UPDATE THESE)
DATAROOT="/path/to/your/mri/data"

# ============================================================================
# Data Configuration
# ============================================================================
DATASET_MODE="mri_unaligned"
DIRECTION="AtoB"
INPUT_NC=1
OUTPUT_NC=1
LOAD_SIZE=256
CROP_SIZE=256
BATCH_SIZE=8  # Increase if you have more GPU memory

# MRI-specific settings
MRI_REPRESENTATION="magnitude"
MRI_NORMALIZE="--mri_normalize_per_slice"
MRI_SLICE_PREFIX="slices_"

# ============================================================================
# I2SB Model Configuration
# ============================================================================
# Network
NETG="resnet_9blocks_cond"
NGF=64
NORMG="instance"

# Diffusion parameters
I2SB_NUM_TIMESTEPS=1000
I2SB_BETA_SCHEDULE="linear"
I2SB_BETA_START=0.0001
I2SB_BETA_END=0.02
I2SB_OBJECTIVE="pred_noise"

# Sampling
I2SB_SAMPLING_TIMESTEPS=250
I2SB_DDIM_ETA=0.0

# ============================================================================
# Loss Configuration
# ============================================================================
LAMBDA_DIFFUSION=1.0
LAMBDA_SIMPLE=1.0
LAMBDA_L1=0.1
LAMBDA_PERCEPTUAL=0.0
LAMBDA_VLB=0.0

# Optional GAN
USE_GAN="--use_gan"
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
# Display and Logging
# ============================================================================
DISPLAY_FREQ=100
PRINT_FREQ=50
SAVE_LATEST_FREQ=5000
SAVE_EPOCH_FREQ=5

WANDB_PROJECT="mri-i2sb"
WANDB_NAME="${NAME}"

# ============================================================================
# Run Training
# ============================================================================
echo "=========================================="
echo "Starting I2SB Training"
echo "=========================================="
echo "Experiment Name: $NAME"
echo "Data Root: $DATAROOT"
echo "Diffusion Steps: $I2SB_NUM_TIMESTEPS"
echo "Objective: $I2SB_OBJECTIVE"
echo "=========================================="

python train.py \
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
  --num_threads 8 \
  --init_type normal \
  --init_gain 0.02 \
  --mri_representation $MRI_REPRESENTATION \
  --mri_slice_prefix $MRI_SLICE_PREFIX \
  $MRI_NORMALIZE \
  --paired_stage \
  --compute_paired_metrics \
  --wandb_project_name $WANDB_PROJECT \
  --wandb_run_name $WANDB_NAME \
  --isTrain

# Print completion information
echo "=========================================="
echo "Training Completed"
echo "End Time: $(date)"
echo "=========================================="
