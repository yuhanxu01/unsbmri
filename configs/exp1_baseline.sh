#!/bin/bash
# Experiment 1: Baseline (No improvements)
# This is the original UNSB without any noise-adaptive modifications

python train.py \
  --name exp1_baseline \
  --dataroot ./datasets/YOUR_DATASET \
  --model sb \
  --dataset_mode mri_unaligned \
  --mri_representation real_imag \
  --mri_normalize_per_case \
  --mri_normalize_method percentile_95 \
  --input_nc 2 \
  --output_nc 2 \
  --ngf 64 \
  --ndf 64 \
  --num_timesteps 20 \
  --netG resnet_9blocks_cond \
  --netD basic_cond \
  --netE basic_cond \
  --tau 0.1 \
  --batch_size 4 \
  --lr 0.0002 \
  --n_epochs 200 \
  --n_epochs_decay 200 \
  --lambda_GAN 1.0 \
  --lambda_NCE 1.0 \
  --lambda_SB 0.1 \
  --nce_idt \
  --wandb_project mri-noise-adaptive-experiments \
  --gpu_ids 0 \
  "$@"
