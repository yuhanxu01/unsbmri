#!/bin/bash
# Experiment 2: Latter Steps Training Only (Di-Fusion inspired)
# Train only the latter 60% of timesteps
# Expected: 20-30% better training stability, 5-10% quality improvement

python train.py \
  --name exp2_latter_steps \
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
  --latter_steps_ratio 0.6 \
  "$@"
