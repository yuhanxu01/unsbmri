#!/bin/bash
# Experiment 4: Combined Nila + Di-Fusion
# Latter steps training + noise-adaptive weighting
# Expected: 40-60% noise reduction, better stability

python train.py \
  --name exp4_combined \
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
  --use_adaptive_sb_weight \
  --data_noise_level 0.03 \
  --noise_adaptive_schedule linear \
  --difusion_weight_schedule linear \
  "$@"
