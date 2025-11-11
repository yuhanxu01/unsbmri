#!/bin/bash
#SBATCH --partition=gpu4_medium
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1

/gpfs/scratch/rl5285/miniconda3/envs/UNSB/bin/python3.8 train.py \
    --dataroot ./datasets/PD2PDFS \
    --name PD2PDFS \
    --mode sb \
    --lambda_SB 1.0 \
    --lambda_NCE 1.0 \
    --mri_normalize_per_slice