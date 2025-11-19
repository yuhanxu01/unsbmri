#!/bin/bash
#SBATCH --partition=gpu4_medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --job-name=exp7_2s10_out

# Experiment 7: Two-Stage 10% - OT Output Only
# Load pretrained, 10% paired data, only OT_output loss

# Change to the working directory
BASE_DIR="/gpfs/scratch/rl5285/test/unsbmri"
cd "$BASE_DIR" || { echo "ERROR: Failed to cd to $BASE_DIR"; exit 1; }

echo "=========================================="
echo "Experiment 7: Two-Stage 10% - OT Output Only"
echo "=========================================="
echo "Node: $SLURM_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo "Working Dir: $(pwd)"
echo "=========================================="
echo ""

# Configuration
export DATAROOT="/gpfs/scratch/rl5285/unsb_mri/datasets/fastmri_knee"
export PYTHON_BIN="/gpfs/scratch/rl5285/miniconda3/envs/UNSB/bin/python3.8"

# Experiment setup
export EXPERIMENT_NAME="ablation_exp7_twostage_10p_OT_output"

# Training parameters
export N_EPOCHS=500          # Keep constant LR until epoch 500
export N_EPOCHS_DECAY=100    # Decay for 100 more epochs (501-600)
export BATCH_SIZE=1

# Paired training configuration
export PAIRED_STAGE="--paired_stage"
export PAIRED_SUBSET_RATIO=0.1   # 10% paired data
export PAIRED_SUBSET_SEED=42
export COMPUTE_METRICS="--compute_paired_metrics"

# Loss configuration for ablation study
export USE_OT_OUTPUT="--use_ot_output"
export DISABLE_GAN="--disable_gan"
export DISABLE_NCE="--disable_nce"

# Load from pre-trained unpaired model
export CONTINUE_TRAIN="--continue_train"
export PRETRAINED_NAME="unpaired"
export LOAD_EPOCH="latest"
export EPOCH_COUNT=401       # Continue from epoch 401

echo "Configuration:"
echo "  Pretrained: checkpoints/unpaired"
echo "  Output: checkpoints/$EXPERIMENT_NAME"
echo "  Paired data: 10%"
echo "  Loss: OT_output only"
echo "  Epochs: 401-500 (constant LR) + 501-600 (decay)"
echo ""

# Run training
bash run_train.sh

exit_code=$?

echo ""
echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "Training completed: Experiment 7"
else
    echo "Training failed: Experiment 7"
fi
echo "=========================================="

exit $exit_code
