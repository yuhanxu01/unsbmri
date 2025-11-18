#!/bin/bash
#SBATCH --partition=gpu4_medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=3-00:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --job-name=exp4_OT_out_E

# Experiment 4: Fully Paired - OT Output + Entropy
# From scratch, 100% paired data, OT_output + entropy loss

# Change to the working directory
BASE_DIR="/gpfs/scratch/rl5285/unsb_mri/unsbmri_2stage"
cd "$BASE_DIR" || { echo "ERROR: Failed to cd to $BASE_DIR"; exit 1; }

echo "=========================================="
echo "Experiment 4: Fully Paired - OT Output + Entropy"
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
export EXPERIMENT_NAME="ablation_exp4_fully_pair_OT_output_E"

# Training parameters
export N_EPOCHS=400          # Train for 400 epochs with constant LR
export N_EPOCHS_DECAY=200    # Decay for 200 more epochs
export BATCH_SIZE=1

# Paired training configuration
export PAIRED_STAGE="--paired_stage"
export PAIRED_SUBSET_RATIO=1.0   # 100% paired data
export PAIRED_SUBSET_SEED=42
export COMPUTE_METRICS="--compute_paired_metrics"

# Loss configuration for ablation study
export USE_OT_OUTPUT="--use_ot_output"
export USE_ENTROPY="--use_entropy_loss"
export DISABLE_GAN="--disable_gan"
export DISABLE_NCE="--disable_nce"

# NO pre-training - train from scratch
export CONTINUE_TRAIN=""
export PRETRAINED_NAME=""
export LOAD_EPOCH=""
export EPOCH_COUNT=1

echo "Configuration:"
echo "  Training: From scratch"
echo "  Output: checkpoints/$EXPERIMENT_NAME"
echo "  Paired data: 100%"
echo "  Loss: OT_output + Entropy"
echo "  Epochs: 1-400 (constant LR) + 401-600 (decay)"
echo ""

# Run training
bash run_train.sh

exit_code=$?

echo ""
echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "Training completed: Experiment 4"
else
    echo "Training failed: Experiment 4"
fi
echo "=========================================="

exit $exit_code
