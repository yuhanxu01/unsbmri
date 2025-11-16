#!/bin/bash
#SBATCH --partition=gpu4_medium
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --mem-per-cpu=4G
#SBATCH --gres=gpu:1

# Scheme A: Two-Stage Training with SB GT Transport
# Stage 1: Unpaired training
# Stage 2: Paired training with GT guidance in SB framework (30% subset)

# ============================================
# STAGE 1: Unpaired Training
# ============================================
export EXPERIMENT_NAME="PDtoPDFS_schemeA"
export STAGE_NAME="stage1_unpaired"
export N_EPOCHS=200
export N_EPOCHS_DECAY=200
export LAMBDA_SB=1.0
export LAMBDA_NCE=1.0
export PAIRED_STRATEGY="none"

echo "Starting Stage 1: Unpaired Training"
bash run_train.sh

if [ $? -ne 0 ]; then
    echo "Stage 1 failed"
    exit 1
fi

# ============================================
# STAGE 2: Paired Fine-tuning (Scheme A)
# ============================================
export STAGE_NAME="stage2_paired_schemeA"
export N_EPOCHS=100
export N_EPOCHS_DECAY=100
export EPOCH_COUNT=$((200 + 200 + 1))  # Continue from stage 1
export PAIRED_STAGE="--paired_stage"
export PAIRED_STRATEGY="sb_gt_transport"  # Scheme A: Use GT in SB transport
export PAIRED_SUBSET_RATIO=0.3
export COMPUTE_METRICS="--compute_paired_metrics"
export CONTINUE_TRAIN="--continue_train"
export PRETRAINED_NAME="PDtoPDFS_schemeA_stage1_unpaired"
export LOAD_EPOCH="latest"

echo ""
echo "Starting Stage 2: Paired Fine-tuning (Scheme A)"
bash run_train.sh

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "Scheme A: Two-Stage Training Complete"
    echo "======================================"
fi
