#!/bin/bash
# Batch launcher for all experiments using 8 GPUs
# All paired experiments use 10% data only

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

# Common settings
export DATAROOT="/gpfs/scratch/rl5285/unsb_mri/datasets/fastmri_knee"
export PYTHON_BIN="/gpfs/scratch/rl5285/miniconda3/envs/UNSB/bin/python3.8"
export BATCH_SIZE=1
export N_EPOCHS_STAGE1=200
export N_EPOCHS_DECAY_STAGE1=200
export N_EPOCHS_STAGE2=100
export N_EPOCHS_DECAY_STAGE2=100
export PAIRED_SUBSET_RATIO=0.1  # Fixed 10% for all paired experiments

mkdir -p logs

echo "=========================================="
echo "8-GPU Batch Experiment Launcher"
echo "=========================================="
echo "All paired experiments use 10% data"
echo "Experiments:"
echo "  1x Unpaired baseline"
echo "  7x Paired strategies (A, L1, B1-B5)"
echo "Total: 8 experiments"
echo "=========================================="
echo ""

# Helper function
launch_single() {
    local gpu=$1
    local name=$2
    local stage_name=$3
    local strategy=$4
    local pretrain=$5

    export CUDA_VISIBLE_DEVICES=$gpu
    export EXPERIMENT_NAME="$name"
    export STAGE_NAME="$stage_name"

    if [ "$stage_name" == "unpaired" ]; then
        export N_EPOCHS=$N_EPOCHS_STAGE1
        export N_EPOCHS_DECAY=$N_EPOCHS_DECAY_STAGE1
        export PAIRED_STRATEGY="none"
        export PAIRED_STAGE=""
        export COMPUTE_METRICS=""
        export CONTINUE_TRAIN=""
        export PRETRAINED_NAME=""
    else
        export N_EPOCHS=$N_EPOCHS_STAGE2
        export N_EPOCHS_DECAY=$N_EPOCHS_DECAY_STAGE2
        export EPOCH_COUNT=$(($N_EPOCHS_STAGE1 + $N_EPOCHS_DECAY_STAGE1 + 1))
        export PAIRED_STRATEGY="$strategy"
        export PAIRED_STAGE="--paired_stage"
        export PAIRED_SUBSET_RATIO=0.1  # Fixed 10%
        export COMPUTE_METRICS="--compute_paired_metrics"
        export CONTINUE_TRAIN="--continue_train"
        export PRETRAINED_NAME="$pretrain"
        export LOAD_EPOCH="latest"
    fi

    bash run_train.sh > "logs/${name}_${stage_name}.log" 2>&1
}

# ============================================
# STAGE 1: Train ONE shared unpaired baseline
# ============================================
echo "STAGE 1: Training shared unpaired baseline on GPU 0..."
launch_single 0 "baseline" "unpaired" "none" ""
echo "STAGE 1 complete"
echo ""

# ============================================
# STAGE 2: Launch all 7 paired experiments (10% data)
# ============================================
echo "STAGE 2: Launching 7 paired experiments in parallel (10% data each)..."
echo ""

# Scheme A
launch_single 0 "baseline" "schemeA" "sb_gt_transport" "baseline_unpaired" &

# Baseline L1
launch_single 1 "baseline" "L1" "l1_loss" "baseline_unpaired" &

# B1: NCE Feature
launch_single 2 "baseline" "B1" "nce_feature" "baseline_unpaired" &

# B2: Frequency
launch_single 3 "baseline" "B2" "frequency" "baseline_unpaired" &

# B3: Gradient
launch_single 4 "baseline" "B3" "gradient" "baseline_unpaired" &

# B4: Multiscale
launch_single 5 "baseline" "B4" "multiscale" "baseline_unpaired" &

# B5: Self-supervised Contrastive
launch_single 6 "baseline" "B5" "selfsup_contrast" "baseline_unpaired" &

echo "Waiting for all 7 paired experiments to complete..."
wait

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=========================================="
echo ""
echo "Results structure:"
echo "  checkpoints/baseline_unpaired/      - Unpaired baseline"
echo "  checkpoints/baseline_schemeA/       - Scheme A (10% data)"
echo "  checkpoints/baseline_L1/            - L1 baseline (10% data)"
echo "  checkpoints/baseline_B1/            - B1: NCE feature (10% data)"
echo "  checkpoints/baseline_B2/            - B2: Frequency (10% data)"
echo "  checkpoints/baseline_B3/            - B3: Gradient (10% data)"
echo "  checkpoints/baseline_B4/            - B4: Multiscale (10% data)"
echo "  checkpoints/baseline_B5/            - B5: Contrastive (10% data)"
echo ""
echo "Next step: Run evaluation"
echo "  bash experiments/test_all.sh"
echo "=========================================="
