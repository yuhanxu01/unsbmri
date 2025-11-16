#!/bin/bash
# Batch launcher for ALL experiments using 18 GPUs
# Total: 18 experiments (1 baseline + 1 L1 + 5 B schemes × 3 data ratios + 1 unpaired)

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

mkdir -p logs

echo "=========================================="
echo "18-GPU Batch Experiment Launcher"
echo "=========================================="
echo "Experiments:"
echo "  1x Unpaired baseline"
echo "  6x Scheme A + L1 (30%, 50%, 100%)"
echo "  5x B1-B5 schemes (30% data each)"
echo "  1x each B scheme at 100% for depth test"
echo "=========================================="
echo ""

# Helper function
launch_single() {
    local gpu=$1
    local name=$2
    local stage_name=$3
    local strategy=$4
    local ratio=$5
    local pretrain=$6

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
        export PAIRED_SUBSET_RATIO="$ratio"
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
launch_single 0 "shared_baseline" "unpaired" "none" "1.0" ""
echo "STAGE 1 complete"
echo ""

# ============================================
# STAGE 2: Launch all 17 paired experiments
# ============================================
echo "STAGE 2: Launching 17 paired experiments in parallel..."
echo ""

# Scheme A (30%, 50%, 100%)
launch_single 0 "shared_baseline" "A_30" "sb_gt_transport" "0.3" "shared_baseline_unpaired" &
launch_single 1 "shared_baseline" "A_50" "sb_gt_transport" "0.5" "shared_baseline_unpaired" &
launch_single 2 "shared_baseline" "A_100" "sb_gt_transport" "1.0" "shared_baseline_unpaired" &

# Baseline L1 (30%, 50%, 100%)
launch_single 3 "shared_baseline" "L1_30" "l1_loss" "0.3" "shared_baseline_unpaired" &
launch_single 4 "shared_baseline" "L1_50" "l1_loss" "0.5" "shared_baseline_unpaired" &
launch_single 5 "shared_baseline" "L1_100" "l1_loss" "1.0" "shared_baseline_unpaired" &

# B1: NCE Feature (30%, 100%)
launch_single 6 "shared_baseline" "B1_30" "nce_feature" "0.3" "shared_baseline_unpaired" &
launch_single 7 "shared_baseline" "B1_100" "nce_feature" "1.0" "shared_baseline_unpaired" &

# B2: Frequency (30%, 100%)
launch_single 8 "shared_baseline" "B2_30" "frequency" "0.3" "shared_baseline_unpaired" &
launch_single 9 "shared_baseline" "B2_100" "frequency" "1.0" "shared_baseline_unpaired" &

# B3: Gradient (30%, 100%)
launch_single 10 "shared_baseline" "B3_30" "gradient" "0.3" "shared_baseline_unpaired" &
launch_single 11 "shared_baseline" "B3_100" "gradient" "1.0" "shared_baseline_unpaired" &

# B4: Multiscale (30%, 100%)
launch_single 12 "shared_baseline" "B4_30" "multiscale" "0.3" "shared_baseline_unpaired" &
launch_single 13 "shared_baseline" "B4_100" "multiscale" "1.0" "shared_baseline_unpaired" &

# B5: Self-supervised Contrastive (30%, 100%)
launch_single 14 "shared_baseline" "B5_30" "selfsup_contrast" "0.3" "shared_baseline_unpaired" &
launch_single 15 "shared_baseline" "B5_100" "selfsup_contrast" "1.0" "shared_baseline_unpaired" &

# Extra: Test B1 at 50% (optimal search)
launch_single 16 "shared_baseline" "B1_50" "nce_feature" "0.5" "shared_baseline_unpaired" &

# Extra: Test best combo if needed later
launch_single 17 "shared_baseline" "B3_50" "gradient" "0.5" "shared_baseline_unpaired" &

echo "Waiting for all 17 paired experiments to complete..."
wait

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=========================================="
echo ""
echo "Results structure:"
echo "  checkpoints/shared_baseline_unpaired/      - Unpaired baseline"
echo "  checkpoints/shared_baseline_A_30/          - Scheme A, 30% data"
echo "  checkpoints/shared_baseline_A_50/          - Scheme A, 50% data"
echo "  checkpoints/shared_baseline_A_100/         - Scheme A, 100% data"
echo "  checkpoints/shared_baseline_L1_30/         - L1 baseline, 30%"
echo "  checkpoints/shared_baseline_B1_30/         - B1 (NCE), 30%"
echo "  checkpoints/shared_baseline_B2_30/         - B2 (Freq), 30%"
echo "  checkpoints/shared_baseline_B3_30/         - B3 (Grad), 30%"
echo "  checkpoints/shared_baseline_B4_30/         - B4 (Multiscale), 30%"
echo "  checkpoints/shared_baseline_B5_30/         - B5 (Contrast), 30%"
echo "  ... and 100% variants"
echo ""
echo "Next step: Run evaluation"
echo "  bash experiments/test_all.sh"
echo "=========================================="
