#!/bin/bash
# Batch launcher for all paired training experiments
# Launches 18 experiments in parallel (requires 18 GPUs)

# Configuration
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

# Function to launch a single experiment
launch_experiment() {
    local gpu_id=$1
    local exp_name=$2
    local strategy=$3
    local subset_ratio=$4
    local stage=$5

    echo "======================================"
    echo "Launching on GPU $gpu_id: $exp_name"
    echo "Strategy: $strategy, Ratio: $subset_ratio, Stage: $stage"
    echo "======================================"

    export CUDA_VISIBLE_DEVICES=$gpu_id

    if [ "$stage" == "1" ]; then
        # Stage 1: Unpaired
        export EXPERIMENT_NAME="$exp_name"
        export STAGE_NAME="stage1_unpaired"
        export N_EPOCHS=$N_EPOCHS_STAGE1
        export N_EPOCHS_DECAY=$N_EPOCHS_DECAY_STAGE1
        export PAIRED_STRATEGY="none"
        export PAIRED_STAGE=""
        export COMPUTE_METRICS=""
        export CONTINUE_TRAIN=""
        export PRETRAINED_NAME=""

        bash run_train.sh > "logs/${exp_name}_stage1.log" 2>&1
    else
        # Stage 2: Paired
        export EXPERIMENT_NAME="$exp_name"
        export STAGE_NAME="stage2_${strategy}_${subset_ratio}"
        export N_EPOCHS=$N_EPOCHS_STAGE2
        export N_EPOCHS_DECAY=$N_EPOCHS_DECAY_STAGE2
        export EPOCH_COUNT=$(($N_EPOCHS_STAGE1 + $N_EPOCHS_DECAY_STAGE1 + 1))
        export PAIRED_STRATEGY="$strategy"
        export PAIRED_STAGE="--paired_stage"
        export PAIRED_SUBSET_RATIO="$subset_ratio"
        export COMPUTE_METRICS="--compute_paired_metrics"
        export CONTINUE_TRAIN="--continue_train"
        export PRETRAINED_NAME="${exp_name}_stage1_unpaired"
        export LOAD_EPOCH="latest"

        bash run_train.sh > "logs/${exp_name}_stage2.log" 2>&1
    fi
}

# Create logs directory
mkdir -p logs

echo "=========================================="
echo "Batch Experiment Launcher"
echo "=========================================="
echo "Total experiments: 18 (3 unpaired + 15 paired)"
echo "Stage 1: Unpaired training (3 experiments)"
echo "Stage 2: Paired fine-tuning (15 experiments)"
echo "=========================================="
echo ""

# ============================================
# STAGE 1: Unpaired Training (3 baselines)
# ============================================
echo "=========================================="
echo "STAGE 1: Launching 3 unpaired baselines"
echo "=========================================="

# GPU 0: Baseline for A (30%)
launch_experiment 0 "baseline_A30" "none" "0.3" "1" &
pid0=$!

# GPU 1: Baseline for A (50%)
launch_experiment 1 "baseline_A50" "none" "0.5" "1" &
pid1=$!

# GPU 2: Baseline for A (100%)
launch_experiment 2 "baseline_A100" "none" "1.0" "1" &
pid2=$!

# Wait for all Stage 1 experiments to complete
echo "Waiting for Stage 1 to complete..."
wait $pid0 $pid1 $pid2

echo ""
echo "=========================================="
echo "STAGE 1: Complete"
echo "=========================================="
echo ""
sleep 5

# ============================================
# STAGE 2: Paired Fine-tuning (15 experiments)
# ============================================
echo "=========================================="
echo "STAGE 2: Launching 15 paired experiments"
echo "=========================================="

# Scheme A: SB GT Transport (3 ratios)
launch_experiment 0 "baseline_A30" "sb_gt_transport" "0.3" "2" &
launch_experiment 1 "baseline_A50" "sb_gt_transport" "0.5" "2" &
launch_experiment 2 "baseline_A100" "sb_gt_transport" "1.0" "2" &

# Baseline: L1 Loss (3 ratios)
launch_experiment 3 "baseline_L1_30" "l1_loss" "0.3" "2" &
launch_experiment 4 "baseline_L1_50" "l1_loss" "0.5" "2" &
launch_experiment 5 "baseline_L1_100" "l1_loss" "1.0" "2" &

# B1: NCE Feature (3 ratios)
launch_experiment 6 "B1_NCE_30" "nce_feature" "0.3" "2" &
launch_experiment 7 "B1_NCE_50" "nce_feature" "0.5" "2" &
launch_experiment 8 "B1_NCE_100" "nce_feature" "1.0" "2" &

# B2: Frequency (3 ratios)
launch_experiment 9 "B2_Freq_30" "frequency" "0.3" "2" &
launch_experiment 10 "B2_Freq_50" "frequency" "0.5" "2" &
launch_experiment 11 "B2_Freq_100" "frequency" "1.0" "2" &

# B3: Gradient (3 ratios)
launch_experiment 12 "B3_Grad_30" "gradient" "0.3" "2" &
launch_experiment 13 "B3_Grad_50" "gradient" "0.5" "2" &
launch_experiment 14 "B3_Grad_100" "gradient" "1.0" "2" &

# Wait for all Stage 2 experiments to complete
echo "Waiting for all Stage 2 experiments to complete..."
wait

echo ""
echo "=========================================="
echo "ALL EXPERIMENTS COMPLETE"
echo "=========================================="
echo "Check logs in: logs/"
echo "Check checkpoints in: checkpoints/"
echo "Run test script to evaluate: bash experiments/test_all.sh"
echo "=========================================="
