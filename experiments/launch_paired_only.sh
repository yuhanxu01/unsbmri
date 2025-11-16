#!/bin/bash
# Launch only the 6 paired experiments (assumes unpaired baseline exists)
# All paired experiments use 10% data

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

# Common settings
export DATAROOT="/gpfs/scratch/rl5285/unsb_mri/datasets/fastmri_knee"
export PYTHON_BIN="/gpfs/scratch/rl5285/miniconda3/envs/UNSB/bin/python3.8"
export BATCH_SIZE=1
export N_EPOCHS=100
export N_EPOCHS_DECAY=100
export PAIRED_SUBSET_RATIO=0.1  # Fixed 10%

# Assuming unpaired baseline was trained for 200+200 epochs
UNPAIRED_EPOCHS=400
export EPOCH_COUNT=$((UNPAIRED_EPOCHS + 1))

mkdir -p logs

echo "=========================================="
echo "Paired Experiments Launcher (5 GPUs)"
echo "=========================================="
echo "All experiments use 10% paired data"
echo "Continuing from unpaired baseline"
echo "Experiments: B1, B2, B3, B4, B5"
echo "=========================================="
echo ""

# Helper function
launch_paired() {
    local gpu=$1
    local stage_name=$2
    local strategy=$3

    export CUDA_VISIBLE_DEVICES=$gpu
    export EXPERIMENT_NAME="baseline"
    export STAGE_NAME="$stage_name"
    export PAIRED_STRATEGY="$strategy"
    export PAIRED_STAGE="--paired_stage"
    export COMPUTE_METRICS="--compute_paired_metrics"
    export CONTINUE_TRAIN="--continue_train"
    export PRETRAINED_NAME="baseline_unpaired"
    export LOAD_EPOCH="latest"

    echo "GPU $gpu: Launching $stage_name ($strategy)"
    bash run_train.sh > "logs/baseline_${stage_name}.log" 2>&1 &
}

# Launch B1-B5 experiments in parallel
echo "Launching 5 paired experiments (B1-B5)..."
echo ""

launch_paired 0 "B1" "nce_feature"
launch_paired 1 "B2" "frequency"
launch_paired 2 "B3" "gradient"
launch_paired 3 "B4" "multiscale"
launch_paired 4 "B5" "selfsup_contrast"

echo "Waiting for all experiments to complete..."
wait

echo ""
echo "=========================================="
echo "ALL B1-B5 EXPERIMENTS COMPLETE"
echo "=========================================="
echo ""
echo "Results:"
echo "  checkpoints/baseline_B1/            - B1: NCE Feature"
echo "  checkpoints/baseline_B2/            - B2: Frequency"
echo "  checkpoints/baseline_B3/            - B3: Gradient"
echo "  checkpoints/baseline_B4/            - B4: Multiscale"
echo "  checkpoints/baseline_B5/            - B5: Contrastive"
echo ""
echo "Next: bash experiments/test_all.sh"
echo "=========================================="
