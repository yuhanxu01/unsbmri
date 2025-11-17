#!/bin/bash
# Batch submit all 7 paired training experiments to SLURM
# Usage: bash submit_all_paired.sh

# Path to the SLURM script (in repo)
SCRIPT_DIR="/home/user/unsbmri"
SLURM_SCRIPT="${SCRIPT_DIR}/slurm_train_paired.sh"

echo "=========================================="
echo "Submitting 7 Paired Training Jobs"
echo "=========================================="
echo "All use 10% paired data"
echo "All continue from: checkpoints/unpaired"
echo "=========================================="
echo ""

# Array of experiments: name, strategy_type
declare -a experiments=(
    "schemeA:sb_gt_transport"
    "L1:l1_loss"
    "B1:nce_feature"
    "B2:frequency"
    "B3:gradient"
    "B4:multiscale"
    "B5:selfsup_contrast"
)

# Submit each experiment
job_ids=()
for exp in "${experiments[@]}"; do
    IFS=':' read -r name strategy <<< "$exp"

    echo "Submitting: $name ($strategy)"
    job_output=$(sbatch --job-name="paired_${name}" "$SLURM_SCRIPT" "$name" "$strategy")
    job_id=$(echo "$job_output" | grep -oP '\d+')

    if [ -n "$job_id" ]; then
        job_ids+=("$job_id")
        echo "  ✓ Job ID: $job_id"
    else
        echo "  ✗ Failed to submit"
    fi

    sleep 1  # Small delay between submissions
done

echo ""
echo "=========================================="
echo "Submission Complete"
echo "=========================================="
echo "Submitted ${#job_ids[@]} jobs"
echo ""
echo "Monitor jobs:"
echo "  squeue -u \$USER"
echo ""
echo "Check specific job:"
for i in "${!job_ids[@]}"; do
    exp=${experiments[$i]}
    IFS=':' read -r name strategy <<< "$exp"
    echo "  squeue -j ${job_ids[$i]}  # $name"
done
echo ""
echo "View logs (once started):"
echo "  tail -f slurm-<job_id>.out"
echo "=========================================="
