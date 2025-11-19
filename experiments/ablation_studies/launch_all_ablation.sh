#!/bin/bash

# ========================================
# Launch all 12 ablation study experiments in parallel
# ========================================

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "========================================"
echo "Launching All 12 Ablation Study Experiments"
echo "========================================"
echo "Script directory: $SCRIPT_DIR"
echo ""

# Array to store job IDs
declare -a job_ids

# Function to submit a job and store its ID
submit_job() {
    local exp_script=$1
    local exp_name=$2

    echo "Submitting: $exp_name"

    # Submit the job and capture the job ID
    job_output=$(sbatch "$SCRIPT_DIR/$exp_script" 2>&1)

    if [ $? -eq 0 ]; then
        # Extract job ID from output (format: "Submitted batch job 12345")
        job_id=$(echo "$job_output" | grep -oP 'Submitted batch job \K\d+')
        job_ids+=("$job_id")
        echo "  ✓ Job ID: $job_id"
    else
        echo "  ✗ Failed to submit: $job_output"
    fi
    echo ""
}

echo "=========================================="
echo "Group 1: Fully Paired Experiments (from scratch, 100% data)"
echo "=========================================="
echo ""

submit_job "exp1_fully_pair_OT_input.sh" "Exp 1: Fully Pair - OT Input"
submit_job "exp2_fully_pair_OT_input_E.sh" "Exp 2: Fully Pair - OT Input + Entropy"
submit_job "exp3_fully_pair_OT_output.sh" "Exp 3: Fully Pair - OT Output"
submit_job "exp4_fully_pair_OT_output_E.sh" "Exp 4: Fully Pair - OT Output + Entropy"

echo "=========================================="
echo "Group 2: Two-Stage 10% Experiments (pretrained, 10% data)"
echo "=========================================="
echo ""

submit_job "exp5_twostage_10p_OT_input.sh" "Exp 5: Two-Stage 10% - OT Input"
submit_job "exp6_twostage_10p_OT_input_E.sh" "Exp 6: Two-Stage 10% - OT Input + Entropy"
submit_job "exp7_twostage_10p_OT_output.sh" "Exp 7: Two-Stage 10% - OT Output"
submit_job "exp8_twostage_10p_OT_output_E.sh" "Exp 8: Two-Stage 10% - OT Output + Entropy"

echo "=========================================="
echo "Group 3: Two-Stage 100% Experiments (pretrained, 100% data)"
echo "=========================================="
echo ""

submit_job "exp9_twostage_100p_OT_input.sh" "Exp 9: Two-Stage 100% - OT Input"
submit_job "exp10_twostage_100p_OT_input_E.sh" "Exp 10: Two-Stage 100% - OT Input + Entropy"
submit_job "exp11_twostage_100p_OT_output.sh" "Exp 11: Two-Stage 100% - OT Output"
submit_job "exp12_twostage_100p_OT_output_E.sh" "Exp 12: Two-Stage 100% - OT Output + Entropy"

echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Total jobs submitted: ${#job_ids[@]}"
echo "Job IDs: ${job_ids[*]}"
echo ""
echo "To check status of all jobs:"
echo "  squeue -j $(IFS=,; echo "${job_ids[*]}")"
echo ""
echo "To cancel all jobs:"
echo "  scancel $(IFS=' '; echo "${job_ids[*]}")"
echo ""
echo "=========================================="
echo "All experiments submitted successfully!"
echo "=========================================="
