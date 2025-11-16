#!/bin/bash
# Test all trained models and find the optimal strategy
# All paired experiments use 10% data

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$BASE_DIR"

DATAROOT="/gpfs/scratch/rl5285/unsb_mri/datasets/fastmri_knee"
PYTHON_BIN="/gpfs/scratch/rl5285/miniconda3/envs/UNSB/bin/python3.8"

# Results directory
RESULTS_DIR="test_results"
mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Testing All Experiments (10% paired data)"
echo "=========================================="
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Function to test a single model
test_model() {
    local exp_name=$1
    local stage_name=$2
    local gpu=$3

    local full_name="${exp_name}_${stage_name}"
    local result_file="$RESULTS_DIR/${full_name}_metrics.txt"

    echo "Testing: $full_name on GPU $gpu"

    export CUDA_VISIBLE_DEVICES=$gpu

    $PYTHON_BIN test_unsb_paired.py \
        --dataroot "$DATAROOT" \
        --name "$full_name" \
        --epoch latest \
        --mri_representation magnitude \
        --input_nc 1 \
        --output_nc 1 \
        --mri_normalize_per_slice \
        --paired_stage \
        --results_dir "$RESULTS_DIR" \
        > "$result_file" 2>&1

    # Extract metrics from the results
    if [ -f "$RESULTS_DIR/$full_name/test_latest_paired_10cases_12slices/metrics.txt" ]; then
        grep "Overall Average" -A 5 "$RESULTS_DIR/$full_name/test_latest_paired_10cases_12slices/metrics.txt" \
            >> "$RESULTS_DIR/summary.txt"
        echo "Strategy: $stage_name" >> "$RESULTS_DIR/summary.txt"
        echo "----------------------------------------" >> "$RESULTS_DIR/summary.txt"
    fi
}

# Initialize summary file
echo "Experiment Comparison Summary" > "$RESULTS_DIR/summary.txt"
echo "All paired experiments use 10% data" >> "$RESULTS_DIR/summary.txt"
echo "========================================" >> "$RESULTS_DIR/summary.txt"
echo "" >> "$RESULTS_DIR/summary.txt"

# Test baseline (unpaired only)
echo "Testing unpaired baseline..."
test_model "baseline" "unpaired" 0 &

# Test all paired experiments in parallel (7 GPUs)
echo "Testing paired experiments in parallel..."

test_model "baseline" "schemeA" 0 &
test_model "baseline" "L1" 1 &
test_model "baseline" "B1" 2 &
test_model "baseline" "B2" 3 &
test_model "baseline" "B3" 4 &
test_model "baseline" "B4" 5 &
test_model "baseline" "B5" 6 &

# Wait for all tests to complete
echo "Waiting for all tests to complete..."
wait

echo ""
echo "=========================================="
echo "All tests complete"
echo "=========================================="
echo ""

# Generate comparison table
echo "Generating comparison table..."
$PYTHON_BIN - <<EOF
import os
import re
import pandas as pd

results_dir = "$RESULTS_DIR"

# Parse all metric files
data = []
for exp_dir in os.listdir(results_dir):
    exp_path = os.path.join(results_dir, exp_dir)
    if not os.path.isdir(exp_path):
        continue

    metrics_file = os.path.join(exp_path, "test_latest_paired_10cases_12slices", "metrics.txt")
    if not os.path.exists(metrics_file):
        continue

    with open(metrics_file, 'r') as f:
        content = f.read()

    # Extract strategy name
    strategy = exp_dir.replace("baseline_", "")

    # Extract metrics
    ssim_match = re.search(r'SSIM:\s+([\d.]+)', content)
    psnr_match = re.search(r'PSNR:\s+([\d.]+)', content)
    nrmse_match = re.search(r'NRMSE:\s+([\d.]+)', content)

    if ssim_match and psnr_match and nrmse_match:
        data.append({
            'Strategy': strategy,
            'SSIM': float(ssim_match.group(1)),
            'PSNR (dB)': float(psnr_match.group(1)),
            'NRMSE': float(nrmse_match.group(1))
        })

# Create DataFrame
df = pd.DataFrame(data)
df = df.sort_values('SSIM', ascending=False)

# Save to CSV
csv_file = os.path.join(results_dir, "comparison_table.csv")
df.to_csv(csv_file, index=False)
print(f"Saved to: {csv_file}")

# Print table
print("\n" + "="*80)
print("EXPERIMENT COMPARISON (10% Paired Data)")
print("="*80)
print(df.to_string(index=False))
print("="*80)

# Find best models
print("\nBEST STRATEGY BY METRIC:")
print("-"*80)
best_ssim = df.loc[df['SSIM'].idxmax()]
best_psnr = df.loc[df['PSNR (dB)'].idxmax()]
best_nrmse = df.loc[df['NRMSE'].idxmin()]

print(f"Best SSIM:  {best_ssim['Strategy']:20s} -> {best_ssim['SSIM']:.4f}")
print(f"Best PSNR:  {best_psnr['Strategy']:20s} -> {best_psnr['PSNR (dB)']:.2f} dB")
print(f"Best NRMSE: {best_nrmse['Strategy']:20s} -> {best_nrmse['NRMSE']:.4f}")
print("="*80)

# Rank strategies
print("\nSTRATEGY RANKING (by SSIM):")
print("-"*80)
for i, row in df.iterrows():
    print(f"{row['Strategy']:20s}  SSIM: {row['SSIM']:.4f}  PSNR: {row['PSNR (dB)']:5.2f}  NRMSE: {row['NRMSE']:.4f}")
print("="*80)

EOF

echo ""
echo "=========================================="
echo "Evaluation Complete!"
echo "=========================================="
echo "Check results in: $RESULTS_DIR/"
echo "  - comparison_table.csv"
echo "  - summary.txt"
echo "  - Individual experiment folders"
echo "=========================================="
