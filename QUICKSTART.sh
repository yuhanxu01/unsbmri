#!/bin/bash
# ==============================================================================
# QUICK START SCRIPT
# ==============================================================================
#
# This script guides you through the complete workflow:
# 1. Estimate data noise level
# 2. Update configuration files
# 3. Launch experiments
# 4. Monitor training
# 5. Evaluate results
#
# ==============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     Noise-Adaptive UNSB MRI Contrast Transfer                 ║"
echo "║     Quick Start Guide                                         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

# ==============================================================================
# Step 1: Setup
# ==============================================================================

echo -e "${BLUE}[Step 1/6] Setup${NC}"
echo ""

# Prompt for dataset path
read -p "Enter your dataset path (e.g., ./datasets/PD_PDFS): " DATAROOT

if [ ! -d "$DATAROOT" ]; then
    echo -e "${RED}Error: Directory $DATAROOT not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Dataset found: $DATAROOT${NC}"
echo ""

# ==============================================================================
# Step 2: Estimate Noise Level
# ==============================================================================

echo -e "${BLUE}[Step 2/6] Estimating Data Noise Level${NC}"
echo ""

python -c "
from noise_estimation import analyze_dataset_noise
import json

print('Analyzing domain A...')
stats_A = analyze_dataset_noise('$DATAROOT', domain='A', num_samples=20)
print('Analyzing domain B...')
stats_B = analyze_dataset_noise('$DATAROOT', domain='B', num_samples=20)

print('\n' + '='*60)
print('Domain A (Source):')
print('  Median noise: {:.4f}'.format(stats_A['median_noise']))
print('  Median SNR: {:.1f}'.format(stats_A['median_snr']))

print('\nDomain B (Target):')
print('  Median noise: {:.4f}'.format(stats_B['median_noise']))
print('  Median SNR: {:.1f}'.format(stats_B['median_snr']))

recommended = max(stats_A['median_noise'], stats_B['median_noise'])
print('\nRecommended --data_noise_level: {:.4f}'.format(recommended))
print('='*60)

# Save to file
with open('.noise_level', 'w') as f:
    f.write(str(recommended))
" || {
    echo -e "${YELLOW}Warning: Could not analyze noise level${NC}"
    echo -e "${YELLOW}Using default: 0.03${NC}"
    echo "0.03" > .noise_level
}

NOISE_LEVEL=$(cat .noise_level)
echo -e "${GREEN}✓ Using noise level: $NOISE_LEVEL${NC}"
echo ""

# ==============================================================================
# Step 3: Select Experiments
# ==============================================================================

echo -e "${BLUE}[Step 3/6] Select Experiments to Run${NC}"
echo ""
echo "Available experiments:"
echo "  1. exp1_baseline         - Original UNSB (no improvements)"
echo "  2. exp2_latter_steps     - Latter steps training only"
echo "  3. exp3_nila_adaptive    - Nila noise-adaptive weighting"
echo "  4. exp4_combined         - Nila + Di-Fusion (recommended)"
echo "  5. exp5_full             - All improvements"
echo "  6. exp6_with_denoise_aug - Full + data augmentation"
echo "  7. all                   - Run all experiments"
echo ""
read -p "Enter experiments to run (e.g., 1 4 5, or 'all'): " SELECTION

if [[ "$SELECTION" == "all" ]]; then
    EXPERIMENTS=("exp1_baseline" "exp2_latter_steps" "exp3_nila_adaptive" "exp4_combined" "exp5_full" "exp6_with_denoise_aug")
else
    EXPERIMENTS=()
    for num in $SELECTION; do
        case $num in
            1) EXPERIMENTS+=("exp1_baseline") ;;
            2) EXPERIMENTS+=("exp2_latter_steps") ;;
            3) EXPERIMENTS+=("exp3_nila_adaptive") ;;
            4) EXPERIMENTS+=("exp4_combined") ;;
            5) EXPERIMENTS+=("exp5_full") ;;
            6) EXPERIMENTS+=("exp6_with_denoise_aug") ;;
        esac
    done
fi

echo -e "${GREEN}✓ Selected ${#EXPERIMENTS[@]} experiments: ${EXPERIMENTS[*]}${NC}"
echo ""

# ==============================================================================
# Step 4: Launch Training
# ==============================================================================

echo -e "${BLUE}[Step 4/6] Launching Training${NC}"
echo ""
echo -e "${YELLOW}Training will start in 5 seconds...${NC}"
echo -e "${YELLOW}Press Ctrl+C to cancel${NC}"
sleep 5

mkdir -p logs

idx=0
GPUS=(0 1 2 3)
rm -f logs/pids.txt

for exp in "${EXPERIMENTS[@]}"; do
    gpu_id=${GPUS[$idx]}

    # Determine config file
    case $exp in
        exp1_baseline) config="configs/exp1_baseline.sh" ;;
        exp2_latter_steps) config="configs/exp2_latter_steps.sh" ;;
        exp3_nila_adaptive) config="configs/exp3_nila_adaptive.sh" ;;
        exp4_combined) config="configs/exp4_combined.sh" ;;
        exp5_full) config="configs/exp5_full.sh" ;;
        exp6_with_denoise_aug) config="configs/exp6_with_denoise_aug.sh" ;;
    esac

    log_file="logs/${exp}_$(date +%Y%m%d_%H%M%S).log"

    echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} Starting ${CYAN}$exp${NC} on GPU $gpu_id"

    # Launch experiment
    CUDA_VISIBLE_DEVICES=$gpu_id bash $config \
        --dataroot $DATAROOT \
        --data_noise_level $NOISE_LEVEL \
        --gpu_ids 0 \
        > $log_file 2>&1 &

    echo $! >> logs/pids.txt

    idx=$(( (idx + 1) % ${#GPUS[@]} ))
    sleep 2
done

echo ""
echo -e "${GREEN}✓ All experiments launched!${NC}"
echo ""

# ==============================================================================
# Step 5: Monitor Training
# ==============================================================================

echo -e "${BLUE}[Step 5/6] Monitoring${NC}"
echo ""
echo "Training logs:"
for exp in "${EXPERIMENTS[@]}"; do
    latest_log=$(ls -t logs/${exp}_*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ]; then
        echo "  - $exp: $latest_log"
    fi
done

echo ""
echo "Monitor commands:"
echo "  ${CYAN}# View specific log${NC}"
echo "  tail -f logs/exp4_combined_*.log"
echo ""
echo "  ${CYAN}# Check running processes${NC}"
echo "  cat logs/pids.txt | xargs ps -p"
echo ""
echo "  ${CYAN}# Stop all experiments${NC}"
echo "  bash stop_experiments.sh"
echo ""
echo "  ${CYAN}# View WandB${NC}"
echo "  https://wandb.ai/YOUR_USERNAME/mri-noise-adaptive-experiments"
echo ""

# ==============================================================================
# Step 6: Next Steps
# ==============================================================================

echo -e "${BLUE}[Step 6/6] Next Steps${NC}"
echo ""
echo "After training completes:"
echo ""
echo "  ${CYAN}1. Test all models:${NC}"
echo "     bash test_all_experiments.sh"
echo ""
echo "  ${CYAN}2. Evaluate results:${NC}"
echo "     python evaluate_experiments.py"
echo ""
echo "  ${CYAN}3. View report:${NC}"
echo "     open evaluation_report.html"
echo ""

echo -e "${GREEN}"
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     Training Started Successfully!                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo ""
echo "Configuration saved:"
echo "  Dataset: $DATAROOT"
echo "  Noise level: $NOISE_LEVEL"
echo "  Experiments: ${EXPERIMENTS[*]}"
echo "  Log directory: logs/"
echo ""
echo -e "${YELLOW}Happy training! 🚀${NC}"
