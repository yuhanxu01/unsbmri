# Ablation Study Experiments

This directory contains 12 ablation study experiments to systematically evaluate different loss components for paired MRI contrast transfer training.

## Overview

The experiments test two types of Optimal Transport (OT) losses:
- **OT_input**: `tau * mean((real_A_noisy - real_B)^2)` - Supervise intermediate diffusion state to GT
  - `real_A_noisy` is computed with gradient when `use_ot_input=True`
  - Directly constrains the forward diffusion process
- **OT_output**: `tau * mean((fake_B - real_B)^2)` - Supervise final network output to GT
  - `fake_B` is the final output from the diffusion process
  - Standard supervised learning on output

Combined with:
- **Entropy loss**: `ET_XY` term from Schrödinger Bridge formulation (energy-based regularization)
- **No entropy**: OT loss only

**Key insight**: OT_input supervises the *intermediate state* (real_A_noisy), while OT_output supervises the *final output* (fake_B).

All experiments disable GAN and NCE losses to isolate the effect of OT and entropy components.

## Experiment Design

### Group 1: Fully Paired (from scratch, 100% paired data)
- `exp1_fully_pair_OT_input.sh` - OT_input only
- `exp2_fully_pair_OT_input_E.sh` - OT_input + Entropy
- `exp3_fully_pair_OT_output.sh` - OT_output only
- `exp4_fully_pair_OT_output_E.sh` - OT_output + Entropy

**Training**: 400 epochs (constant LR) + 200 epochs (decay) = 600 epochs total

### Group 2: Two-Stage (pretrained, 10% paired data)
- `exp5_twostage_10p_OT_input.sh` - OT_input only
- `exp6_twostage_10p_OT_input_E.sh` - OT_input + Entropy
- `exp7_twostage_10p_OT_output.sh` - OT_output only
- `exp8_twostage_10p_OT_output_E.sh` - OT_output + Entropy

**Training**: Load unpaired pretrained model (400 epochs), continue from epoch 401-500 (constant LR) + 501-600 (decay) = 200 additional epochs

### Group 3: Two-Stage (pretrained, 100% paired data)
- `exp9_twostage_100p_OT_input.sh` - OT_input only
- `exp10_twostage_100p_OT_input_E.sh` - OT_input + Entropy
- `exp11_twostage_100p_OT_output.sh` - OT_output only
- `exp12_twostage_100p_OT_output_E.sh` - OT_output + Entropy

**Training**: Load unpaired pretrained model (400 epochs), continue from epoch 401-500 (constant LR) + 501-600 (decay) = 200 additional epochs

## Usage

### Submit all experiments in parallel
```bash
cd experiments/ablation_studies
bash launch_all_ablation.sh
```

This will submit all 12 experiments simultaneously. The script will output job IDs and provide commands to:
- Check status: `squeue -j <job_ids>`
- Cancel all: `scancel <job_ids>`

### Submit individual experiment
```bash
sbatch exp1_fully_pair_OT_input.sh
```

### Prerequisites
- Unpaired pretrained model at `checkpoints/unpaired/latest_net_*.pth` (for Group 2 & 3 experiments)
- FastMRI knee dataset at `/gpfs/scratch/rl5285/unsb_mri/datasets/fastmri_knee`
- Python environment at `/gpfs/scratch/rl5285/miniconda3/envs/UNSB`

## Expected Outcomes

The experiments will generate checkpoints in:
```
checkpoints/ablation_exp1_fully_pair_OT_input/
checkpoints/ablation_exp2_fully_pair_OT_input_E/
...
checkpoints/ablation_exp12_twostage_100p_OT_output_E/
```

Each checkpoint directory contains:
- Model weights (`latest_net_*.pth`, `<epoch>_net_*.pth`)
- Training logs (`loss_log.txt`)
- Evaluation metrics (`metrics.txt` - if paired metrics enabled)
- Web visualizations (`web/` directory)

## Analysis Questions

These experiments are designed to answer:

1. **OT_input vs OT_output**: Which formulation is more effective?
2. **Entropy contribution**: Does entropy loss improve results when added to OT loss?
3. **Pretraining benefit**: Do two-stage experiments outperform fully paired training?
4. **Data efficiency**: How does 10% vs 100% paired data affect performance?
5. **Interaction effects**: Are there synergistic effects between different components?

## Monitoring

Check SLURM output logs:
```bash
# View real-time output
tail -f slurm-<job_id>.out

# Check all running jobs
squeue -u $USER

# View job details
scontrol show job <job_id>
```

Check training progress:
```bash
# View loss curves
cat checkpoints/ablation_exp1_fully_pair_OT_input/loss_log.txt

# View latest metrics
tail checkpoints/ablation_exp1_fully_pair_OT_input/loss_log.txt
```

## Troubleshooting

### H5 file reading errors
The code includes retry logic for concurrent h5 file access. If multiple nodes read the same file:
- Wait 5 seconds and retry (up to 10 attempts)
- Skip file if still locked after max retries
- Warning messages logged but training continues

### Out of memory
Reduce batch size in experiment scripts (default is 1, already minimal)

### Missing pretrained model
For Group 2 & 3 experiments, ensure unpaired model exists:
```bash
ls checkpoints/unpaired/latest_net_*.pth
```

If missing, train unpaired model first or modify experiment scripts to train from scratch.
