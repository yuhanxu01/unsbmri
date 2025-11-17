# Experiment Configurations

Modular experiment system for MRI contrast transfer with multiple paired training strategies.

## Quick Start

```bash
# Run pre-configured experiment
sbatch experiments/scheme_a_twostage.sh

# Or customize via environment variables
export EXPERIMENT_NAME=my_test
export PAIRED_STRATEGY=sb_gt_transport
bash run_train.sh
```

## Available Strategies

| Strategy | Description | Key Parameters |
|----------|-------------|----------------|
| `none` | Unpaired training (default) | - |
| `sb_gt_transport` | **[Scheme A]** Use GT to guide SB transport cost | `--paired_stage` |
| `l1_loss` | **[Baseline]** Add simple L1 loss | `--lambda_L1 1.0` |
| `regularization` | **[Scheme B]** Enhanced regularization (TBD) | `--lambda_perceptual` |
| `weight_schedule` | **[Scheme D]** Dynamic loss weights (TBD) | `--sb_weight_schedule` |
| `hybrid` | Combine multiple strategies (TBD) | Custom |

## Pre-configured Experiments

### Scheme A: SB GT Transport
```bash
sbatch experiments/scheme_a_twostage.sh
```
- Stage 1: Unpaired (200+200 epochs)
- Stage 2: Paired with GT guidance in SB framework (100+100 epochs, 30% data)

### Baseline: L1 Loss
```bash
sbatch experiments/baseline_l1.sh
```
- Stage 1: Unpaired (200+200 epochs)
- Stage 2: Paired with naive L1 loss (100+100 epochs, 30% data)

## Configuration Variables

Key environment variables for `run_train.sh`:

```bash
# Experiment
EXPERIMENT_NAME="my_experiment"
STAGE_NAME="stage1"                    # Optional stage suffix

# Training
N_EPOCHS=200
N_EPOCHS_DECAY=200
BATCH_SIZE=1

# Loss weights
LAMBDA_SB=1.0
LAMBDA_NCE=1.0
LAMBDA_L1=0.0
LAMBDA_PERCEPTUAL=0.0

# Paired training
PAIRED_STAGE="--paired_stage"          # Enable paired mode
PAIRED_STRATEGY="sb_gt_transport"      # Strategy selection
PAIRED_SUBSET_RATIO=0.3                # Use 30% of paired data
COMPUTE_METRICS="--compute_paired_metrics"  # Log SSIM/PSNR/NRMSE

# Resume training
CONTINUE_TRAIN="--continue_train"
PRETRAINED_NAME="previous_experiment"
LOAD_EPOCH="latest"
```

## Adding New Strategies

1. Add strategy to `options/train_options.py`:
   ```python
   choices=['none', 'sb_gt_transport', 'l1_loss', 'your_new_strategy']
   ```

2. Implement in `models/sb_model.py`:
   ```python
   elif paired_strategy == 'your_new_strategy':
       self.loss_custom = compute_your_loss()
       extra_loss += self.loss_custom
   ```

3. Create experiment script:
   ```bash
   cp experiments/scheme_a_twostage.sh experiments/your_strategy.sh
   # Edit PAIRED_STRATEGY and parameters
   ```

## WandB Monitoring

Metrics logged for paired training:
- **Losses**: `loss/SB`, `loss/SB_guidance`, `loss/L1`, etc.
- **Metrics**: `loss/metric_SSIM`, `loss/metric_PSNR`, `loss/metric_NRMSE`
- **Visuals**: Input | Output | GT comparison

## File Structure

```
unsbmri/
├── experiments/                # Pre-configured experiments
│   ├── scheme_a_twostage.sh    # Scheme A config
│   ├── baseline_l1.sh          # Baseline config
│   └── README.md               # This file
├── run_train.sh                # Main training script
├── train.py                    # Training loop
├── models/sb_model.py          # Model with strategy implementation
└── options/train_options.py   # Training options
```
