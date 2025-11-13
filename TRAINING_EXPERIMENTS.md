# MRI Contrast Transfer Training Experiments

This document describes the two-stage and paired training experiments for MRI contrast transfer (PD to PDFS).

## Overview

We conduct two experiments to compare different training strategies:

1. **Two-Stage Training**: Unpaired training followed by paired fine-tuning on 30% subset
2. **Full Paired Training**: Paired training on full dataset from the beginning

Both experiments use L1 loss in the paired training phase for supervised learning with ground truth.

## Experiment Details

### Experiment 1: Two-Stage Training

**Script**: `run_twostage_training.sh`

**Stages**:
- **Stage 1 (Unpaired)**:
  - Training mode: Unpaired (full dataset)
  - Epochs: 200 + 200 (decay)
  - Loss: GAN + NCE + SB
  - Saves to: `./checkpoints/PDtoPDFS_mag_stage1_unpaired/`

- **Stage 2 (Paired Fine-tuning)**:
  - Training mode: Paired (30% subset)
  - Epochs: 100 + 100 (decay)
  - Loss: GAN + NCE + SB + **L1 (λ=1.0)**
  - Continues from Stage 1 checkpoint
  - Saves to: `./checkpoints/PDtoPDFS_mag_stage2_paired_30pct/`
  - **Metrics logged**: SSIM, PSNR, NRMSE (to WandB)

**Rationale**:
- Stage 1 learns general domain transfer from unpaired data
- Stage 2 refines with paired supervision on a subset for better alignment

### Experiment 2: Full Paired Training

**Script**: `run_paired_training.sh`

**Training**:
- Training mode: Paired (full dataset, 100%)
- Epochs: 200 + 200 (decay)
- Loss: GAN + NCE + SB + **L1 (λ=1.0)**
- Saves to: `./checkpoints/PDtoPDFS_mag_paired_full/`
- **Metrics logged**: SSIM, PSNR, NRMSE (to WandB)

**Rationale**:
- Baseline to compare against two-stage approach
- Tests if starting with paired data from the beginning is better

## Usage

### Running on SLURM

Submit the jobs to SLURM:

```bash
# Two-stage training
sbatch run_twostage_training.sh

# Full paired training
sbatch run_paired_training.sh
```

### Running Locally (without SLURM)

Remove the `#SBATCH` lines and run directly:

```bash
# Two-stage training
bash run_twostage_training.sh

# Full paired training
bash run_paired_training.sh
```

## Key Parameters

### Loss Weights
- `--lambda_SB 1.0`: Schrödinger Bridge loss weight
- `--lambda_NCE 1.0`: Contrastive learning (NCE) loss weight
- `--lambda_L1 1.0`: L1 loss weight for paired training (GT supervision)

### Paired Training Options
- `--paired_stage`: Enable paired training mode (matches A/B slices by filename)
- `--paired_subset_ratio 0.3`: Use 30% of paired data (for Stage 2)
- `--paired_subset_seed 42`: Random seed for subset selection
- `--compute_paired_metrics`: Compute SSIM/PSNR/NRMSE during training

### Data
- `--mri_representation magnitude`: Use magnitude (single channel)
- `--mri_normalize_per_slice`: Per-slice max normalization (matches PNG workflow)

## Monitoring

### WandB Logging

Both experiments log to WandB with the following metrics:

**Losses**:
- `loss/G_GAN`: Generator GAN loss
- `loss/D_real`, `loss/D_fake`: Discriminator losses
- `loss/NCE`: Contrastive learning loss
- `loss/SB`: Schrödinger Bridge loss
- `loss/L1`: L1 reconstruction loss (paired training only)
- `loss/G`: Total generator loss

**Metrics** (paired training only):
- `loss/metric_SSIM`: Structural Similarity Index
- `loss/metric_PSNR`: Peak Signal-to-Noise Ratio (dB)
- `loss/metric_NRMSE`: Normalized Root Mean Squared Error

**Visualizations**:
- `visuals/real_A`: Input (PD)
- `visuals/fake_B`: Generated output (PDFS)
- `visuals/real_B`: Ground truth (PDFS) - **shown in paired mode**
- `visuals/real_A_noisy`: Noisy intermediate state

### Local Logging

Training logs are saved to:
- `./checkpoints/{experiment_name}/loss_log.txt`

Checkpoints are saved to:
- `./checkpoints/{experiment_name}/{epoch}_net_*.pth`
- `./checkpoints/{experiment_name}/latest_net_*.pth`

## Data Requirements

### Directory Structure

```
datasets/fastmri_knee/
├── trainA/          # Source domain (PD)
│   ├── case001.h5
│   ├── case002.h5
│   └── ...
└── trainB/          # Target domain (PDFS)
    ├── case001.h5
    ├── case002.h5
    └── ...
```

### Paired Data Matching

For paired training, A and B slices are matched by:
1. **Case token**: Extracted from filename (e.g., `case001` from `case001.h5`)
2. **Slice token**: Extracted from HDF5 key (e.g., `5` from `slices_5`)

Example matching:
- `trainA/case001.h5::slices_5` ↔ `trainB/case001.h5::slices_5`

The dataset automatically:
- Removes domain suffixes (`_a`, `_b`, etc.) from filenames
- Normalizes case names (lowercase)
- Excludes first/last 5 slices from each volume (if >10 slices)

## Expected Outcomes

### Two-Stage Training
- **Hypothesis**: Stage 1 learns robust unpaired features, Stage 2 refines with paired supervision
- **Expected**: Good generalization with efficient use of paired data (only 30%)

### Full Paired Training
- **Hypothesis**: More paired data from the start may improve alignment
- **Expected**: Better metrics but potentially less robust to unpaired test data

## Testing

After training, test the models:

```bash
python test_unsb_paired.py \
  --dataroot ./datasets/fastmri_knee \
  --name PDtoPDFS_mag_stage2_paired_30pct \
  --epoch latest \
  --mri_representation magnitude \
  --input_nc 1 \
  --output_nc 1 \
  --mri_normalize_per_slice
```

This generates:
- Visualizations comparing Input | Output | Ground Truth
- Quantitative metrics (SSIM, PSNR, NRMSE) per case and overall

## Customization

### Adjusting Training Duration

Edit the scripts to change epochs:

```bash
# In run_twostage_training.sh
N_EPOCHS_STAGE1=100          # Reduce Stage 1 duration
N_EPOCHS_DECAY_STAGE1=100
N_EPOCHS_STAGE2=50           # Reduce Stage 2 duration
N_EPOCHS_DECAY_STAGE2=50
```

### Adjusting Paired Subset Ratio

For different amounts of paired data in Stage 2:

```bash
--paired_subset_ratio 0.1    # Use 10% paired data
--paired_subset_ratio 0.5    # Use 50% paired data
--paired_subset_ratio 1.0    # Use 100% paired data
```

### Adjusting Loss Weights

To change the balance of losses:

```bash
LAMBDA_SB=1.0      # Schrödinger Bridge weight
LAMBDA_NCE=1.0     # Contrastive learning weight
LAMBDA_L1=1.0      # L1 reconstruction weight (try 0.1, 1.0, 10.0)
```

## Troubleshooting

### No Paired Data Found

**Error**: `RuntimeError: No matching slices found between domain A and B`

**Solution**: Ensure filenames in trainA/ and trainB/ match (e.g., `case001.h5` in both)

### L1 Loss Not Working

**Error**: L1 loss remains 0.0

**Solution**: Ensure both flags are set:
```bash
--paired_stage \
--lambda_L1 1.0
```

### Metrics Not Logged

**Error**: SSIM/PSNR/NRMSE not appearing in WandB

**Solution**: Add the flag:
```bash
--compute_paired_metrics
```

## Implementation Details

### Modified Files

1. **options/train_options.py**:
   - Added `--paired_stage`, `--paired_subset_ratio`, `--lambda_L1`, `--compute_paired_metrics`

2. **models/sb_model.py**:
   - Added L1 loss computation in `compute_G_loss()`
   - Added `compute_paired_metrics()` method for SSIM/PSNR/NRMSE

3. **train.py**:
   - Compute and log paired metrics at `print_freq` intervals

4. **data/mri_unaligned_dataset.py**:
   - Existing paired mode support (`_initialize_paired_indices()`)
   - Supports `paired_subset_ratio` for subset selection

### Continuing Training

To resume training from a checkpoint:

```bash
--continue_train \
--epoch latest                    # or specific epoch number
--pretrained_name experiment_name # or omit to continue same experiment
```

For Stage 2 to continue from Stage 1:
```bash
--continue_train \
--pretrained_name PDtoPDFS_mag_stage1_unpaired \
--epoch latest \
--epoch_count 401  # Start counting from after Stage 1
```

## References

- Schrödinger Bridge: [Paper link]
- FastCUT/CUT: Contrastive unpaired translation
- FastMRI Dataset: https://fastmri.med.nyu.edu/
