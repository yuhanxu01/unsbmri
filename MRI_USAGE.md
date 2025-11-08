# MRI Contrast Transfer - Usage Guide

This guide explains the MRI-specific modifications and how to use them for MRI contrast transfer tasks.

## Key Modifications

### 1. Per-Case Normalization

**Why**: Natural images use fixed normalization (mean=0.5, std=0.5), but MRI data has different intensity ranges per case/sequence.

**Options**:
```bash
--mri_normalize_per_case       # Enable per-case normalization
--mri_normalize_method median   # Method: median (default) | percentile_95 | max
--mri_hard_normalize            # Force normalize to [-1,1] range after scaling
```

**How it works**:
- Each H5 case computes its own normalization constant (e.g., median of magnitude across all slices)
- Each slice is divided by this constant
- At inference time, test cases use their own median (no dependency on training stats)

### 2. Phase Alignment (Optional)

**Why**: Unpaired MRI data may have arbitrary global phase offsets.

**Option**:
```bash
--mri_phase_align    # Enable global phase alignment between domain A and B
```

**Note**: Only useful for paired training or when A/B have meaningful phase relationship.

### 3. Wandb Logging

**Why**: HTML visualization doesn't support complex-valued MRI data well.

**Options**:
```bash
--use_wandb                          # Use wandb instead of HTML/Visdom
--wandb_project mri-contrast-transfer  # Wandb project name
--wandb_run_id <id>                  # Resume specific run
```

**Features**:
- For `--mri_representation real_imag`: logs both magnitude and phase images
- For `--mri_representation magnitude`: logs magnitude images
- Automatic loss curves and training metrics

### 4. Network Output Activation

**Why**: Default Tanh clips output to [-1,1], but MRI data may be in [0, 1.4] range.

**Option**:
```bash
--no_tanh    # Remove Tanh from generator output layer
```

**When to use**:
- Use `--no_tanh` if data is NOT normalized to [-1,1] (i.e., no `--mri_hard_normalize`)
- Keep Tanh if using `--mri_hard_normalize`

## Usage Examples

### Example 1: Real/Imag Mode with Per-Case Normalization

```bash
python train.py \
  --dataroot ./datasets/T1_to_T2 \
  --name T1toT2_realimag \
  --dataset_mode mri_unaligned \
  --mri_representation real_imag \
  --mri_normalize_per_case \
  --mri_normalize_method median \
  --no_tanh \
  --use_wandb \
  --input_nc 2 \
  --output_nc 2 \
  --batch_size 4 \
  --n_epochs 200 \
  --n_epochs_decay 200
```

**Explanation**:
- Uses 2-channel real/imag representation
- Each case normalized by its median
- No Tanh (data in ~[0, 1.4] range)
- Logs to wandb with magnitude+phase visualization

### Example 2: Magnitude Mode with Hard Normalization

```bash
python train.py \
  --dataroot ./datasets/PD_to_PDFS \
  --name PDtoPDFS_mag \
  --dataset_mode mri_unaligned \
  --mri_representation magnitude \
  --mri_normalize_per_case \
  --mri_hard_normalize \
  --use_wandb \
  --input_nc 1 \
  --output_nc 1 \
  --batch_size 8 \
  --n_epochs 200 \
  --n_epochs_decay 200
```

**Explanation**:
- Uses 1-channel magnitude representation
- Per-case median normalization + hard normalize to [-1,1]
- Keeps Tanh (data in [-1,1] range)

### Example 3: With Phase Alignment and Paired Training

```bash
python train.py \
  --dataroot ./datasets/paired_T1_T2 \
  --name T1toT2_paired \
  --dataset_mode mri_unaligned \
  --mri_representation real_imag \
  --mri_normalize_per_case \
  --mri_phase_align \
  --paired_stage \
  --no_tanh \
  --use_wandb \
  --input_nc 2 \
  --output_nc 2
```

**Explanation**:
- Paired training mode (requires matching case/slice names)
- Phase alignment enabled (useful for paired data)

## Data Format Requirements

Your H5 files should contain:
- Keys like `slices_0`, `slices_1`, ... (configurable with `--mri_slice_prefix`)
- Each slice: shape `[H, W, 2]` where last dimension = [real, imaginary]
- Directory structure:
  ```
  datasets/YourDataset/
    trainA/
      case001.h5
      case002.h5
    trainB/
      case001.h5
      case002.h5
    testA/
      ...
    testB/
      ...
  ```

## Recommendations

### For Most MRI Tasks:
```bash
--mri_representation real_imag      # Use complex data
--mri_normalize_per_case            # Per-case normalization
--mri_normalize_method median       # Robust to outliers
--no_tanh                           # Allow flexible output range
--use_wandb                         # Better visualization
```

### For Quick Experiments:
```bash
--mri_representation magnitude      # Simpler (1 channel)
--mri_normalize_per_case
--mri_hard_normalize                # Normalize to [-1,1]
--use_wandb
```

## Debugging Tips

1. **Check normalization constants**: They're printed during dataset initialization
2. **Visualize first batch**: Use wandb to check if magnitude/phase look reasonable
3. **Monitor loss scale**: If losses explode, try `--mri_hard_normalize`
4. **Phase issues**: If using real_imag, try with/without `--mri_phase_align`

## Migration from Natural Images

Original code changes:
- ✅ Removed: HTML visualization
- ✅ Removed: Hard-coded `Normalize((0.5,), (0.5,))` from transforms
- ✅ Removed: Fixed `*3000` scaling
- ✅ Added: Per-case median normalization
- ✅ Added: Optional Tanh removal
- ✅ Added: MRI-specific wandb visualization
