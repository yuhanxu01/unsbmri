# UNSB Paired Testing Guide

This guide describes how to perform paired testing on a trained UNSB model with automatic case and slice selection.

## Overview

The paired testing script (`test_unsb_paired.py`) performs:
- **Automatic case selection**: Selects 10 paired cases from your test data
- **Middle slice selection**: Tests only the middle 12 slices from each case
- **Total slices tested**: 10 cases × 12 slices = **120 slices**
- **Evaluation metrics**: Computes SSIM, PSNR, NRMSE for each slice
- **Visualization**: Saves 120 images with 3 subplots (Input | Output | Ground Truth)
- **No interpolation**: Original pixel resolution preserved

## Prerequisites

```bash
# Install scikit-image for metrics
pip install scikit-image>=0.19.0

# Install matplotlib if not already installed
pip install matplotlib
```

## Quick Start

### 1. Basic Usage

```bash
# Test with default settings
bash test_unsb_paired.sh
```

### 2. Custom Parameters

```bash
# Specify your data and model
DATAROOT=./datasets/knee_mri \
NAME=my_unsb_experiment \
EPOCH=latest \
bash test_unsb_paired.sh
```

### 3. Full Command

```bash
python test_unsb_paired.py \
  --dataroot ./datasets/knee_mri \
  --name my_unsb_experiment \
  --model sb \
  --epoch latest \
  --mri_representation real_imag \
  --mri_normalize_per_slice \
  --dataset_mode mri_unaligned \
  --phase test \
  --eval
```

## Output Structure

After running the test, you'll get:

```
results/my_unsb_experiment/test_latest_paired_10cases_12slices/
├── metrics.txt              # Detailed metrics log
├── metrics.csv              # CSV format for analysis
├── case_summary.txt         # List of selected cases
└── visualizations/          # 120 visualization images
    ├── case001_slice00.png
    ├── case001_slice01.png
    ├── ...
    └── case010_slice11.png
```

## Output Files

### 1. metrics.txt
Human-readable log with per-slice and average metrics:

```
UNSB Paired Testing Results
10 Cases × 12 Middle Slices per Case = 120 Total Slices
================================================================================

[001/120] case001 Slice 00: SSIM=0.8542, PSNR=28.34 dB, NRMSE=0.0521
[002/120] case001 Slice 01: SSIM=0.8321, PSNR=27.12 dB, NRMSE=0.0634
...

================================================================================
Overall Average Metrics (120 slices):
--------------------------------------------------------------------------------
SSIM: 0.8234 ± 0.0421
PSNR: 27.56 ± 2.34
NRMSE: 0.0612 ± 0.0123
================================================================================

Per-Case Average Metrics:
--------------------------------------------------------------------------------
case001: SSIM=0.8234, PSNR=27.56 dB, NRMSE=0.0612
case002: SSIM=0.8156, PSNR=26.89 dB, NRMSE=0.0645
...
```

### 2. metrics.csv
CSV format for easy analysis in Excel/Python:

```csv
case_name,slice_idx,ssim,psnr,nrmse
case001,0,0.854200,28.340000,0.052100
case001,1,0.832100,27.120000,0.063400
...
```

### 3. case_summary.txt
List of selected cases for reproducibility:

```
Selected Cases for Testing
================================================================================
- case001
- case002
- case003
...
- case010
================================================================================
```

### 4. Visualization Images

Each PNG file contains 3 subplots side-by-side:
- **Left**: Input (source contrast A)
- **Middle**: Output (generated contrast B)
- **Right**: Ground Truth (real contrast B)

Features:
- Original pixel resolution (no interpolation)
- Grayscale colormap
- Clear labels and titles
- 150 DPI for publication quality

## Understanding the Metrics

### SSIM (Structural Similarity Index)
- **Range**: [-1, 1], higher is better (1 = perfect match)
- **Typical values**: 0.7-0.95 for good MRI synthesis
- **Measures**: Structural similarity (brightness, contrast, structure)
- **Best for**: Perceptual quality assessment

### PSNR (Peak Signal-to-Noise Ratio)
- **Range**: [0, ∞] dB, higher is better
- **Typical values**: 25-35 dB for good MRI synthesis
- **Measures**: Pixel-wise intensity similarity
- **Best for**: Quantitative comparison

### NRMSE (Normalized Root Mean Squared Error)
- **Range**: [0, ∞], lower is better (0 = perfect match)
- **Typical values**: 0.03-0.10 for good MRI synthesis
- **Measures**: Normalized pixel-wise error
- **Best for**: Relative error assessment

## Data Requirements

### Directory Structure

```
datasets/knee_mri/
├── testA/              # Source contrast (paired)
│   ├── case001.h5
│   ├── case002.h5
│   └── ...
└── testB/              # Target contrast (paired)
    ├── case001.h5
    ├── case002.h5
    └── ...
```

**Important**:
- Files in `testA` and `testB` must have **matching filenames**
- Each h5 file must contain **the same slice indices** in both domains
- The script will automatically select the middle 12 slices from each case

### H5 File Format

```python
case001.h5
  ├── slices_0: [H, W, 2]   # [real, imaginary]
  ├── slices_1: [H, W, 2]
  ├── ...
  └── slices_N: [H, W, 2]
```

## How It Works

### 1. Case Selection
```
All available cases → Sort alphabetically → Select first 10 cases
```

### 2. Slice Selection (Per Case)
```
Total slices in case → Select middle 12 slices

Example: If case has 20 slices (indices 0-19)
  Middle 12: slices 4-15 (start at (20-12)//2 = 4)
```

### 3. Testing Process
```
For each of 10 cases:
  For each of 12 middle slices:
    1. Load paired data (A and B)
    2. Run model inference: A → fake_B
    3. Compute metrics: compare fake_B with real_B
    4. Save visualization: [A | fake_B | real_B]
```

## Advanced Usage

### Test Different Epoch

```bash
EPOCH=100 bash test_unsb_paired.sh
```

### Test with Magnitude Representation

```bash
MRI_REPR=magnitude bash test_unsb_paired.sh
```

### Use Specific GPU

```bash
GPU_IDS=1 bash test_unsb_paired.sh
```

### Modify Number of Cases/Slices

Edit `test_unsb_paired.py` and change:

```python
# Line ~150: Select more/fewer cases
if len(all_cases) > 10:
    selected_cases = all_cases[:10]  # Change 10 to desired number

# Line ~160: Select more/fewer slices per case
middle_slices = select_middle_slices(slice_indices, num_slices=12)  # Change 12
```

## Analyzing Results

### Python Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load metrics
df = pd.read_csv('results/my_experiment/test_latest_paired_10cases_12slices/metrics.csv')

# Analyze by case
case_avg = df.groupby('case_name')[['ssim', 'psnr', 'nrmse']].mean()
print(case_avg)

# Plot SSIM distribution
plt.figure(figsize=(10, 6))
plt.hist(df['ssim'], bins=30)
plt.xlabel('SSIM')
plt.ylabel('Frequency')
plt.title('SSIM Distribution (120 slices)')
plt.show()

# Find best/worst slices
best_ssim = df.nlargest(5, 'ssim')
worst_ssim = df.nsmallest(5, 'ssim')
print("Best SSIM:", best_ssim)
print("Worst SSIM:", worst_ssim)
```

### Excel Analysis

1. Open `metrics.csv` in Excel
2. Create pivot table with:
   - Rows: case_name
   - Values: Average of ssim, psnr, nrmse
3. Create charts to visualize per-case performance

## Troubleshooting

### Issue: "No matching slices found"
**Solution**: Ensure testA and testB have matching filenames and slice keys

### Issue: "Less than 10 cases available"
**Solution**: The script will use all available cases. Check your data directory.

### Issue: "Less than 12 slices in some cases"
**Solution**: The script will use all available slices for that case.

### Issue: Out of memory
**Solution**:
```bash
# Process one slice at a time (batch_size=1 is already set)
# Or use CPU mode:
GPU_IDS=-1 bash test_unsb_paired.sh
```

### Issue: Metrics seem unrealistic
**Solution**:
- Check if normalization matches training (`--mri_normalize_per_slice`)
- Verify data range is correct
- Ensure paired data truly corresponds to the same anatomy

## Comparison with Other Models

To compare UNSB with CUT or other models:

```bash
# Test UNSB
NAME=unsb_model bash test_unsb_paired.sh

# Test CUT
NAME=cut_model bash test_cut_paired.sh

# Compare metrics.csv files
python -c "
import pandas as pd
unsb = pd.read_csv('results/unsb_model/test_latest_paired_10cases_12slices/metrics.csv')
cut = pd.read_csv('results/cut_model/test_latest_paired/metrics.csv')

print('UNSB SSIM:', unsb['ssim'].mean())
print('CUT SSIM:', cut['ssim'].mean())
"
```

## Publication Tips

For including results in papers:

1. **Report all three metrics**: SSIM, PSNR, NRMSE with std dev
2. **Show visualizations**: Select 3-5 representative cases
3. **Include failure cases**: Show worst-performing slices for transparency
4. **Per-case analysis**: Report metric variance across cases
5. **Visual quality**: Use 300 DPI for publication (modify script DPI setting)

## Citation

If you use this testing methodology, consider citing:

```bibtex
@article{your_paper,
  title={Your Paper Title},
  author={Your Name},
  journal={Journal Name},
  year={2024},
  note={Evaluation performed on 10 cases with 12 middle slices each,
        using SSIM, PSNR, and NRMSE metrics}
}
```

## References

- SSIM: Wang et al., "Image quality assessment: from error visibility to structural similarity", IEEE TIP 2004
- PSNR: Standard image quality metric
- NRMSE: Normalized root mean squared error
