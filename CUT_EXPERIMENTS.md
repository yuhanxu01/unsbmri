# CUT Model Experiments for MRI Contrast Transfer

This document describes how to use the CUT (Contrastive Unpaired Translation) model for MRI contrast transfer experiments and compare it with other models like UNSB.

## Overview

The CUT model has been integrated into this project to provide a baseline for comparison with UNSB and other models. While training uses unpaired data, testing is performed with paired data to compute quantitative metrics.

## Key Features

- **Unpaired Training**: Train on unpaired MRI data (trainA and trainB don't need to match)
- **Paired Testing**: Test with paired data to compute evaluation metrics
- **H5 Data Support**: Uses the same h5 knee data loader as UNSB
- **Evaluation Metrics**: Computes SSIM, PSNR, and NRMSE on paired test data

## Data Format

The CUT model uses the same h5 data format as UNSB:

```
datasets/knee_mri/
├── trainA/          # Source contrast (unpaired)
│   ├── case001.h5
│   └── case002.h5
├── trainB/          # Target contrast (unpaired)
│   ├── case001.h5
│   └── case002.h5
├── testA/           # Source contrast (paired with testB)
│   └── case001.h5
└── testB/           # Target contrast (paired with testA)
    └── case001.h5
```

Each h5 file contains:
```
case001.h5
  ├── slices_0: [H, W, 2]  # [real, imaginary]
  ├── slices_1: [H, W, 2]
  └── ...
```

## Training

### Quick Start

```bash
# Train with default settings
bash train_cut.sh

# Or with custom parameters
DATAROOT=./datasets/knee_mri \
NAME=cut_t1_t2 \
BATCH_SIZE=8 \
bash train_cut.sh
```

### Full Training Command

```bash
python train.py \
  --dataroot ./datasets/knee_mri \
  --name cut_experiment \
  --model cut \
  --CUT_mode CUT \
  --dataset_mode mri_unaligned \
  --mri_representation real_imag \
  --mri_normalize_per_slice \
  --wandb_project cut-mri \
  --batch_size 4 \
  --n_epochs 200 \
  --n_epochs_decay 200 \
  --netG resnet_9blocks \
  --netD basic \
  --lambda_GAN 1.0 \
  --lambda_NCE 1.0 \
  --nce_idt
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `cut` | Model type |
| `--CUT_mode` | `CUT` | CUT or FastCUT |
| `--lambda_GAN` | `1.0` | Weight for GAN loss |
| `--lambda_NCE` | `1.0` | Weight for NCE (contrastive) loss |
| `--nce_idt` | `True` | Use identity NCE loss |
| `--nce_layers` | `0,4,8,12,16` | Layers for NCE loss |
| `--num_patches` | `256` | Number of patches per layer |
| `--netG` | `resnet_9blocks` | Generator architecture |
| `--netD` | `basic` | Discriminator architecture |

## Testing with Paired Data

### Quick Start

```bash
# Test with default settings
bash test_cut_paired.sh

# Or with custom parameters
DATAROOT=./datasets/knee_mri \
NAME=cut_t1_t2 \
EPOCH=latest \
bash test_cut_paired.sh
```

### Full Testing Command

```bash
python test_paired.py \
  --dataroot ./datasets/knee_mri \
  --name cut_experiment \
  --model cut \
  --dataset_mode mri_unaligned \
  --mri_representation real_imag \
  --mri_normalize_per_slice \
  --epoch latest \
  --phase test \
  --eval
```

### Output

The paired testing script generates:

1. **metrics.txt**: Human-readable metrics log
   ```
   Image 0000 (case001_slice10): SSIM=0.8542, PSNR=28.34, NRMSE=0.0521
   ...
   Average Metrics:
   SSIM: 0.8234 ± 0.0421
   PSNR: 27.56 ± 2.34
   NRMSE: 0.0612 ± 0.0123
   ```

2. **metrics.csv**: CSV file for easy analysis
   ```csv
   image_name,ssim,psnr,nrmse
   case001_slice10,0.854200,28.340000,0.052100
   ...
   ```

3. **Image outputs**: Saved in `results/experiment_name/test_latest_paired/`
   - `real_A/`: Input images (source contrast)
   - `fake_B/`: Generated images (synthesized target contrast)
   - `real_B/`: Ground truth images (real target contrast)

## Evaluation Metrics

### SSIM (Structural Similarity Index)
- Range: [-1, 1], higher is better
- Measures structural similarity between images
- Good metric for perceptual quality

### PSNR (Peak Signal-to-Noise Ratio)
- Range: [0, ∞] dB, higher is better
- Measures pixel-wise similarity
- Common metric in image processing

### NRMSE (Normalized Root Mean Squared Error)
- Range: [0, ∞], lower is better
- Normalized by mean of reference image
- Measures pixel-wise error

## Comparison with UNSB

| Aspect | CUT | UNSB |
|--------|-----|------|
| Training | Unpaired | Unpaired |
| Method | Contrastive Learning + GAN | Score-Based Diffusion |
| Speed | Fast | Slower (iterative sampling) |
| Quality | Good | Potentially better |
| NCE Loss | Yes | No |
| SB Loss | No | Yes |

### Running Comparison Experiments

1. Train both models on the same data:
   ```bash
   # Train CUT
   bash train_cut.sh

   # Train UNSB
   bash run_train.sh
   ```

2. Test both models with paired evaluation:
   ```bash
   # Test CUT
   bash test_cut_paired.sh

   # Test UNSB (you may need to create a similar paired test script)
   python test_paired.py --model sb --name unsb_experiment
   ```

3. Compare metrics from the generated CSV files

## MRI-Specific Options

### Representation Mode

```bash
# Complex (real/imaginary) - 2 channels
--mri_representation real_imag --input_nc 2 --output_nc 2

# Magnitude only - 1 channel
--mri_representation magnitude --input_nc 1 --output_nc 1
```

### Normalization

```bash
# Per-slice max normalization (recommended for consistency with original workflow)
--mri_normalize_per_slice

# Per-case normalization
--mri_normalize_per_case --mri_normalize_method median

# Hard normalization to [-1, 1]
--mri_hard_normalize
```

## Troubleshooting

### Issue: Generator output is blank
- Check normalization settings
- Try adjusting `--lambda_GAN` and `--lambda_NCE`
- Increase training time

### Issue: Discriminator wins too early
- Reduce `--lambda_GAN`
- Increase `--lambda_NCE`
- Try different learning rates

### Issue: NCE loss doesn't decrease
- Check if `--nce_layers` are valid for your generator
- Try reducing `--num_patches`
- Adjust `--nce_T` (temperature)

### Issue: Out of memory
- Reduce `--batch_size`
- Use smaller generator (e.g., `--netG resnet_6blocks`)
- Reduce image size

## Advanced Usage

### FastCUT Mode

FastCUT is faster but may sacrifice some quality:

```bash
python train.py \
  --model cut \
  --CUT_mode FastCUT \
  --lambda_NCE 10.0 \
  --flip_equivariance \
  --n_epochs 150 \
  --n_epochs_decay 50 \
  ...
```

### Custom Network Architecture

```bash
# Use UNet generator
--netG unet_256

# Use multi-scale discriminator
--netD n_layers --n_layers_D 4
```

### Resume Training

```bash
python train.py \
  --continue_train \
  --epoch latest \
  --name cut_experiment \
  ...
```

## Citation

If you use the CUT model, please cite:

```bibtex
@inproceedings{park2020cut,
  title={Contrastive Learning for Unpaired Image-to-Image Translation},
  author={Park, Taesung and Efros, Alexei A and Zhang, Richard and Zhu, Jun-Yan},
  booktitle={European Conference on Computer Vision},
  year={2020}
}
```

## References

- [CUT Official Repository](https://github.com/taesungp/contrastive-unpaired-translation)
- [CUT Paper](https://arxiv.org/abs/2007.15651)
