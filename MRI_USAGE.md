# UNSB for MRI Contrast Transfer

Minimal implementation of UNSB (Unsupervised Score-Based) for MRI contrast transfer.

## Quick Start

### Training
```bash
python train.py \
  --dataroot ./datasets/YOUR_DATA \
  --name experiment_name \
  --dataset_mode mri_unaligned \
  --mri_representation real_imag \
  --mri_normalize_per_case \
  --wandb_project my-mri-project \
  --batch_size 4
```

### Testing
```bash
python test.py \
  --dataroot ./datasets/YOUR_DATA \
  --name experiment_name \
  --dataset_mode mri_unaligned \
  --mri_representation real_imag \
  --mri_normalize_per_case \
  --epoch latest
```

## MRI-Specific Options

### Data Representation
```bash
--mri_representation real_imag  # 2-channel complex data (default)
--mri_representation magnitude  # 1-channel magnitude only
```

### Normalization
```bash
--mri_normalize_per_case        # Per-case median normalization (recommended)
--mri_normalize_method median   # median | percentile_95 | max
--mri_hard_normalize            # Force to [-1,1] range
```

### Optional Features
```bash
--mri_phase_align              # Phase alignment (for paired data)
--mri_crop_size 256            # Paired random crop
--no_tanh                      # Remove Tanh from generator output
```

### Logging
```bash
--wandb_project PROJECT_NAME   # Wandb project (required)
--wandb_run_id RUN_ID         # Resume specific run
```

## Network Architecture

Available generators (`--netG`):
- `resnet_9blocks_cond` - UNSB ResNet with time conditioning (default)
- `resnet_9blocks` - Standard ResNet
- `resnet_6blocks` - Lighter ResNet
- `unet_256` / `unet_128` - UNet variants

Available discriminators (`--netD`):
- `basic_cond` - Conditional PatchGAN (default for UNSB)
- `basic` - Standard PatchGAN
- `n_layers` - Customizable depth PatchGAN
- `pixel` - PixelGAN

## Data Format

HDF5 files with structure:
```
case001.h5
  ├── slices_0: [H, W, 2]  # [real, imaginary]
  ├── slices_1: [H, W, 2]
  └── ...
```

Directory structure:
```
datasets/YOUR_DATA/
  ├── trainA/
  │   ├── case001.h5
  │   └── case002.h5
  ├── trainB/
  │   ├── case001.h5
  │   └── case002.h5
  ├── testA/
  └── testB/
```

## Common Configurations

### Real/Imag with per-case normalization
```bash
python train.py \
  --dataroot ./datasets/T1_T2 \
  --name T1toT2 \
  --mri_representation real_imag \
  --mri_normalize_per_case \
  --no_tanh \
  --input_nc 2 --output_nc 2
```

### Magnitude with hard normalization
```bash
python train.py \
  --dataroot ./datasets/PD_PDFS \
  --name PDtoPDFS \
  --mri_representation magnitude \
  --mri_normalize_per_case \
  --mri_hard_normalize \
  --input_nc 1 --output_nc 1
```

## Code Structure

```
unsbmri/
├── train.py              # Training script
├── test.py               # Testing script
├── data/
│   ├── mri_unaligned_dataset.py  # MRI data loader
│   └── base_dataset.py           # Dataset base class
├── models/
│   ├── sb_model.py              # UNSB model
│   ├── networks.py              # Network definitions
│   ├── ncsn_networks.py         # NCSN generator
│   └── patchnce.py              # Contrastive loss
├── util/
│   ├── wandb_logger.py          # Wandb logging
│   └── mri_visualize.py         # MRI visualization
└── options/
    ├── base_options.py
    ├── train_options.py
    └── test_options.py
```

## Key Changes from Original UNSB

1. **Removed**: HTML visualization, Visdom, StyleGAN networks
2. **Simplified**: Only core UNSB components retained
3. **Added**: Per-case normalization, wandb integration, MRI visualization
4. **Modified**: Generator output (optional Tanh), data loading (H5 support)

## Tips

1. **Start with per-case normalization**: `--mri_normalize_per_case --no_tanh`
2. **Use wandb**: Provides magnitude + phase visualization for complex data
3. **Adjust learning rate**: Default is 0.0002, may need tuning
4. **Monitor losses**: Check that SB loss, NCE loss, and GAN loss are balanced
5. **Phase alignment**: Only use with paired data or clear phase relationship
