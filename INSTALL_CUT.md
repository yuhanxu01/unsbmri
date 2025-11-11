# Installation Guide for CUT Experiments

## Quick Install

```bash
# Install additional dependencies for CUT experiments
pip install scikit-image>=0.19.0

# Or install all requirements
pip install -r requirements_cut.txt
```

## Verify Installation

```bash
python -c "from skimage.metrics import structural_similarity; print('scikit-image installed successfully')"
```

## Dependencies

### Core Dependencies (should already be installed)
- torch >= 1.10.0
- torchvision >= 0.11.0
- numpy >= 1.21.0
- Pillow >= 8.0.0
- h5py >= 3.0.0
- wandb >= 0.12.0

### Additional for CUT Experiments
- scikit-image >= 0.19.0 (for SSIM, PSNR, NRMSE metrics)

## Testing the Installation

```bash
# Test if you can import the CUT model
python -c "from models.cut_model import CUTModel; print('CUT model imported successfully')"

# Test if metrics are available
python -c "from skimage.metrics import structural_similarity, peak_signal_noise_ratio, normalized_root_mse; print('All metrics available')"
```

## Troubleshooting

### Issue: ModuleNotFoundError: No module named 'skimage'
```bash
pip install scikit-image
```

### Issue: CUDA out of memory
Reduce batch size in training script:
```bash
BATCH_SIZE=2 bash train_cut.sh
```

### Issue: Cannot import CUT model
Make sure you're in the correct directory:
```bash
cd /path/to/unsbmri
python -c "from models.cut_model import CUTModel"
```
