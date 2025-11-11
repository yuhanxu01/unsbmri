"""Paired test script with evaluation metrics for MRI contrast transfer.

This script performs paired testing on MRI data and computes evaluation metrics:
- SSIM (Structural Similarity Index)
- PSNR (Peak Signal-to-Noise Ratio)
- NRMSE (Normalized Root Mean Squared Error)

Usage:
    python test_paired.py --dataroot ./datasets/YOUR_DATA --name experiment_name --model cut --epoch latest
"""
import os
import numpy as np
import torch
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import util.util as util
from util.mri_visualize import visuals_to_wandb_dict
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nrmse


def tensor_to_numpy(tensor):
    """Convert tensor to numpy array for metric computation.

    Args:
        tensor: [C, H, W] tensor

    Returns:
        numpy array [H, W] (for single channel) or [H, W, C] (for multi-channel)
    """
    if isinstance(tensor, torch.Tensor):
        img = tensor.cpu().float().numpy()
    else:
        img = np.array(tensor)

    # Handle different channel configurations
    if img.ndim == 3:
        if img.shape[0] == 1:  # Single channel [1, H, W]
            img = img[0]
        elif img.shape[0] == 2:  # Complex data [2, H, W] - convert to magnitude
            real, imag = img[0], img[1]
            img = np.sqrt(real * real + imag * imag)
        else:  # Multi-channel, transpose to [H, W, C]
            img = np.transpose(img, (1, 2, 0))

    return img


def compute_metrics(fake_B, real_B, data_range=None):
    """Compute SSIM, PSNR, and NRMSE between fake_B and real_B.

    Args:
        fake_B: Generated image tensor [C, H, W]
        real_B: Ground truth image tensor [C, H, W]
        data_range: Data range for PSNR computation (if None, computed from real_B)

    Returns:
        Dictionary with metrics: {'ssim': float, 'psnr': float, 'nrmse': float}
    """
    # Convert tensors to numpy
    fake_np = tensor_to_numpy(fake_B)
    real_np = tensor_to_numpy(real_B)

    # Ensure same shape
    assert fake_np.shape == real_np.shape, f"Shape mismatch: {fake_np.shape} vs {real_np.shape}"

    # Compute data range if not provided
    if data_range is None:
        data_range = real_np.max() - real_np.min()
        if data_range == 0:
            data_range = 1.0

    # Compute metrics
    # For multi-channel images, compute metrics across channels
    multichannel = (fake_np.ndim == 3 and fake_np.shape[2] > 1)

    ssim_val = ssim(real_np, fake_np, data_range=data_range,
                    channel_axis=2 if multichannel else None)
    psnr_val = psnr(real_np, fake_np, data_range=data_range)
    nrmse_val = nrmse(real_np, fake_np, normalization='mean')

    return {
        'ssim': float(ssim_val),
        'psnr': float(psnr_val),
        'nrmse': float(nrmse_val)
    }


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True

    # Enable paired mode for testing
    opt.paired_stage = True

    dataset = create_dataset(opt)
    dataset2 = create_dataset(opt)
    train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)

    # Create results directory
    save_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.epoch}_paired')
    os.makedirs(save_dir, exist_ok=True)
    print(f'Saving results to {save_dir}')

    # Initialize metrics storage
    all_metrics = {
        'ssim': [],
        'psnr': [],
        'nrmse': []
    }

    # Create metrics log file
    metrics_log_path = os.path.join(save_dir, 'metrics.txt')
    metrics_csv_path = os.path.join(save_dir, 'metrics.csv')

    with open(metrics_log_path, 'w') as log_file, open(metrics_csv_path, 'w') as csv_file:
        log_file.write('Paired Testing Results with Evaluation Metrics\n')
        log_file.write('=' * 80 + '\n\n')

        csv_file.write('image_name,ssim,psnr,nrmse\n')

        for i, (data, data2) in enumerate(zip(dataset, dataset2)):
            if i == 0:
                model.data_dependent_initialize(data, data2)
                model.setup(opt)
                model.parallelize()
                if opt.eval:
                    model.eval()
            if i >= opt.num_test:
                break

            model.set_input(data, data2)
            model.test()
            visuals = model.get_current_visuals()
            img_path = model.get_image_paths()

            # Compute metrics on the tensors
            fake_B = visuals['fake_B']
            real_B = visuals['real_B']

            metrics = compute_metrics(fake_B, real_B)

            # Store metrics
            for key in all_metrics:
                all_metrics[key].append(metrics[key])

            # Log per-image metrics
            base_name = os.path.splitext(os.path.basename(str(img_path[0])))[0]
            log_line = f'Image {i:04d} ({base_name}): SSIM={metrics["ssim"]:.4f}, PSNR={metrics["psnr"]:.2f}, NRMSE={metrics["nrmse"]:.4f}\n'
            log_file.write(log_line)
            log_file.flush()

            csv_file.write(f'{base_name},{metrics["ssim"]:.6f},{metrics["psnr"]:.6f},{metrics["nrmse"]:.6f}\n')
            csv_file.flush()

            if i % 5 == 0:
                print(f'Processing ({i:04d})-th image... {img_path}')
                print(f'  SSIM: {metrics["ssim"]:.4f}, PSNR: {metrics["psnr"]:.2f} dB, NRMSE: {metrics["nrmse"]:.4f}')

            # Save images
            mri_mode = getattr(opt, 'mri_representation', 'magnitude')
            images_dict = visuals_to_wandb_dict(visuals, mri_representation=mri_mode)

            for label, img_array in images_dict.items():
                label_dir = os.path.join(save_dir, label)
                os.makedirs(label_dir, exist_ok=True)
                save_path = os.path.join(label_dir, f'{base_name}.png')
                Image.fromarray(img_array).save(save_path)

        # Compute and log average metrics
        log_file.write('\n' + '=' * 80 + '\n')
        log_file.write('Average Metrics:\n')
        log_file.write('-' * 80 + '\n')

        print('\n' + '=' * 80)
        print('Average Metrics:')
        print('-' * 80)

        for key in all_metrics:
            mean_val = np.mean(all_metrics[key])
            std_val = np.std(all_metrics[key])

            log_line = f'{key.upper()}: {mean_val:.4f} ± {std_val:.4f}\n'
            log_file.write(log_line)

            print(f'{key.upper()}: {mean_val:.4f} ± {std_val:.4f}')

        log_file.write('=' * 80 + '\n')
        print('=' * 80)

    print(f'\nTest complete. Results saved to {save_dir}')
    print(f'Metrics log: {metrics_log_path}')
    print(f'Metrics CSV: {metrics_csv_path}')
