"""Paired test script for UNSB model with 10 cases, 12 middle slices per case.

This script:
1. Selects 10 paired cases
2. Tests only the middle 12 slices from each case
3. Computes SSIM, PSNR, NRMSE metrics
4. Saves 120 visualization images (10 cases × 12 slices)
5. Each image shows: Input (A), Output (fake_B), Ground Truth (B)
6. Original pixel resolution, no interpolation

Usage:
    python test_unsb_paired.py --dataroot ./datasets/YOUR_DATA --name experiment_name --epoch latest
"""
import os
import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import util.util as util
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import normalized_root_mse as nrmse


def tensor_to_numpy(tensor):
    """Convert tensor to numpy array for visualization and metric computation.

    Args:
        tensor: [C, H, W] tensor

    Returns:
        numpy array [H, W]
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
        else:  # Multi-channel
            img = img[0]  # Take first channel

    return img


def normalize_for_display(img):
    """Normalize image to [0, 1] for display without interpolation."""
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        return (img - img_min) / (img_max - img_min)
    return img


def compute_metrics(fake_B, real_B, data_range=None):
    """Compute SSIM, PSNR, and NRMSE between fake_B and real_B."""
    fake_np = tensor_to_numpy(fake_B)
    real_np = tensor_to_numpy(real_B)

    if data_range is None:
        data_range = real_np.max() - real_np.min()
        if data_range == 0:
            data_range = 1.0

    ssim_val = ssim(real_np, fake_np, data_range=data_range)
    psnr_val = psnr(real_np, fake_np, data_range=data_range)
    nrmse_val = nrmse(real_np, fake_np, normalization='mean')

    return {
        'ssim': float(ssim_val),
        'psnr': float(psnr_val),
        'nrmse': float(nrmse_val)
    }


def get_case_slice_mapping(dataset):
    """Build mapping from case to slice indices in dataset.

    Returns:
        dict: {case_name: [dataset_indices]}
    """
    case_to_indices = defaultdict(list)

    for idx in range(len(dataset)):
        if hasattr(dataset.dataset, '_paired_keys'):
            # Paired mode
            key_id = dataset.dataset._paired_keys[idx]
            case_name = key_id.split('::')[0]
            case_to_indices[case_name].append(idx)

    return case_to_indices


def select_middle_slices(slice_indices, num_slices=12):
    """Select middle num_slices from the list of slice indices.

    Args:
        slice_indices: List of dataset indices for a case
        num_slices: Number of slices to select from the middle

    Returns:
        List of selected indices
    """
    if len(slice_indices) <= num_slices:
        return slice_indices

    # Calculate start index to center the selection
    start_idx = (len(slice_indices) - num_slices) // 2
    return slice_indices[start_idx:start_idx + num_slices]


def save_visualization(real_A, fake_B, real_B, save_path, case_name, slice_idx):
    """Save 3-subplot visualization: Input | Output | Ground Truth.

    Args:
        real_A: Input image tensor [C, H, W]
        fake_B: Generated image tensor [C, H, W]
        real_B: Ground truth image tensor [C, H, W]
        save_path: Path to save the figure
        case_name: Name of the case
        slice_idx: Slice index
    """
    # Convert to numpy and normalize
    img_A = normalize_for_display(tensor_to_numpy(real_A))
    img_fake_B = normalize_for_display(tensor_to_numpy(fake_B))
    img_real_B = normalize_for_display(tensor_to_numpy(real_B))

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot Input (A)
    axes[0].imshow(img_A, cmap='gray', interpolation='none')
    axes[0].set_title('Input (A)', fontsize=14)
    axes[0].axis('off')

    # Plot Output (fake_B)
    axes[1].imshow(img_fake_B, cmap='gray', interpolation='none')
    axes[1].set_title('Output (Generated B)', fontsize=14)
    axes[1].axis('off')

    # Plot Ground Truth (B)
    axes[2].imshow(img_real_B, cmap='gray', interpolation='none')
    axes[2].set_title('Ground Truth (B)', fontsize=14)
    axes[2].axis('off')

    # Add super title
    fig.suptitle(f'{case_name} - Slice {slice_idx}', fontsize=16, y=0.98)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.num_threads = 0
    opt.batch_size = 1
    opt.serial_batches = True
    opt.no_flip = True

    # Enable paired mode for testing
    opt.paired_stage = True

    print("="*80)
    print("UNSB Paired Testing: 10 Cases × 12 Middle Slices")
    print("="*80)
    print(f"Data root: {opt.dataroot}")
    print(f"Model: {opt.name}")
    print(f"Epoch: {opt.epoch}")
    print(f"MRI representation: {getattr(opt, 'mri_representation', 'real_imag')}")
    print("="*80)
    print()

    # Create dataset
    dataset = create_dataset(opt)
    dataset2 = create_dataset(opt)
    train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)

    # Create results directory
    save_dir = os.path.join(opt.results_dir, opt.name, f'{opt.phase}_{opt.epoch}_paired_10cases_12slices')
    vis_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    print(f'Results will be saved to: {save_dir}')
    print(f'Visualizations will be saved to: {vis_dir}')
    print()

    # Build case-to-slice mapping
    print("Building case-to-slice mapping...")
    case_to_indices = get_case_slice_mapping(dataset)

    # Select 10 cases
    all_cases = sorted(case_to_indices.keys())
    if len(all_cases) > 10:
        selected_cases = all_cases[:10]
    else:
        selected_cases = all_cases

    print(f"Total cases available: {len(all_cases)}")
    print(f"Selected cases: {len(selected_cases)}")
    print(f"Selected case names: {selected_cases}")
    print()

    # Build test indices: 10 cases × 12 middle slices
    test_indices = []
    case_slice_info = []  # Store (case_name, slice_idx, dataset_idx) for tracking

    for case_name in selected_cases:
        slice_indices = case_to_indices[case_name]
        middle_slices = select_middle_slices(slice_indices, num_slices=12)

        for local_idx, dataset_idx in enumerate(middle_slices):
            test_indices.append(dataset_idx)
            case_slice_info.append((case_name, local_idx, dataset_idx))

        print(f"Case {case_name}: Total {len(slice_indices)} slices, selected middle {len(middle_slices)} slices")

    print()
    print(f"Total slices to test: {len(test_indices)} (expected: {len(selected_cases) * 12})")
    print("="*80)
    print()

    # Initialize metrics storage
    all_metrics = {
        'ssim': [],
        'psnr': [],
        'nrmse': []
    }
    case_metrics = defaultdict(lambda: {'ssim': [], 'psnr': [], 'nrmse': []})

    # Create metrics log files
    metrics_log_path = os.path.join(save_dir, 'metrics.txt')
    metrics_csv_path = os.path.join(save_dir, 'metrics.csv')
    case_summary_path = os.path.join(save_dir, 'case_summary.txt')

    with open(metrics_log_path, 'w') as log_file, \
         open(metrics_csv_path, 'w') as csv_file, \
         open(case_summary_path, 'w') as summary_file:

        log_file.write('UNSB Paired Testing Results\n')
        log_file.write('10 Cases × 12 Middle Slices per Case = 120 Total Slices\n')
        log_file.write('='*80 + '\n\n')

        csv_file.write('case_name,slice_idx,ssim,psnr,nrmse\n')

        summary_file.write('Selected Cases for Testing\n')
        summary_file.write('='*80 + '\n')
        for case_name in selected_cases:
            summary_file.write(f'- {case_name}\n')
        summary_file.write('='*80 + '\n\n')

        # Test each slice
        for test_idx, (case_name, slice_idx, dataset_idx) in enumerate(case_slice_info):
            # Get data
            data = dataset.dataset[dataset_idx]
            data2 = dataset2.dataset[dataset_idx]

            # Wrap in batch
            for key in data:
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].unsqueeze(0)
                else:
                    data[key] = [data[key]]
            for key in data2:
                if isinstance(data2[key], torch.Tensor):
                    data2[key] = data2[key].unsqueeze(0)
                else:
                    data2[key] = [data2[key]]

            # Initialize model on first iteration
            if test_idx == 0:
                model.data_dependent_initialize(data, data2)
                model.setup(opt)
                model.parallelize()
                if opt.eval:
                    model.eval()

            # Run inference
            model.set_input(data, data2)
            model.test()
            visuals = model.get_current_visuals()

            # Extract images (UNSB uses different variable names in test mode)
            # In test mode: visual_names = ['real', 'fake_1', 'fake_2', ..., 'fake_N']
            real_A = visuals['real'].squeeze(0)  # Input image

            # Get the final timestep output as the generated result
            num_timesteps = opt.num_timesteps
            fake_key = f'fake_{num_timesteps}'
            fake_B = visuals[fake_key].squeeze(0)  # Generated output at final timestep

            # Get ground truth from original data (not in visuals during test)
            real_B = data['B'].squeeze(0)

            # Compute metrics
            metrics = compute_metrics(fake_B, real_B)

            # Store metrics
            for key in all_metrics:
                all_metrics[key].append(metrics[key])
                case_metrics[case_name][key].append(metrics[key])

            # Log per-slice metrics
            log_line = (f'[{test_idx+1:03d}/{len(test_indices)}] '
                       f'{case_name} Slice {slice_idx:02d}: '
                       f'SSIM={metrics["ssim"]:.4f}, '
                       f'PSNR={metrics["psnr"]:.2f} dB, '
                       f'NRMSE={metrics["nrmse"]:.4f}\n')
            log_file.write(log_line)
            log_file.flush()

            csv_file.write(f'{case_name},{slice_idx},{metrics["ssim"]:.6f},'
                          f'{metrics["psnr"]:.6f},{metrics["nrmse"]:.6f}\n')
            csv_file.flush()

            # Print progress
            if test_idx % 10 == 0 or test_idx < 5:
                print(f'[{test_idx+1:03d}/{len(test_indices)}] '
                      f'{case_name} Slice {slice_idx:02d}: '
                      f'SSIM={metrics["ssim"]:.4f}, '
                      f'PSNR={metrics["psnr"]:.2f} dB, '
                      f'NRMSE={metrics["nrmse"]:.4f}')

            # Save visualization
            vis_filename = f'{case_name}_slice{slice_idx:02d}.png'
            vis_path = os.path.join(vis_dir, vis_filename)
            save_visualization(real_A, fake_B, real_B, vis_path, case_name, slice_idx)

        # Compute and log overall average metrics
        print()
        print('='*80)
        print('Overall Average Metrics (120 slices):')
        print('-'*80)

        log_file.write('\n' + '='*80 + '\n')
        log_file.write('Overall Average Metrics (120 slices):\n')
        log_file.write('-'*80 + '\n')

        for key in all_metrics:
            mean_val = np.mean(all_metrics[key])
            std_val = np.std(all_metrics[key])

            log_line = f'{key.upper()}: {mean_val:.4f} ± {std_val:.4f}\n'
            log_file.write(log_line)
            print(f'{key.upper()}: {mean_val:.4f} ± {std_val:.4f}')

        log_file.write('='*80 + '\n\n')
        print('='*80)
        print()

        # Compute and log per-case average metrics
        print('Per-Case Average Metrics:')
        print('-'*80)

        log_file.write('Per-Case Average Metrics:\n')
        log_file.write('-'*80 + '\n')

        for case_name in selected_cases:
            case_ssim = np.mean(case_metrics[case_name]['ssim'])
            case_psnr = np.mean(case_metrics[case_name]['psnr'])
            case_nrmse = np.mean(case_metrics[case_name]['nrmse'])

            log_line = (f'{case_name}: '
                       f'SSIM={case_ssim:.4f}, '
                       f'PSNR={case_psnr:.2f} dB, '
                       f'NRMSE={case_nrmse:.4f}\n')
            log_file.write(log_line)
            print(f'{case_name}: SSIM={case_ssim:.4f}, PSNR={case_psnr:.2f} dB, NRMSE={case_nrmse:.4f}')

        log_file.write('='*80 + '\n')

    print('='*80)
    print()
    print('Testing Complete!')
    print('='*80)
    print(f'Results saved to: {save_dir}')
    print(f'  - Metrics log: {metrics_log_path}')
    print(f'  - Metrics CSV: {metrics_csv_path}')
    print(f'  - Case summary: {case_summary_path}')
    print(f'  - Visualizations ({len(test_indices)} images): {vis_dir}')
    print('='*80)
