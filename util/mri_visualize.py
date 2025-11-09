"""MRI-specific visualization utilities for complex-valued and magnitude data."""
import numpy as np
import torch


def prepare_image_for_logging(tensor: torch.Tensor, vmin_percentile: float = 1.0, vmax_percentile: float = 99.0) -> np.ndarray:
    """
    Prepare MRI tensor for visualization with robust percentile-based normalization.

    Args:
        tensor: Input tensor, can be complex or real
        vmin_percentile: Lower percentile for clipping (default 1.0)
        vmax_percentile: Upper percentile for clipping (default 99.0)

    Returns:
        RGB numpy array in [0, 255] range with shape [H, W, 3]
    """
    # Detach and move to CPU
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()
    else:
        tensor = torch.from_numpy(tensor)

    # Remove batch dimension if present
    if tensor.ndim == 4:
        tensor = tensor[0]

    # Convert complex to magnitude
    if torch.is_complex(tensor):
        magnitude = tensor.abs()
    else:
        # If real representation with 2 channels (real, imag), compute magnitude
        if tensor.shape[0] == 2:
            real = tensor[0]
            imag = tensor[1]
            magnitude = torch.sqrt(real**2 + imag**2)
        else:
            # Single channel magnitude - do NOT take abs(), keep original values
            magnitude = tensor[0] if tensor.shape[0] == 1 else tensor

    # Convert to numpy
    img = magnitude.numpy().astype(np.float32)

    # Robust normalization using percentiles
    vmin = np.percentile(img, vmin_percentile)
    vmax = np.percentile(img, vmax_percentile)

    # Handle edge cases
    if vmax <= vmin:
        vmax = vmin + 1e-8

    # Clip and normalize to [0, 1]
    img_norm = np.clip((img - vmin) / (vmax - vmin), 0, 1)

    # Convert to uint8 [0, 255]
    img_uint8 = (img_norm * 255).astype(np.uint8)

    # Convert grayscale to RGB
    img_rgb = np.stack([img_uint8] * 3, axis=-1)

    return img_rgb


def complex_to_magnitude_phase(tensor: torch.Tensor) -> tuple:
    """
    Convert complex tensor to magnitude and phase with robust normalization.

    Args:
        tensor: [2, H, W] complex tensor (real, imag)

    Returns:
        magnitude: [H, W, 3] RGB array
        phase: [H, W, 3] RGB array with HSV colormap
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.from_numpy(tensor)

    tensor = tensor.detach().cpu()
    real = tensor[0].numpy()
    imag = tensor[1].numpy()

    # Compute magnitude with robust normalization
    mag = np.sqrt(real**2 + imag**2)
    vmin = np.percentile(mag, 1.0)
    vmax = np.percentile(mag, 99.0)
    if vmax <= vmin:
        vmax = vmin + 1e-8
    mag_norm = np.clip((mag - vmin) / (vmax - vmin), 0, 1)
    mag_uint8 = (mag_norm * 255).astype(np.uint8)
    mag_rgb = np.stack([mag_uint8] * 3, axis=-1)

    # Compute phase and convert to HSV colormap
    phase = np.arctan2(imag, real)
    phase_norm = (phase + np.pi) / (2 * np.pi)  # Normalize to [0, 1]

    # Use matplotlib colormap if available
    try:
        from matplotlib import cm
        colormap = cm.get_cmap('hsv')
        phase_rgb = colormap(phase_norm)[:, :, :3]
        phase_rgb = (phase_rgb * 255).astype(np.uint8)
    except ImportError:
        # Fallback: simple HSV to RGB conversion
        phase_rgb = (phase_norm * 255).astype(np.uint8)
        phase_rgb = np.stack([phase_rgb] * 3, axis=-1)

    return mag_rgb, phase_rgb


def tensor_to_wandb_image(tensor: torch.Tensor, is_complex: bool = False) -> dict:
    """
    Convert MRI tensor to wandb-compatible images.

    Args:
        tensor: [C, H, W] tensor
        is_complex: if True, generate magnitude + phase

    Returns:
        dict with 'magnitude'/'phase' or 'image'
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.from_numpy(tensor)

    if tensor.ndim == 4:
        tensor = tensor[0]

    result = {}

    if is_complex and tensor.shape[0] == 2:
        # Complex representation: magnitude and phase
        mag_rgb, phase_rgb = complex_to_magnitude_phase(tensor)
        result['magnitude'] = mag_rgb
        result['phase'] = phase_rgb
    else:
        # Single channel magnitude
        img_rgb = prepare_image_for_logging(tensor)
        result['image'] = img_rgb

    return result


def visuals_to_wandb_dict(visuals: dict, mri_representation: str = 'magnitude') -> dict:
    """
    Convert model visuals dict to wandb-compatible format.

    Args:
        visuals: OrderedDict from model.get_current_visuals()
        mri_representation: 'real_imag' or 'magnitude'

    Returns:
        dict ready for wandb.log()
    """
    result = {}
    is_complex = (mri_representation == 'real_imag')

    for label, tensor in visuals.items():
        images = tensor_to_wandb_image(tensor, is_complex=is_complex)

        for img_type, img_array in images.items():
            if img_type == 'image':
                result[label] = img_array
            else:
                result[f"{label}_{img_type}"] = img_array

    return result
