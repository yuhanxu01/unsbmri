"""MRI-specific visualization utilities for complex-valued and magnitude data."""
import numpy as np
import torch


def complex_to_magnitude_phase(tensor):
    """
    Convert complex tensor (real/imag) to magnitude and phase images.

    Args:
        tensor: torch.Tensor with shape [2, H, W] where channel 0=real, 1=imag

    Returns:
        magnitude: np.array [H, W] with magnitude values
        phase: np.array [H, W] with phase values in [-pi, pi]
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.from_numpy(tensor)

    real = tensor[0].cpu().numpy()
    imag = tensor[1].cpu().numpy()

    magnitude = np.sqrt(real**2 + imag**2)
    phase = np.arctan2(imag, real)

    return magnitude, phase


def normalize_for_display(img, percentile_clip=99.5):
    """
    Normalize image to [0, 255] for display, with optional percentile clipping.

    Args:
        img: np.array with arbitrary range
        percentile_clip: clip values above this percentile (reduces outlier effects)

    Returns:
        np.array with dtype=uint8, range [0, 255]
    """
    if percentile_clip is not None and percentile_clip < 100:
        vmax = np.percentile(img, percentile_clip)
        img = np.clip(img, 0, vmax)

    vmin = img.min()
    vmax = img.max()

    if vmax > vmin:
        img_norm = (img - vmin) / (vmax - vmin) * 255.0
    else:
        img_norm = np.zeros_like(img)

    return img_norm.astype(np.uint8)


def phase_to_colormap(phase):
    """
    Convert phase [-pi, pi] to RGB colormap for visualization.

    Args:
        phase: np.array [H, W] with phase values in [-pi, pi]

    Returns:
        np.array [H, W, 3] with RGB values in [0, 255]
    """
    # Normalize phase to [0, 1]
    phase_norm = (phase + np.pi) / (2 * np.pi)

    # Use HSV colormap: phase -> hue, full saturation and value
    from matplotlib import cm
    colormap = cm.get_cmap('hsv')
    rgb = colormap(phase_norm)[:, :, :3]  # Drop alpha channel

    return (rgb * 255).astype(np.uint8)


def tensor_to_wandb_image(tensor, is_complex=False):
    """
    Convert MRI tensor to image dict for wandb logging.

    Args:
        tensor: torch.Tensor [C, H, W] where C=1 (magnitude) or C=2 (real/imag)
        is_complex: if True, treat as real/imag and generate magnitude + phase

    Returns:
        dict with keys like 'magnitude', 'phase' (for complex) or 'image' (for magnitude)
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.from_numpy(tensor)

    # Remove batch dimension if present
    if tensor.ndim == 4:
        tensor = tensor[0]

    result = {}

    if is_complex and tensor.shape[0] == 2:
        # Complex representation: generate magnitude and phase
        mag, phase = complex_to_magnitude_phase(tensor)

        # Normalize magnitude for display
        mag_display = normalize_for_display(mag)

        # Convert phase to colormap
        phase_display = phase_to_colormap(phase)

        # Convert magnitude to RGB (grayscale)
        mag_rgb = np.stack([mag_display] * 3, axis=-1)

        result['magnitude'] = mag_rgb
        result['phase'] = phase_display

    else:
        # Magnitude representation or single channel
        if tensor.shape[0] == 1:
            img = tensor[0].cpu().numpy()
        else:
            # If multi-channel but not complex, take first channel
            img = tensor[0].cpu().numpy()

        img_display = normalize_for_display(img)
        img_rgb = np.stack([img_display] * 3, axis=-1)

        result['image'] = img_rgb

    return result


def visuals_to_wandb_dict(visuals, mri_representation='magnitude'):
    """
    Convert OrderedDict of visuals to wandb-compatible image dict.

    Args:
        visuals: OrderedDict from model.get_current_visuals()
        mri_representation: 'real_imag' or 'magnitude'

    Returns:
        dict ready for wandb.log({'images': result})
    """
    result = {}
    is_complex = (mri_representation == 'real_imag')

    for label, tensor in visuals.items():
        images = tensor_to_wandb_image(tensor, is_complex=is_complex)

        # Flatten the dict: 'real_A' -> {'real_A_magnitude': ..., 'real_A_phase': ...}
        for img_type, img_array in images.items():
            if img_type == 'image':
                # For single channel, use original label
                result[label] = img_array
            else:
                # For complex, append magnitude/phase suffix
                result[f"{label}_{img_type}"] = img_array

    return result
