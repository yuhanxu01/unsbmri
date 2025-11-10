import os
import random
import re
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch
import warnings

from data.base_dataset import BaseDataset



SliceIndex = Tuple[str, str]


class MriUnalignedDataset(BaseDataset):
    """Dataset for MRI slices stored as complex data in HDF5 volumes."""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument(
            '--mri_representation',
            type=str,
            default='real_imag',
            choices=['real_imag', 'magnitude'],
            help='Representation for complex MRI data: either two-channel real/imag or single-channel magnitude.'
        )
        parser.add_argument(
            '--mri_slice_prefix',
            type=str,
            default='slices_',
            help='Prefix pattern used for slice keys inside each HDF5 file.'
        )
        parser.add_argument(
            '--mri_normalize_per_case',
            action='store_true',
            help='If true, normalize each case by its own median value'
        )
        parser.add_argument(
            '--mri_normalize_method',
            type=str,
            default='median',
            choices=['median', 'percentile_95', 'max'],
            help='Method to compute normalization constant for each case'
        )
        parser.add_argument(
            '--mri_hard_normalize',
            action='store_true',
            help='If true, force normalize data to [-1,1] range after scaling'
        )
        parser.add_argument(
            '--mri_normalize_per_slice',
            action='store_true',
            help='If true, normalize each slice by its own max (mimics PNG workflow: x/max -> [0,1] -> Normalize)'
        )
        parser.add_argument(
            '--mri_phase_align',
            action='store_true',
            help='If true, apply global phase alignment between A and B'
        )
        parser.set_defaults(preprocess='none', no_flip=True, input_nc=2, output_nc=2)
        parser.add_argument('--mri_crop_size', type=int, default=0, help='if > 0, apply paired random crop of this size to MRI slices from both domains')
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.slice_prefix = opt.mri_slice_prefix

        if self.opt.mri_representation == 'magnitude':
            if self.opt.input_nc != 1 or self.opt.output_nc != 1:
                print('Adjusting input/output channels to 1 for magnitude MRI representation.')
                self.opt.input_nc = 1
                self.opt.output_nc = 1
        else:
            if self.opt.input_nc != 2 or self.opt.output_nc != 2:
                print('Adjusting input/output channels to 2 for real/imag MRI representation.')
                self.opt.input_nc = 2
                self.opt.output_nc = 2

        phase_dir = self.opt.phase
        self.dir_A = os.path.join(opt.dataroot, f'{phase_dir}A')
        self.dir_B = os.path.join(opt.dataroot, f'{phase_dir}B')
        if phase_dir == 'test' and (not os.path.exists(self.dir_A)) and os.path.exists(os.path.join(opt.dataroot, 'valA')):
            self.dir_A = os.path.join(opt.dataroot, 'valA')
            self.dir_B = os.path.join(opt.dataroot, 'valB')

        self.A_indices = self._index_directory(self.dir_A)
        self.B_indices = self._index_directory(self.dir_B)
        self.A_size = len(self.A_indices)
        self.B_size = len(self.B_indices)

        if self.A_size == 0:
            raise RuntimeError(f'No MRI slices found under {self.dir_A}. Ensure .h5 files contain keys starting with {self.slice_prefix}.')
        if self.B_size == 0:
            raise RuntimeError(f'No MRI slices found under {self.dir_B}. Ensure .h5 files contain keys starting with {self.slice_prefix}.')

        # Compute per-case normalization constants if enabled
        self.norm_constants_A = {}
        self.norm_constants_B = {}
        if getattr(self.opt, 'mri_normalize_per_case', False):
            print('Computing per-case normalization constants...')
            self.norm_constants_A = self._compute_normalization_constants(self.A_indices, 'A')
            self.norm_constants_B = self._compute_normalization_constants(self.B_indices, 'B')
            print(f'Computed normalization for {len(self.norm_constants_A)} cases in A, {len(self.norm_constants_B)} cases in B')

        if getattr(self.opt, 'paired_stage', False):
            self._initialize_paired_indices()

    def _compute_normalization_constants(self, indices: List[SliceIndex], domain: str) -> dict:
        """Compute normalization constant for each case (h5 file)."""
        # Group slices by h5 file
        case_slices = {}
        for h5_path, slice_key in indices:
            if h5_path not in case_slices:
                case_slices[h5_path] = []
            case_slices[h5_path].append(slice_key)

        norm_constants = {}
        method = self.opt.mri_normalize_method

        for h5_path, slice_keys in case_slices.items():
            magnitudes = []
            with h5py.File(h5_path, 'r') as handle:
                for key in slice_keys:
                    data = handle[key][...]
                    # Compute magnitude regardless of representation mode
                    if data.shape[-1] >= 2:
                        real = data[..., 0]
                        imag = data[..., 1]
                        mag = np.sqrt(real * real + imag * imag)
                    else:
                        mag = data[..., 0]
                    magnitudes.append(mag.flatten())

            # Concatenate all magnitude values from this case
            all_mags = np.concatenate(magnitudes)

            # Compute normalization constant based on method
            if method == 'median':
                norm_const = np.median(all_mags)
            elif method == 'percentile_95':
                norm_const = np.percentile(all_mags, 95)
            elif method == 'max':
                norm_const = np.max(all_mags)
            else:
                norm_const = 1.0

            # Avoid division by zero
            if norm_const < 1e-8:
                norm_const = 1.0

            norm_constants[h5_path] = norm_const

        return norm_constants

    def __len__(self) -> int:
        if getattr(self.opt, 'paired_stage', False):
            size = getattr(self, '_paired_size', 0)
            if size == 0:
                raise RuntimeError('Paired stage requested but no matching case/slice pairs were found.')
            return size
        return max(self.A_size, self.B_size)

    def __getitem__(self, index: int):
        if getattr(self.opt, 'paired_stage', False):
            if not getattr(self, '_paired_keys', None):
                raise RuntimeError('Paired indices are not initialized. This should not happen; please check dataset setup.')
            key_id = self._paired_keys[index % self._paired_size]
            A_path, A_key = self._paired_lookup_A[key_id]
            B_path, B_key = self._paired_lookup_B[key_id]
        else:
            A_idx = index % self.A_size
            A_path, A_key = self.A_indices[A_idx]

            if self.opt.serial_batches:
                B_path, B_key = self.B_indices[index % self.B_size]
            else:
                # Select B from nearby slices (±2 from A index)
                nearby_range = 2
                B_start = max(0, A_idx - nearby_range)
                B_end = min(self.B_size - 1, A_idx + nearby_range)
                B_idx = random.randint(B_start, B_end)
                B_path, B_key = self.B_indices[B_idx]

        A_tensor = self._load_slice(A_path, A_key, self.norm_constants_A)
        B_tensor = self._load_slice(B_path, B_key, self.norm_constants_B)

        # Apply center crop to 256x256
        A_tensor = self._center_crop(A_tensor, 256)
        B_tensor = self._center_crop(B_tensor, 256)

        if getattr(self.opt, 'paired_stage', False):
            case_token, slice_token = key_id.split('::', 1)
            display_token = f"{case_token}_slice{slice_token}"
            a_path_label = display_token
            b_path_label = display_token
        else:
            a_path_label = f"{A_path}:{A_key}"
            b_path_label = f"{B_path}:{B_key}"

        # Apply phase alignment only if enabled
        if getattr(self.opt, 'mri_phase_align', False):
            A_tensor, B_tensor = self._align_phase_global(A_tensor, B_tensor)
        return {
            'A': A_tensor,
            'B': B_tensor,
            'A_paths': a_path_label,
            'B_paths': b_path_label
        }

    def _initialize_paired_indices(self) -> None:
        """Build lookup tables to align A/B slices when paired training is enabled."""

        def build_lookup(indices, root, domain_label):
            lookup = {}
            collisions = 0
            for file_path, key in indices:
                rel = os.path.relpath(file_path, root)
                pair_key = self._canonical_pair_key(rel, key)
                if pair_key in lookup:
                    collisions += 1
                    if collisions <= 5:
                        warnings.warn(
                            f"Duplicate slice identifier '{pair_key}' detected in domain {domain_label}; keeping first occurrence."
                        )
                    continue
                lookup[pair_key] = (file_path, key)
            if collisions > 5:
                warnings.warn(f"{collisions} duplicate slice identifiers encountered in domain {domain_label}; duplicates ignored.")
            return lookup

        a_lookup = build_lookup(self.A_indices, self.dir_A, 'A')
        b_lookup = build_lookup(self.B_indices, self.dir_B, 'B')
        shared = sorted(set(a_lookup.keys()) & set(b_lookup.keys()))

        if not shared:
            raise RuntimeError('No matching slices found between domain A and B for paired training. Ensure case names and slice indices align.')

        missing_a = sorted(set(b_lookup.keys()) - set(a_lookup.keys()))
        missing_b = sorted(set(a_lookup.keys()) - set(b_lookup.keys()))
        if missing_a:
            warnings.warn(f"{len(missing_a)} B slices have no A counterpart; they will be ignored during paired training.")
        if missing_b:
            warnings.warn(f"{len(missing_b)} A slices have no B counterpart; they will be ignored during paired training.")

        subset_ratio = float(getattr(self.opt, 'paired_subset_ratio', 1.0))
        subset_ratio = min(max(subset_ratio, 0.0), 1.0)
        if subset_ratio <= 0.0:
            raise RuntimeError('paired_subset_ratio must be greater than 0 when paired training is enabled.')
        if subset_ratio < 1.0:
            subset_count = max(1, int(round(len(shared) * subset_ratio)))
            subset_seed = getattr(self.opt, 'paired_subset_seed', 0)
            rng = random.Random(subset_seed)
            if subset_count < len(shared):
                shared = sorted(rng.sample(shared, subset_count))

        self._paired_lookup_A = a_lookup
        self._paired_lookup_B = b_lookup
        self._paired_keys = shared
        self._paired_size = len(shared)

    def _canonical_pair_key(self, rel_path: str, slice_key: str) -> str:
        case_token = self._extract_case_token(rel_path)
        slice_token = self._extract_slice_token(slice_key)
        return f"{case_token}::{slice_token}"

    def _extract_case_token(self, rel_path: str) -> str:
        rel_obj = Path(rel_path)
        candidates = [rel_obj.stem] + [Path(part).stem for part in reversed(rel_obj.parts[:-1])]
        for token in candidates:
            normalized = self._normalize_case_token(token)
            if normalized:
                return normalized
        fallback = rel_obj.stem.lower()
        return fallback or 'unknown-case'

    def _normalize_case_token(self, token: str) -> str:
        token = token.lower().strip()
        if not token or token in {'train', 'val', 'test', 'images'}:
            return ''
        for suffix in ('_a', '_b', '-a', '-b', '_source', '_target', '_domaina', '_domainb'):
            if token.endswith(suffix):
                token = token[:-len(suffix)]
                break
        return token

    def _extract_slice_token(self, slice_key: str) -> str:
        base = slice_key
        if self.slice_prefix and base.startswith(self.slice_prefix):
            base = base[len(self.slice_prefix):]
        base = base.strip().lower()
        match = re.search(r'(\d+)$', base)
        if match:
            digits = match.group(1).lstrip('0')
            return digits or '0'
        return base or '0'

    def _index_directory(self, directory: str) -> List[SliceIndex]:
        if not os.path.isdir(directory):
            raise RuntimeError(f'Directory {directory} not found for MRI dataset.')
        indices: List[SliceIndex] = []
        for root, _, files in os.walk(directory):
            for fname in sorted(files):
                if not fname.lower().endswith(('.h5', '.hdf5')):
                    continue
                path = os.path.join(root, fname)
                indices.extend(self._list_slices(path))
        indices.sort()
        return indices
    def _list_slices(self, h5_path: str) -> List[SliceIndex]:
        slices: List[SliceIndex] = []
        with h5py.File(h5_path, 'r') as handle:
            valid_keys = [
                key for key in sorted(handle.keys())
                if key.startswith(self.slice_prefix)
                and handle[key].ndim == 3
                and handle[key].shape[-1] >= 1
            ]

            if len(valid_keys) > 10:
                valid_keys = valid_keys[5:-5]
            else:
                warnings.warn(f"File {os.path.basename(h5_path)} has only {len(valid_keys)} slices; skipping exclusion.")
            
            for key in valid_keys:
                slices.append((h5_path, key))
        return slices
    def _load_slice(self, file_path: str, key: str, norm_constants: dict) -> torch.Tensor:
        with h5py.File(file_path, 'r') as handle:
            data = handle[key][...]

        if data.ndim != 3:
            raise ValueError(f'Slice {key} in {file_path} has unexpected ndim {data.ndim}; expected 3.')

        if self.opt.mri_representation == 'real_imag':
            if data.shape[-1] != 2:
                raise ValueError(f'Slice {key} in {file_path} expected last dim 2 for real/imag but found {data.shape[-1]}.')
            tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(data, (2, 0, 1))))
        else:
            if data.shape[-1] < 2:
                raise ValueError(f'Slice {key} in {file_path} needs both real and imaginary components for magnitude computation.')
            real = data[..., 0]
            imag = data[..., 1]
            magnitude = np.sqrt(real * real + imag * imag).astype(data.dtype, copy=False)
            tensor = torch.from_numpy(np.ascontiguousarray(magnitude[None, ...]))

        if tensor.dtype != torch.float32:
            raise TypeError(
                f'MRI slice {key} in {file_path} has dtype {tensor.dtype}; expected torch.float32 to match network weights without implicit casting.'
            )

        # Normalization strategy selection
        if getattr(self.opt, 'mri_normalize_per_slice', False):
            # Per-slice max normalization (mimics original PNG workflow)
            # This is what you used before: img/max(img) -> [0,1] -> (x-0.5)/0.5 -> [-1,1]
            tensor_max = tensor.max()
            if tensor_max > 0:
                tensor = tensor / tensor_max  # [0, 1]
            tensor = (tensor - 0.5) / 0.5  # [-1, 1]

        elif getattr(self.opt, 'mri_normalize_per_case', False):
            # Per-case normalization (using median/percentile/max of entire case)
            if file_path in norm_constants:
                tensor = tensor / norm_constants[file_path]
            # After per-case normalization, apply final normalization to [-1, 1]
            if not getattr(self.opt, 'mri_hard_normalize', False):
                # Normalize to [-1, 1] (mimics Normalize((0.5,), (0.5,)) for images in [0,1])
                # Assume tensor is roughly in [0, 1.5] after median normalization
                tensor = torch.clamp(tensor, 0, 1)  # Clip to [0, 1]
                tensor = (tensor - 0.5) / 0.5  # [-1, 1]
        else:
            # Default: no per-case normalization (legacy behavior)
            # This path should NOT be used - it's kept for backward compatibility
            tensor = tensor * 3000.0

        # Apply hard normalization to [-1, 1] if requested (for per-case mode)
        if getattr(self.opt, 'mri_hard_normalize', False) and not getattr(self.opt, 'mri_normalize_per_slice', False):
            # Per-slice min-max normalization (NOT recommended - loses inter-slice intensity contrast)
            tensor_min = tensor.min()
            tensor_max = tensor.max()
            if tensor_max > tensor_min:
                tensor = (tensor - tensor_min) / (tensor_max - tensor_min)  # [0, 1]
                tensor = tensor * 2.0 - 1.0  # [-1, 1]

        return tensor

    def _align_phase_global(self, A_tensor: torch.Tensor, B_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align the phase of B to A using global mean phase difference.
        
        Args:
            A_tensor: Complex tensor with shape [2, H, W] where channel 0 is real, 1 is imaginary
            B_tensor: Complex tensor with shape [2, H, W] where channel 0 is real, 1 is imaginary
        
        Returns:
            A_tensor (unchanged), B_tensor (phase-aligned)
        """
        if self.opt.mri_representation != 'real_imag':
            # Phase alignment only applies to complex (real/imag) representation
            return A_tensor, B_tensor
        
        # Extract real and imaginary parts
        A_real, A_imag = A_tensor[0], A_tensor[1]
        B_real, B_imag = B_tensor[0], B_tensor[1]
        
        # Compute phase for both images
        A_phase = torch.atan2(A_imag, A_real)  # Shape: [H, W]
        B_phase = torch.atan2(B_imag, B_real)  # Shape: [H, W]
        
        # Compute magnitude for B (we'll keep this unchanged)
        B_mag = torch.sqrt(B_real ** 2 + B_imag ** 2)
        
        # Create a mask to ignore background/noise (optional but recommended)
        # Only consider pixels with significant magnitude
        A_mag = torch.sqrt(A_real ** 2 + A_imag ** 2)
        threshold = 0.1 * torch.max(torch.stack([A_mag.max(), B_mag.max()]))
        mask = (A_mag > threshold) & (B_mag > threshold)
        
        if mask.sum() > 0:
            # Compute phase difference only on valid pixels
            phase_diff = A_phase[mask] - B_phase[mask]
            
            # Compute mean phase difference (wrapped to [-π, π])
            mean_phase_diff = torch.atan2(
                torch.sin(phase_diff).mean(),
                torch.cos(phase_diff).mean()
            )
        else:
            # If no valid pixels, no adjustment needed
            mean_phase_diff = 0.0
        
        # Apply phase correction to B
        B_phase_aligned = B_phase + mean_phase_diff
        
        # Convert back to real/imaginary representation
        B_real_aligned = B_mag * torch.cos(B_phase_aligned)
        B_imag_aligned = B_mag * torch.sin(B_phase_aligned)
        
        # Stack back into tensor format
        B_tensor_aligned = torch.stack([B_real_aligned, B_imag_aligned], dim=0)

        return A_tensor, B_tensor_aligned

    def _center_crop(self, tensor: torch.Tensor, crop_size: int) -> torch.Tensor:
        """Apply center crop to tensor.

        Args:
            tensor: [C, H, W] tensor
            crop_size: target size

        Returns:
            Cropped tensor [C, crop_size, crop_size]
        """
        _, h, w = tensor.shape

        if h < crop_size or w < crop_size:
            raise ValueError(f'Cannot crop tensor of size {tensor.shape} to {crop_size}x{crop_size}')

        top = (h - crop_size) // 2
        left = (w - crop_size) // 2

        return tensor[:, top:top + crop_size, left:left + crop_size]

    def _paired_random_crop(self, tensor_a: torch.Tensor, tensor_b: torch.Tensor, crop_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the same random crop to both tensors."""
        if crop_size <= 0:
            return tensor_a, tensor_b

        min_h = min(tensor_a.shape[1], tensor_b.shape[1])
        min_w = min(tensor_a.shape[2], tensor_b.shape[2])
        if min_h < crop_size or min_w < crop_size:
            raise ValueError(
                f'Cannot crop tensors of shapes {tuple(tensor_a.shape)} and {tuple(tensor_b.shape)} to {crop_size}.'
            )

        max_top = min_h - crop_size
        max_left = min_w - crop_size
        top = random.randint(0, max_top) if max_top > 0 else 0
        left = random.randint(0, max_left) if max_left > 0 else 0

        return (
            tensor_a[:, top:top + crop_size, left:left + crop_size],
            tensor_b[:, top:top + crop_size, left:left + crop_size]
        )
'''        
import matplotlib.pyplot as plt

        # --- assume A_tensor, B_tensor are [2, H, W] complex tensors ---
        A_real, A_imag = A_tensor[0], A_tensor[1]
        B_real, B_imag = B_tensor[0], B_tensor[1]

        A_mag = torch.sqrt(A_real ** 2 + A_imag ** 2)
        B_mag = torch.sqrt(B_real ** 2 + B_imag ** 2)

        # --- optional: normalization for consistent display ---
        def normalize(img):
            img = img - img.min()
            img = img / (img.max() + 1e-8)
            return img

        A_real_vis = normalize(A_real).cpu()
        A_imag_vis = normalize(A_imag).cpu()
        A_mag_vis  = normalize(A_mag).cpu()

        B_real_vis = normalize(B_real).cpu()
        B_imag_vis = normalize(B_imag).cpu()
        B_mag_vis  = normalize(B_mag).cpu()

        # --- visualization: 2 rows (A/B), 3 columns (real/imag/mag) ---
        plt.figure(figsize=(10, 6))

        titles = ["Real", "Imag", "Magnitude"]

        for i, (a_img, b_img) in enumerate(
            [(A_real_vis, B_real_vis), (A_imag_vis, B_imag_vis), (A_mag_vis, B_mag_vis)]
        ):
            # Row 1: A
            plt.subplot(2, 3, i + 1)
            plt.imshow(a_img, cmap='gray')
            plt.title(f"A - {titles[i]}")
            plt.axis('off')

            # Row 2: B
            plt.subplot(2, 3, i + 4)
            plt.imshow(b_img, cmap='gray')
            plt.title(f"B - {titles[i]}")
            plt.axis('off')

        plt.tight_layout()
        plt.show() 
'''