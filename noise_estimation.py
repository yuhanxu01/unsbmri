"""
噪音估计工具
用于估计MRI数据中的噪音水平，支持Nila风格的自适应处理
"""

import numpy as np
import torch
import h5py
from pathlib import Path
from typing import Dict, Tuple
import scipy.ndimage as ndimage


def estimate_noise_mad(image: np.ndarray) -> float:
    """
    使用Median Absolute Deviation (MAD)估计噪音标准差

    Args:
        image: 2D magnitude image (numpy array)

    Returns:
        噪音标准差估计值

    原理:
        假设背景区域只包含噪音，使用MAD估计:
        σ = 1.4826 * median(|x - median(x)|)
        系数1.4826使得MAD等价于高斯分布的标准差
    """
    # 使用较低的像素值作为背景
    threshold = np.percentile(image, 20)
    background = image[image < threshold]

    if len(background) < 100:
        # 如果背景像素太少，使用整幅图像的较低百分位
        background = image[image < np.percentile(image, 10)]

    if len(background) == 0:
        return 0.0

    median_bg = np.median(background)
    mad = np.median(np.abs(background - median_bg))
    sigma = 1.4826 * mad

    return float(sigma)


def estimate_noise_background(image: np.ndarray, corner_size: int = 20) -> float:
    """
    使用图像四角的背景区域估计噪音

    Args:
        image: 2D magnitude image
        corner_size: 每个角落的区域大小

    Returns:
        噪音标准差估计值

    原理:
        假设图像四角主要是背景（无信号），计算这些区域的标准差
    """
    h, w = image.shape
    corners = [
        image[:corner_size, :corner_size],           # 左上
        image[:corner_size, -corner_size:],          # 右上
        image[-corner_size:, :corner_size],          # 左下
        image[-corner_size:, -corner_size:]          # 右下
    ]

    # 计算每个角落的标准差
    stds = [np.std(corner) for corner in corners]

    # 使用中位数作为鲁棒估计
    sigma = np.median(stds)

    return float(sigma)


def estimate_noise_laplacian(image: np.ndarray) -> float:
    """
    使用Laplacian算子估计噪音

    Args:
        image: 2D image

    Returns:
        噪音标准差估计值

    原理:
        Laplacian算子对噪音敏感，对信号不敏感
        σ = median(|Laplacian(image)|) / 0.6745 / √2

    参考:
        J. Immerkaer, "Fast Noise Variance Estimation",
        Computer Vision and Image Understanding, 1996
    """
    # Laplacian kernel
    laplacian = ndimage.laplace(image)

    # 使用绝对值的中位数
    sigma = np.median(np.abs(laplacian)) / 0.6745 / np.sqrt(2)

    return float(sigma)


def estimate_noise_rician(magnitude: np.ndarray, method: str = 'background') -> Tuple[float, float]:
    """
    估计Rician噪音参数

    Args:
        magnitude: MRI magnitude image
        method: 'background' 或 'mad'

    Returns:
        (sigma, A) - 噪音标准差和信号幅度估计

    原理:
        MRI magnitude遵循Rician分布:
        R ~ Rician(A, σ)
        其中A是真实信号幅度，σ是高斯噪音标准差

        在背景区域(A≈0): E[R] ≈ σ√(π/2)
    """
    if method == 'background':
        # 使用背景区域
        threshold = np.percentile(magnitude, 10)
        background = magnitude[magnitude < threshold]

        if len(background) > 100:
            # Rician分布在A=0时的期望
            mean_bg = np.mean(background)
            sigma = mean_bg / np.sqrt(np.pi / 2)
        else:
            sigma = estimate_noise_mad(magnitude)
    else:
        sigma = estimate_noise_mad(magnitude)

    # 估计信号幅度 (使用高信号区域)
    signal_region = magnitude[magnitude > np.percentile(magnitude, 90)]
    A = np.mean(signal_region)

    return float(sigma), float(A)


def estimate_noise_complex(real: np.ndarray, imag: np.ndarray) -> Tuple[float, float]:
    """
    从复数MRI数据估计噪音

    Args:
        real: 实部
        imag: 虚部

    Returns:
        (sigma_real, sigma_imag) - 实部和虚部的噪音标准差

    原理:
        复数MRI中，实部和虚部的噪音是独立的高斯分布
        直接估计两个分量的噪音标准差
    """
    sigma_real = estimate_noise_mad(real)
    sigma_imag = estimate_noise_mad(imag)

    return sigma_real, sigma_imag


def analyze_dataset_noise(dataroot: str, domain: str = 'A',
                          num_samples: int = 100) -> Dict[str, float]:
    """
    分析整个数据集的噪音水平

    Args:
        dataroot: 数据集根目录
        domain: 'A' 或 'B'
        num_samples: 采样的切片数量

    Returns:
        噪音统计信息字典
    """
    data_dir = Path(dataroot) / f'train{domain}'

    if not data_dir.exists():
        raise ValueError(f"Directory {data_dir} does not exist")

    noise_estimates = []
    snr_estimates = []

    h5_files = list(data_dir.glob('*.h5'))

    for h5_path in h5_files[:num_samples]:
        with h5py.File(h5_path, 'r') as handle:
            for key in handle.keys():
                if key.startswith('slices_'):
                    data = handle[key][...]

                    # 计算magnitude
                    if data.shape[-1] >= 2:
                        real = data[..., 0]
                        imag = data[..., 1]
                        magnitude = np.sqrt(real**2 + imag**2)
                    else:
                        magnitude = data[..., 0]

                    # 估计噪音
                    sigma = estimate_noise_mad(magnitude)
                    noise_estimates.append(sigma)

                    # 估计SNR
                    signal = np.percentile(magnitude, 95)
                    snr = signal / sigma if sigma > 0 else float('inf')
                    snr_estimates.append(snr)

    if len(noise_estimates) == 0:
        return {
            'mean_noise': 0.0,
            'median_noise': 0.0,
            'std_noise': 0.0,
            'mean_snr': 0.0,
            'median_snr': 0.0
        }

    return {
        'mean_noise': float(np.mean(noise_estimates)),
        'median_noise': float(np.median(noise_estimates)),
        'std_noise': float(np.std(noise_estimates)),
        'min_noise': float(np.min(noise_estimates)),
        'max_noise': float(np.max(noise_estimates)),
        'mean_snr': float(np.mean(snr_estimates)),
        'median_snr': float(np.median(snr_estimates)),
        'num_samples': len(noise_estimates)
    }


def compute_adaptive_weight(t: int, T: int, tau: float,
                           data_noise_level: float,
                           schedule: str = 'linear') -> float:
    """
    计算Nila风格的自适应权重

    Args:
        t: 当前时间步 (0 to T-1)
        T: 总时间步数
        tau: SB噪音参数
        data_noise_level: 数据噪音水平σ_y
        schedule: 'linear' 或 'exponential'

    Returns:
        自适应权重λ_t ∈ [0, 1]

    原理 (类似Nila Eq. 10):
        在时间步t，人工噪音水平约为: σ_t ≈ √(τ * t/T * (1 - t/T))

        if σ_t > σ_y:  # 人工噪音占主导
            λ_t = 1.0  # 全强度重建损失
        else:          # 数据噪音占主导
            λ_t 线性衰减到0  # 减少对含噪数据的拟合
    """
    # 计算当前时间步的人工噪音水平
    t_normalized = t / T
    artificial_noise = np.sqrt(tau * t_normalized * (1 - t_normalized))

    # 噪音比率
    noise_ratio = artificial_noise / (data_noise_level + 1e-8)

    if noise_ratio >= 1.0:
        # 人工噪音足够大，使用全强度
        return 1.0
    else:
        # 人工噪音小于数据噪音，应该衰减
        if schedule == 'linear':
            # 线性衰减: λ = noise_ratio
            weight = noise_ratio
        elif schedule == 'exponential':
            # 指数衰减: λ = exp(-k*(1-ratio))
            k = 3.0  # 衰减速率
            weight = np.exp(-k * (1 - noise_ratio))
        elif schedule == 'step':
            # 阶跃: 超过阈值就截断
            threshold = 0.5
            weight = 1.0 if noise_ratio > threshold else 0.0
        else:
            weight = noise_ratio

        return float(np.clip(weight, 0.0, 1.0))


def visualize_adaptive_schedule(T: int = 20, tau: float = 0.1,
                                data_noise_levels: list = [0.01, 0.05, 0.1]):
    """
    可视化自适应权重调度

    Args:
        T: 时间步数
        tau: SB参数
        data_noise_levels: 不同的数据噪音水平
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    t_range = np.arange(T)

    # 左图: 噪音水平随时间变化
    artificial_noise = [np.sqrt(tau * t/T * (1 - t/T)) for t in t_range]
    ax1.plot(t_range, artificial_noise, 'b-', linewidth=2, label='Artificial noise (σ_t)')

    for sigma_y in data_noise_levels:
        ax1.axhline(y=sigma_y, linestyle='--', label=f'Data noise σ_y={sigma_y}')

    ax1.set_xlabel('Time step t')
    ax1.set_ylabel('Noise level')
    ax1.set_title('Noise levels during SB process')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 右图: 自适应权重
    for sigma_y in data_noise_levels:
        weights = [compute_adaptive_weight(t, T, tau, sigma_y) for t in t_range]
        ax2.plot(t_range, weights, linewidth=2, label=f'σ_y={sigma_y}')

    ax2.set_xlabel('Time step t')
    ax2.set_ylabel('Adaptive weight λ_t')
    ax2.set_title('Nila-style adaptive weights')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-0.05, 1.05])

    plt.tight_layout()
    plt.savefig('adaptive_schedule_visualization.png', dpi=150)
    print("Saved visualization to adaptive_schedule_visualization.png")
    plt.show()


if __name__ == '__main__':
    """
    使用示例:

    # 1. 估计单张图像的噪音
    import h5py
    with h5py.File('datasets/trainA/case001.h5', 'r') as f:
        data = f['slices_10'][...]
        magnitude = np.sqrt(data[...,0]**2 + data[...,1]**2)

        noise_mad = estimate_noise_mad(magnitude)
        noise_bg = estimate_noise_background(magnitude)
        noise_lap = estimate_noise_laplacian(magnitude)

        print(f"MAD estimate: {noise_mad:.4f}")
        print(f"Background estimate: {noise_bg:.4f}")
        print(f"Laplacian estimate: {noise_lap:.4f}")

    # 2. 分析整个数据集
    stats_A = analyze_dataset_noise('./datasets/PD_PDFS', domain='A')
    stats_B = analyze_dataset_noise('./datasets/PD_PDFS', domain='B')

    print("Domain A noise statistics:")
    print(stats_A)
    print("\nDomain B noise statistics:")
    print(stats_B)

    # 3. 可视化自适应调度
    visualize_adaptive_schedule(T=20, tau=0.1,
                                data_noise_levels=[0.01, 0.05, 0.1])
    """

    # 示例: 可视化自适应权重调度
    print("Generating adaptive weight schedule visualization...")
    print("This shows how the reconstruction loss weight should change")
    print("at different time steps for different data noise levels.")
    print()

    visualize_adaptive_schedule(T=20, tau=0.1,
                                data_noise_levels=[0.01, 0.05, 0.1])
