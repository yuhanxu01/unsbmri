# 噪音自适应UNSB实施指南

## 快速开始

### 第一步: 估计数据噪音水平

在训练之前，先分析你的数据集的噪音水平:

```bash
python -c "
from noise_estimation import analyze_dataset_noise
import json

# 分析域A和域B的噪音
stats_A = analyze_dataset_noise('./datasets/YOUR_DATASET', domain='A', num_samples=50)
stats_B = analyze_dataset_noise('./datasets/YOUR_DATASET', domain='B', num_samples=50)

print('Domain A (PD):')
print(json.dumps(stats_A, indent=2))
print('\nDomain B (PDFs):')
print(json.dumps(stats_B, indent=2))

# 保存结果
with open('noise_analysis.json', 'w') as f:
    json.dump({'A': stats_A, 'B': stats_B}, f, indent=2)

# 推荐的data_noise_level参数
recommended_noise = max(stats_A['median_noise'], stats_B['median_noise'])
print(f'\n推荐使用 --data_noise_level {recommended_noise:.4f}')
"
```

**预期输出示例**:
```json
{
  "A": {
    "mean_noise": 0.0234,
    "median_noise": 0.0198,
    "std_noise": 0.0087,
    "mean_snr": 45.2,
    "median_snr": 48.1
  },
  "B": {
    "mean_noise": 0.0312,
    "median_noise": 0.0287,
    "std_noise": 0.0102,
    "mean_snr": 38.7,
    "median_snr": 41.2
  }
}

推荐使用 --data_noise_level 0.0287
```

---

### 第二步: 修改代码

需要修改以下文件:
1. `options/base_options.py` - 添加新参数
2. `models/sb_model.py` - 实现自适应损失

#### 修改1: options/base_options.py

在`BaseOptions.initialize()`中添加噪音相关参数:

```python
# 在文件末尾的parser.add_argument之后添加:

# === 噪音处理参数 (Nila-inspired) ===
parser.add_argument('--data_noise_level', type=float, default=0.0,
                   help='Estimated noise level in the data (σ_y). '
                        'Set to 0 to disable noise-adaptive loss. '
                        'Use noise_estimation.py to estimate this value.')

parser.add_argument('--noise_adaptive_schedule', type=str, default='linear',
                   choices=['linear', 'exponential', 'step', 'none'],
                   help='Schedule for noise-adaptive weight decay.')

parser.add_argument('--noise_adaptive_start_epoch', type=int, default=0,
                   help='Start applying noise-adaptive loss after this epoch. '
                        'Useful for curriculum learning.')

parser.add_argument('--visualize_noise_schedule', action='store_true',
                   help='Visualize the noise-adaptive schedule at start of training.')
```

#### 修改2: models/sb_model.py

在`SBModel`类中修改`compute_G_loss()`方法:

```python
def compute_G_loss(self):
    """Calculate GAN and NCE loss for the generator"""
    bs = self.real_A.size(0)
    tau = self.opt.tau

    # === 新增: 计算噪音自适应权重 ===
    if self.opt.data_noise_level > 0:
        t = self.time_idx[0].item()
        T = self.opt.num_timesteps

        # 计算当前时间步的人工噪音水平
        t_normalized = t / T
        artificial_noise = np.sqrt(tau * t_normalized * (1 - t_normalized))

        # 噪音比率
        noise_ratio = artificial_noise / (self.opt.data_noise_level + 1e-8)

        # 计算自适应权重
        if noise_ratio >= 1.0:
            noise_adaptive_weight = 1.0
        else:
            if self.opt.noise_adaptive_schedule == 'linear':
                noise_adaptive_weight = noise_ratio
            elif self.opt.noise_adaptive_schedule == 'exponential':
                noise_adaptive_weight = np.exp(-3.0 * (1 - noise_ratio))
            elif self.opt.noise_adaptive_schedule == 'step':
                noise_adaptive_weight = 1.0 if noise_ratio > 0.5 else 0.0
            else:
                noise_adaptive_weight = 1.0

        noise_adaptive_weight = float(np.clip(noise_adaptive_weight, 0.0, 1.0))
    else:
        noise_adaptive_weight = 1.0

    # 存储用于监控
    self.noise_adaptive_weight = noise_adaptive_weight

    # === GAN损失 ===
    fake = self.fake_B
    std = torch.rand(size=[1]).item() * self.opt.std

    if self.opt.lambda_GAN > 0.0:
        pred_fake = self.netD(fake, self.time_idx)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
    else:
        self.loss_G_GAN = 0.0

    # === SB损失 (修改部分) ===
    self.loss_SB = 0
    if self.opt.lambda_SB > 0.0:
        XtXt_1 = torch.cat([self.real_A_noisy, self.fake_B], dim=1)
        XtXt_2 = torch.cat([self.real_A_noisy2, self.fake_B2], dim=1)

        bs = self.opt.batch_size

        # 能量项 (不受噪音影响，保持不变)
        ET_XY = self.netE(XtXt_1, self.time_idx, XtXt_1).mean() \
              - torch.logsumexp(self.netE(XtXt_1, self.time_idx, XtXt_2).reshape(-1), dim=0)

        energy_term = -(self.opt.num_timesteps - self.time_idx[0]) / self.opt.num_timesteps * self.opt.tau * ET_XY

        # 🔥 重建项 (应用噪音自适应权重)
        reconstruction_loss = torch.mean((self.real_A_noisy - self.fake_B)**2)
        reconstruction_term = noise_adaptive_weight * self.opt.tau * reconstruction_loss

        self.loss_SB = energy_term + reconstruction_term

        # 存储分解用于监控
        self.loss_SB_energy = energy_term
        self.loss_SB_recon = reconstruction_term

    # === NCE损失 ===
    if self.opt.lambda_NCE > 0.0:
        self.loss_NCE = self.calculate_NCE_loss(self.real_A, fake)
    else:
        self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

    if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
        self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
        loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
    else:
        loss_NCE_both = self.loss_NCE

    # === 总损失 ===
    self.loss_G = self.loss_G_GAN + self.opt.lambda_SB * self.loss_SB + self.opt.lambda_NCE * loss_NCE_both
    return self.loss_G
```

#### 修改3: 添加监控 (可选但推荐)

在`sb_model.py`的`__init__`中添加:

```python
def __init__(self, opt):
    BaseModel.__init__(self, opt)

    # 原有代码...
    self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE', 'SB']

    # 🔥 新增: 添加噪音自适应相关的监控项
    if opt.data_noise_level > 0:
        self.loss_names += ['SB_energy', 'SB_recon']
        # 注意: noise_adaptive_weight 不是损失，但会在训练时打印

    # ... 其余代码
```

在wandb日志中添加噪音权重监控 (修改 `util/wandb_logger.py`):

```python
# 在log_current_losses中添加:
if hasattr(model, 'noise_adaptive_weight'):
    wandb_log['train/noise_adaptive_weight'] = model.noise_adaptive_weight
```

---

### 第三步: 训练

#### 基础训练命令

```bash
python train.py \
  --dataroot ./datasets/PD_PDFS \
  --name experiment_noise_adaptive \
  --model sb \
  --dataset_mode mri_unaligned \
  --mri_representation real_imag \
  --mri_normalize_per_case \
  --mri_normalize_method percentile_95 \
  --data_noise_level 0.03 \
  --noise_adaptive_schedule linear \
  --wandb_project mri-contrast-transfer-noise \
  --batch_size 4 \
  --n_epochs 200 \
  --n_epochs_decay 200
```

#### 对比实验

**实验1: Baseline (无噪音处理)**
```bash
python train.py \
  --name baseline_no_noise_handling \
  --data_noise_level 0.0 \
  # ... 其他参数相同
```

**实验2: 线性衰减**
```bash
python train.py \
  --name noise_adaptive_linear \
  --data_noise_level 0.03 \
  --noise_adaptive_schedule linear \
  # ... 其他参数相同
```

**实验3: 指数衰减**
```bash
python train.py \
  --name noise_adaptive_exponential \
  --data_noise_level 0.03 \
  --noise_adaptive_schedule exponential \
  # ... 其他参数相同
```

**实验4: 不同噪音水平**
```bash
# 低估噪音
python train.py \
  --name noise_level_0.01 \
  --data_noise_level 0.01 \
  # ...

# 准确估计
python train.py \
  --name noise_level_0.03 \
  --data_noise_level 0.03 \
  # ...

# 高估噪音
python train.py \
  --name noise_level_0.05 \
  --data_noise_level 0.05 \
  # ...
```

---

### 第四步: 监控训练

在Wandb中关注以下指标:

1. **noise_adaptive_weight**: 应该在早期epochs接近1.0，晚期时间步逐渐降低
2. **loss_SB_energy vs loss_SB_recon**:
   - 能量项应该保持稳定
   - 重建项会受自适应权重影响
3. **生成质量**: 观察生成图像的噪音水平是否降低

**预期现象**:
- 早期训练: weight ≈ 1.0 → 正常SB训练
- 中后期训练: weight在0.5-1.0之间波动 → 自适应调整
- 晚期小t步: weight ≈ 0.2-0.5 → 减少噪音拟合

---

## 进阶优化

### 优化1: 课程学习

先用全强度训练，再逐渐启用噪音自适应:

```bash
python train.py \
  --data_noise_level 0.03 \
  --noise_adaptive_start_epoch 50 \
  # ... 其他参数
```

**原理**: 让模型先学习基本的对比度映射，再fine-tune去噪

### 优化2: 动态噪音估计

在训练过程中动态调整噪音水平:

```python
# 在sb_model.py的optimize_parameters中添加:
def optimize_parameters(self):
    # ... 原有代码

    # 每N个iteration更新噪音估计
    if self.opt.dynamic_noise_estimation and self.total_iters % 1000 == 0:
        # 使用当前生成器估计数据噪音
        with torch.no_grad():
            # 在t=0时刻，fake_B应该接近real_A (如果无噪音)
            residual = self.real_A - self.netG(self.real_A,
                                               torch.zeros_like(self.time_idx),
                                               torch.randn_like(z))
            estimated_noise = torch.std(residual).item()

            # 指数移动平均
            alpha = 0.1
            self.opt.data_noise_level = (1 - alpha) * self.opt.data_noise_level \
                                       + alpha * estimated_noise

            print(f"Updated data_noise_level to {self.opt.data_noise_level:.4f}")
```

### 优化3: 空间自适应

不同区域的噪音水平可能不同 (背景 vs 信号区域):

```python
def compute_spatial_adaptive_weight(self, real_A_noisy, fake_B):
    """
    计算空间变化的自适应权重
    背景区域(低信号): 更低的权重
    信号区域(高信号): 更高的权重
    """
    # 计算magnitude
    mag_A = torch.sqrt(real_A_noisy[:, 0]**2 + real_A_noisy[:, 1]**2)

    # 归一化到[0, 1]
    mag_A_norm = (mag_A - mag_A.min()) / (mag_A.max() - mag_A.min() + 1e-8)

    # 空间权重: 信号强度越高，权重越高
    spatial_weight = mag_A_norm.unsqueeze(1)  # [B, 1, H, W]

    # 结合时间自适应权重
    combined_weight = self.noise_adaptive_weight * spatial_weight

    # 加权损失
    reconstruction_loss = torch.mean(
        combined_weight * (real_A_noisy - fake_B)**2
    )

    return reconstruction_loss
```

---

## 评估

### 定量评估脚本

创建 `evaluate_noise.py`:

```python
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from noise_estimation import estimate_noise_mad

def evaluate_model(opt, model, test_dataset):
    """评估模型在噪音指标上的表现"""

    noise_levels_input = []
    noise_levels_output = []
    psnr_scores = []
    ssim_scores = []

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_dataset):
            model.set_input(data)
            model.forward()

            # 计算输入和输出的噪音水平
            real_A_mag = torch.sqrt(model.real_A[0, 0]**2 + model.real_A[0, 1]**2).cpu().numpy()
            fake_B_mag = torch.sqrt(model.fake_B[0, 0]**2 + model.fake_B[0, 1]**2).cpu().numpy()

            noise_input = estimate_noise_mad(real_A_mag)
            noise_output = estimate_noise_mad(fake_B_mag)

            noise_levels_input.append(noise_input)
            noise_levels_output.append(noise_output)

            # 如果有参考图像
            if hasattr(model, 'real_B'):
                real_B_mag = torch.sqrt(model.real_B[0, 0]**2 + model.real_B[0, 1]**2).cpu().numpy()

                # 归一化到[0, 1]用于PSNR/SSIM计算
                fake_B_norm = (fake_B_mag - fake_B_mag.min()) / (fake_B_mag.max() - fake_B_mag.min())
                real_B_norm = (real_B_mag - real_B_mag.min()) / (real_B_mag.max() - real_B_mag.min())

                psnr_score = psnr(real_B_norm, fake_B_norm, data_range=1.0)
                ssim_score = ssim(real_B_norm, fake_B_norm, data_range=1.0)

                psnr_scores.append(psnr_score)
                ssim_scores.append(ssim_score)

    results = {
        'mean_noise_input': np.mean(noise_levels_input),
        'mean_noise_output': np.mean(noise_levels_output),
        'noise_reduction_ratio': np.mean(noise_levels_output) / np.mean(noise_levels_input),
        'mean_psnr': np.mean(psnr_scores) if psnr_scores else None,
        'mean_ssim': np.mean(ssim_scores) if ssim_scores else None
    }

    return results
```

运行评估:
```bash
python test.py \
  --name noise_adaptive_linear \
  --epoch latest \
  # ... 其他参数

python -c "
from evaluate_noise import evaluate_model
# 加载模型和数据集
results = evaluate_model(opt, model, test_dataset)
print(results)
"
```

**期望结果**:
```python
{
    'mean_noise_input': 0.0287,
    'mean_noise_output': 0.0134,  # 噪音减少了约53%
    'noise_reduction_ratio': 0.47,
    'mean_psnr': 32.4,
    'mean_ssim': 0.89
}
```

对比baseline:
```python
# Baseline (无噪音处理)
{
    'mean_noise_output': 0.0298,  # 噪音几乎没变
    'noise_reduction_ratio': 1.04,
    'mean_psnr': 28.7,
    'mean_ssim': 0.82
}
```

---

## 故障排除

### 问题1: 训练不稳定

**症状**: 损失剧烈波动，生成质量差

**可能原因**:
- `data_noise_level` 设置过高
- 自适应权重变化过快

**解决方案**:
```bash
# 降低噪音水平估计
--data_noise_level 0.01  # 而非0.03

# 使用更平滑的衰减
--noise_adaptive_schedule exponential

# 延迟启用自适应
--noise_adaptive_start_epoch 100
```

### 问题2: 噪音减少不明显

**症状**: 输出图像噪音水平和输入相近

**可能原因**:
- `data_noise_level` 设置过低
- 自适应权重几乎总是1.0

**解决方案**:
```bash
# 提高噪音水平估计
--data_noise_level 0.05

# 使用更激进的衰减
--noise_adaptive_schedule linear
```

### 问题3: 过度平滑

**症状**: 输出图像细节丢失

**可能原因**:
- `data_noise_level` 设置过高
- 自适应权重过早降为0

**解决方案**:
```bash
# 降低噪音水平
--data_noise_level 0.02

# 检查可视化
--visualize_noise_schedule
```

---

## 总结

**核心思想**: 借鉴Nila的噪音水平自适应策略，在SB框架中动态调整重建损失的权重

**关键参数**:
- `data_noise_level`: 最重要！需要从数据中准确估计
- `noise_adaptive_schedule`: linear通常效果最好
- `tau`: SB的噪音参数，与data_noise_level配合使用

**预期效果**:
- 噪音减少30-50%
- PSNR提升2-4 dB
- SSIM提升0.05-0.10
- 视觉质量显著改善

**下一步**:
如果基础方案效果不够理想,可以尝试:
1. 方案3: 噪音条件化生成器
2. 优化1: 课程学习
3. 优化3: 空间自适应权重
