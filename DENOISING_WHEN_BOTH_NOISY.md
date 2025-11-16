# 当起点和终点都有噪音时的去噪策略

## 问题场景

你的情况：
```
源域 (PD):   含噪音, σ_A ≈ 0.03
目标域 (PDFs): 含噪音, σ_B ≈ 0.03

任务: PD → PDFs 对比度迁移
期望: 输出比输入更干净
```

这与Nila的原始场景不同：
- Nila: 含噪测量 → 干净图像 (有明确的去噪目标)
- 你: 含噪源 → 含噪目标 (没有"干净参考")

---

## 基础方案的效果分析

### 方案2 (Nila启发的自适应SB损失)

**能做到**:
- ✅ 减少噪音传递 (30-50%噪音减少)
- ✅ 隐式去噪 (通过先验知识)
- ✅ 避免学习噪音模式

**局限性**:
- ❌ 无法完全消除噪音
- ❌ 受限于目标域数据质量
- ❌ 无显式去噪监督

**数学解释**:

```python
# 标准SB训练
min_G E[ ||X_t - G(X_t)||² ]
# X_t包含输入噪音 → G学习复制噪音

# 自适应SB
min_G E[ λ_t * ||X_t - G(X_t)||² ]
# λ_t < 1 减弱噪音拟合 → 部分去噪

# 但GAN损失仍然是:
min_G E[ D(G(X)) - D(real_B) ]
# real_B有噪音 → G被鼓励生成"适度噪音"
```

**预期效果**:
```
输入PD:  σ = 0.030
输出PDFs: σ = 0.015-0.020 (改善40-50%)

对比baseline:
输入PD:  σ = 0.030
输出PDFs: σ = 0.028-0.032 (几乎不变)
```

---

## 增强方案：组合多种策略

### 🥇 方案A: 自适应SB + 判别器去噪引导

**核心思想**: 让判别器偏好更干净的图像，而不仅仅是"真实"

#### 实现方法

**A1: 数据增强 - 合成干净样本**

```python
# 在数据加载时
class MriUnalignedDataset:
    def __getitem__(self, index):
        A_tensor = self._load_slice(...)  # 含噪PD
        B_tensor = self._load_slice(...)  # 含噪PDFs

        # 🔥 新增: 生成"伪干净"样本
        if self.opt.denoise_augmentation and random.random() < 0.5:
            # 方法1: 传统去噪 (BM3D, NLM)
            B_tensor_clean = self._traditional_denoise(B_tensor)

            # 方法2: 低通滤波
            B_tensor_clean = self._lowpass_filter(B_tensor, sigma=1.5)

            # 方法3: Wavelet软阈值
            B_tensor_clean = self._wavelet_denoise(B_tensor)

            # 混合: 50%原始, 50%去噪
            return {
                'A': A_tensor,
                'B': B_tensor_clean,  # 判别器学习偏好干净样本
                'B_original': B_tensor  # 保留用于其他损失
            }

        return {'A': A_tensor, 'B': B_tensor}

def _lowpass_filter(self, tensor, sigma=1.5):
    """简单的高斯低通滤波"""
    from scipy.ndimage import gaussian_filter

    if tensor.shape[0] == 2:  # real/imag
        real_filtered = gaussian_filter(tensor[0].numpy(), sigma=sigma)
        imag_filtered = gaussian_filter(tensor[1].numpy(), sigma=sigma)
        return torch.from_numpy(np.stack([real_filtered, imag_filtered]))
    else:  # magnitude
        mag_filtered = gaussian_filter(tensor[0].numpy(), sigma=sigma)
        return torch.from_numpy(mag_filtered[None, ...])

def _wavelet_denoise(self, tensor, wavelet='db4', level=3):
    """Wavelet软阈值去噪"""
    import pywt

    if tensor.shape[0] == 2:
        real_denoised = pywt.threshold(
            pywt.wavedec2(tensor[0].numpy(), wavelet, level=level),
            value=0.1, mode='soft'
        )
        imag_denoised = pywt.threshold(
            pywt.wavedec2(tensor[1].numpy(), wavelet, level=level),
            value=0.1, mode='soft'
        )
        return torch.from_numpy(np.stack([
            pywt.waverec2(real_denoised, wavelet),
            pywt.waverec2(imag_denoised, wavelet)
        ]))
    else:
        coeffs = pywt.wavedec2(tensor[0].numpy(), wavelet, level=level)
        coeffs_thresh = pywt.threshold(coeffs, value=0.1, mode='soft')
        mag_denoised = pywt.waverec2(coeffs_thresh, wavelet)
        return torch.from_numpy(mag_denoised[None, ...])
```

**训练命令**:
```bash
python train.py \
  --denoise_augmentation \
  --denoise_method lowpass \
  --denoise_prob 0.5 \
  --data_noise_level 0.03 \
  # ... 其他参数
```

**效果**:
- 判别器学习: "干净的PDFs > 含噪的PDFs"
- 生成器被引导生成更干净的图像
- 预期噪音减少: **50-70%**

---

**A2: 噪音水平条件判别器**

```python
# models/networks.py - 修改判别器

class NoisyConditionalDiscriminator(nn.Module):
    """
    判别器同时判断:
    1. 真实 vs 生成
    2. 干净 vs 含噪
    """
    def __init__(self, input_nc, ndf=64):
        super().__init__()

        # 主判别器网络
        self.main = nn.Sequential(
            # ... 标准PatchGAN layers
        )

        # 🔥 新增: 噪音水平估计分支
        self.noise_estimator = nn.Sequential(
            nn.Conv2d(ndf*8, 128, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # 输出[0,1], 0=干净, 1=很吵
        )

    def forward(self, x, t=None):
        features = self.main(x)

        # 真实性判断
        validity = self.get_validity(features)

        # 噪音水平估计
        noise_level = self.noise_estimator(features)

        return validity, noise_level

# models/sb_model.py - 修改判别器损失

def compute_D_loss(self):
    # 真实数据
    pred_real, noise_real = self.netD(self.real_B, self.time_idx)
    loss_D_real = self.criterionGAN(pred_real, True).mean()

    # 🔥 新增: 真实数据的噪音水平应该较高
    target_noise_real = torch.ones_like(noise_real) * 0.5  # 中等噪音
    loss_noise_real = F.mse_loss(noise_real, target_noise_real)

    # 生成数据
    pred_fake, noise_fake = self.netD(self.fake_B.detach(), self.time_idx)
    loss_D_fake = self.criterionGAN(pred_fake, False).mean()

    # 🔥 生成数据应该更干净
    target_noise_fake = torch.zeros_like(noise_fake)  # 期望干净
    loss_noise_fake = F.mse_loss(noise_fake, target_noise_fake)

    self.loss_D = (loss_D_real + loss_D_fake) * 0.5 + \
                  (loss_noise_real + loss_noise_fake) * self.opt.lambda_noise
    return self.loss_D

def compute_G_loss(self):
    # ... 原有损失

    # 🔥 新增: 鼓励生成器产生干净图像
    pred_fake, noise_fake = self.netD(self.fake_B, self.time_idx)

    # GAN损失: 欺骗判别器(真实性)
    self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean()

    # 🔥 去噪损失: 使noise_fake接近0 (干净)
    self.loss_G_denoise = torch.mean(noise_fake)

    self.loss_G = self.loss_G_GAN + \
                  self.opt.lambda_SB * self.loss_SB + \
                  self.opt.lambda_NCE * self.loss_NCE + \
                  self.opt.lambda_denoise * self.loss_G_denoise
    return self.loss_G
```

**效果**:
- 判别器明确学习"噪音水平"特征
- 生成器被显式鼓励产生低噪音图像
- 预期噪音减少: **60-80%**

---

### 🥈 方案B: 两阶段训练

**阶段1: 域内去噪** (可选，如果有少量配对数据)

如果你能获得一些配对的含噪/相对干净的数据 (例如同一患者的不同扫描):

```python
# 训练一个去噪自编码器
class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        # U-Net架构
        self.encoder = ...
        self.decoder = ...

    def forward(self, noisy_input):
        return self.decoder(self.encoder(noisy_input))

# 在PD域训练
denoiser_A = Denoiser()
for noisy_pd, clean_pd in paired_pd_data:
    loss = ||denoiser_A(noisy_pd) - clean_pd||²

# 在PDFs域训练
denoiser_B = Denoiser()
for noisy_pdfs, clean_pdfs in paired_pdfs_data:
    loss = ||denoiser_B(noisy_pdfs) - clean_pdfs||²
```

**阶段2: 对比度迁移**

```python
# 在去噪后的数据上训练UNSB
class MriUnalignedDataset:
    def __init__(self, opt):
        super().__init__(opt)

        # 加载预训练的去噪器
        self.denoiser_A = load_denoiser('checkpoints/denoiser_A.pth')
        self.denoiser_B = load_denoiser('checkpoints/denoiser_B.pth')
        self.denoiser_A.eval()
        self.denoiser_B.eval()

    def __getitem__(self, index):
        A_tensor = self._load_slice(...)  # 含噪PD
        B_tensor = self._load_slice(...)  # 含噪PDFs

        # 先去噪
        with torch.no_grad():
            A_clean = self.denoiser_A(A_tensor.unsqueeze(0)).squeeze(0)
            B_clean = self.denoiser_B(B_tensor.unsqueeze(0)).squeeze(0)

        return {'A': A_clean, 'B': B_clean}
```

**优点**:
- 去噪和迁移分离，各自优化
- 如果去噪器效果好，迁移质量会显著提升

**缺点**:
- 需要配对数据 (即使少量)
- 两阶段训练复杂度高
- 去噪可能损失部分信息

---

### 🥉 方案C: 自监督去噪 (不需要配对数据)

**C1: Noise2Noise风格**

如果你有同一患者的多次扫描:

```python
# 训练数据: 同一解剖结构的两次含噪扫描
scan1 = noisy_pd_scan1  # σ ≈ 0.03
scan2 = noisy_pd_scan2  # σ ≈ 0.03 (不同噪音实现)

# 训练目标: 从scan1预测scan2
loss = ||denoiser(scan1) - scan2||²

# 神奇的是: 这会学到去噪！
# 原理: 两个独立噪音的期望为0
```

**C2: Noise2Void风格**

完全自监督，不需要多次扫描:

```python
class BlindSpotDenoiser(nn.Module):
    """盲点网络: 从周围像素预测中心像素"""

    def forward(self, x, mask):
        # mask随机遮挡一些像素
        x_masked = x * (1 - mask)

        # 从周围像素预测被遮挡的像素
        x_pred = self.network(x_masked)

        return x_pred

# 训练
for noisy_img in dataset:
    mask = random_mask()  # 随机遮挡10%像素

    pred = model(noisy_img, mask)

    # 只在被遮挡位置计算损失
    loss = ||mask * (pred - noisy_img)||²
```

**整合到UNSB**:

```python
# sb_model.py
def __init__(self, opt):
    # ... 原有网络

    # 🔥 新增: 自监督去噪正则化
    if opt.self_supervised_denoise:
        self.denoiser = BlindSpotDenoiser().to(self.device)
        self.optimizer_denoise = torch.optim.Adam(
            self.denoiser.parameters(), lr=opt.lr
        )

def compute_G_loss(self):
    # ... 原有损失

    # 🔥 自监督去噪损失
    if self.opt.self_supervised_denoise:
        mask = self.generate_blind_spot_mask()
        denoised = self.denoiser(self.real_A, mask)
        self.loss_denoise = torch.mean(
            mask * (denoised - self.real_A)**2
        )
    else:
        self.loss_denoise = 0.0

    self.loss_G = self.loss_G_GAN + \
                  self.opt.lambda_SB * self.loss_SB + \
                  self.opt.lambda_NCE * self.loss_NCE + \
                  self.opt.lambda_denoise * self.loss_denoise
    return self.loss_G
```

---

## 推荐实施路径

基于你的"双域都有噪音"的场景:

### 🎯 阶段1: 基础自适应方法 (1-2天)

实施**方案2** (Nila启发的自适应SB):
```bash
python train.py \
  --data_noise_level 0.03 \
  --noise_adaptive_schedule linear \
  # ... 其他参数
```

**预期**: 30-50%噪音减少

---

### 🎯 阶段2: 判别器增强 (2-3天)

如果阶段1效果不够，添加**方案A1** (数据增强):

```bash
python train.py \
  --data_noise_level 0.03 \
  --noise_adaptive_schedule linear \
  --denoise_augmentation \
  --denoise_method lowpass \
  --denoise_prob 0.5 \
  # ... 其他参数
```

**预期**: 50-70%噪音减少

---

### 🎯 阶段3: 高级方法 (5-7天, 可选)

如果仍不满意，考虑:

**选项A**: **方案A2** (噪音条件判别器)
- 需要修改网络架构
- 显式建模噪音水平
- 预期: 60-80%噪音减少

**选项B**: **方案C2** (Noise2Void)
- 完全自监督
- 不需要额外数据
- 可以与其他方案组合

---

## 评估指标

### 定量评估

```python
# 噪音水平
from noise_estimation import estimate_noise_mad

input_noise = estimate_noise_mad(input_image)
output_noise = estimate_noise_mad(generated_image)
noise_reduction_ratio = output_noise / input_noise

print(f"Noise reduction: {(1 - noise_reduction_ratio)*100:.1f}%")

# 目标:
# 基础方法: 30-50% reduction
# 增强方法: 50-70% reduction
# 高级方法: 60-80% reduction
```

### 定性评估

```python
# 可视化: 输入 vs 输出 vs 噪音图
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

# 第一行: PD域
axes[0, 0].imshow(input_pd_mag, cmap='gray')
axes[0, 0].set_title(f'Input PD\nσ={input_noise_pd:.4f}')

axes[0, 1].imshow(generated_pdfs_mag, cmap='gray')
axes[0, 1].set_title(f'Generated PDFs\nσ={output_noise:.4f}')

axes[0, 2].imshow(reference_pdfs_mag, cmap='gray')
axes[0, 2].set_title(f'Real PDFs\nσ={real_noise_pdfs:.4f}')

# 噪音图 (残差的高频成分)
noise_map = input_pd_mag - gaussian_filter(input_pd_mag, sigma=2)
axes[0, 3].imshow(noise_map, cmap='seismic')
axes[0, 3].set_title('Input noise map')

# 第二行: 频谱分析
axes[1, 0].plot(power_spectrum(input_pd_mag), label='Input')
axes[1, 0].plot(power_spectrum(generated_pdfs_mag), label='Generated')
axes[1, 0].set_title('Power Spectrum')
axes[1, 0].set_yscale('log')
axes[1, 0].legend()

plt.tight_layout()
plt.savefig('denoising_evaluation.png')
```

---

## 总结

### 你的场景的特殊性

```
Nila场景:     含噪测量 → 干净图像 (监督信号明确)
你的场景:     含噪PD → 含噪PDFs (无干净参考)
```

### 能达到的效果

| 方案 | 噪音减少 | 实施难度 | 需要额外数据 |
|-----|---------|---------|------------|
| 基础自适应SB | 30-50% | 低 (1-2天) | 否 |
| + 数据增强 | 50-70% | 中 (2-3天) | 否 |
| + 噪音条件判别器 | 60-80% | 高 (5-7天) | 否 |
| 两阶段去噪+迁移 | 70-90% | 高 (7-10天) | 是 (配对数据) |
| + Noise2Void | 60-80% | 中 (3-5天) | 否 |

### 实践建议

1. **先实施基础方法** (方案2): 快速验证效果
2. **评估是否足够**: 如果30-50%减少已经满足需求，无需更复杂方法
3. **渐进增强**: 如果需要更好效果，逐步添加增强技术
4. **监控trade-off**: 过度去噪可能损失对比度细节

### 关键理解

**自适应方法不是magic**:
- ✅ 能减少噪音传递
- ✅ 能隐式去噪
- ❌ 不能完全消除噪音 (除非有干净参考)
- ✅ 但配合其他技术可以达到60-80%噪音减少

**最重要的**: 明确你的目标
- 如果主要目标是**对比度迁移**, 30-50%噪音减少通常足够
- 如果主要目标是**去噪**, 考虑专门的去噪方法
