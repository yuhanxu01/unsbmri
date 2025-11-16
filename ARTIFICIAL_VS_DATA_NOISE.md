# 人工噪音 vs 数据噪音: 详细解释

## 问题: 什么是人工噪音？

在Schrödinger Bridge (SB)和扩散模型中，有两种完全不同的噪音：

---

## 1. 人工噪音 (Artificial Noise) - σ_t

### 定义
**训练/推理过程中人为添加的随机噪音**，用于构建从源域到目标域的随机桥接。

### 在你的代码中
```python
# sb_model.py, line 192
Xt = (1-inter) * Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * torch.randn_like(Xt)
#                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#                                               人工噪音: N(0, τ·scale·I)
```

### 数学形式
```
X_t = (1-α_t)·X_{t-1} + α_t·G(X_{t-1}) + √(τ·Δ_t·(1-Δ_t))·ε

其中:
- ε ~ N(0, I): 标准高斯噪音
- τ: 温度参数 (opt.tau, 通常0.1)
- Δ_t = t/T: 归一化时间
- σ_t = √(τ·Δ_t·(1-Δ_t)): 人工噪音的标准差
```

### 时间演化
```python
import numpy as np
import matplotlib.pyplot as plt

tau = 0.1
T = 20
t_range = np.arange(T+1)
delta_t = t_range / T

# 人工噪音的标准差
sigma_t = np.sqrt(tau * delta_t * (1 - delta_t))

plt.figure(figsize=(10, 5))
plt.plot(t_range, sigma_t, 'b-', linewidth=2, marker='o')
plt.xlabel('Time step t')
plt.ylabel('Artificial noise σ_t')
plt.title('Artificial Noise Schedule (τ=0.1, T=20)')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)

# 标注关键点
max_idx = np.argmax(sigma_t)
plt.annotate(f'Max: σ_{{{max_idx}}} = {sigma_t[max_idx]:.3f}',
            xy=(max_idx, sigma_t[max_idx]),
            xytext=(max_idx+2, sigma_t[max_idx]+0.02),
            arrowprops=dict(arrowstyle='->', color='red'))

plt.savefig('artificial_noise_schedule.png', dpi=150)
print("Saved to artificial_noise_schedule.png")
```

**输出**:
```
t=0:   σ_0  = 0.000  (起点，干净的PD图像)
t=5:   σ_5  = 0.122
t=10:  σ_10 = 0.158  (最大噪音，50%时刻)
t=15:  σ_15 = 0.122
t=20:  σ_20 = 0.000  (终点，目标PDFs图像)
```

### 特点
- ✅ **可控**: 由超参数τ控制
- ✅ **动态**: 随时间步变化（倒U型曲线）
- ✅ **有目的**: 实现随机插值，探索路径空间
- ✅ **可逆**: 训练时前向添加，推理时后向去除

### 作用
1. **正则化**: 避免模式崩塌
2. **探索**: 增加生成多样性
3. **桥接**: 构建从A到B的平滑过渡

---

## 2. 数据噪音 (Data/Measurement Noise) - σ_y

### 定义
**MRI采集过程中固有的噪音**，已经存在于你的原始数据文件（.h5）中。

### 物理来源
```
真实解剖结构: x_true (理想的、无噪音的MRI图像)
         ↓
    MRI采集过程
         ↓
观测数据: y = x_true + η
            ^^^^^^^^^^^
            你的.h5文件中的数据

其中: η ~ N(0, σ_y²I) 或 Rician分布
```

**具体来源**:
| 噪音类型 | 来源 | 相对贡献 |
|---------|------|---------|
| 热噪音 | MRI线圈的热运动 (约37°C) | 主要 |
| 量子噪音 | 质子自旋的随机涨落 | 次要 |
| 电子噪音 | 放大器、ADC转换 | 中等 |
| 运动伪影 | 病人呼吸、心跳 | 可变 |
| k空间欠采样 | 快速成像的代价 | 主要（加速扫描）|

### 在你的数据中
```python
# 当你加载.h5文件时
import h5py
with h5py.File('datasets/trainA/PD_case001.h5', 'r') as f:
    data = f['slices_10'][...]  # Shape: [H, W, 2]
    real = data[..., 0]
    imag = data[..., 1]

    # 这个magnitude已经包含噪音了！
    magnitude = np.sqrt(real**2 + imag**2)

    # 真实情况:
    # magnitude = sqrt((x_true + η_real)² + (y_true + η_imag)²)
    #           ≈ sqrt(x_true² + y_true²) + noise_floor
```

### 估计方法

**方法1: MAD (Median Absolute Deviation)**
```python
from noise_estimation import estimate_noise_mad

# 假设背景区域只有噪音
background = magnitude[magnitude < np.percentile(magnitude, 20)]
σ_y = 1.4826 * median(|background - median(background)|)
```

**方法2: 背景区域标准差**
```python
# 使用图像四角（假设是背景）
corners = [magnitude[:20, :20],      # 左上
           magnitude[:20, -20:],     # 右上
           magnitude[-20:, :20],     # 左下
           magnitude[-20:, -20:]]    # 右下
σ_y = median([std(corner) for corner in corners])
```

**方法3: Rician分布拟合**
```python
# MRI magnitude遵循Rician分布
# 在背景区域 (信号≈0): E[R] = σ_y·sqrt(π/2)
mean_bg = mean(background)
σ_y = mean_bg / sqrt(π/2) ≈ mean_bg / 1.253
```

### 典型值范围

| 场景 | σ_y (归一化后) | SNR |
|-----|---------------|-----|
| 高场强(3T), 慢扫描 | 0.01 - 0.02 | >50 dB |
| 标准临床(1.5T) | 0.03 - 0.05 | 30-40 dB |
| 低场强(0.3T), 快速扫描 | 0.05 - 0.10 | 20-30 dB |
| 极端加速(R=8) | 0.10 - 0.20 | <20 dB |

### 特点
- ❌ **不可控**: 在数据采集时就已经固定
- ❌ **恒定**: 对每个数据集是常数
- ❌ **有害**: 降低图像质量和诊断价值
- ✅ **可估计**: 使用统计方法

---

## 3. Nila的关键洞察: 两种噪音的竞争

### 问题场景

在SB/Diffusion的**后期步骤**(小t)，会出现问题：

```
时间步 t=2 (接近终点):
  人工噪音: σ_2  = sqrt(0.1 * 2/20 * 18/20) = 0.095
  数据噪音: σ_y  = 0.03 (从数据估计)

  比率: σ_2 / σ_y = 0.095 / 0.03 = 3.17 ✅ OK

时间步 t=1 (非常接近终点):
  人工噪音: σ_1  = sqrt(0.1 * 1/20 * 19/20) = 0.069
  数据噪音: σ_y  = 0.03

  比率: σ_1 / σ_y = 0.069 / 0.03 = 2.30 ✅ OK

时间步 t=0.5 (假设允许小数步):
  人工噪音: σ_0.5 = sqrt(0.1 * 0.5/20 * 19.5/20) = 0.049
  数据噪音: σ_y  = 0.03

  比率: σ_0.5 / σ_y = 0.049 / 0.03 = 1.63 ⚠️ 接近临界

时间步 t=0.2:
  人工噪音: σ_0.2 = sqrt(0.1 * 0.2/20 * 19.8/20) = 0.031
  数据噪音: σ_y  = 0.03

  比率: σ_0.2 / σ_y = 0.031 / 0.03 = 1.03 ⚠️ 几乎相等！

时间步 t=0.1:
  人工噪音: σ_0.1 = sqrt(0.1 * 0.1/20 * 19.9/20) = 0.022
  数据噪音: σ_y  = 0.03

  比率: σ_0.1 / σ_y = 0.022 / 0.03 = 0.73 ❌ 数据噪音占主导！
```

### 为什么这是问题？

当 σ_t < σ_y 时，**数据噪音占主导**：

```python
# 标准SB损失 (你的代码, line 317)
loss_reconstruction = tau * ||X_t - G(X_t)||²

# 实际上在拟合:
# X_t ≈ x_true + η_artificial + η_data
#                ^^^^^^^^^^^^   ^^^^^^^^
#                人工噪音        数据噪音
#                (正常，训练需要) (有害，应该忽略！)

# 当 σ_artificial < σ_data:
# 模型会努力拟合 η_data，因为它比 η_artificial 更显著！
# 结果: 学习了噪音模式 → 生成结果也有噪音
```

### Nila的解决方案

**自适应权重**:
```python
if σ_t < σ_y:
    # 数据噪音占主导，减少重建损失的权重
    λ_t = σ_t / σ_y  # 线性衰减
else:
    # 人工噪音占主导，正常训练
    λ_t = 1.0

# 修改损失
loss_reconstruction = λ_t * tau * ||X_t - G(X_t)||²
```

**效果**:
```
t=10: λ_10 = 1.0    → 全强度重建损失
t=5:  λ_5  = 1.0    → 全强度
t=2:  λ_2  = 1.0    → 全强度
t=1:  λ_1  = 1.0    → 全强度
t=0.5:λ_0.5= 1.0    → 全强度
t=0.2:λ_0.2= 1.03   → 几乎全强度
t=0.1:λ_0.1= 0.73   → 减弱到73% ✅
t=0.05:λ_0.05=0.52  → 减弱到52% ✅
t=0.01:λ_0.01=0.15  → 大幅减弱 ✅
```

---

## 4. 可视化对比

### 代码示例
```python
import numpy as np
import matplotlib.pyplot as plt

# 参数
tau = 0.1
T = 20
sigma_y = 0.03  # 数据噪音水平

# 时间范围
t_range = np.linspace(0, T, 200)
delta_t = t_range / T

# 人工噪音
sigma_artificial = np.sqrt(tau * delta_t * (1 - delta_t))

# 自适应权重
ratio = sigma_artificial / sigma_y
lambda_t = np.minimum(ratio, 1.0)  # 上限为1.0

# 绘图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# 上图: 噪音水平对比
ax1.plot(t_range, sigma_artificial, 'b-', linewidth=2, label='Artificial noise σ_t')
ax1.axhline(y=sigma_y, color='r', linestyle='--', linewidth=2, label=f'Data noise σ_y = {sigma_y}')
ax1.fill_between(t_range, 0, sigma_y, alpha=0.2, color='red', label='Data noise dominates')
ax1.set_xlabel('Time step t')
ax1.set_ylabel('Noise level')
ax1.set_title('Artificial Noise vs Data Noise')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 标注交叉点
crossover_idx = np.where(np.diff(sigma_artificial < sigma_y))[0]
if len(crossover_idx) > 0:
    for idx in crossover_idx:
        ax1.axvline(x=t_range[idx], color='orange', linestyle=':', alpha=0.7)
        ax1.annotate(f't={t_range[idx]:.1f}',
                    xy=(t_range[idx], sigma_y),
                    xytext=(t_range[idx]+1, sigma_y+0.01))

# 下图: 自适应权重
ax2.plot(t_range, lambda_t, 'g-', linewidth=2)
ax2.fill_between(t_range, 0, lambda_t, alpha=0.3, color='green')
ax2.set_xlabel('Time step t')
ax2.set_ylabel('Adaptive weight λ_t')
ax2.set_title('Nila-style Adaptive Weight (reduces overfitting to noisy data)')
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1.1])
ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.3)

# 标注关键区域
decay_start = t_range[np.where(lambda_t < 0.99)[0][0]] if any(lambda_t < 0.99) else T
ax2.axvline(x=decay_start, color='orange', linestyle=':', alpha=0.7)
ax2.annotate(f'Weight decay starts\nat t≈{decay_start:.1f}',
            xy=(decay_start, 0.5),
            xytext=(decay_start+2, 0.3),
            arrowprops=dict(arrowstyle='->', color='orange'))

plt.tight_layout()
plt.savefig('noise_comparison_detailed.png', dpi=150)
print("Saved to noise_comparison_detailed.png")
plt.show()
```

---

## 5. 实际例子: 你的PD/PDFs数据

### 场景

假设你的数据分析结果为:
```json
{
  "A": {"median_noise": 0.0287},  // PD
  "B": {"median_noise": 0.0312}   // PDFs
}

建议: data_noise_level = 0.03 (取两者的最大值)
```

### 训练过程中的噪音演化

**在epoch 1, iteration 100**:
```
随机抽取时间步: t = 8 (共T=20步)

1. 人工噪音水平:
   δ_8 = 8/20 = 0.4
   σ_8 = sqrt(0.1 * 0.4 * 0.6) = 0.155

2. 数据噪音水平:
   σ_y = 0.03 (固定)

3. 噪音比率:
   ratio = 0.155 / 0.03 = 5.17

4. 自适应权重:
   λ_8 = min(5.17, 1.0) = 1.0 ✅

5. SB损失:
   L_SB = -能量项 + 1.0 * τ * ||X_8 - G(X_8)||²
          ^^^^^^^^^^  ^^^
          正常        全强度重建损失
```

**在同一个epoch, iteration 101**:
```
随机抽取时间步: t = 1 (接近终点)

1. 人工噪音水平:
   δ_1 = 1/20 = 0.05
   σ_1 = sqrt(0.1 * 0.05 * 0.95) = 0.069

2. 数据噪音水平:
   σ_y = 0.03 (固定)

3. 噪音比率:
   ratio = 0.069 / 0.03 = 2.30

4. 自适应权重:
   λ_1 = min(2.30, 1.0) = 1.0 ✅

5. SB损失:
   L_SB = -能量项 + 1.0 * τ * ||X_1 - G(X_1)||²
          仍然是全强度
```

**假设允许更细粒度的时间步, t = 0.5**:
```
1. 人工噪音:
   σ_0.5 = sqrt(0.1 * 0.025 * 0.975) = 0.049

2. 噪音比率:
   ratio = 0.049 / 0.03 = 1.63

3. 自适应权重:
   λ_0.5 = 1.0 ✅ (仍然>1)

4. SB损失: 全强度
```

**假设 t = 0.2** (非常接近终点):
```
1. 人工噪音:
   σ_0.2 = sqrt(0.1 * 0.01 * 0.99) = 0.0315

2. 噪音比率:
   ratio = 0.0315 / 0.03 = 1.05

3. 自适应权重:
   λ_0.2 = 1.0 ✅ (刚好>1)

4. SB损失: 全强度
```

**关键时刻: t = 0.1**:
```
1. 人工噪音:
   σ_0.1 = sqrt(0.1 * 0.005 * 0.995) = 0.0223

2. 噪音比率:
   ratio = 0.0223 / 0.03 = 0.74 ❌ <1

3. 自适应权重:
   λ_0.1 = 0.74 ⚠️ 减弱到74%！

4. SB损失:
   L_SB = -能量项 + 0.74 * τ * ||X_0.1 - G(X_0.1)||²
                    ^^^^
                    减弱了26%，避免过拟合数据噪音
```

---

## 6. 总结

| 特性 | 人工噪音 σ_t | 数据噪音 σ_y |
|-----|-------------|-------------|
| **来源** | 训练算法添加 | MRI采集固有 |
| **可控性** | 可控 (τ参数) | 不可控 |
| **时间依赖** | 动态变化 (倒U型) | 恒定 |
| **作用** | 构建随机桥接 | 有害，降低质量 |
| **处理方式** | 正常训练目标 | 需要减轻影响 |
| **典型值** | 0.0 - 0.158 (τ=0.1) | 0.01 - 0.10 |

### 关键公式
```python
# 人工噪音标准差
σ_t = sqrt(τ * t/T * (1 - t/T))

# Nila自适应权重
λ_t = min(σ_t / σ_y, 1.0)

# 修改后的SB重建损失
L_recon = λ_t * τ * ||X_t - G(X_t)||²
```

### 直观理解

想象你在学习画画:
- **人工噪音**: 老师故意让你练习在有网格线的纸上画画，然后逐渐去掉网格（教学工具）
- **数据噪音**: 纸张本身有污渍和折痕（质量缺陷）

**问题**: 如果网格线（人工噪音）太淡，你可能会开始临摹纸张的污渍（数据噪音）！

**Nila的解决**: 当网格线变淡时，老师告诉你"不要太在意纸上画得完美"（减少重建损失权重）

---

## 7. 实践建议

### 检查你的数据噪音水平
```bash
python -c "
from noise_estimation import analyze_dataset_noise
stats = analyze_dataset_noise('./datasets/YOUR_DATA', 'A', num_samples=20)
print(f'Data noise level: {stats[\"median_noise\"]:.4f}')
print(f'Use --data_noise_level {stats[\"median_noise\"]:.4f}')
"
```

### 可视化你的噪音调度
```bash
python -c "
from noise_estimation import visualize_adaptive_schedule
visualize_adaptive_schedule(
    T=20,                    # 你的num_timesteps
    tau=0.1,                 # 你的tau参数
    data_noise_levels=[0.01, 0.03, 0.05]  # 不同假设
)
"
```

### 调试技巧
在训练时打印噪音比率:
```python
# 在sb_model.py的compute_G_loss中添加
if self.total_iters % 100 == 0:
    print(f"[Iter {self.total_iters}] t={self.time_idx[0]}, "
          f"σ_t={artificial_noise:.4f}, σ_y={self.opt.data_noise_level:.4f}, "
          f"ratio={noise_ratio:.2f}, λ_t={noise_adaptive_weight:.2f}")
```

**预期输出**:
```
[Iter 100] t=8,  σ_t=0.1549, σ_y=0.0300, ratio=5.16, λ_t=1.00
[Iter 200] t=15, σ_t=0.1225, σ_y=0.0300, ratio=4.08, λ_t=1.00
[Iter 300] t=3,  σ_t=0.1161, σ_y=0.0300, ratio=3.87, λ_t=1.00
[Iter 400] t=1,  σ_t=0.0688, σ_y=0.0300, ratio=2.29, λ_t=1.00
[Iter 500] t=12, σ_t=0.1549, σ_y=0.0300, ratio=5.16, λ_t=1.00
```

---

希望这个文档彻底解释清楚了人工噪音和数据噪音的区别！
