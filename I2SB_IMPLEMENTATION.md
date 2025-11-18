# I2SB模型实现文档

## 概述

本文档说明了基于论文 "Guided MRI Reconstruction via Schrödinger Bridge" 实现的 I2SB (Image-to-Image Schrödinger Bridge) 模型。

## 论文方法总结

### 核心思想

I²SB方法将配对MRI重建问题建模为Schrödinger Bridge，通过以下方式工作：

1. **配对数据传输**：在源图像X（如欠采样MRI）和目标图像Y（如全采样MRI）之间建立最优传输
2. **条件扩散过程**：
   - **前向过程**：从目标图像Y_0添加噪声到Y_T
   - **反向过程**：以源图像X为条件，从Y_T去噪回Y_0
3. **显式结构约束**：利用配对数据提供的像素级对应关系

### 与传统方法的区别

| 方面 | 传统Unpaired SB | I²SB (本实现) |
|------|----------------|--------------|
| 数据需求 | 无配对数据 | 配对数据 |
| 网络结构 | 需要能量网络E | 仅需生成器G（可选判别器D） |
| 条件方式 | 无条件或弱条件 | 强条件（源图像作为输入） |
| 损失函数 | 对比损失 + SB损失 | 简单扩散损失（MSE） |
| 训练难度 | 较高（需要平衡多个损失） | 较低（类似标准扩散模型） |

## 实现架构

### 文件结构

```
models/
├── i2sb_model.py          # 新的I2SB模型实现（核心）
├── sb_model.py            # 原有的unpaired SB模型
├── ncsn_networks.py       # 时间条件网络架构
└── networks.py            # 通用网络定义

train.py                   # 修改后的训练脚本（支持I2SB）
run_train_i2sb.sh         # I2SB训练脚本
slurm_train_i2sb.sh       # SLURM集群训练脚本
test_i2sb_model.py        # 模型测试脚本
```

### 核心组件

#### 1. 模型架构 (`I2SBModel`)

**输入**：
- `source`（X）：源图像（如欠采样MRI）- 作为条件
- `target`（Y）：目标图像（如全采样MRI）- 用于训练

**网络**：
- `netG`：条件U-Net生成器
  - 输入：`[Y_t, X]`（拼接噪声目标和源图像）
  - 输出：预测的噪声ε或预测的Y_0
  - 时间嵌入：通过`time_idx`编码当前时间步

- `netD`（可选）：时间条件判别器
  - 用于GAN损失，提升生成质量

#### 2. 扩散过程

**前向扩散** `q(y_t | y_0)`:
```python
y_t = sqrt(alpha_bar_t) * y_0 + sqrt(1 - alpha_bar_t) * epsilon
```

**反向去噪** `p(y_{t-1} | y_t, x)`:
```python
# 使用网络预测噪声或y_0
model_output = netG([y_t, x], t)

# DDIM采样（确定性）
y_{t-1} = sqrt(alpha_{t-1}) * pred_y_0 + sqrt(1 - alpha_{t-1}) * pred_epsilon
```

#### 3. 训练目标

根据`i2sb_objective`参数，支持三种训练目标：

**Objective 1: 预测噪声** (推荐)
```python
target = epsilon  # 真实噪声
prediction = model_output
loss = MSE(prediction, target)
```

**Objective 2: 预测x0**
```python
target = y_0  # 真实目标图像
prediction = model_output
loss = MSE(prediction, target)
```

**Objective 3: 预测v** (velocity)
```python
v = sqrt(alpha_bar_t) * epsilon - sqrt(1-alpha_bar_t) * y_0
target = v
prediction = model_output
loss = MSE(prediction, target)
```

#### 4. 损失函数

**主损失**（diffusion loss）：
```python
loss_diffusion = MSE(prediction, target)
```

**可选损失**：
1. L1损失：`L1(pred_y_0, y_0)`
2. 感知损失：`LPIPS(pred_y_0, y_0)`（需要安装lpips）
3. GAN损失：`GAN(D(pred_y_0), True)`

**总损失**：
```python
loss_G = lambda_diffusion * loss_diffusion
         + lambda_l1 * loss_l1
         + lambda_perceptual * loss_perceptual
         + lambda_gan * loss_gan
```

## 使用方法

### 1. 准备数据

数据应组织为配对的HDF5文件：

```
dataroot/
├── trainA/
│   ├── case_001.h5    # 源图像（如欠采样）
│   ├── case_002.h5
│   └── ...
├── trainB/
│   ├── case_001.h5    # 目标图像（如全采样）
│   ├── case_002.h5
│   └── ...
├── valA/
└── valB/
```

每个HDF5文件包含多个切片：
```python
with h5py.File('case_001.h5', 'r') as f:
    # 键: slices_0, slices_1, slices_2, ...
    # 值: [H, W, 2] 张量 (实部/虚部) 或 [H, W, C]
```

### 2. 配置训练参数

编辑 `run_train_i2sb.sh`：

```bash
# 数据路径
DATAROOT="/path/to/your/data"

# 扩散参数
I2SB_NUM_TIMESTEPS=1000         # 训练时间步数
I2SB_BETA_SCHEDULE="linear"     # 噪声调度：linear/cosine/quadratic
I2SB_OBJECTIVE="pred_noise"     # 训练目标：pred_noise/pred_x0/pred_v

# 采样参数
I2SB_SAMPLING_TIMESTEPS=250     # 推理时间步（可少于训练步数）
I2SB_DDIM_ETA=0.0              # DDIM参数：0=确定性，1=随机

# 损失权重
LAMBDA_DIFFUSION=1.0
LAMBDA_L1=0.1
LAMBDA_PERCEPTUAL=0.0           # 设为>0启用LPIPS（需安装）
USE_GAN="--use_gan"             # 使用GAN损失
LAMBDA_GAN=0.1
```

### 3. 开始训练

**本地训练**：
```bash
bash run_train_i2sb.sh
```

**集群训练**（SLURM）：
```bash
# 1. 编辑slurm_train_i2sb.sh，更新数据路径
# 2. 提交任务
sbatch slurm_train_i2sb.sh
```

### 4. 监控训练

训练过程会自动记录到Weights & Biases：

```python
# 损失
- loss_diffusion    # 主扩散损失
- loss_l1          # L1损失
- loss_perceptual  # 感知损失
- loss_G_GAN       # 生成器GAN损失
- loss_D_real      # 判别器真实损失
- loss_D_fake      # 判别器虚假损失

# 指标（如果启用compute_paired_metrics）
- metric_SSIM      # 结构相似度
- metric_PSNR      # 峰值信噪比
- metric_NRMSE     # 归一化均方根误差
```

### 5. 推理/测试

```python
from models import create_model
from options.test_options import TestOptions

# 加载模型
opt = TestOptions().parse()
opt.model = 'i2sb'
model = create_model(opt)
model.setup(opt)

# 采样
with torch.no_grad():
    generated = model.sample(
        source=source_image,
        num_steps=250,      # 采样步数
        eta=0.0            # DDIM eta参数
    )
```

## 关键参数说明

### 扩散参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `i2sb_num_timesteps` | 1000 | 训练时的扩散步数，越大越慢但可能更准确 |
| `i2sb_beta_schedule` | linear | 噪声调度策略：<br>• linear: 线性增长<br>• cosine: 余弦调度<br>• quadratic: 二次增长 |
| `i2sb_beta_start` | 0.0001 | 起始β值 |
| `i2sb_beta_end` | 0.02 | 终止β值 |
| `i2sb_objective` | pred_noise | 训练目标：<br>• pred_noise: 预测噪声（推荐）<br>• pred_x0: 预测目标图像<br>• pred_v: 预测速度场 |

### 采样参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `i2sb_sampling_timesteps` | 250 | 推理时的采样步数，可小于训练步数以加速 |
| `i2sb_ddim_sampling_eta` | 0.0 | DDIM随机性参数：<br>• 0: 完全确定性（推荐）<br>• 1: 等同于DDPM（随机） |

### 条件方式

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `condition_method` | concat | 如何将源图像作为条件：<br>• concat: 通道拼接（当前实现）<br>• cross_attention: 交叉注意力（待实现）<br>• film: FiLM调制（待实现） |

### 损失权重

| 参数 | 默认值 | 推荐范围 | 说明 |
|------|-------|---------|------|
| `lambda_diffusion` | 1.0 | [0.5, 2.0] | 主扩散损失权重 |
| `lambda_simple` | 1.0 | 固定为1.0 | 简单MSE损失权重 |
| `lambda_l1` | 0.1 | [0, 1.0] | L1正则化权重 |
| `lambda_perceptual` | 0.0 | [0, 0.1] | 感知损失权重（需安装LPIPS） |
| `lambda_gan` | 0.1 | [0.01, 0.5] | GAN损失权重（需启用use_gan） |

## 与原SB模型的对比

### 原SB模型（unpaired）
```bash
python train.py \
  --model sb \
  --netG resnet_9blocks_cond \
  --lambda_SB 0.1 \
  --lambda_NCE 1.0 \
  --lambda_GAN 1.0
```

需要的组件：
- netG（生成器）
- netE（能量网络）
- netD（判别器）
- netF（特征提取）
- 两个数据集（dataset和dataset2）

### 新I2SB模型（paired）
```bash
python train.py \
  --model i2sb \
  --netG resnet_9blocks_cond \
  --lambda_diffusion 1.0 \
  --lambda_l1 0.1
```

需要的组件：
- netG（生成器）
- netD（判别器，可选）
- 一个数据集（配对）

**简化点**：
1. ✓ 移除了能量网络netE
2. ✓ 移除了特征网络netF
3. ✓ 移除了NCE损失
4. ✓ 移除了复杂的SB对比损失
5. ✓ 简化为标准扩散模型训练

## 常见问题

### Q1: 什么时候使用I2SB而不是原SB？

**使用I2SB当**：
- ✓ 你有配对的训练数据
- ✓ 想要更简单、更稳定的训练
- ✓ 需要更快的收敛速度
- ✓ 关注重建质量而非域适应

**使用原SB当**：
- ✓ 只有非配对数据
- ✓ 需要域适应能力
- ✓ 处理unpaired image translation任务

### Q2: 如何选择训练目标（objective）？

推荐顺序：
1. **pred_noise**（默认）：最稳定，与大多数扩散模型论文一致
2. **pred_x0**：可能训练更快，但可能不稳定
3. **pred_v**：理论上更优，但实践中差异不大

### Q3: 如何调整扩散步数？

- **训练步数**（i2sb_num_timesteps）：
  - 开始：1000（标准）
  - 如果GPU内存不足：500-800
  - 如果想要更高质量：2000-4000

- **采样步数**（i2sb_sampling_timesteps）：
  - 快速测试：50-100
  - 常规推理：250
  - 高质量：500-1000

### Q4: 需要使用GAN损失吗？

- **不使用GAN**（更简单）：
  ```bash
  USE_GAN=""
  ```
  优点：训练更稳定，收敛更快
  缺点：可能生成图像稍微模糊

- **使用GAN**（更精细）：
  ```bash
  USE_GAN="--use_gan"
  LAMBDA_GAN=0.1
  ```
  优点：生成图像更清晰，细节更好
  缺点：训练需要更小心调参

### Q5: 如何处理不同的MRI数据格式？

**幅度图像**：
```bash
MRI_REPRESENTATION="magnitude"
INPUT_NC=1
OUTPUT_NC=1
```

**复数图像（实部+虚部）**：
```bash
MRI_REPRESENTATION="real_imag"
INPUT_NC=2
OUTPUT_NC=2
```

### Q6: 训练需要多久？

取决于：
- 数据集大小
- GPU性能
- 扩散步数
- 是否使用GAN

**典型时间**（单GPU V100）：
- 小数据集（<1000对）：2-4小时
- 中等数据集（1000-5000对）：8-16小时
- 大数据集（>5000对）：24-48小时

## 测试模型

运行测试脚本验证实现：

```bash
python test_i2sb_model.py
```

这将测试：
1. ✓ 扩散参数设置
2. ✓ 模型创建
3. ✓ 前向传播
4. ✓ 损失计算
5. ✓ 优化步骤
6. ✓ 采样推理

## 技术细节

### DDIM采样

使用DDIM（Denoising Diffusion Implicit Models）进行快速采样：

```python
# 确定性采样（eta=0，推荐）
y_{t-1} = sqrt(alpha_{t-1}) * pred_y_0
          + sqrt(1 - alpha_{t-1}) * pred_epsilon

# 随机采样（eta=1，等同于DDPM）
y_{t-1} = sqrt(alpha_{t-1}) * pred_y_0
          + sqrt(1 - alpha_{t-1} - sigma_t^2) * pred_epsilon
          + sigma_t * noise
```

### 时间嵌入

网络通过正弦位置编码接收时间信息：

```python
def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = log(10000) / (half_dim - 1)
    emb = exp(arange(half_dim) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = concat([sin(emb), cos(emb)], dim=1)
    return emb
```

### 噪声调度

**Linear schedule**：
```python
betas = linspace(beta_start, beta_end, T)
```

**Cosine schedule** (推荐用于高分辨率)：
```python
alphas_cumprod = cos^2((t/T + s)/(1+s) * π/2)
betas = 1 - alphas_cumprod[t] / alphas_cumprod[t-1]
```

## 参考文献

1. **I²SB原论文**：
   - "Guided MRI Reconstruction via Schrödinger Bridge"
   - arXiv:2411.14269

2. **相关工作**：
   - DDPM: "Denoising Diffusion Probabilistic Models"
   - DDIM: "Denoising Diffusion Implicit Models"
   - Schrödinger Bridge: "Schrödinger Bridge Matching"

## 开发者信息

- **实现日期**：2025-11
- **基于代码库**：unsbmri (Unpaired SB MRI)
- **主要修改**：
  - 新增 `models/i2sb_model.py`
  - 修改 `train.py` 支持单数据集
  - 新增训练脚本和测试脚本

## 许可证

遵循原项目许可证。
