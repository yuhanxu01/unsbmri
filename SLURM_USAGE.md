# SLURM Training Guide

## Quick Start

### 一次性提交所有7个实验
```bash
bash submit_all_paired.sh
```

这会自动提交7个SLURM任务：
- Scheme A (sb_gt_transport)
- Baseline L1 (l1_loss)
- B1-B5 (nce_feature, frequency, gradient, multiscale, selfsup_contrast)

### 单独提交实验
```bash
sbatch slurm_train_paired.sh <实验名称> <策略类型>
```

例如：
```bash
sbatch slurm_train_paired.sh schemeA sb_gt_transport
sbatch slurm_train_paired.sh B2 frequency
```

## 配置说明

**预训练模型**: `/gpfs/scratch/rl5285/unsb_mri/unsbmri_2stage/checkpoints/unpaired`
- 所有7个实验都从这个模型继续训练
- 从epoch 401开始

**训练参数**:
- Epochs: 100 + 100 (decay)
- Paired data: 10%
- Batch size: 1
- GPU: 1 per job

**输出**:
- Checkpoints: `checkpoints/baseline_<实验名称>/`
- Logs: `slurm-<job_id>.out`

## 监控任务

### 查看所有任务
```bash
squeue -u $USER
```

### 查看特定任务
```bash
squeue -j <job_id>
```

### 实时查看日志
```bash
tail -f slurm-<job_id>.out
```

### 取消任务
```bash
scancel <job_id>
```

## 实验对应表

| 实验名称 | 策略类型 | 描述 |
|---------|---------|------|
| schemeA | sb_gt_transport | GT guidance in SB transport |
| L1 | l1_loss | Simple pixel L1 loss |
| B1 | nce_feature | Enhanced NCE in feature space |
| B2 | frequency | Frequency domain (FFT/k-space) |
| B3 | gradient | Gradient/structure matching |
| B4 | multiscale | Multi-scale pyramid |
| B5 | selfsup_contrast | Self-supervised contrastive |

## 训练完成后

### 测试所有模型
```bash
bash experiments/test_all.sh
```

生成结果：
- `test_results/comparison_table.csv` - 所有策略的SSIM/PSNR/NRMSE对比
- 按SSIM排序，找到最优策略

## 故障排查

### 任务失败
1. 检查日志: `cat slurm-<job_id>.out`
2. 查看错误信息
3. 确认预训练模型路径正确

### GPU不足
- 每个任务需要1个GPU
- 如果队列满了，任务会等待

### 磁盘空间
- 每个实验约需10-20GB (checkpoints + logs)
- 7个实验总共约100GB
