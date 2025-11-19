# HPC集群代码更新指南

## 问题诊断

如果你看到以下错误之一，说明HPC集群上的代码**没有更新到最新版本**：

1. `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`
2. `AttributeError: 'SBModel' object has no attribute 'optimizer_F'`
3. 实验立即完成但没有训练任何epoch

## 快速修复步骤

### 步骤1: SSH登录到HPC集群

```bash
ssh rl5285@greene.hpc.nyu.edu
# 或你的HPC登录命令
```

### 步骤2: 进入项目目录

```bash
cd /gpfs/scratch/rl5285/test/unsbmri
```

### 步骤3: 强制更新到最新版本

```bash
# 获取最新代码
git fetch origin

# 强制重置到最新版本（这会丢弃所有本地修改）
git reset --hard origin/claude/setup-mri-training-pipeline-01SPqpGQe22LVbdgKBHDkPF1

# 验证更新成功
git log -1
# 应该显示: "Fix optimizer_F AttributeError when NCE is disabled"
```

### 步骤4: 验证所有修复已应用

```bash
# 运行验证脚本
bash verify_fixes.sh
```

**期望输出**:
```
========================================
✓ ALL CHECKS PASSED
Code is up to date with all fixes applied!
You can now run the experiments.
========================================
```

如果看到 `❌ SOME CHECKS FAILED`，请按照脚本输出的指示操作。

### 步骤5: 重新运行实验

```bash
# 运行单个实验
sbatch experiments/ablation_studies/exp1_fully_pair_OT_input.sh

# 或运行所有实验
bash experiments/ablation_studies/launch_all_ablation.sh
```

## 为什么需要使用 git reset --hard？

- `git pull` 可能会因为本地修改而失败
- `git reset --hard` 确保代码与远程仓库**完全一致**
- 这是最可靠的更新方法

## 已修复的Bug列表

本次更新修复了3个关键bug：

1. **梯度追踪错误**: 所有标量损失现在都是torch.tensor
2. **优化器错误**: 禁用NCE/GAN/Entropy时不会创建对应的优化器
3. **Epoch配置错误**: 两阶段实验现在正确训练200个epoch (401-600)

## 需要帮助？

如果更新后仍有问题，请提供：
1. `verify_fixes.sh` 的完整输出
2. 实验的完整错误日志
3. `git log -3` 的输出
