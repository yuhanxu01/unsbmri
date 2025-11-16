# MRI Paired Training Experiments

Modular framework for testing strategies to utilize paired data in Schrödinger Bridge training.

**All paired experiments use 10% data** for controlled comparison.

## Available Strategies

| ID | Strategy | Description | Loss Formula |
|----|----------|-------------|--------------|
| **A** | `sb_gt_transport` | GT guidance in SB transport | `τ·‖fake_B - real_B‖²` added to SB |
| **Baseline** | `l1_loss` | Simple pixel L1 | `λ_L1·‖fake_B - real_B‖₁` |
| **B1** | `nce_feature` | Enhanced NCE in feature space | `NCE(feat(fake_B), feat(real_B))` |
| **B2** | `frequency` | Frequency domain (FFT) | `‖FFT(fake_B) - FFT(real_B)‖` |
| **B3** | `gradient` | Gradient/structure | `‖∇fake_B - ∇real_B‖` |
| **B4** | `multiscale` | Multi-scale pyramid | `Σ w_i·‖pyramid_i(fake_B) - pyramid_i(real_B)‖` |
| **B5** | `selfsup_contrast` | Self-supervised contrastive | `1 - cosine_sim(feat(fake_B), feat(real_B))` |

## Experiment Setup (8 Total)

### Workflow
```
1. Unpaired baseline (200+200 epochs)  ← Already trained
   └─ checkpoints/unpaired/

2. 7 Paired strategies (100+100 epochs each, 10% data)
   ├─ Scheme A: sb_gt_transport
   ├─ Baseline L1: l1_loss
   ├─ B1: nce_feature
   ├─ B2: frequency
   ├─ B3: gradient
   ├─ B4: multiscale
   └─ B5: selfsup_contrast
```

### Submit All 7 Jobs (Recommended)
```bash
# Submit all 7 paired training jobs to SLURM
bash submit_all_paired.sh
```

### Submit Individual Jobs
```bash
# Submit one at a time
sbatch slurm_train_paired.sh schemeA sb_gt_transport
sbatch slurm_train_paired.sh L1 l1_loss
sbatch slurm_train_paired.sh B1 nce_feature
sbatch slurm_train_paired.sh B2 frequency
sbatch slurm_train_paired.sh B3 gradient
sbatch slurm_train_paired.sh B4 multiscale
sbatch slurm_train_paired.sh B5 selfsup_contrast
```

### Monitor Jobs
```bash
# Check job status
squeue -u $USER

# View live logs
tail -f slurm-<job_id>.out
```

### Testing & Evaluation
```bash
# Test all models and generate comparison
bash experiments/test_all.sh

# Results: test_results/comparison_table.csv
```

**Output**:
- Strategy ranking by SSIM/PSNR/NRMSE
- Best strategy identification
- Complete comparison table

## Single Experiment

```bash
export EXPERIMENT_NAME="my_test"
export PAIRED_STRATEGY="frequency"  # Choose strategy
export PAIRED_SUBSET_RATIO=0.1      # 10% paired data
export PAIRED_STAGE="--paired_stage"
bash run_train.sh
```

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `paired_strategy` | see table | Strategy selection |
| `paired_subset_ratio` | `0.1` | Fixed 10% paired data |
| `lambda_L1` | `1.0` | L1 loss weight |
| `lambda_reg` | `1.0` | B1-B5 regularization weight |

## WandB Monitoring

**Losses**:
- `loss/SB`, `loss/SB_guidance` (Scheme A)
- `loss/L1` (Baseline)
- `loss/NCE_paired` (B1), `loss/freq` (B2), `loss/gradient` (B3), `loss/multiscale` (B4), `loss/contrast` (B5)

**Metrics**:
- `loss/metric_SSIM`, `loss/metric_PSNR`, `loss/metric_NRMSE`

**Visuals**: Input | Output | Ground Truth

## File Structure

```
experiments/
├── launch_all_8gpu.sh       # Main launcher (8 GPUs, 10% data)
├── slurm_launch_all.sh      # SLURM wrapper
├── test_all.sh              # Batch testing & ranking
├── scheme_a_twostage.sh     # Individual: Scheme A
└── baseline_l1.sh           # Individual: Baseline L1
```

## Design Principles

1. **Controlled comparison**: All use 10% paired data, shared unpaired baseline
2. **Mathematical rigor**: Each strategy complements SB framework
3. **MRI-appropriate**: No VGG/ImageNet, B2 uses k-space physics
4. **Efficient**: 8 experiments find optimal strategy

---

**Implementation**: `models/sb_model.py` (B1-B5 methods)
**Configuration**: `options/train_options.py`
