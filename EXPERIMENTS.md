# MRI Paired Training Experiments

Modular framework for testing strategies to utilize paired data in Schrödinger Bridge training.

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

## Batch Experiments (18 GPUs)

### Quick Launch
```bash
# Train all experiments in parallel
sbatch experiments/slurm_launch_all.sh

# Or without SLURM (requires 18 GPUs)
bash experiments/launch_all_18gpu.sh
```

### Experiment Matrix
```
Shared unpaired baseline (1x)
├─ Scheme A: 30%, 50%, 100% (3x)
├─ Baseline L1: 30%, 50%, 100% (3x)
├─ B1 (NCE): 30%, 50%, 100% (3x)
├─ B2 (Freq): 30%, 100% (2x)
├─ B3 (Grad): 30%, 50%, 100% (3x)
├─ B4 (Multi): 30%, 100% (2x)
└─ B5 (Contrast): 30%, 100% (2x)
Total: 18 experiments
```

### Testing & Evaluation
```bash
# Test all models and generate comparison table
bash experiments/test_all.sh

# Results will be in test_results/comparison_table.csv
```

## Single Experiment

```bash
export EXPERIMENT_NAME="my_test"
export PAIRED_STRATEGY="frequency"  # Choose: sb_gt_transport, l1_loss, nce_feature, frequency, gradient, multiscale, selfsup_contrast
export PAIRED_SUBSET_RATIO=0.3
export PAIRED_STAGE="--paired_stage"
bash run_train.sh
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `paired_strategy` | `none` | Strategy selection (see table above) |
| `paired_subset_ratio` | `1.0` | Fraction of paired data (0.0-1.0) |
| `lambda_L1` | `1.0` | L1 loss weight |
| `lambda_reg` | `1.0` | B1-B5 regularization weight |
| `lambda_SB` | `1.0` | SB loss weight |
| `lambda_NCE` | `1.0` | NCE loss weight |

## WandB Monitoring

**Losses**:
- `loss/SB`, `loss/SB_guidance` (Scheme A)
- `loss/L1` (Baseline)
- `loss/NCE_paired` (B1), `loss/freq` (B2), `loss/gradient` (B3), `loss/multiscale` (B4), `loss/contrast` (B5)

**Metrics** (paired only):
- `loss/metric_SSIM`, `loss/metric_PSNR`, `loss/metric_NRMSE`

**Visuals**: Input | Output | Ground Truth

## File Structure

```
experiments/
├── launch_all_18gpu.sh       # Parallel launcher (18 GPUs)
├── slurm_launch_all.sh       # SLURM wrapper
├── test_all.sh               # Batch testing & comparison
├── scheme_a_twostage.sh      # Individual: Scheme A
└── baseline_l1.sh            # Individual: Baseline L1
```

## Design Principles

1. **Mathematical rigor**: Each strategy preserves or complements SB framework
2. **Modularity**: Easy to add new strategies
3. **Controlled comparison**: Shared baseline, consistent configuration
4. **Scalability**: Batch processing with parallel execution

---

**Implementation**: `models/sb_model.py` (compute_G_loss, B1-B5 methods)
**Configuration**: `options/train_options.py` (--paired_strategy)
