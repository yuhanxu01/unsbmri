# MRI Paired Training Experiments

Modular framework for testing different strategies to utilize paired data in Schrödinger Bridge training.

## Philosophy

Based on UNSB's mathematical framework (entropy-regularized optimal transport), we avoid naive approaches that break the SB structure. Instead, we implement scientifically-grounded strategies that:
1. Maintain SB's physical/mathematical meaning
2. Leverage paired GT within the existing framework
3. Allow controlled comparison between strategies

## Implemented Strategies

### Scheme A: SB GT Transport (`sb_gt_transport`)
**Mathematical basis**: Add GT guidance to transport cost while preserving SB structure

```
ℒ_SB = -(T-t)/T·τ·ET_XY + τ·||x_t - fake_B||² + τ·||fake_B - real_B||²
                                                   └─ GT guidance term
```

**Advantage**: Uses paired data to guide transport without breaking SB's mathematical framework

**Usage**:
```bash
export PAIRED_STRATEGY="sb_gt_transport"
export PAIRED_STAGE="--paired_stage"
bash run_train.sh
```

### Baseline: L1 Loss (`l1_loss`)
**Mathematical basis**: Direct pixel-wise supervision (naive approach for comparison)

```
ℒ_total = ℒ_GAN + λ_SB·ℒ_SB + λ_NCE·ℒ_NCE + λ_L1·||fake_B - real_B||₁
```

**Usage**:
```bash
export PAIRED_STRATEGY="l1_loss"
export LAMBDA_L1=1.0
export PAIRED_STAGE="--paired_stage"
bash run_train.sh
```

### Future Strategies (Extensible Design)

- **Scheme B** (`regularization`): Enhanced regularization with perceptual/LPIPS loss
- **Scheme D** (`weight_schedule`): Dynamic loss weight annealing
- **Hybrid** (`hybrid`): Combine multiple strategies

## Quick Examples

### Two-Stage Training: Unpaired → Paired (Scheme A)
```bash
sbatch experiments/scheme_a_twostage.sh
```

### Two-Stage Training: Unpaired → Paired (Baseline)
```bash
sbatch experiments/baseline_l1.sh
```

### Custom Configuration
```bash
export EXPERIMENT_NAME="custom_test"
export PAIRED_STRATEGY="sb_gt_transport"
export PAIRED_SUBSET_RATIO=0.5  # Use 50% paired data
export N_EPOCHS=100
bash run_train.sh
```

## Key Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `paired_strategy` | `none`, `sb_gt_transport`, `l1_loss`, ... | Strategy selection |
| `paired_subset_ratio` | `0.0-1.0` | Fraction of paired data to use |
| `lambda_SB` | `1.0` (default) | SB loss weight |
| `lambda_NCE` | `1.0` (default) | NCE loss weight |
| `lambda_L1` | `0.0` (default) | L1 loss weight (for `l1_loss` strategy) |

## Monitoring

**WandB Logs**:
- `loss/SB`: Schrödinger Bridge loss
- `loss/SB_guidance`: GT guidance term (Scheme A)
- `loss/L1`: L1 reconstruction loss (Baseline)
- `loss/metric_SSIM`: Structural similarity (paired only)
- `loss/metric_PSNR`: Peak signal-to-noise ratio (paired only)

**Visualizations**: Input | Output | Ground Truth (3-column comparison)

## Design Principles

1. **Modularity**: Easy to add new strategies without modifying core code
2. **Reproducibility**: All experiments use consistent base configuration
3. **Extensibility**: Configuration system supports future schemes (B, C, D, ...)
4. **Scientific rigor**: Each strategy has clear mathematical justification

## File Organization

```
experiments/
├── README.md              # Detailed configuration guide
├── scheme_a_twostage.sh   # Scheme A: SB GT transport
└── baseline_l1.sh         # Baseline: L1 loss

run_train.sh               # Main training script (configurable via env vars)
```

---

**Full documentation**: See `experiments/README.md`
**Code details**: See `models/sb_model.py` (compute_G_loss method)
