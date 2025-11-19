# Ablation Study: Detailed Experiment Design

## Scientific Motivation

This ablation study aims to answer two key research questions:

1. **Component Analysis**: What are the individual contributions of different loss components in Schrödinger Bridge (SB) for paired MRI contrast transfer?
2. **Supervision Strategy**: Where should we apply supervision - at the intermediate diffusion state or at the final output?

---

## Loss Component Definitions

### OT_input: Intermediate State Supervision
```python
loss_OT_input = tau * mean((real_A_noisy - real_B)^2)
```

**What it does**:
- Supervises the **intermediate noisy state** from forward diffusion
- `real_A_noisy` is computed through multiple diffusion steps: A → X₁ → X₂ → ... → Xₜ
- When `use_ot_input=True`, this diffusion is computed **with gradient**
- Directly constrains the forward diffusion process to move toward ground truth

**Implementation**:
- Forward diffusion computed in `forward()` with `compute_noisy_with_grad=True`
- Uses gradient checkpointing: only keeps gradient for final Xₜ, detaches intermediate states
- Memory-efficient: `Xt = (1-inter) * Xt.detach() + inter * Xt_1 + noise`

### OT_output: Final Output Supervision
```python
loss_OT_output = tau * mean((fake_B - real_B)^2)
```

**What it does**:
- Supervises the **final network output**
- `fake_B` is the result after full diffusion process
- Standard supervised learning: minimizes L2 distance to ground truth
- Does NOT require gradient in forward diffusion (uses `no_grad` version)

### Entropy Loss: Energy-Based Regularization
```python
loss_entropy = -tau * ET_XY
```

**What it does**:
- Energy-based regularization from Schrödinger Bridge formulation
- `ET_XY = E(X,X|X,X) - logsumexp(E(X,X|X,X'))`
- Encourages smooth transport between domains
- Based on netE (energy network)

---

## 12 Experiments Breakdown

### Group 1: Fully Paired (100% data, from scratch)

| Exp | OT_input | OT_output | Entropy | Training | Research Question |
|-----|----------|-----------|---------|----------|-------------------|
| 1   | ✓        |           |         | 1-600    | Can intermediate state supervision alone work? |
| 2   | ✓        |           | ✓       | 1-600    | Does entropy help intermediate supervision? |
| 3   |          | ✓         |         | 1-600    | Can output supervision alone work? |
| 4   |          | ✓         | ✓       | 1-600    | Does entropy help output supervision? |

**Comparison**:
- **Exp1 vs Exp3**: Intermediate vs output supervision (both without entropy)
- **Exp2 vs Exp4**: Intermediate vs output supervision (both with entropy)
- **Exp1 vs Exp2**: Effect of adding entropy to intermediate supervision
- **Exp3 vs Exp4**: Effect of adding entropy to output supervision

### Group 2: Two-Stage (10% data, pretrained)

| Exp | OT_input | OT_output | Entropy | Training | Research Question |
|-----|----------|-----------|---------|----------|-------------------|
| 5   | ✓        |           |         | 401-600  | Low-data: intermediate supervision? |
| 6   | ✓        |           | ✓       | 401-600  | Low-data: intermediate + entropy? |
| 7   |          | ✓         |         | 401-600  | Low-data: output supervision? |
| 8   |          | ✓         | ✓       | 401-600  | Low-data: output + entropy? |

**Comparison**:
- **Exp5 vs Exp7**: Intermediate vs output (10% data, no entropy)
- **Exp6 vs Exp8**: Intermediate vs output (10% data, with entropy)
- **Exp5 vs Exp1**: Effect of data scarcity on intermediate supervision
- **Exp7 vs Exp3**: Effect of data scarcity on output supervision

### Group 3: Two-Stage (100% data, pretrained)

| Exp | OT_input | OT_output | Entropy | Training | Research Question |
|-----|----------|-----------|---------|----------|-------------------|
| 9   | ✓        |           |         | 401-600  | Pretrained: intermediate supervision? |
| 10  | ✓        |           | ✓       | 401-600  | Pretrained: intermediate + entropy? |
| 11  |          | ✓         |         | 401-600  | Pretrained: output supervision? |
| 12  |          | ✓         | ✓       | 401-600  | Pretrained: output + entropy? |

**Comparison**:
- **Exp9 vs Exp11**: Intermediate vs output (pretrained, no entropy)
- **Exp10 vs Exp12**: Intermediate vs output (pretrained, with entropy)
- **Exp1 vs Exp9**: From-scratch vs pretrained (intermediate, no entropy)
- **Exp3 vs Exp11**: From-scratch vs pretrained (output, no entropy)

---

## Expected Insights

### 1. Supervision Location (OT_input vs OT_output)
- **Hypothesis**: OT_output should work better as it directly supervises the final task
- **OT_input** may help learn better diffusion dynamics but might be harder to optimize

### 2. Role of Entropy
- **Hypothesis**: Entropy regularization provides smoother transport
- May be more important when supervision is weaker (OT_input)

### 3. Data Efficiency
- **Group 2 (10% data)**: Which supervision strategy is more data-efficient?
- **Group 3 (100% data)**: Can pretrained model benefit from paired fine-tuning?

### 4. Training Dynamics
- **Fully Paired**: Learn everything from paired data
- **Two-Stage**: Build on unpaired pretrained model, adapt with paired data

---

## Implementation Notes

### Memory Efficiency for OT_input

When `use_ot_input=True`, forward diffusion is computed with gradient:

```python
# Gradient checkpointing: only keep gradient for current step
Xt = (1-inter) * Xt.detach() + inter * Xt_1 + noise
#                    ^^^^^^^ Detach previous state to save memory
#                                    ^^^^ Keep gradient for network output
```

This allows gradient to flow through `real_A_noisy` without storing all intermediate states.

### Conditional Gradient Computation

```python
compute_noisy_with_grad = use_ot_input and self.opt.isTrain
```

- Only enabled for OT_input experiments
- Other experiments use faster `no_grad` version
- Automatic switching based on experiment config

---

## Metrics to Compare

All experiments log:
- **SSIM**: Structural similarity
- **PSNR**: Peak signal-to-noise ratio
- **NRMSE**: Normalized root mean square error
- **Loss components**: OT_input, OT_output, Entropy (when applicable)

Compare across:
- Training efficiency (loss curves)
- Final performance (SSIM/PSNR)
- Generalization (validation metrics)
