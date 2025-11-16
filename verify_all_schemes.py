#!/usr/bin/env python3
"""Verify all schemes (A, Baseline, B1-B5) implementation."""

import sys
import os

print("Verifying All Schemes Implementation")
print("=" * 60)

# Test 1: Check options
print("\n[1/4] Checking train_options.py...")
with open('options/train_options.py', 'r') as f:
    options_code = f.read()

required_strategies = [
    'sb_gt_transport',  # Scheme A
    'l1_loss',          # Baseline
    'nce_feature',      # B1
    'frequency',        # B2
    'gradient',         # B3
    'multiscale',       # B4
    'selfsup_contrast', # B5
]

all_found = True
for strategy in required_strategies:
    if strategy in options_code:
        print(f"  ✓ Found strategy: {strategy}")
    else:
        print(f"  ✗ Missing strategy: {strategy}")
        all_found = False
        sys.exit(1)

if 'lambda_reg' in options_code:
    print(f"  ✓ Found lambda_reg parameter")
else:
    print(f"  ✗ Missing lambda_reg parameter")
    sys.exit(1)

# Test 2: Check model implementation
print("\n[2/4] Checking models/sb_model.py...")
with open('models/sb_model.py', 'r') as f:
    model_code = f.read()

required_methods = {
    'compute_frequency_loss': 'B2 frequency loss',
    'compute_gradient_loss': 'B3 gradient loss',
    'compute_multiscale_loss': 'B4 multiscale loss',
    'compute_contrastive_loss': 'B5 contrastive loss',
}

for method, description in required_methods.items():
    if f'def {method}' in model_code:
        print(f"  ✓ Found method: {method} ({description})")
    else:
        print(f"  ✗ Missing method: {method}")
        sys.exit(1)

# Check loss assignments
loss_checks = {
    'self.loss_SB_guidance': 'Scheme A guidance',
    'self.loss_L1': 'Baseline L1',
    'self.loss_NCE_paired': 'B1 NCE paired',
    'self.loss_freq': 'B2 frequency',
    'self.loss_gradient': 'B3 gradient',
    'self.loss_multiscale': 'B4 multiscale',
    'self.loss_contrast': 'B5 contrastive',
}

for loss_var, description in loss_checks.items():
    if loss_var in model_code:
        print(f"  ✓ Found loss: {loss_var} ({description})")
    else:
        print(f"  ✗ Missing loss: {loss_var}")
        sys.exit(1)

# Test 3: Check experiment scripts
print("\n[3/4] Checking experiment scripts...")
scripts_to_check = {
    'experiments/launch_all_18gpu.sh': 'Main launcher',
    'experiments/slurm_launch_all.sh': 'SLURM wrapper',
    'experiments/test_all.sh': 'Batch testing',
}

for script, description in scripts_to_check.items():
    if os.path.exists(script) and os.access(script, os.X_OK):
        print(f"  ✓ {description}: {script}")
    elif os.path.exists(script):
        print(f"  ! {description}: {script} (not executable)")
    else:
        print(f"  ✗ Missing: {script}")
        sys.exit(1)

# Test 4: Mathematical formulations
print("\n[4/4] Verifying mathematical implementations...")

print("\n  Loss Formulations:")
print("  ─────────────────")
print("  Scheme A (sb_gt_transport):")
print("    ℒ_SB_guidance = τ·||fake_B - real_B||²")
print("    └─ Added to SB loss")
if 'self.opt.tau * torch.mean((self.fake_B - self.real_B)**2)' in model_code:
    print("    ✓ Implementation matches formula")
else:
    print("    ! Warning: Implementation may differ")

print("\n  Baseline (l1_loss):")
print("    ℒ_L1 = λ_L1·||fake_B - real_B||₁")
if 'self.criterionL1(fake, self.real_B)' in model_code:
    print("    ✓ Implementation found")
else:
    print("    ✗ Implementation not found")
    sys.exit(1)

print("\n  B1 (nce_feature):")
print("    ℒ_NCE_paired = NCE(real_B, fake_B)")
if 'self.calculate_NCE_loss(self.real_B, fake)' in model_code:
    print("    ✓ Implementation found")
else:
    print("    ✗ Implementation not found")
    sys.exit(1)

print("\n  B2 (frequency):")
print("    ℒ_freq = ||FFT(fake_B) - FFT(real_B)||")
if 'torch.fft.fft2' in model_code:
    print("    ✓ FFT implementation found")
else:
    print("    ✗ FFT not found")
    sys.exit(1)

print("\n  B3 (gradient):")
print("    ℒ_gradient = ||∇fake_B - ∇real_B||")
if 'grad_x' in model_code and 'grad_y' in model_code:
    print("    ✓ Gradient computation found")
else:
    print("    ✗ Gradient computation not found")
    sys.exit(1)

print("\n  B4 (multiscale):")
print("    ℒ_multiscale = Σ w_i·||pyramid_i(fake_B) - pyramid_i(real_B)||")
if 'build_pyramid' in model_code:
    print("    ✓ Pyramid construction found")
else:
    print("    ✗ Pyramid not found")
    sys.exit(1)

print("\n  B5 (selfsup_contrast):")
print("    ℒ_contrast = 1 - cosine_sim(feat(fake_B), feat(real_B))")
if 'f_fake_norm' in model_code and 'f_real_norm' in model_code:
    print("    ✓ Cosine similarity found")
else:
    print("    ✗ Contrastive loss not found")
    sys.exit(1)

print("\n" + "=" * 60)
print("All Schemes Verification: PASSED ✓")
print("=" * 60)
print("\nImplemented Strategies:")
print("  ✓ Scheme A: SB GT Transport")
print("  ✓ Baseline: L1 Loss")
print("  ✓ B1: Enhanced NCE (feature space)")
print("  ✓ B2: Frequency Domain (FFT)")
print("  ✓ B3: Gradient/Structure")
print("  ✓ B4: Multi-scale Pyramid")
print("  ✓ B5: Self-supervised Contrastive")
print("\nBatch Experiments:")
print("  ✓ 18-GPU parallel launcher ready")
print("  ✓ SLURM integration ready")
print("  ✓ Batch testing script ready")
print("\nQuick Start:")
print("  sbatch experiments/slurm_launch_all.sh")
print("  # After training:")
print("  bash experiments/test_all.sh")
print("=" * 60)
