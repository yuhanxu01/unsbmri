#!/usr/bin/env python3
"""Quick verification of 8-experiment setup (10% paired data)."""

import sys
import os

print("Verifying 8-Experiment Setup (10% Paired Data)")
print("=" * 60)

# Test 1: Check launcher script
print("\n[1/3] Checking launcher script...")
with open('experiments/launch_all_8gpu.sh', 'r') as f:
    launcher = f.read()

if 'PAIRED_SUBSET_RATIO=0.1' in launcher:
    print("  ✓ Uses 10% paired data")
else:
    print("  ✗ Wrong paired data ratio")
    sys.exit(1)

strategies = ['sb_gt_transport', 'l1_loss', 'nce_feature', 'frequency', 'gradient', 'multiscale', 'selfsup_contrast']
for strategy in strategies:
    if strategy in launcher:
        print(f"  ✓ Found strategy: {strategy}")
    else:
        print(f"  ✗ Missing strategy: {strategy}")
        sys.exit(1)

# Test 2: Check test script
print("\n[2/3] Checking test script...")
with open('experiments/test_all.sh', 'r') as f:
    test_script = f.read()

if '10% paired data' in test_script or '10% Paired Data' in test_script:
    print("  ✓ Test script configured for 10% data")
else:
    print("  ! Warning: Test script may not mention 10% explicitly")

if 'STRATEGY RANKING' in test_script:
    print("  ✓ Includes strategy ranking")
else:
    print("  ✗ Missing ranking functionality")
    sys.exit(1)

# Test 3: Check documentation
print("\n[3/3] Checking documentation...")
with open('EXPERIMENTS.md', 'r') as f:
    doc = f.read()

if '10% data' in doc or '10% paired data' in doc:
    print("  ✓ Documentation specifies 10% data")
else:
    print("  ✗ Documentation doesn't specify 10%")
    sys.exit(1)

if '8 experiments' in doc or '8 GPUs' in doc:
    print("  ✓ Documentation mentions 8 experiments")
else:
    print("  ✗ Documentation doesn't mention 8 experiments")
    sys.exit(1)

print("\n" + "=" * 60)
print("Verification: PASSED ✓")
print("=" * 60)
print("\nExperiment Setup:")
print("  Total: 8 experiments")
print("  Paired data: 10% (fixed)")
print("  Strategies: A, L1, B1, B2, B3, B4, B5")
print("\nQuick Start:")
print("  sbatch experiments/slurm_launch_all.sh")
print("  # After training:")
print("  bash experiments/test_all.sh")
print("=" * 60)
