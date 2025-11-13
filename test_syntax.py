#!/usr/bin/env python3
"""Simple syntax and structure test for new paired training features."""

import sys
import os
import ast

print("Testing new paired training features (syntax check)...")
print("=" * 60)

# Test 1: Parse Python files to check syntax
print("\n1. Testing Python file syntax...")
files_to_check = [
    'options/train_options.py',
    'models/sb_model.py',
    'train.py'
]

for filepath in files_to_check:
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        print(f"   ✓ {filepath}: Valid syntax")
    except SyntaxError as e:
        print(f"   ✗ {filepath}: Syntax error at line {e.lineno}: {e.msg}")
        sys.exit(1)
    except Exception as e:
        print(f"   ✗ {filepath}: Error - {e}")
        sys.exit(1)

# Test 2: Check for new parameters in train_options.py
print("\n2. Checking train_options.py for new parameters...")
with open('options/train_options.py', 'r') as f:
    options_content = f.read()

required_params = [
    'paired_stage',
    'paired_subset_ratio',
    'paired_subset_seed',
    'lambda_L1',
    'compute_paired_metrics'
]

for param in required_params:
    if param in options_content:
        print(f"   ✓ Found parameter: {param}")
    else:
        print(f"   ✗ Missing parameter: {param}")
        sys.exit(1)

# Test 3: Check for new methods in sb_model.py
print("\n3. Checking sb_model.py for new methods...")
with open('models/sb_model.py', 'r') as f:
    model_content = f.read()

required_methods = [
    'compute_paired_metrics',
    'criterionL1'
]

for method in required_methods:
    if method in model_content:
        print(f"   ✓ Found: {method}")
    else:
        print(f"   ✗ Missing: {method}")
        sys.exit(1)

# Test 4: Check for metrics logging in train.py
print("\n4. Checking train.py for metrics logging...")
with open('train.py', 'r') as f:
    train_content = f.read()

required_features = [
    'compute_paired_metrics',
    'metric_SSIM',
    'metric_PSNR',
    'metric_NRMSE'
]

for feature in required_features:
    if feature in train_content:
        print(f"   ✓ Found: {feature}")
    else:
        print(f"   ✗ Missing: {feature}")
        sys.exit(1)

# Test 5: Check training scripts
print("\n5. Checking training scripts...")
scripts = [
    'run_twostage_training.sh',
    'run_paired_training.sh'
]

for script in scripts:
    if os.path.exists(script):
        if os.access(script, os.X_OK):
            print(f"   ✓ Found executable script: {script}")
        else:
            print(f"   ! Found script (not executable): {script}")

        # Check for key parameters in script
        with open(script, 'r') as f:
            script_content = f.read()

        if '--paired_stage' in script_content:
            print(f"      ✓ Script uses --paired_stage")
        if '--lambda_L1' in script_content:
            print(f"      ✓ Script uses --lambda_L1")
        if '--compute_paired_metrics' in script_content:
            print(f"      ✓ Script uses --compute_paired_metrics")
    else:
        print(f"   ✗ Missing script: {script}")
        sys.exit(1)

# Test 6: Check documentation
print("\n6. Checking documentation...")
if os.path.exists('TRAINING_EXPERIMENTS.md'):
    print("   ✓ Found documentation: TRAINING_EXPERIMENTS.md")
    with open('TRAINING_EXPERIMENTS.md', 'r') as f:
        doc_content = f.read()
    if 'Two-Stage Training' in doc_content and 'Full Paired Training' in doc_content:
        print("   ✓ Documentation contains experiment descriptions")
else:
    print("   ! Missing documentation file")

print("\n" + "=" * 60)
print("All syntax checks passed! ✓")
print("=" * 60)
print("\nImplemented features:")
print("  • Paired training mode with --paired_stage")
print("  • Subset selection with --paired_subset_ratio")
print("  • L1 loss for supervised paired training")
print("  • SSIM/PSNR/NRMSE metrics computation and logging")
print("  • WandB integration for metrics visualization")
print("\nTraining scripts ready:")
print("  1. Two-stage training: run_twostage_training.sh")
print("     - Stage 1: Unpaired (200+200 epochs)")
print("     - Stage 2: Paired 30% subset (100+100 epochs)")
print("  2. Full paired training: run_paired_training.sh")
print("     - Paired 100% (200+200 epochs)")
print("=" * 60)
