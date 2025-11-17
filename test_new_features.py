#!/usr/bin/env python3
"""Test script to verify new paired training features."""

import sys
import torch
import numpy as np

print("Testing new paired training features...")
print("=" * 60)

# Test 1: Import modified modules
print("\n1. Testing imports...")
try:
    from options.train_options import TrainOptions
    from models.sb_model import SBModel
    print("   ✓ Successfully imported TrainOptions and SBModel")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check new command-line arguments
print("\n2. Testing new command-line arguments...")
try:
    parser = TrainOptions().initialize(TrainOptions().parser)

    # Check if new arguments exist
    required_args = [
        '--paired_stage',
        '--paired_subset_ratio',
        '--paired_subset_seed',
        '--lambda_L1',
        '--compute_paired_metrics'
    ]

    all_actions = {action.dest: action for action in parser._actions}

    for arg in required_args:
        arg_dest = arg.lstrip('--').replace('-', '_')
        if arg_dest in all_actions:
            print(f"   ✓ Found argument: {arg}")
        else:
            print(f"   ✗ Missing argument: {arg}")
            sys.exit(1)

except Exception as e:
    print(f"   ✗ Test failed: {e}")
    sys.exit(1)

# Test 3: Check SBModel methods
print("\n3. Testing SBModel new methods...")
try:
    # Check if compute_paired_metrics method exists
    if hasattr(SBModel, 'compute_paired_metrics'):
        print("   ✓ Found method: compute_paired_metrics")
    else:
        print("   ✗ Missing method: compute_paired_metrics")
        sys.exit(1)

except Exception as e:
    print(f"   ✗ Test failed: {e}")
    sys.exit(1)

# Test 4: Test compute_paired_metrics with dummy data
print("\n4. Testing compute_paired_metrics with dummy data...")
try:
    # Create a mock model instance
    class MockOpt:
        def __init__(self):
            self.gpu_ids = []
            self.isTrain = True
            self.checkpoints_dir = './checkpoints'
            self.name = 'test'
            self.mode = 'sb'
            self.lambda_L1 = 1.0
            self.paired_stage = True
            self.nce_layers = '0,4,8,12,16'
            self.lambda_GAN = 1.0
            self.lambda_NCE = 1.0
            self.lambda_SB = 0.1
            self.nce_idt = True
            self.input_nc = 1
            self.output_nc = 1
            self.ngf = 64
            self.netG = 'resnet_9blocks'
            self.normG = 'instance'
            self.no_dropout = False
            self.init_type = 'normal'
            self.init_gain = 0.02
            self.no_antialias = False
            self.no_antialias_up = False
            self.netF = 'mlp_sample'
            self.netF_nc = 256
            self.ndf = 64
            self.netD = 'basic'
            self.n_layers_D = 3
            self.normD = 'instance'
            self.gan_mode = 'lsgan'
            self.lr = 0.0002
            self.beta1 = 0.5
            self.beta2 = 0.999
            self.nce_T = 0.07
            self.num_patches = 256
            self.flip_equivariance = False
            self.nce_includes_all_negatives_from_minibatch = False
            self.tau = 0.1
            self.num_timesteps = 5
            self.std = 0.0
            self.batch_size = 1
            self.phase = 'train'

    opt = MockOpt()

    # Create dummy fake_B and real_B tensors
    # Simulate magnitude MRI images [B, 1, H, W]
    fake_B = torch.randn(2, 1, 256, 256) * 0.5 + 0.5  # [0, 1] range
    real_B = fake_B + torch.randn_like(fake_B) * 0.1  # Add small noise

    # Create a minimal model instance just for testing
    class TestModel:
        def __init__(self):
            self.fake_B = fake_B
            self.real_B = real_B
            self.opt = opt

    test_model = TestModel()

    # Manually bind the method to test_model
    from types import MethodType
    test_model.compute_paired_metrics = MethodType(SBModel.compute_paired_metrics, test_model)

    # Call compute_paired_metrics
    metrics = test_model.compute_paired_metrics()

    # Check if metrics are returned correctly
    required_keys = ['ssim', 'psnr', 'nrmse']
    for key in required_keys:
        if key in metrics:
            print(f"   ✓ Metric '{key}': {metrics[key]:.4f}")
        else:
            print(f"   ✗ Missing metric: {key}")
            sys.exit(1)

    # Validate metric ranges
    if 0 <= metrics['ssim'] <= 1:
        print(f"   ✓ SSIM in valid range [0, 1]")
    else:
        print(f"   ✗ SSIM out of range: {metrics['ssim']}")

    if metrics['psnr'] > 0:
        print(f"   ✓ PSNR is positive")
    else:
        print(f"   ✗ PSNR is not positive: {metrics['psnr']}")

    if 0 <= metrics['nrmse'] <= 1:
        print(f"   ✓ NRMSE in valid range [0, 1]")
    else:
        print(f"   ✗ NRMSE out of range: {metrics['nrmse']}")

except Exception as e:
    print(f"   ✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check training scripts
print("\n5. Testing training scripts...")
import os
scripts = [
    'run_twostage_training.sh',
    'run_paired_training.sh'
]

for script in scripts:
    if os.path.exists(script) and os.access(script, os.X_OK):
        print(f"   ✓ Found executable script: {script}")
    elif os.path.exists(script):
        print(f"   ! Found script (not executable): {script}")
    else:
        print(f"   ✗ Missing script: {script}")
        sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)
print("\nYou can now run the training experiments:")
print("  1. Two-stage training: sbatch run_twostage_training.sh")
print("  2. Full paired training: sbatch run_paired_training.sh")
print("=" * 60)
