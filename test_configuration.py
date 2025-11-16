#!/usr/bin/env python3
"""Test configuration system for paired training experiments."""

import sys
import os

print("Testing Paired Training Configuration System")
print("=" * 60)

# Test 1: Import modules
print("\n[1/5] Testing imports...")
try:
    from options.train_options import TrainOptions
    from models.sb_model import SBModel
    print("  ✓ Successfully imported modules")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check paired_strategy options
print("\n[2/5] Testing paired_strategy parameter...")
try:
    parser = TrainOptions().initialize(TrainOptions().parser)

    # Find the paired_strategy action
    paired_strategy_action = None
    for action in parser._actions:
        if action.dest == 'paired_strategy':
            paired_strategy_action = action
            break

    if paired_strategy_action:
        expected_choices = ['none', 'sb_gt_transport', 'l1_loss', 'regularization', 'weight_schedule', 'hybrid']
        if paired_strategy_action.choices == expected_choices:
            print(f"  ✓ Found paired_strategy with correct choices")
            print(f"    Strategies: {', '.join(expected_choices)}")
        else:
            print(f"  ✗ Unexpected choices: {paired_strategy_action.choices}")
            sys.exit(1)
    else:
        print(f"  ✗ paired_strategy parameter not found")
        sys.exit(1)
except Exception as e:
    print(f"  ✗ Test failed: {e}")
    sys.exit(1)

# Test 3: Check strategy-specific parameters
print("\n[3/5] Testing strategy-specific parameters...")
try:
    required_params = {
        'lambda_L1': 'l1_loss strategy',
        'lambda_perceptual': 'regularization strategy',
        'sb_weight_schedule': 'weight_schedule strategy',
        'sb_weight_end': 'weight_schedule strategy'
    }

    all_actions = {action.dest: action for action in parser._actions}

    for param, purpose in required_params.items():
        if param in all_actions:
            print(f"  ✓ Found {param} ({purpose})")
        else:
            print(f"  ✗ Missing {param}")
            sys.exit(1)
except Exception as e:
    print(f"  ✗ Test failed: {e}")
    sys.exit(1)

# Test 4: Verify SBModel loss implementation
print("\n[4/5] Testing SBModel implementation...")
try:
    # Check if compute_G_loss handles strategies
    import inspect
    source = inspect.getsource(SBModel.compute_G_loss)

    if 'paired_strategy' in source and 'sb_gt_transport' in source:
        print("  ✓ SBModel.compute_G_loss includes strategy handling")
    else:
        print("  ✗ Strategy handling not found in compute_G_loss")
        sys.exit(1)

    if 'loss_SB_guidance' in source:
        print("  ✓ Found loss_SB_guidance for Scheme A")
    else:
        print("  ! Warning: loss_SB_guidance not found")

except Exception as e:
    print(f"  ✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Check experiment scripts
print("\n[5/5] Testing experiment scripts...")
experiment_scripts = [
    'experiments/scheme_a_twostage.sh',
    'experiments/baseline_l1.sh',
    'run_train.sh'
]

for script in experiment_scripts:
    if os.path.exists(script) and os.access(script, os.X_OK):
        # Check if script contains key configuration
        with open(script, 'r') as f:
            content = f.read()

        if 'PAIRED_STRATEGY' in content:
            print(f"  ✓ {script}: Executable and configured")
        else:
            print(f"  ! {script}: Missing PAIRED_STRATEGY")
    elif os.path.exists(script):
        print(f"  ! {script}: Not executable")
    else:
        print(f"  ✗ {script}: Not found")
        sys.exit(1)

# Summary
print("\n" + "=" * 60)
print("Configuration System Test: PASSED ✓")
print("=" * 60)
print("\nImplemented Strategies:")
print("  • none            : Default unpaired training")
print("  • sb_gt_transport : [Scheme A] GT guidance in SB transport")
print("  • l1_loss         : [Baseline] Simple L1 loss")
print("  • regularization  : [Scheme B] Enhanced regularization (TBD)")
print("  • weight_schedule : [Scheme D] Dynamic weights (TBD)")
print("  • hybrid          : Combination of strategies (TBD)")
print("\nQuick Start:")
print("  sbatch experiments/scheme_a_twostage.sh")
print("  sbatch experiments/baseline_l1.sh")
print("=" * 60)
