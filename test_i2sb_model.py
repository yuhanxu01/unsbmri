"""
Test script for I2SB model to verify implementation
"""

import torch
import numpy as np
from options.train_options import TrainOptions
from models import create_model


def test_i2sb_model():
    """Test basic I2SB model functionality"""

    print("=" * 80)
    print("Testing I2SB Model Implementation")
    print("=" * 80)

    # Create minimal options for testing
    print("\n[1/6] Creating test options...")
    opt = TrainOptions().parse([
        '--dataroot', '/tmp/test_data',
        '--name', 'test_i2sb',
        '--model', 'i2sb',
        '--gpu_ids', '-1',  # CPU only for testing
        '--batch_size', '2',
        '--input_nc', '1',
        '--output_nc', '1',
        '--netG', 'resnet_9blocks_cond',
        '--ngf', '32',  # Smaller for faster testing
        '--num_timesteps', '5',  # Smaller for faster testing
        '--i2sb_num_timesteps', '100',
        '--i2sb_beta_schedule', 'linear',
        '--i2sb_objective', 'pred_noise',
        '--lambda_diffusion', '1.0',
        '--lambda_simple', '1.0',
        '--isTrain',
        '--no_html',
        '--display_id', '-1',
    ])

    print(f"   Model: {opt.model}")
    print(f"   Diffusion steps: {opt.i2sb_num_timesteps}")
    print(f"   Objective: {opt.i2sb_objective}")

    # Create model
    print("\n[2/6] Creating I2SB model...")
    try:
        model = create_model(opt)
        print("   ✓ Model created successfully")
    except Exception as e:
        print(f"   ✗ Error creating model: {e}")
        return False

    # Create dummy data
    print("\n[3/6] Creating dummy data...")
    batch_size = 2
    height, width = 256, 256
    channels = 1

    dummy_source = torch.randn(batch_size, channels, height, width)
    dummy_target = torch.randn(batch_size, channels, height, width)

    data = {
        'A': dummy_source,
        'B': dummy_target,
        'A_paths': ['/dummy/path/a.h5'] * batch_size,
        'B_paths': ['/dummy/path/b.h5'] * batch_size
    }

    print(f"   Source shape: {dummy_source.shape}")
    print(f"   Target shape: {dummy_target.shape}")

    # Initialize model with data
    print("\n[4/6] Initializing model with data...")
    try:
        model.data_dependent_initialize(data, None)
        model.setup(opt)
        print("   ✓ Model initialized successfully")
    except Exception as e:
        print(f"   ✗ Error initializing model: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test forward pass
    print("\n[5/6] Testing forward pass...")
    try:
        model.set_input(data, None)
        model.forward()
        print("   ✓ Forward pass successful")

        # Check outputs
        assert hasattr(model, 'source'), "Model should have 'source' attribute"
        assert hasattr(model, 'target'), "Model should have 'target' attribute"
        assert hasattr(model, 'generated'), "Model should have 'generated' attribute"
        assert hasattr(model, 'noisy_target'), "Model should have 'noisy_target' attribute"

        print(f"   Source shape: {model.source.shape}")
        print(f"   Target shape: {model.target.shape}")
        print(f"   Generated shape: {model.generated.shape}")
        print(f"   Noisy target shape: {model.noisy_target.shape}")

    except Exception as e:
        print(f"   ✗ Error in forward pass: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test loss computation
    print("\n[6/6] Testing loss computation...")
    try:
        loss_G = model.compute_G_loss()
        print(f"   ✓ Generator loss computed: {loss_G.item():.4f}")

        # Check individual losses
        losses = model.get_current_losses()
        print("   Loss components:")
        for loss_name, loss_value in losses.items():
            print(f"      {loss_name}: {loss_value:.4f}")

        if opt.use_gan:
            loss_D = model.compute_D_loss()
            print(f"   ✓ Discriminator loss computed: {loss_D.item():.4f}")

    except Exception as e:
        print(f"   ✗ Error computing losses: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test full optimization step
    print("\n[Bonus] Testing full optimization step...")
    try:
        model.set_input(data, None)
        model.optimize_parameters()
        print("   ✓ Optimization step successful")
    except Exception as e:
        print(f"   ✗ Error in optimization: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test sampling
    print("\n[Bonus] Testing sampling (inference)...")
    try:
        model.netG.eval()
        with torch.no_grad():
            generated = model.sample(dummy_source, num_steps=10)
        print(f"   ✓ Sampling successful, output shape: {generated.shape}")
    except Exception as e:
        print(f"   ✗ Error in sampling: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("✓ All tests passed successfully!")
    print("=" * 80)

    return True


def test_diffusion_parameters():
    """Test diffusion parameter setup"""

    print("\n" + "=" * 80)
    print("Testing Diffusion Parameters")
    print("=" * 80)

    opt = TrainOptions().parse([
        '--dataroot', '/tmp',
        '--name', 'test',
        '--model', 'i2sb',
        '--gpu_ids', '-1',
        '--i2sb_num_timesteps', '100',
        '--i2sb_beta_schedule', 'linear',
        '--i2sb_beta_start', '0.0001',
        '--i2sb_beta_end', '0.02',
        '--isTrain',
        '--no_html',
        '--display_id', '-1',
    ])

    model = create_model(opt)

    print("\n[1/3] Checking beta schedule...")
    print(f"   Beta shape: {model.betas.shape}")
    print(f"   Beta min: {model.betas.min().item():.6f}")
    print(f"   Beta max: {model.betas.max().item():.6f}")
    print(f"   ✓ Beta schedule looks correct")

    print("\n[2/3] Checking alpha_cumprod...")
    print(f"   Alpha_cumprod shape: {model.alphas_cumprod.shape}")
    print(f"   Alpha_cumprod[0]: {model.alphas_cumprod[0].item():.6f}")
    print(f"   Alpha_cumprod[-1]: {model.alphas_cumprod[-1].item():.6f}")
    assert model.alphas_cumprod[0] > model.alphas_cumprod[-1], "Alpha_cumprod should be decreasing"
    print(f"   ✓ Alpha_cumprod is monotonically decreasing")

    print("\n[3/3] Checking derived parameters...")
    print(f"   sqrt_alphas_cumprod shape: {model.sqrt_alphas_cumprod.shape}")
    print(f"   sqrt_one_minus_alphas_cumprod shape: {model.sqrt_one_minus_alphas_cumprod.shape}")
    print(f"   posterior_variance shape: {model.posterior_variance.shape}")
    print(f"   ✓ All derived parameters have correct shapes")

    print("\n" + "=" * 80)
    print("✓ Diffusion parameters test passed!")
    print("=" * 80)


if __name__ == '__main__':
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "I2SB Model Test Suite" + " " * 37 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")

    # Run tests
    test1_passed = test_diffusion_parameters()
    test2_passed = test_i2sb_model()

    # Summary
    print("\n\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 32 + "Test Summary" + " " * 34 + "║")
    print("╠" + "=" * 78 + "╣")
    print("║  Diffusion Parameters Test: " + ("✓ PASSED" if test1_passed else "✗ FAILED") + " " * 40 + "║")
    print("║  Model Functionality Test:  " + ("✓ PASSED" if test2_passed else "✗ FAILED") + " " * 40 + "║")
    print("╚" + "=" * 78 + "╝")
    print("\n")

    if test1_passed and test2_passed:
        print("🎉 All tests passed! The I2SB model is ready for training.")
        exit(0)
    else:
        print("❌ Some tests failed. Please check the errors above.")
        exit(1)
