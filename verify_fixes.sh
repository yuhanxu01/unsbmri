#!/bin/bash
# Verification script to check if all fixes are applied
# Run this on the HPC cluster to verify the code is up to date

echo "========================================"
echo "Verifying Code Fixes"
echo "========================================"
echo ""

FAIL=0

# Check 1: Verify git branch and commit
echo "1. Checking git branch and latest commit..."
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
LATEST_COMMIT=$(git log -1 --format="%h %s")
echo "   Current branch: $CURRENT_BRANCH"
echo "   Latest commit: $LATEST_COMMIT"

if [[ "$LATEST_COMMIT" != *"Fix optimizer_F AttributeError"* ]]; then
    echo "   ❌ FAIL: Code is NOT up to date!"
    echo "   Expected latest commit to contain 'Fix optimizer_F AttributeError'"
    FAIL=1
else
    echo "   ✓ PASS: Latest commit is correct"
fi
echo ""

# Check 2: Verify tensor fix in compute_G_loss
echo "2. Checking compute_G_loss() tensor fixes..."
if grep -q "self.loss_G_GAN = torch.tensor(0.0, device=self.real_A.device)" models/sb_model.py; then
    echo "   ✓ PASS: loss_G_GAN tensor fix found"
else
    echo "   ❌ FAIL: loss_G_GAN still using scalar 0.0"
    FAIL=1
fi

if grep -q "self.loss_SB = torch.tensor(0.0, device=self.real_A.device)" models/sb_model.py; then
    echo "   ✓ PASS: loss_SB tensor fix found"
else
    echo "   ❌ FAIL: loss_SB still using scalar 0"
    FAIL=1
fi

if grep -q "self.loss_NCE = torch.tensor(0.0, device=self.real_A.device)" models/sb_model.py; then
    echo "   ✓ PASS: loss_NCE tensor fix found"
else
    echo "   ❌ FAIL: loss_NCE still using scalar 0.0"
    FAIL=1
fi

# Check critical OT_input gradient fix
if grep -q "self.loss_OT_input = self.opt.tau \* torch.mean((self.real_A_noisy - self.real_B)\*\*2)" models/sb_model.py; then
    echo "   ✓ PASS: loss_OT_input uses real_B (supervises intermediate state)"
else
    echo "   ❌ FAIL: loss_OT_input definition incorrect"
    FAIL=1
fi

# Check gradient-enabled forward diffusion
if grep -q "compute_noisy_with_grad = use_ot_input and self.opt.isTrain" models/sb_model.py; then
    echo "   ✓ PASS: Gradient-enabled forward diffusion for OT_input"
else
    echo "   ❌ FAIL: Missing gradient-enabled forward diffusion"
    FAIL=1
fi
echo ""

# Check 3: Verify optimizer_F fix in data_dependent_initialize
echo "3. Checking data_dependent_initialize() NCE fix..."
if grep -q "if self.opt.lambda_NCE > 0.0 and not getattr(self.opt, 'disable_nce', False):" models/sb_model.py | head -1; then
    echo "   ✓ PASS: optimizer_F creation has disable_nce check"
else
    echo "   ❌ FAIL: optimizer_F creation missing disable_nce check"
    FAIL=1
fi
echo ""

# Check 4: Verify optimize_parameters refactoring
echo "4. Checking optimize_parameters() refactoring..."
if grep -q "use_gan = self.opt.lambda_GAN > 0.0 and not getattr(self.opt, 'disable_gan', False)" models/sb_model.py; then
    echo "   ✓ PASS: use_gan flag found in optimize_parameters()"
else
    echo "   ❌ FAIL: optimize_parameters() not refactored with use_gan flag"
    FAIL=1
fi

if grep -q "use_nce = self.opt.lambda_NCE > 0.0 and not getattr(self.opt, 'disable_nce', False)" models/sb_model.py; then
    echo "   ✓ PASS: use_nce flag found in optimize_parameters()"
else
    echo "   ❌ FAIL: optimize_parameters() not refactored with use_nce flag"
    FAIL=1
fi
echo ""

# Check 5: Verify epoch configuration in two-stage experiments
echo "5. Checking two-stage experiment epoch configuration..."
if grep -q "export N_EPOCHS=500" experiments/ablation_studies/exp5_twostage_10p_OT_input.sh; then
    echo "   ✓ PASS: exp5 has N_EPOCHS=500"
else
    echo "   ❌ FAIL: exp5 still has N_EPOCHS=100"
    FAIL=1
fi

if grep -q "export N_EPOCHS=500" experiments/ablation_studies/exp9_twostage_100p_OT_input.sh; then
    echo "   ✓ PASS: exp9 has N_EPOCHS=500"
else
    echo "   ❌ FAIL: exp9 still has N_EPOCHS=100"
    FAIL=1
fi
echo ""

# Final result
echo "========================================"
if [ $FAIL -eq 0 ]; then
    echo "✓ ALL CHECKS PASSED"
    echo "Code is up to date with all fixes applied!"
    echo "You can now run the experiments."
else
    echo "❌ SOME CHECKS FAILED"
    echo "Code is NOT up to date. Please update following the instructions below."
fi
echo "========================================"

# If failed, show update instructions
if [ $FAIL -ne 0 ]; then
    echo ""
    echo "To update the code, run these commands:"
    echo ""
    echo "  cd /gpfs/scratch/rl5285/test/unsbmri"
    echo "  git fetch origin"
    echo "  git reset --hard origin/claude/setup-mri-training-pipeline-01SPqpGQe22LVbdgKBHDkPF1"
    echo "  bash verify_fixes.sh  # Run this script again to verify"
    echo ""
fi

exit $FAIL
