"""
Improvements to sb_model.py - Nila + Di-Fusion Inspired

This file contains the modified methods that should replace the corresponding
methods in sb_model.py to implement all noise-adaptive improvements.

To apply:
1. Backup original sb_model.py
2. Apply these modifications to the respective methods
3. Test with baseline config first, then progressive improvements
"""

import numpy as np
import torch
import math


# ============================================================================
# MODIFICATION 1: Enhanced forward() method
# Location: Replace the forward() method in SBModel class
# ============================================================================

def forward_improved(self):
    """
    Enhanced forward pass with:
    - Latter steps training (Di-Fusion)
    - Continuous time sampling (Di-Fusion)
    """
    tau = self.opt.tau
    T = self.opt.num_timesteps

    # 🔥 Di-Fusion Improvement 1: Latter Steps Training
    if hasattr(self.opt, 'latter_steps_ratio') and self.opt.latter_steps_ratio < 1.0:
        T_c = int(T * self.opt.latter_steps_ratio)
        if not hasattr(self, '_latter_steps_logged'):
            print(f"[Di-Fusion] Training latter {T_c}/{T} steps only (ratio={self.opt.latter_steps_ratio})")
            self._latter_steps_logged = True
    else:
        T_c = T

    # Time schedule (unchanged)
    incs = np.array([0] + [1/(i+1) for i in range(T-1)])
    times = np.cumsum(incs)
    times = times / times[-1]
    times = 0.5 * times[-1] + 0.5 * times
    times = np.concatenate([np.zeros(1), times])
    times = torch.tensor(times).float().cuda()
    self.times = times
    bs = self.real_A.size(0)

    # 🔥 Modified: Sample from latter steps only
    time_idx = (torch.randint(T_c, size=[1]).cuda() * torch.ones(size=[1]).cuda()).long()
    self.time_idx = time_idx

    # 🔥 Di-Fusion Improvement 2: Continuous Time Sampling
    if hasattr(self.opt, 'continuous_time_sampling') and self.opt.continuous_time_sampling:
        # Sample continuous alpha value
        if time_idx < T - 1:
            alpha_t = times[time_idx]
            alpha_t_next = times[time_idx + 1]
            continuous_offset = torch.rand(1).cuda()
            self.timestep = alpha_t + continuous_offset * (alpha_t_next - alpha_t)
        else:
            self.timestep = times[time_idx]
    else:
        self.timestep = times[time_idx]

    # === Original forward process (with timestep modification) ===
    with torch.no_grad():
        self.netG.eval()
        for t in range(self.time_idx.int().item() + 1):

            if t > 0:
                delta = times[t] - times[t-1]
                denom = times[-1] - times[t-1]
                inter = (delta / denom).reshape(-1,1,1,1)
                scale = (delta * (1 - delta / denom)).reshape(-1,1,1,1)

            Xt = self.real_A if (t == 0) else (1-inter) * Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * torch.randn_like(Xt).to(self.real_A.device)
            time_idx_iter = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
            time = times[time_idx_iter]
            z = torch.randn(size=[self.real_A.shape[0], 4*self.opt.ngf]).to(self.real_A.device)
            Xt_1 = self.netG(Xt, time_idx_iter, z)

            Xt2 = self.real_A2 if (t == 0) else (1-inter) * Xt2 + inter * Xt_12.detach() + (scale * tau).sqrt() * torch.randn_like(Xt2).to(self.real_A.device)
            time_idx_iter = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
            time = times[time_idx_iter]
            z = torch.randn(size=[self.real_A.shape[0], 4*self.opt.ngf]).to(self.real_A.device)
            Xt_12 = self.netG(Xt2, time_idx_iter, z)

            if self.opt.nce_idt:
                XtB = self.real_B if (t == 0) else (1-inter) * XtB + inter * Xt_1B.detach() + (scale * tau).sqrt() * torch.randn_like(XtB).to(self.real_A.device)
                time_idx_iter = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
                time = times[time_idx_iter]
                z = torch.randn(size=[self.real_A.shape[0], 4*self.opt.ngf]).to(self.real_A.device)
                Xt_1B = self.netG(XtB, time_idx_iter, z)

        if self.opt.nce_idt:
            self.XtB = XtB.detach()
        self.real_A_noisy = Xt.detach()
        self.real_A_noisy2 = Xt2.detach()

    z_in = torch.randn(size=[2*bs, 4*self.opt.ngf]).to(self.real_A.device)
    z_in2 = torch.randn(size=[bs, 4*self.opt.ngf]).to(self.real_A.device)
    """Run forward pass"""
    self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A

    self.realt = torch.cat((self.real_A_noisy, self.XtB), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A_noisy

    if self.opt.flip_equivariance:
        self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
        if self.flipped_for_equivariance:
            self.real = torch.flip(self.real, [3])
            self.realt = torch.flip(self.realt, [3])

    self.fake = self.netG(self.realt, self.time_idx, z_in)
    self.fake_B2 = self.netG(self.real_A_noisy2, self.time_idx, z_in2)
    self.fake_B = self.fake[:self.real_A.size(0)]
    if self.opt.nce_idt:
        self.idt_B = self.fake[self.real_A.size(0):]

    # === Test phase adaptive inference ===
    if self.opt.phase == 'test':
        self._test_phase_adaptive_inference()


def _test_phase_adaptive_inference(self):
    """
    Test phase with adaptive inference (Di-Fusion inspired)
    """
    tau = self.opt.tau
    T = self.opt.num_timesteps
    incs = np.array([0] + [1/(i+1) for i in range(T-1)])
    times = np.cumsum(incs)
    times = times / times[-1]
    times = 0.5 * times[-1] + 0.5 * times
    times = np.concatenate([np.zeros(1), times])
    times = torch.tensor(times).float().cuda()
    self.times = times
    bs = self.real.size(0)

    # 🔥 Adaptive inference schedule
    if hasattr(self.opt, 'adaptive_inference') and self.opt.adaptive_inference:
        dense_steps = int(T * self.opt.dense_steps_ratio)
        schedule = []

        for i in range(T):
            if i < dense_steps:
                schedule.append(i)  # Dense sampling
            else:
                # Sparse sampling
                stride = self.opt.sparse_stride
                mapped = dense_steps + (i - dense_steps) * stride
                if mapped < T:
                    schedule.append(mapped)

        if not hasattr(self, '_adaptive_logged'):
            print(f"[Adaptive Inference] Using {len(schedule)}/{T} steps "
                  f"(dense ratio={self.opt.dense_steps_ratio}, sparse stride={self.opt.sparse_stride})")
            self._adaptive_logged = True
    else:
        schedule = range(T)

    # 🔥 Convergence threshold for early termination
    convergence_threshold = getattr(self.opt, 'convergence_threshold', 0.01) if hasattr(self.opt, 'early_termination') and self.opt.early_termination else None

    with torch.no_grad():
        self.netG.eval()

        for idx, t in enumerate(schedule):
            if t > 0:
                delta = times[t] - times[t-1]
                denom = times[-1] - times[t-1]
                inter = (delta / denom).reshape(-1,1,1,1)
                scale = (delta * (1 - delta / denom)).reshape(-1,1,1,1)

            Xt_prev = Xt.clone() if t > 0 else None

            Xt = self.real_A if (t == 0) else (1-inter) * Xt + inter * Xt_1.detach() + (scale * tau).sqrt() * torch.randn_like(Xt).to(self.real_A.device)
            time_idx = (t * torch.ones(size=[self.real_A.shape[0]]).to(self.real_A.device)).long()
            time = times[time_idx]
            z = torch.randn(size=[self.real_A.shape[0], 4*self.opt.ngf]).to(self.real_A.device)
            Xt_1 = self.netG(Xt, time_idx, z)

            # 🔥 Early termination check
            if convergence_threshold is not None and Xt_prev is not None:
                change = torch.mean((Xt_1 - Xt_prev)**2).item()
                if change < convergence_threshold:
                    print(f"[Early Termination] Converged at step {idx+1}/{len(schedule)} (change={change:.6f})")
                    break

            # Save intermediate if requested
            if hasattr(self.opt, 'save_intermediate') and self.opt.save_intermediate:
                setattr(self, f"fake_{t+1}", Xt_1)


# ============================================================================
# MODIFICATION 2: Enhanced compute_G_loss() method
# Location: Replace compute_G_loss() in SBModel class
# ============================================================================

def compute_G_loss_improved(self):
    """
    Enhanced generator loss with:
    - Nila-inspired noise-adaptive weighting
    - Di-Fusion-inspired timestep weighting
    - Combined adaptive strategy
    """
    bs = self.real_A.size(0)
    tau = self.opt.tau

    fake = self.fake_B

    # === GAN loss (unchanged) ===
    if self.opt.lambda_GAN > 0.0:
        pred_fake = self.netD(fake, self.time_idx)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
    else:
        self.loss_G_GAN = 0.0

    # === SB loss with adaptive weighting ===
    self.loss_SB = 0
    if self.opt.lambda_SB > 0.0:
        XtXt_1 = torch.cat([self.real_A_noisy, self.fake_B], dim=1)
        XtXt_2 = torch.cat([self.real_A_noisy2, self.fake_B2], dim=1)

        bs = self.opt.batch_size

        # Energy term (unchanged)
        ET_XY = self.netE(XtXt_1, self.time_idx, XtXt_1).mean() - \
                torch.logsumexp(self.netE(XtXt_1, self.time_idx, XtXt_2).reshape(-1), dim=0)
        energy_term = -(self.opt.num_timesteps - self.time_idx[0]) / self.opt.num_timesteps * tau * ET_XY

        # Reconstruction loss
        reconstruction_loss = torch.mean((self.real_A_noisy - self.fake_B)**2)

        # 🔥 Compute adaptive weight (Nila + Di-Fusion combined)
        if hasattr(self.opt, 'use_adaptive_sb_weight') and self.opt.use_adaptive_sb_weight:
            t = self.time_idx[0].item()
            T = self.opt.num_timesteps

            # === Nila-inspired: Noise ratio adaptive ===
            t_normalized = t / T
            artificial_noise = np.sqrt(tau * t_normalized * (1 - t_normalized))

            if hasattr(self.opt, 'data_noise_level') and self.opt.data_noise_level > 0:
                noise_ratio = artificial_noise / (self.opt.data_noise_level + 1e-8)

                if self.opt.noise_adaptive_schedule == 'linear':
                    nila_weight = min(noise_ratio, 1.0)
                elif self.opt.noise_adaptive_schedule == 'exponential':
                    nila_weight = min(np.exp(-3.0 * (1 - noise_ratio)), 1.0) if noise_ratio < 1.0 else 1.0
                elif self.opt.noise_adaptive_schedule == 'step':
                    nila_weight = 1.0 if noise_ratio > 0.5 else 0.0
                else:
                    nila_weight = 1.0
            else:
                nila_weight = 1.0

            # === Di-Fusion-inspired: Timestep adaptive ===
            if hasattr(self.opt, 'difusion_weight_schedule') and self.opt.difusion_weight_schedule != 'none':
                if self.opt.difusion_weight_schedule == 'linear':
                    difusion_weight = 1.0 - (t / T)
                elif self.opt.difusion_weight_schedule == 'quadratic':
                    difusion_weight = (1.0 - (t / T)) ** 2
                elif self.opt.difusion_weight_schedule == 'exponential':
                    difusion_weight = np.exp(-2.0 * t / T)
                else:
                    difusion_weight = 1.0
            else:
                difusion_weight = 1.0

            # Combined weight
            adaptive_weight = nila_weight * difusion_weight

            # Store for monitoring
            self.nila_weight = nila_weight
            self.difusion_weight = difusion_weight
            self.adaptive_weight = adaptive_weight
        else:
            adaptive_weight = 1.0
            self.adaptive_weight = 1.0

        # Apply adaptive weight to reconstruction term
        reconstruction_term = adaptive_weight * tau * reconstruction_loss

        self.loss_SB = energy_term + reconstruction_term

        # Store components for monitoring
        self.loss_SB_energy = energy_term
        self.loss_SB_recon = reconstruction_term

    # === NCE loss (unchanged) ===
    if self.opt.lambda_NCE > 0.0:
        self.loss_NCE = self.calculate_NCE_loss(self.real_A, fake)
    else:
        self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

    if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
        self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
        loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
    else:
        loss_NCE_both = self.loss_NCE

    # === Total loss ===
    self.loss_G = self.loss_G_GAN + self.opt.lambda_SB * self.loss_SB + self.opt.lambda_NCE * loss_NCE_both
    return self.loss_G


# ============================================================================
# MODIFICATION 3: Enhanced __init__() method
# Location: Add to __init__() in SBModel class
# ============================================================================

def init_improvements(self):
    """
    Add these lines to the __init__() method after loss_names definition
    """
    # Add adaptive weight monitoring to loss names
    if hasattr(self.opt, 'use_adaptive_sb_weight') and self.opt.use_adaptive_sb_weight:
        if 'SB' in self.loss_names:
            # Add decomposed SB losses
            idx = self.loss_names.index('SB')
            self.loss_names.insert(idx + 1, 'SB_energy')
            self.loss_names.insert(idx + 2, 'SB_recon')

    # Visualize noise schedule if requested
    if hasattr(self.opt, 'visualize_noise_schedule') and self.opt.visualize_noise_schedule:
        self._visualize_noise_schedule()


def _visualize_noise_schedule(self):
    """Visualize the adaptive weight schedule"""
    import matplotlib.pyplot as plt

    T = self.opt.num_timesteps
    tau = self.opt.tau
    t_range = np.arange(T)

    # Compute noise levels
    t_normalized = t_range / T
    artificial_noise = np.sqrt(tau * t_normalized * (1 - t_normalized))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Plot 1: Noise levels
    ax1.plot(t_range, artificial_noise, 'b-', linewidth=2, label='Artificial noise σ_t')
    if hasattr(self.opt, 'data_noise_level') and self.opt.data_noise_level > 0:
        ax1.axhline(y=self.opt.data_noise_level, color='r', linestyle='--',
                   linewidth=2, label=f'Data noise σ_y={self.opt.data_noise_level}')
    ax1.set_xlabel('Timestep t')
    ax1.set_ylabel('Noise level')
    ax1.set_title('Noise Levels')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Adaptive weights
    if hasattr(self.opt, 'use_adaptive_sb_weight') and self.opt.use_adaptive_sb_weight:
        weights = []
        for t in t_range:
            # Nila weight
            noise_at_t = np.sqrt(tau * (t/T) * (1 - t/T))
            if self.opt.data_noise_level > 0:
                ratio = noise_at_t / self.opt.data_noise_level
                nila_w = min(ratio, 1.0)
            else:
                nila_w = 1.0

            # Di-Fusion weight
            if self.opt.difusion_weight_schedule == 'linear':
                difusion_w = 1.0 - (t / T)
            elif self.opt.difusion_weight_schedule == 'quadratic':
                difusion_w = (1.0 - (t / T)) ** 2
            elif self.opt.difusion_weight_schedule == 'exponential':
                difusion_w = np.exp(-2.0 * t / T)
            else:
                difusion_w = 1.0

            weights.append(nila_w * difusion_w)

        ax2.plot(t_range, weights, 'g-', linewidth=2)
        ax2.set_xlabel('Timestep t')
        ax2.set_ylabel('Adaptive weight λ_t')
        ax2.set_title('Combined Adaptive Weight')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.1])

    plt.tight_layout()
    save_path = f'noise_schedule_{self.opt.name}.png'
    plt.savefig(save_path, dpi=150)
    print(f"[Visualization] Saved noise schedule to {save_path}")
    plt.close()


# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================
"""
To apply these modifications to sb_model.py:

1. Backup original file:
   cp models/sb_model.py models/sb_model.py.backup

2. Replace methods in SBModel class:
   - Replace forward() with forward_improved()
   - Add _test_phase_adaptive_inference() method
   - Replace compute_G_loss() with compute_G_loss_improved()
   - Add init_improvements() to __init__()
   - Add _visualize_noise_schedule() method

3. Test progressively:
   - Start with baseline (no new flags)
   - Add --latter_steps_ratio 0.6
   - Add --use_adaptive_sb_weight --data_noise_level 0.03
   - Add other flags as needed

See experiment configs for full examples.
"""
