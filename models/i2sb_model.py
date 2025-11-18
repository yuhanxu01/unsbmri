"""
I2SB Model: Image-to-Image Schrödinger Bridge for Paired MRI Reconstruction
Based on the paper "Guided MRI Reconstruction via Schrödinger Bridge"

This implementation uses paired data (source X, target Y) to train a conditional
diffusion model that learns the optimal transport from X to Y.

Key differences from unpaired SB:
- Uses paired data exclusively
- Source image X is used as guidance condition
- Forward process: Y_0 -> Y_T (add noise to target)
- Reverse process: Y_T + X -> Y_0 (denoise with source as condition)
- Simplified loss: direct diffusion matching loss
"""

import numpy as np
import torch
import torch.nn as nn
from .base_model import BaseModel
from . import networks
import util.util as util


class I2SBModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Configure options specific for I2SB model"""

        # I2SB specific parameters
        parser.add_argument('--i2sb_num_timesteps', type=int, default=1000,
                          help='number of diffusion timesteps')
        parser.add_argument('--i2sb_beta_schedule', type=str, default='linear',
                          choices=['linear', 'cosine', 'quadratic'],
                          help='noise schedule type')
        parser.add_argument('--i2sb_beta_start', type=float, default=1e-4,
                          help='beta start value')
        parser.add_argument('--i2sb_beta_end', type=float, default=0.02,
                          help='beta end value')
        parser.add_argument('--i2sb_objective', type=str, default='pred_noise',
                          choices=['pred_noise', 'pred_x0', 'pred_v'],
                          help='training objective: predict noise, x0, or v')

        # Loss weights
        parser.add_argument('--lambda_diffusion', type=float, default=1.0,
                          help='weight for diffusion loss')
        parser.add_argument('--lambda_vlb', type=float, default=0.0,
                          help='weight for variational lower bound')
        parser.add_argument('--lambda_simple', type=float, default=1.0,
                          help='weight for simple MSE loss')
        parser.add_argument('--lambda_perceptual', type=float, default=0.0,
                          help='weight for perceptual loss (LPIPS)')
        parser.add_argument('--lambda_l1', type=float, default=0.0,
                          help='weight for L1 loss on x0 prediction')

        # Optional GAN loss for quality improvement
        parser.add_argument('--use_gan', type=util.str2bool, nargs='?', const=True, default=False,
                          help='whether to use GAN discriminator')
        parser.add_argument('--lambda_gan', type=float, default=0.1,
                          help='weight for GAN loss')

        # Conditioning strategy
        parser.add_argument('--condition_method', type=str, default='concat',
                          choices=['concat', 'cross_attention', 'film'],
                          help='how to condition on source image')

        # Sampling parameters
        parser.add_argument('--i2sb_sampling_timesteps', type=int, default=250,
                          help='number of sampling steps (can be < training steps)')
        parser.add_argument('--i2sb_ddim_sampling_eta', type=float, default=0.0,
                          help='eta for DDIM sampling (0=deterministic)')

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # Specify training losses
        self.loss_names = ['diffusion']
        if opt.lambda_vlb > 0:
            self.loss_names.append('vlb')
        if opt.lambda_l1 > 0:
            self.loss_names.append('l1')
        if opt.lambda_perceptual > 0:
            self.loss_names.append('perceptual')
        if opt.use_gan:
            self.loss_names.extend(['G_GAN', 'D_real', 'D_fake'])

        # Specify visual outputs
        self.visual_names = ['source', 'target', 'generated']

        # Specify models
        if self.isTrain:
            self.model_names = ['G']
            if opt.use_gan:
                self.model_names.append('D')
        else:
            self.model_names = ['G']

        # Define the generator (U-Net with conditioning)
        # Input channels: output_nc (target) + input_nc (source condition)
        if opt.condition_method == 'concat':
            effective_input_nc = opt.input_nc + opt.output_nc
        else:
            effective_input_nc = opt.output_nc

        self.netG = networks.define_G(
            effective_input_nc, opt.output_nc, opt.ngf,
            opt.netG, opt.normG, not opt.no_dropout,
            opt.init_type, opt.init_gain, opt.no_antialias,
            opt.no_antialias_up, self.gpu_ids, opt
        )

        # Define discriminator if using GAN
        if self.isTrain and opt.use_gan:
            self.netD = networks.define_D(
                opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D,
                opt.normD, opt.init_type, opt.init_gain,
                opt.no_antialias, self.gpu_ids, opt
            )

        # Initialize diffusion parameters
        self._setup_diffusion_parameters()

        if self.isTrain:
            # Define loss functions
            self.criterionL1 = nn.L1Loss()
            self.criterionMSE = nn.MSELoss()

            if opt.use_gan:
                self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)

            # Initialize perceptual loss if needed
            if opt.lambda_perceptual > 0:
                self._init_perceptual_loss()

            # Setup optimizers
            self.optimizer_G = torch.optim.Adam(
                self.netG.parameters(),
                lr=opt.lr,
                betas=(opt.beta1, opt.beta2)
            )
            self.optimizers.append(self.optimizer_G)

            if opt.use_gan:
                self.optimizer_D = torch.optim.Adam(
                    self.netD.parameters(),
                    lr=opt.lr,
                    betas=(opt.beta1, opt.beta2)
                )
                self.optimizers.append(self.optimizer_D)

    def _setup_diffusion_parameters(self):
        """Initialize diffusion process parameters (betas, alphas, etc.)"""
        opt = self.opt
        timesteps = opt.i2sb_num_timesteps

        # Define beta schedule
        if opt.i2sb_beta_schedule == 'linear':
            betas = np.linspace(opt.i2sb_beta_start, opt.i2sb_beta_end, timesteps)
        elif opt.i2sb_beta_schedule == 'cosine':
            betas = self._cosine_beta_schedule(timesteps)
        elif opt.i2sb_beta_schedule == 'quadratic':
            betas = np.linspace(opt.i2sb_beta_start**0.5, opt.i2sb_beta_end**0.5, timesteps)**2

        # Pre-compute diffusion parameters
        betas = torch.from_numpy(betas).float()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        # Register as buffers (will be moved to device automatically)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Calculations for diffusion q(x_t | x_0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)
        self.register_buffer('posterior_log_variance_clipped',
                           torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                           betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                           (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod))

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
        steps = timesteps + 1
        x = np.linspace(0, timesteps, steps)
        alphas_cumprod = np.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return np.clip(betas, 0, 0.999)

    def _init_perceptual_loss(self):
        """Initialize LPIPS perceptual loss"""
        try:
            import lpips
            self.lpips_loss = lpips.LPIPS(net='alex').to(self.device)
            self.lpips_loss.eval()
            for param in self.lpips_loss.parameters():
                param.requires_grad = False
        except ImportError:
            print("Warning: LPIPS not available. Install with: pip install lpips")
            self.opt.lambda_perceptual = 0.0

    def data_dependent_initialize(self, data, data2):
        """Initialize model with data (required by base model)"""
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data, data2)
        self.source = self.source[:bs_per_gpu]
        self.target = self.target[:bs_per_gpu]

        if self.opt.isTrain:
            # Do a forward pass to initialize any adaptive parameters
            self.forward()
            self.compute_G_loss().backward()
            if self.opt.use_gan:
                self.compute_D_loss().backward()

    def set_input(self, input, input2=None):
        """Unpack input data from dataloader"""
        AtoB = self.opt.direction == 'AtoB'
        self.source = input['A' if AtoB else 'B'].to(self.device)  # X (guidance)
        self.target = input['B' if AtoB else 'A'].to(self.device)  # Y (target)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        # For I2SB, we don't need input2 (second dataloader)
        # since we're using paired data exclusively

    def forward(self):
        """
        Forward pass for training:
        1. Sample timestep t
        2. Add noise to target Y_0 to get Y_t (forward diffusion)
        3. Predict noise/x0 conditioned on source X
        """
        batch_size = self.source.size(0)

        # Sample random timesteps
        t = torch.randint(0, self.opt.i2sb_num_timesteps, (batch_size,), device=self.device).long()
        self.t = t

        # Add noise to target (forward diffusion): q(y_t | y_0)
        noise = torch.randn_like(self.target)
        self.noise = noise

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, self.target.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, self.target.shape)

        # y_t = sqrt(alpha_bar_t) * y_0 + sqrt(1 - alpha_bar_t) * epsilon
        self.noisy_target = sqrt_alphas_cumprod_t * self.target + sqrt_one_minus_alphas_cumprod_t * noise

        # Prepare input for network: condition on source
        if self.opt.condition_method == 'concat':
            model_input = torch.cat([self.noisy_target, self.source], dim=1)
        else:
            model_input = self.noisy_target

        # Predict using the network
        # Note: netG needs to accept time embedding
        # For now, we'll use the existing architecture which takes time_idx
        # We need to adapt this
        time_idx = (t.float() / self.opt.i2sb_num_timesteps * self.opt.num_timesteps).long()
        time_idx = torch.clamp(time_idx, 0, self.opt.num_timesteps - 1)

        # Generate dummy z for compatibility with existing netG
        z = torch.randn(batch_size, 4 * self.opt.ngf, device=self.device)

        # Network prediction
        self.model_output = self.netG(model_input, time_idx, z)

        # Interpret model output based on objective
        if self.opt.i2sb_objective == 'pred_noise':
            self.pred_noise = self.model_output
            # Predict x0 from noise prediction
            self.pred_x0 = self._predict_x0_from_noise(self.noisy_target, t, self.pred_noise)
        elif self.opt.i2sb_objective == 'pred_x0':
            self.pred_x0 = self.model_output
            # Predict noise from x0 prediction
            self.pred_noise = self._predict_noise_from_x0(self.noisy_target, t, self.pred_x0)
        elif self.opt.i2sb_objective == 'pred_v':
            # v-prediction: v = sqrt(alpha_bar_t) * epsilon - sqrt(1-alpha_bar_t) * x0
            v = self.model_output
            self.pred_x0 = sqrt_alphas_cumprod_t * self.noisy_target - sqrt_one_minus_alphas_cumprod_t * v
            self.pred_noise = sqrt_alphas_cumprod_t * v + sqrt_one_minus_alphas_cumprod_t * self.noisy_target

        # For visualization
        self.generated = self.pred_x0.detach()

    def _extract(self, a, t, x_shape):
        """Extract appropriate t index for a batch of indices"""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def _predict_x0_from_noise(self, x_t, t, noise):
        """Predict x0 from noise prediction"""
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t

    def _predict_noise_from_x0(self, x_t, t, x0):
        """Predict noise from x0 prediction"""
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        return (x_t - sqrt_alphas_cumprod_t * x0) / sqrt_one_minus_alphas_cumprod_t

    def compute_G_loss(self):
        """Calculate losses for generator"""

        # Main diffusion loss
        if self.opt.i2sb_objective == 'pred_noise':
            target = self.noise
            prediction = self.pred_noise
        elif self.opt.i2sb_objective == 'pred_x0':
            target = self.target
            prediction = self.pred_x0
        elif self.opt.i2sb_objective == 'pred_v':
            sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, self.t, self.target.shape)
            sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, self.t, self.target.shape)
            target = sqrt_alphas_cumprod_t * self.noise - sqrt_one_minus_alphas_cumprod_t * self.target
            prediction = self.model_output

        # Simple MSE loss
        self.loss_diffusion = self.criterionMSE(prediction, target) * self.opt.lambda_simple

        # Optional L1 loss on x0 prediction
        if self.opt.lambda_l1 > 0:
            self.loss_l1 = self.criterionL1(self.pred_x0, self.target) * self.opt.lambda_l1
        else:
            self.loss_l1 = 0.0

        # Optional perceptual loss
        if self.opt.lambda_perceptual > 0 and hasattr(self, 'lpips_loss'):
            # LPIPS expects input in [-1, 1] range
            # If using magnitude, convert to 3 channels
            pred_x0_3ch = self.pred_x0.repeat(1, 3, 1, 1) if self.pred_x0.size(1) == 1 else self.pred_x0
            target_3ch = self.target.repeat(1, 3, 1, 1) if self.target.size(1) == 1 else self.target
            self.loss_perceptual = self.lpips_loss(pred_x0_3ch, target_3ch).mean() * self.opt.lambda_perceptual
        else:
            self.loss_perceptual = 0.0

        # Optional GAN loss
        if self.opt.use_gan:
            # Use the same time conditioning for discriminator
            time_idx = (self.t.float() / self.opt.i2sb_num_timesteps * self.opt.num_timesteps).long()
            time_idx = torch.clamp(time_idx, 0, self.opt.num_timesteps - 1)
            pred_fake = self.netD(self.pred_x0, time_idx)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_gan
        else:
            self.loss_G_GAN = 0.0

        # Total generator loss
        self.loss_G = self.loss_diffusion + self.loss_l1 + self.loss_perceptual + self.loss_G_GAN

        return self.loss_G

    def compute_D_loss(self):
        """Calculate GAN loss for discriminator"""
        if not self.opt.use_gan:
            return 0.0

        time_idx = (self.t.float() / self.opt.i2sb_num_timesteps * self.opt.num_timesteps).long()
        time_idx = torch.clamp(time_idx, 0, self.opt.num_timesteps - 1)

        # Fake
        pred_fake = self.netD(self.pred_x0.detach(), time_idx)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()

        # Real
        pred_real = self.netD(self.target, time_idx)
        self.loss_D_real = self.criterionGAN(pred_real, True).mean()

        # Combined
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        return self.loss_D

    def optimize_parameters(self):
        """Update network weights"""
        # Forward pass
        self.forward()

        # Update discriminator
        if self.opt.use_gan:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.loss_D = self.compute_D_loss()
            self.loss_D.backward()
            self.optimizer_D.step()
            self.set_requires_grad(self.netD, False)

        # Update generator
        self.optimizer_G.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()

    @torch.no_grad()
    def sample(self, source, num_steps=None, eta=None):
        """
        Generate target image from source using DDIM sampling

        Args:
            source: source/guidance image (X)
            num_steps: number of sampling steps (default: use training steps)
            eta: DDIM eta parameter (0=deterministic, 1=DDPM)

        Returns:
            generated target image
        """
        if num_steps is None:
            num_steps = self.opt.i2sb_sampling_timesteps
        if eta is None:
            eta = self.opt.i2sb_ddim_sampling_eta

        batch_size = source.size(0)

        # Start from pure noise
        y_t = torch.randn_like(source)

        # Create time schedule
        times = torch.linspace(self.opt.i2sb_num_timesteps - 1, 0, num_steps, dtype=torch.long, device=self.device)

        for i, t in enumerate(times):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

            # Prepare input
            if self.opt.condition_method == 'concat':
                model_input = torch.cat([y_t, source], dim=1)
            else:
                model_input = y_t

            # Convert to time_idx for existing network
            time_idx = (t_batch.float() / self.opt.i2sb_num_timesteps * self.opt.num_timesteps).long()
            time_idx = torch.clamp(time_idx, 0, self.opt.num_timesteps - 1)

            # Dummy z
            z = torch.randn(batch_size, 4 * self.opt.ngf, device=self.device)

            # Predict
            model_output = self.netG(model_input, time_idx, z)

            # Get x0 prediction
            if self.opt.i2sb_objective == 'pred_noise':
                pred_noise = model_output
                pred_x0 = self._predict_x0_from_noise(y_t, t_batch, pred_noise)
            elif self.opt.i2sb_objective == 'pred_x0':
                pred_x0 = model_output
                pred_noise = self._predict_noise_from_x0(y_t, t_batch, pred_x0)
            else:  # pred_v
                sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t_batch, y_t.shape)
                sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t_batch, y_t.shape)
                pred_x0 = sqrt_alphas_cumprod_t * y_t - sqrt_one_minus_alphas_cumprod_t * model_output

            # DDIM sampling step
            if i < len(times) - 1:
                t_next = times[i + 1]
                t_next_batch = torch.full((batch_size,), t_next, device=self.device, dtype=torch.long)
                y_t = self._ddim_step(y_t, t_batch, t_next_batch, pred_x0, pred_noise, eta)
            else:
                y_t = pred_x0

        return y_t

    def _ddim_step(self, x_t, t, t_next, pred_x0, pred_noise, eta):
        """Single DDIM sampling step"""
        alpha_t = self._extract(self.alphas_cumprod, t, x_t.shape)
        alpha_t_next = self._extract(self.alphas_cumprod, t_next, x_t.shape)

        sigma_t = eta * torch.sqrt((1 - alpha_t_next) / (1 - alpha_t) * (1 - alpha_t / alpha_t_next))

        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_t_next - sigma_t**2) * pred_noise

        # Random noise
        noise = torch.randn_like(x_t) if eta > 0 else 0

        # Compute x_{t-1}
        x_t_next = torch.sqrt(alpha_t_next) * pred_x0 + dir_xt + sigma_t * noise

        return x_t_next

    def compute_paired_metrics(self):
        """Compute SSIM, PSNR, and NRMSE metrics"""
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio as psnr
        from skimage.metrics import normalized_root_mse as nrmse

        if not hasattr(self, 'generated') or not hasattr(self, 'target'):
            return {'ssim': 0.0, 'psnr': 0.0, 'nrmse': 0.0}

        def tensor_to_numpy(tensor):
            img = tensor.detach().cpu().float().numpy()
            if img.ndim == 4:
                return img
            elif img.ndim == 3:
                if img.shape[0] == 1:
                    img = img[0]
                elif img.shape[0] == 2:
                    real, imag = img[0], img[1]
                    img = np.sqrt(real * real + imag * imag)
                else:
                    img = img[0]
            return img

        generated_np = tensor_to_numpy(self.generated)
        target_np = tensor_to_numpy(self.target)

        batch_size = generated_np.shape[0] if generated_np.ndim == 4 else 1

        ssim_vals, psnr_vals, nrmse_vals = [], [], []

        for i in range(batch_size):
            if generated_np.ndim == 4:
                gen_img = generated_np[i]
                tgt_img = target_np[i]
            else:
                gen_img = generated_np
                tgt_img = target_np

            if gen_img.ndim == 3:
                if gen_img.shape[0] == 1:
                    gen_img = gen_img[0]
                    tgt_img = tgt_img[0]
                elif gen_img.shape[0] == 2:
                    gen_img = np.sqrt(gen_img[0]**2 + gen_img[1]**2)
                    tgt_img = np.sqrt(tgt_img[0]**2 + tgt_img[1]**2)

            data_range = tgt_img.max() - tgt_img.min()
            if data_range == 0:
                data_range = 1.0

            try:
                ssim_val = ssim(tgt_img, gen_img, data_range=data_range)
                psnr_val = psnr(tgt_img, gen_img, data_range=data_range)
                nrmse_val = nrmse(tgt_img, gen_img, normalization='mean')

                ssim_vals.append(float(ssim_val))
                psnr_vals.append(float(psnr_val))
                nrmse_vals.append(float(nrmse_val))
            except Exception as e:
                print(f"Warning: Failed to compute metrics: {e}")
                ssim_vals.append(0.0)
                psnr_vals.append(0.0)
                nrmse_vals.append(1.0)

        return {
            'ssim': float(np.mean(ssim_vals)),
            'psnr': float(np.mean(psnr_vals)),
            'nrmse': float(np.mean(nrmse_vals))
        }
