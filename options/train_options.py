from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=None, help='window id of the web display. Default is random window id')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--evaluation_freq', type=int, default=5000, help='evaluation freq')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--pretrained_name', type=str, default=None, help='resume training from another checkpoint')

        # training parameters
        parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=200, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        # Paired training parameters
        parser.add_argument('--paired_stage', action='store_true', help='Enable paired training mode (requires matching A/B slices)')
        parser.add_argument('--paired_subset_ratio', type=float, default=1.0, help='Ratio of paired data to use (0.0-1.0). For two-stage training, use 0.3 in stage 2')
        parser.add_argument('--paired_subset_seed', type=int, default=42, help='Random seed for paired subset selection')
        parser.add_argument('--compute_paired_metrics', action='store_true', help='Compute and log SSIM/PSNR/NRMSE metrics during training (requires paired_stage)')

        # Paired training strategy (modular design for multiple schemes)
        parser.add_argument('--paired_strategy', type=str, default='none',
                          choices=['none', 'sb_gt_transport', 'l1_loss', 'nce_feature', 'frequency', 'gradient', 'multiscale', 'selfsup_contrast', 'hybrid'],
                          help='Strategy for using paired data:\n'
                               '  none: No paired training (default unpaired)\n'
                               '  sb_gt_transport: [Scheme A] Use GT in SB transport cost\n'
                               '  l1_loss: [Baseline] Add simple L1 loss\n'
                               '  nce_feature: [B1] Enhanced NCE in feature space\n'
                               '  frequency: [B2] Frequency domain (FFT) loss\n'
                               '  gradient: [B3] Gradient/structure loss\n'
                               '  multiscale: [B4] Multi-scale pyramid loss\n'
                               '  selfsup_contrast: [B5] Self-supervised contrastive\n'
                               '  hybrid: Combine multiple strategies')

        # Strategy-specific parameters
        parser.add_argument('--lambda_L1', type=float, default=1.0, help='[l1_loss] Weight for L1 loss')
        parser.add_argument('--lambda_reg', type=float, default=1.0, help='[B1-B5] Weight for regularization losses')

        # Loss control options for paired experiments (ablation studies)
        parser.add_argument('--use_ot_input', action='store_true', help='Use OT input loss: tau * mean((real_A_noisy - real_B)^2)')
        parser.add_argument('--use_ot_output', action='store_true', help='Use OT output loss: tau * mean((fake_B - real_B)^2)')
        parser.add_argument('--use_entropy_loss', action='store_true', help='Use entropy loss (ET_XY term from SB formulation)')
        parser.add_argument('--disable_gan', action='store_true', help='Disable GAN adversarial loss')
        parser.add_argument('--disable_nce', action='store_true', help='Disable NCE contrastive loss')

        self.isTrain = True
        return parser
