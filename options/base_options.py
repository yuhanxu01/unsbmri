import argparse
import os
from util import util
import torch
import models
import data


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self, cmd_line=None):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False
        self.cmd_line = None
        if cmd_line is not None:
            self.cmd_line = cmd_line.split()

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot', default='placeholder', help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--easy_label', type=str, default='experiment_name', help='Interpretable name')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        # model parameters
        parser.add_argument('--model', type=str, default='sb', help='chooses which model to use.')
        parser.add_argument('--input_nc', type=int, default=1, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=1, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--num_timesteps', type=int, default=5, help='# of discrim filters in the first conv layer')
        parser.add_argument('--embedding_dim', type=int, default=512, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--netD', type=str, default='basic_cond', choices=['basic', 'n_layers', 'pixel', 'basic_cond'], help='specify discriminator architecture')
        parser.add_argument('--netE', type=str, default='basic_cond', choices=['basic', 'n_layers', 'pixel', 'basic_cond'], help='specify energy network architecture')
        parser.add_argument('--netG', type=str, default='resnet_9blocks_cond', choices=['resnet_9blocks', 'resnet_6blocks', 'unet_256', 'unet_128', 'resnet_9blocks_cond'], help='specify generator architecture')
        parser.add_argument('--embedding_type', type=str, default='positional', choices=['fourier', 'positional'], help='time embedding type')
        parser.add_argument('--n_layers_D', type=int, default=3, help='number of layers in discriminator')
        parser.add_argument('--n_mlp', type=int, default=3, help='number of MLP layers for time embedding')
        parser.add_argument('--normG', type=str, default='instance', choices=['instance', 'batch', 'none'], help='instance normalization or batch normalization for G')
        parser.add_argument('--normD', type=str, default='instance', choices=['instance', 'batch', 'none'], help='instance normalization or batch normalization for D')
        parser.add_argument('--init_type', type=str, default='xavier', choices=['normal', 'xavier', 'kaiming', 'orthogonal'], help='network initialization')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', type=util.str2bool, nargs='?', const=True, default=True)
        parser.add_argument('--std', type=float, default=0.25, help='Scale of Gaussian noise added to data')
        parser.add_argument('--tau', type=float, default=0.01, help='Entropy parameter')
        parser.add_argument('--no_antialias', action='store_true', help='if specified, use stride=2 convs instead of antialiased-downsampling (sad)')
        parser.add_argument('--no_antialias_up', action='store_true', help='if specified, use [upconv(learned filter)] instead of [upconv(hard-coded [1,3,3,1] filter), conv]')
        parser.add_argument('--no_tanh', action='store_true', help='if specified, remove Tanh activation from generator output (for MRI data not in [-1,1] range)')
        # dataset parameters
        parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single | colorization]')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        parser.add_argument('--random_scale_max', type=float, default=3.0,
                            help='(used for single image translation) Randomly scale the image by the specified factor as data augmentation.')
        # wandb parameters (required for logging)
        parser.add_argument('--wandb_project', type=str, default='mri-contrast-transfer', help='wandb project name')
        parser.add_argument('--wandb_run_id', type=str, default=None, help='wandb run id for resuming')
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')

        # ========================================================================
        # Noise-Adaptive Training Parameters (Nila + Di-Fusion Inspired)
        # ========================================================================

        # Nila-inspired parameters
        parser.add_argument('--data_noise_level', type=float, default=0.0,
                           help='Estimated noise level in the data (σ_y). '
                                'Set to 0 to disable noise-adaptive loss (default). '
                                'Use noise_estimation.py to estimate this value. '
                                'Typical values: 0.01-0.05 for high quality, 0.05-0.10 for fast scans.')

        parser.add_argument('--noise_adaptive_schedule', type=str, default='linear',
                           choices=['linear', 'exponential', 'step', 'none'],
                           help='Schedule for Nila-style noise-adaptive weight decay. '
                                'linear: λ=ratio (recommended), '
                                'exponential: λ=exp(-k*(1-ratio)), '
                                'step: threshold-based, '
                                'none: disable.')

        parser.add_argument('--noise_adaptive_start_epoch', type=int, default=0,
                           help='Start applying noise-adaptive loss after this epoch. '
                                'Useful for curriculum learning (train normally first, then adapt).')

        # Di-Fusion-inspired parameters
        parser.add_argument('--latter_steps_ratio', type=float, default=1.0,
                           help='Ratio of latter diffusion steps to train (Di-Fusion inspired). '
                                '1.0 = train all steps (default UNSB), '
                                '0.6 = train latter 60%% (recommended for denoising), '
                                '0.3 = train latter 30%% (aggressive, for strong denoising). '
                                'Focuses training on refinement rather than generation.')

        parser.add_argument('--difusion_weight_schedule', type=str, default='none',
                           choices=['linear', 'quadratic', 'exponential', 'none'],
                           help='Di-Fusion inspired schedule for timestep-dependent SB weight. '
                                'linear: 1-t/T (emphasize latter steps), '
                                'quadratic: (1-t/T)^2 (more aggressive), '
                                'exponential: exp(-2t/T) (smooth decay), '
                                'none: uniform weight (default).')

        parser.add_argument('--use_adaptive_sb_weight', action='store_true',
                           help='Use combined Nila + Di-Fusion adaptive weighting for SB reconstruction loss. '
                                'Combines noise-ratio weighting with timestep weighting.')

        parser.add_argument('--continuous_time_sampling', action='store_true',
                           help='Use continuous timestep sampling (Di-Fusion inspired). '
                                'Samples alpha values continuously within [alpha_{t-1}, alpha_t] '
                                'instead of discrete timesteps for smoother training.')

        # Data augmentation for denoising
        parser.add_argument('--denoise_augmentation', action='store_true',
                           help='Apply denoising data augmentation to target domain B. '
                                'Generates pseudo-clean samples to guide discriminator toward cleaner outputs.')

        parser.add_argument('--denoise_method', type=str, default='lowpass',
                           choices=['lowpass', 'wavelet', 'bilateral', 'nlm'],
                           help='Method for denoising augmentation. '
                                'lowpass: Gaussian low-pass filter (fast), '
                                'wavelet: Wavelet soft-thresholding (better), '
                                'bilateral: Edge-preserving bilateral filter, '
                                'nlm: Non-local means (slow but best).')

        parser.add_argument('--denoise_prob', type=float, default=0.5,
                           help='Probability of applying denoising augmentation to each sample. '
                                '0.5 = 50%% clean, 50%% original (recommended).')

        parser.add_argument('--denoise_sigma', type=float, default=1.5,
                           help='Sigma parameter for denoising methods (controls smoothing strength).')

        # Visualization and monitoring
        parser.add_argument('--visualize_noise_schedule', action='store_true',
                           help='Visualize the noise-adaptive schedule at start of training. '
                                'Generates a plot showing how weights change with timesteps.')

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        if self.cmd_line is None:
            opt, _ = parser.parse_known_args()
        else:
            opt, _ = parser.parse_known_args(self.cmd_line)

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        if self.cmd_line is None:
            opt, _ = parser.parse_known_args()  # parse again with new defaults
        else:
            opt, _ = parser.parse_known_args(self.cmd_line)  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        if self.cmd_line is None:
            return parser.parse_args()
        else:
            return parser.parse_args(self.cmd_line)

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        try:
            with open(file_name, 'wt') as opt_file:
                opt_file.write(message)
                opt_file.write('\n')
        except PermissionError as error:
            print("permission error {}".format(error))
            pass

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
