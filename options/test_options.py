from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')

        # ========================================================================
        # Adaptive Inference Parameters (Di-Fusion Inspired)
        # ========================================================================

        parser.add_argument('--adaptive_inference', action='store_true',
                           help='Use Di-Fusion inspired adaptive inference schedule. '
                                'Applies non-uniform sampling (dense in low-noise, sparse in high-noise).')

        parser.add_argument('--dense_steps_ratio', type=float, default=0.3,
                           help='Ratio of dense sampling steps (0.3 = first 30%% uses stride 1). '
                                'Remaining steps use larger stride for faster inference.')

        parser.add_argument('--sparse_stride', type=int, default=2,
                           help='Stride for sparse sampling in high-noise region. '
                                '2 = every other step (2x speedup), 3 = every third step (3x speedup).')

        parser.add_argument('--early_termination', action='store_true',
                           help='Enable early termination when convergence detected. '
                                'Stops iteration when change between steps falls below threshold.')

        parser.add_argument('--convergence_threshold', type=float, default=0.01,
                           help='Threshold for early termination convergence check. '
                                'Lower = stricter convergence, higher = earlier stopping.')

        parser.add_argument('--save_intermediate', action='store_true',
                           help='Save intermediate results at each timestep for visualization.')

        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser
