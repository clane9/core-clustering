"""Global command line configuration set-up."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import utils as ut


global_args_dict = {
    'out-dir': dict(type=str, required=True,
        help='Output directory.'),

    # data settings
    'img-dataset': dict(type=str, default='mnist',
        help='Real image dataset [default: mnist].',
        choices=['mnist', 'cifar10', 'coil100', 'coil20', 'yaleb']),
    'mc-dataset': dict(type=str, default='nf_1k',
        help='Real matrix completion dataset [default: nf_1k].',
        choices=['nf_1k', 'nf_17k']),
    'k': dict(type=int, default=10,
        help='Number of subspaces [default: 10]'),
    'd': dict(type=int, default=10,
        help='Subspace dimension [default: 10]'),
    'D': dict(type=int, default=100,
        help='Ambient dimension [default: 100]'),
    'Ng': dict(type=int, default=1000,
        help='Points per group [default: 1000]'),
    'N': dict(type=int, default=None,
        help='Total data points [default: k*Ng]'),
    'affine': dict(type=ut.boolarg, default=False,
        help='Affine setting [default: 0]'),
    'sigma': dict(type=float, default=0.4,
        help='Data noise sigma [default: 0.4]'),
    'sep': dict(type=float, default=2.0,
        help=('k-means separation in gaussian radius units '
            '[default: 2.0]')),
    'theta': dict(type=float, default=None,
        help='Principal angles between subspaces [default: None]'),
    'miss-rate': dict(type=float, default=0.0,
        help='Data missing rate [default: 0.0]'),
    'online': dict(type=ut.boolarg, default=False,
        help='Online data generation [default: 0]'),
    'center': dict(type=ut.boolarg, default=False,
        help='Center dataset [default: 0].'),
    'normalize': dict(type=ut.boolarg, default=False,
        help='Project data onto sphere [default: 0]'),
    'sv-range': dict(nargs=2, type=int, default=None,
        help=('Singular vector range for pseudo-whitening '
            '[default: None].')),

    # model settings
    'form': dict(type=str, default='mf',
        help='Model formulation [default: mf]',
        choices=['mf', 'proj']),
    'init': dict(type=str, default='random',
        help='Initialization [default: random]',
        choices=['random', 'pca', 'pfi']),
    'kpp-n-trials': dict(type=int, default=None,
        help='Number of k-means++ (pfi) samples [default: 2 log n]'),
    'model-k': dict(type=int, default=None,
        help='Model number of subspaces [default: k]'),
    'model-d': dict(type=int, default=None,
        help='Model subspace dimension [default: d]'),
    'auto-reg': dict(type=ut.boolarg, default=True,
        help='Auto regularization mode [default: 1]'),
    'sigma-hat': dict(type=float, default=None,
        help='Noise estimate [default: sigma]'),
    'min-size': dict(type=float, default=0.0,
        help=('Minimum cluster size as fraction relative to 1/n '
            '[default: 0.0]')),
    'U-frosqr-in-lamb': dict(type=float, default=0.001,
        help=('Frobenius squared U reg parameter, inside assignment '
            '[default: 0.001]')),
    'U-frosqr-out-lamb': dict(type=float, default=0.0,
        help=('Frobenius squared U reg parameter, outside assignment '
            '[default: 0]')),
    'z-lamb': dict(type=float, default=0.01,
        help=('L2 squared coefficient reg parameter, inside assignment '
            '[default: 0.01]')),
    'b-frosqr-out-lamb': dict(type=float, default=0.0,
        help=('Frobenius squared b reg parameter, outside assignment '
            '[default: 0.0]')),

    # training settings
    'epochs': dict(type=int, default=20,
        help='Number of epochs to train [default: 20]'),
    'epoch-size': dict(type=int, default=None,
        help='Number of samples to consider an "epoch" [default: N]'),
    'optim': dict(type=str, default='sgd',
        help='Optimizer [default: sgd]',
        choices=['sgd', 'adam', 'batch-alt']),
    'init-lr': dict(type=float, default=0.1,
        help='Initial learning rate [default: 0.1]'),
    'lr-step-size': dict(type=int, default=None,
        help='How often to decay LR [default: None]'),
    'lr-gamma': dict(type=float, default=0.1,
        help='How much to decay LR [default: 0.1]'),
    'lr-min': dict(type=float, default=1e-8,
        help='Minimum LR [default: 1e-8]'),
    'init-bs': dict(type=int, default=100,
        help='Initial batch size [default: 100]'),
    'bs-step-size': dict(type=int, default=None,
        help='How often to increase batch size [default: None]'),
    'bs-gamma': dict(type=int, default=4,
        help='How much to increase batch size [default: 4]'),
    'bs-max': dict(type=int, default=8192,
        help='Maximum batch size [default: 8192]'),
    'scale-grad-mode': dict(type=str, default=None,
        help='SGD gradient scaling mode for MF formulation [default: None]',
        choices=['none', 'lip', 'newton']),
    'scale-grad-update-freq': dict(type=int, default=20,
        help='How often to re-compute gradient scaling [default: 20]'),
    'sparse-encode': dict(type=ut.boolarg, default=True,
        help='Sparse encoding in MC setting [default: 1]'),
    'sparse-decode': dict(type=ut.boolarg, default=True,
        help='Sparse decoding in MC setting  [default: 1]'),

    # reset settings
    'reps': dict(type=int, default=6,
        help='Number of model replicates [default: 6]'),
    'core-reset': dict(type=ut.boolarg, default=True,
        help='Re-initialize using CoRe [default: 1]'),
    'reset-patience': dict(type=int, default=100,
        help=('Steps to wait without obj decrease before trying to reset '
            '[default: 100]')),
    'reset-try-tol': dict(type=float, default=0.01,
        help=('Objective decrease tolerance for deciding when to reset, '
            'set <= 0 to disable [default: 0.01]')),
    'reset-steps': dict(type=int, default=50,
        help='Number of reset terations [default: 50]'),
    'reset-accept-tol': dict(type=float, default=0.001,
        help=('Objective decrease tolerance for accepting a reset '
            '[default: 0.001]')),
    'reset-cache-size': dict(type=int, default=500,
        help='Num samples for reset assign obj [default: 500]'),

    # general configuration
    'cuda': dict(type=ut.boolarg, default=False,
        help='Enables CUDA training [default: 0]'),
    'num-threads': dict(type=int, default=1,
        help='Number of parallel threads to use [default: 1]'),
    'num-workers': dict(type=int, default=1,
        help='Number of workers for data loading [default: 1]'),
    'data-seed': dict(type=int, default=2001,
        help='Data random seed [default: 2001]'),
    'seed': dict(type=int, default=2018,
        help='Training random seed [default: 2018]'),
    'eval-rank': dict(type=ut.boolarg, default=False,
        help='Evaluate ranks of subspace models [default: 0]'),
    'chkp-freq': dict(type=int, default=None,
        help='How often to save checkpoints [default: None]'),
    'stop-freq': dict(type=int, default=None,
        help='How often to stop in ipdb [default: None]'),
    'save-large-data': dict(type=ut.boolarg, default=True,
        help='Save larger data [default: 1]'),
    'config': dict(type=str, default=None,
        help=('Json file containing args (command line takes precedent) '
          '[default: None]')),
}


def add_args(parser, args):
  """Add a list of args from the global args_dict to a parser."""
  for arg in args:
    parser.add_argument('--' + arg, **global_args_dict[arg])
  return parser
