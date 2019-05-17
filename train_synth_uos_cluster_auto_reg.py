"""Train and evaluate k-subspace model on synthetic union of subspaces."""

from __future__ import absolute_import

import argparse
import numpy as np
import train_synth_uos_cluster as tr

import utils as ut


def set_args(args):
  """Compute "optimal" regularization parameters based on expected singular
  value distribution.

  Key resource is (Gavish & Donoho, 2014). In general, we know that the noise
  singular values will have a right bulk edge at:
      (sqrt{N_j} + sqrt{D - d}) (\sigma / sqrt{D})
  whereas the data singular values will have a right bulk edge at:
      (sqrt{N_j} + sqrt{d}) sqrt{1/ d + \sigma^2 / D}
  The regularization parameters are set so that the "inside reg" will threshold
  all noise svs, whereas the "outside reg" will threshold all noise + data svs
  as soon as the cluster becomes too small.
  """
  if args.N is None:
    args.N = args.k * args.Ng
  else:
    args.Ng = args.N // args.k

  if args.sigma_hat is None or args.sigma_hat < 0:
    args.sigma_hat = args.sigma
  if args.min_size is None or args.min_size < 0:
    args.min_size = 0.01
  if args.min_size >= 1:
    raise ValueError("min size {} should be < 1.".format(args.min_size))

  Ng_total = min(10, args.epochs)*args.Ng if args.online else args.Ng
  args.U_frosqr_in_lamb = ((1.0 + np.sqrt((args.D - args.d)/Ng_total))**2 *
      (args.sigma_hat**2 / args.D))
  args.U_frosqr_out_lamb = ((args.min_size / args.k) *
      (1.0 / args.d + args.sigma_hat**2 / args.D))

  # other fixed args
  args.z_lamb = 0.01
  args.eval_rank = True
  return args


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Cluster synthetic UoS data')
  parser.add_argument('--out-dir', type=str, required=True,
                      help='Output directory.')
  # data settings
  parser.add_argument('--k', type=int, default=10,
                      help='Number of subspaces [default: 10]')
  parser.add_argument('--d', type=int, default=10,
                      help='Subspace dimension [default: 10]')
  parser.add_argument('--D', type=int, default=100,
                      help='Ambient dimension [default: 100]')
  parser.add_argument('--Ng', type=int, default=1000,
                      help='Points per group [default: 1000]')
  parser.add_argument('--N', type=int, default=None,
                      help='Alternatively, total data points [default: None]')
  parser.add_argument('--affine', type=ut.boolarg, default=False,
                      help='Affine setting [default: 0]')
  parser.add_argument('--sigma', type=float, default=0.01,
                      help='Data noise sigma [default: 0.01]')
  parser.add_argument('--theta', type=float, default=None,
                      help=('Principal angles between subspaces '
                      '[default: None]'))
  parser.add_argument('--miss-rate', type=float, default=0.0,
                      help='Data missing rate [default: 0.0]')
  parser.add_argument('--normalize', type=ut.boolarg, default=False,
                      help='Project data onto sphere [default: 0]')
  parser.add_argument('--data-seed', type=int, default=1904,
                      help='Data random seed [default: 1904]')
  parser.add_argument('--online', type=ut.boolarg, default=False,
                      help='Online data generation [default: 0]')
  # model settings
  parser.add_argument('--form', type=str, default='mf',
                      help=('Model formulation (proj, mf, batch-alt-proj, '
                      'batch-alt-mf) [default: mf]'))
  parser.add_argument('--prob-farthest-insert', type=ut.boolarg, default=False,
                      help=('Initialize by probabilistic farthest insertion '
                      '[default: 0]'))
  parser.add_argument('--reps', type=int, default=6,
                      help='Number of model replicates [default: 6]')
  parser.add_argument('--model-k', type=int, default=None,
                      help='Model number of subspaces [default: k]')
  parser.add_argument('--model-d', type=int, default=None,
                      help='Model subspace dimension [default: d]')
  parser.add_argument('--sigma-hat', type=float, default=None,
                      help='Noise estimate [default: sigma]')
  parser.add_argument('--min-size', type=float, default=None,
                      help=('Minimum cluster size as fraction relative to 1/n'
                      '[default: 0.01]'))
  # training settings
  parser.add_argument('--batch-size', type=int, default=100,
                      help='Input batch size for training [default: 100]')
  parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs to train [default: 50]')
  parser.add_argument('--optim', type=str, default='SGD',
                      help='Optimizer [default: SGD]')
  parser.add_argument('--init-lr', type=float, default=1.0,
                      help='Initial learning rate [default: 1.0]')
  parser.add_argument('--scale-grad-freq', type=int, default=100,
                      help=('How often to re-compute local Lipschitz for MF '
                      'formulation [default: 100]'))
  parser.add_argument('--reset-unused', type=ut.boolarg, default=False,
                      help='Reset nearly unused clusters [default: 0]')
  parser.add_argument('--reset-patience', type=int, default=50,
                      help=('Steps to wait without obj decrease '
                      'before trying to reset [default: 50]'))
  parser.add_argument('--reset-jitter', type=ut.boolarg, default=True,
                      help='Jitter reset patience [default: 1]')
  parser.add_argument('--reset-low-value', type=ut.boolarg, default=True,
                      help='Choose low value cluster to reset [default: 1]')
  parser.add_argument('--reset-value-thr', type=float, default=0.1,
                      help=('Value threshold for low value clusters '
                      '[default: 0.1]'))
  parser.add_argument('--reset-stochastic', type=ut.boolarg, default=False,
                      help=('Choose reset substitution stochastically '
                      '[default: 0]'))
  parser.add_argument('--reset-try-tol', type=float, default=0.01,
                      help=('Objective decrease tolerance for deciding'
                      'when to reset [default: 0.01]'))
  parser.add_argument('--reset-accept-tol', type=float, default=0.01,
                      help=('Objective decrease tolerance for accepting'
                      'a reset [default: 0.01]'))
  parser.add_argument('--reset-sigma', type=float, default=0.05,
                      help=('Scale of perturbation to add after reset '
                      '[default: 0.05]'))
  parser.add_argument('--reset-cache-size', type=int, default=None,
                      help=('Num samples for reset assign obj '
                      '[default: 4 k log k]'))
  parser.add_argument('--cuda', type=ut.boolarg, default=False,
                      help='Enables CUDA training [default: 0]')
  parser.add_argument('--num-threads', type=int, default=1,
                      help='Number of parallel threads to use [default: 1]')
  parser.add_argument('--num-workers', type=int, default=1,
                      help='Number of workers for data loading [default: 1]')
  parser.add_argument('--seed', type=int, default=2018,
                      help='Training random seed [default: 2018]')
  parser.add_argument('--eval-rank', type=ut.boolarg, default=False,
                      help='Evaluate ranks of subspace models [default: 0]')
  parser.add_argument('--chkp-freq', type=int, default=None,
                      help='How often to save checkpoints [default: None]')
  parser.add_argument('--stop-freq', type=int, default=None,
                      help='How often to stop in ipdb [default: None]')
  parser.add_argument('--save-large-data', type=ut.boolarg, default=True,
                      help='Save larger data [default: 1]')
  args = parser.parse_args()

  args = set_args(args)

  tr.train_synth_uos_cluster(args)
