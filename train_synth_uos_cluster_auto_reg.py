"""Train and evaluate k-subspace model on synthetic union of subspaces."""

from __future__ import absolute_import

import argparse
import numpy as np
import train_synth_uos_cluster as tr


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
    args.N = args.n * args.Ng
  else:
    args.Ng = args.N // args.n
    args.N = args.n * args.Ng
  if args.online:
    args.N *= args.epochs
    args.Ng *= args.epochs

  if args.sigma_hat is None or args.sigma_hat < 0:
    args.sigma_hat = args.sigma
  if args.min_size is None or args.min_size < 0:
    args.min_size = 0.01 / args.n
  if args.min_size >= 1:
    raise ValueError("min size {} must be < 1.".format(args.min_size))

  args.U_frosqr_in_lamb = ((1.0 + np.sqrt((args.D - args.d)/args.Ng))**2 *
      (args.sigma_hat**2 / args.D))
  args.U_frosqr_out_lamb = (args.min_size *
      (1.0 / args.d + args.sigma_hat**2 / args.D))

  args.U_fro_out_lamb = 0.0
  args.U_gram_fro_out_lamb = 0.0
  args.z_lamb = 0.01

  # other fixed args
  args.reset_obj = 'full'
  args.eval_rank = True
  return args


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Cluster synthetic UoS data')
  parser.add_argument('--out-dir', type=str, required=True,
                      help='Output directory.')
  # data settings
  parser.add_argument('--n', type=int, default=10,
                      help='Number of subspaces [default: 10]')
  parser.add_argument('--d', type=int, default=10,
                      help='Subspace dimension [default: 10]')
  parser.add_argument('--D', type=int, default=100,
                      help='Ambient dimension [default: 100]')
  parser.add_argument('--Ng', type=int, default=1000,
                      help='Points per group [default: 1000]')
  parser.add_argument('--N', type=int, default=None,
                      help='Alternatively, total data points [default: None]')
  parser.add_argument('--affine', action='store_true',
                      help='Affine setting')
  parser.add_argument('--sigma', type=float, default=0.01,
                      help='Data noise sigma [default: 0.01]')
  parser.add_argument('--data-seed', type=int, default=1904,
                      help='Data random seed [default: 1904]')
  parser.add_argument('--online', action='store_true',
                      help='Online data generation')
  # model settings
  parser.add_argument('--form', type=str, default='proj',
                      help=('Model formulation (proj, mf, batch_alt) '
                      '[default: proj]'))
  parser.add_argument('--reps', type=int, default=5,
                      help='Number of model replicates [default: 5]')
  parser.add_argument('--model-n', type=int, default=None,
                      help='Model number of subspaces [default: n]')
  parser.add_argument('--model-d', type=int, default=None,
                      help='Model subspace dimension [default: d]')
  parser.add_argument('--sigma-hat', type=float, default=None,
                      help='Noise estimate [default: sigma]')
  parser.add_argument('--min-size', type=float, default=None,
                      help=('Minimum cluster size as fraction '
                      '[default: 0.01/n]'))
  # training settings
  parser.add_argument('--batch-size', type=int, default=100,
                      help='Input batch size for training [default: 100]')
  parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs to train [default: 50]')
  parser.add_argument('--optim', type=str, default='SGD',
                      help='Optimizer [default: SGD]')
  parser.add_argument('--init-lr', type=float, default=0.5,
                      help='Initial learning rate [default: 0.5]')
  parser.add_argument('--reset-metric', type=str, default='value',
                      help='Reset metric (value, size, none) [default: value]')
  parser.add_argument('--unused-thr', type=float, default=0.1,
                      help=('Threshold for identifying unused clusters, '
                      '[default: 0.1]'))
  parser.add_argument('--reset-patience', type=int, default=None,
                      help='Steps to wait between resets [default: 1 epoch]')
  parser.add_argument('--reset-decr-tol', type=float, default=1e-4,
                      help=('Relative objective decrease tolerance to reset '
                      '[default: 1e-4]'))
  parser.add_argument('--reset-sigma', type=float, default=0.05,
                      help=('Scale of perturbation to add after reset '
                      '[default: 0.05]'))
  parser.add_argument('--cuda', action='store_true', default=False,
                      help='Enables CUDA training')
  parser.add_argument('--num-threads', type=int, default=1,
                      help='Number of parallel threads to use [default: 1]')
  parser.add_argument('--num-workers', type=int, default=1,
                      help='Number of workers for data loading [default: 1]')
  parser.add_argument('--seed', type=int, default=2018,
                      help='Training random seed [default: 2018]')
  parser.add_argument('--chkp-freq', type=int, default=None,
                      help='How often to save checkpoints [default: None]')
  parser.add_argument('--stop-freq', type=int, default=None,
                      help='How often to stop in ipdb [default: None]')
  args = parser.parse_args()

  args = set_args(args)

  tr.train_synth_uos_cluster(args)
