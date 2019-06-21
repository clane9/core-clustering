"""Train and evaluate online k-subspace clustering & completion on movie
recommendation datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import datasets as dat
import models as mod
import training as tr
import utils as ut


def main():
  use_cuda = args.cuda and torch.cuda.is_available()
  device = torch.device('cuda' if use_cuda else 'cpu')
  torch.set_num_threads(args.num_threads)

  if args.dataset not in {'nf_17k', 'nf_1k'}:
    raise ValueError("Invalid dataset {}".format(args.dataset))

  # create output directory, deleting any existing results.
  if os.path.exists(args.out_dir):
    shutil.rmtree(args.out_dir)
  os.mkdir(args.out_dir)

  # determines data sampling, initialization
  torch.manual_seed(args.seed)

  # load dataset
  store_dense = not (args.sparse_encode and args.sparse_decode)
  store_sparse = args.sparse_encode or args.sparse_decode
  if args.dataset in {'nf_17k', 'nf_1k'}:
    fname = {'nf_17k': 'nf_prize_446460x16885',
        'nf_1k': 'nf_prize_422889x889'}[args.dataset]
    dataset = dat.NetflixDataset(fname=fname, center=args.center,
        normalize=args.normalize, store_sparse=store_sparse,
        store_dense=store_dense)
  kwargs = {
      'num_workers': args.num_workers,
      'batch_size': args.batch_size,
      'collate_fn': dat.missing_data_collate,
      'shuffle': True,
      'drop_last': True}
  if use_cuda:
    kwargs['pin_memory'] = True
  data_loader = DataLoader(dataset, **kwargs)

  # construct model
  reset_kwargs = dict(reset_patience=args.reset_patience,
      reset_try_tol=args.reset_try_tol,
      reset_max_steps=args.reset_max_steps,
      reset_accept_tol=args.reset_accept_tol,
      reset_sigma=args.reset_sigma,
      reset_cache_size=args.reset_cache_size,
      temp_scheduler=mod.GeoTempScheduler(init_temp=0.1, replicates=args.reps,
          patience=1, gamma=0.9))

  # set reg params based on dataset, using properties of singular value
  # distribution for gaussian random matrices. In particular, know
  # singular values will have a right bulk edge at:
  #     (sqrt{N_j} + sqrt{D - d}) (\sigma / sqrt{D})
  # whereas the data singular values will have a right bulk edge at:
  #     (sqrt{N_j} + sqrt{d}) sqrt{1/ d + \sigma^2 / D}
  # where we assume X = U Y + Z, Y ~ N(0, 1/d I_d), Z ~ N(0, \sigma/D I_D)
  # NOTE: data should be projected to sphere for this to make sense (?)
  if args.sigma_hat is not None and args.normalize:
    Ng = dataset.N / args.model_k
    args.U_frosqr_in_lamb = (
        (1.0 + np.sqrt((dataset.D - args.model_d)/Ng))**2 *
        (args.sigma_hat**2 / dataset.D))
    args.U_frosqr_out_lamb = ((args.min_size / args.model_k) *
        (1.0 / args.model_d + args.sigma_hat**2 / dataset.D))
    args.z_lamb = 0.01

  if args.z_lamb is None or args.z_lamb <= 0:
    args.z_lamb = 0.01
  reg_params = {
      'U_frosqr_in': args.U_frosqr_in_lamb / args.z_lamb,
      'U_frosqr_out': args.U_frosqr_out_lamb / args.z_lamb,
      'z': (args.z_lamb if
          max(args.U_frosqr_in_lamb, args.U_frosqr_out_lamb) > 0
          else 0.0)}

  model = mod.KSubspaceMCModel(args.model_k, args.model_d, dataset.D,
      affine=args.affine, replicates=args.reps, reg_params=reg_params,
      scale_grad_freq=args.scale_grad_freq, sparse_encode=args.sparse_encode,
      sparse_decode=args.sparse_decode, norm_comp_error=args.normalize,
      **reset_kwargs)
  model = model.to(device)

  if args.optim == 'SGD':
    optimizer = torch.optim.SGD(model.parameters(), lr=args.init_lr,
        momentum=0.9, nesterov=True)
  elif args.optim == 'Adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr,
        betas=(0.9, 0.9), amsgrad=True)
  else:
    raise ValueError("Invalid optimizer {}.".format(args.optim))

  min_lr = max(1e-8, 0.1**4 * args.init_lr)
  if args.reset_unused:
    patience = int(np.ceil(5 * args.reset_patience /
        (dataset.N // args.batch_size)))
    threshold = args.reset_try_tol/10
  else:
    patience = 10
    threshold = 1e-4
  scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=patience,
      threshold=threshold, min_lr=min_lr)

  if args.chkp_freq is None or args.chkp_freq <= 0:
    args.chkp_freq = args.epochs + 1
  if args.stop_freq is None or args.stop_freq <= 0:
    args.stop_freq = -1
  if args.epoch_size is None or args.epoch_size <= 0:
    args.epoch_size = len(dataset)
  args.epoch_steps = args.epoch_size // args.batch_size

  # save args
  with open('{}/args.json'.format(args.out_dir), 'w') as f:
    json.dump(args.__dict__, f, sort_keys=True, indent=4)

  tr.train_loop(model, data_loader, device, optimizer, args.out_dir,
      args.epochs, args.chkp_freq, args.stop_freq, scheduler=scheduler,
      epoch_steps=args.epoch_steps, eval_rank=args.eval_rank,
      reset_unused=args.reset_unused, save_data=args.save_large_data)
  return


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      description='Cluster and complete movie rating data')
  parser.add_argument('--out-dir', type=str, required=True,
                      help='Output directory.')
  parser.add_argument('--dataset', type=str, default='nf_1k',
                      help='Real dataset [default: nf_1k].',
                      choices=['nf_1k', 'nf_17k'])
  parser.add_argument('--center', type=ut.boolarg, default=True,
                      help='Center dataset [default: 1].')
  parser.add_argument('--normalize', type=ut.boolarg, default=False,
                      help='Normalize dataset [default: 0].')
  # model settings
  parser.add_argument('--reps', type=int, default=6,
                      help='Number of model replicates [default: 6]')
  parser.add_argument('--model-k', type=int, required=True,
                      help='Model number of subspaces')
  parser.add_argument('--model-d', type=int, required=True,
                      help='Model subspace dimension')
  parser.add_argument('--affine', type=ut.boolarg, default=False,
                      help='Affine setting [default: 0]')
  parser.add_argument('--U-frosqr-in-lamb', type=float, default=0.01,
                      help=('Frobenius squared U reg parameter, '
                      'inside assignment [default: 0.01]'))
  parser.add_argument('--U-frosqr-out-lamb', type=float, default=1e-4,
                      help=('Frobenius squared U reg parameter, '
                      'outside assignment [default: 1e-4]'))
  parser.add_argument('--z-lamb', type=float, default=0.01,
                      help=('L2 squared coefficient reg parameter, '
                      'inside assignment [default: 0.01]'))
  parser.add_argument('--sigma-hat', type=float, default=None,
                      help='Noise estimate [default: None]')
  parser.add_argument('--min-size', type=float, default=0.0,
                      help=('Minimum cluster size as fraction relative to 1/n'
                      '[default: 0.0]'))
  # training settings
  parser.add_argument('--batch-size', type=int, default=100,
                      help='Input batch size for training [default: 100]')
  parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs to train [default: 50]')
  parser.add_argument('--epoch-size', type=int, default=None,
                      help=('Number of samples to consider an "epoch" '
                      '[default: N]'))
  parser.add_argument('--optim', type=str, default='SGD',
                      help='Optimizer [default: SGD]')
  parser.add_argument('--init-lr', type=float, default=0.5,
                      help='Initial learning rate [default: 0.5]')
  parser.add_argument('--scale-grad-freq', type=int, default=20,
                      help=('How often to re-compute local Lipschitz for MF '
                      'formulation [default: 20]'))
  parser.add_argument('--sparse-encode', type=ut.boolarg, default=True,
                      help='Sparse encoding [default: 1]')
  parser.add_argument('--sparse-decode', type=ut.boolarg, default=True,
                      help='Sparse decoding [default: 1]')
  parser.add_argument('--reset-unused', type=ut.boolarg, default=True,
                      help='Reset nearly unused clusters [default: 1]')
  parser.add_argument('--reset-patience', type=int, default=100,
                      help=('Steps to wait without obj decrease '
                      'before trying to reset [default: 100]'))
  parser.add_argument('--reset-try-tol', type=float, default=0.01,
                      help=('Objective decrease tolerance for deciding'
                      'when to reset [default: 0.01]'))
  parser.add_argument('--reset-max-steps', type=int, default=50,
                      help='Number of reset SA iterations [default: 50]')
  parser.add_argument('--reset-accept-tol', type=float, default=0.01,
                      help=('Objective decrease tolerance for accepting'
                      'a reset [default: 0.01]'))
  parser.add_argument('--reset-sigma', type=float, default=0.0,
                      help=('Scale of perturbation to add after reset '
                      '[default: 0.0]'))
  parser.add_argument('--reset-cache-size', type=int, default=500,
                      help='Num samples for reset assign obj [default: 500]')
  parser.add_argument('--cuda', type=ut.boolarg, default=False,
                      help='Enables CUDA training [default: 0]')
  parser.add_argument('--num-threads', type=int, default=1,
                      help='Number of parallel threads to use [default: 1]')
  parser.add_argument('--num-workers', type=int, default=1,
                      help='Number of workers for data loading [default: 1]')
  parser.add_argument('--eval-rank', type=ut.boolarg, default=False,
                      help='Evaluate ranks of subspace models [default: 0]')
  parser.add_argument('--seed', type=int, default=2018,
                      help='Training random seed [default: 2018]')
  parser.add_argument('--chkp-freq', type=int, default=None,
                      help='How often to save checkpoints [default: None]')
  parser.add_argument('--stop-freq', type=int, default=None,
                      help='How often to stop in ipdb [default: None]')
  parser.add_argument('--save-large-data', type=ut.boolarg, default=True,
                      help='Save larger data [default: 1]')
  args = parser.parse_args()

  main()
