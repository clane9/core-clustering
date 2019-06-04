"""Train and evaluate online k-subspace clustering on range of real
datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import json
import time

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

  if args.dataset not in {'mnist', 'coil100', 'coil20', 'yaleb'}:
    raise ValueError("Invalid dataset {}".format(args.dataset))
  if args.form not in {'batch-alt-proj', 'batch-alt-mf', 'proj', 'mf'}:
    raise ValueError("Invalid form {}".format(args.form))
  batch_alt_mode = args.form in {'batch-alt-proj', 'batch-alt-mf'}

  # create output directory, deleting any existing results.
  if os.path.exists(args.out_dir):
    shutil.rmtree(args.out_dir)
  os.mkdir(args.out_dir)

  # determines data sampling, initialization
  torch.manual_seed(args.seed)

  # load dataset
  dataset = dat.ImageUoSDataset(args.dataset, center=args.center,
      sv_range=args.sv_range, normalize=args.normalize)
  kwargs = {'num_workers': args.num_workers}
  if use_cuda:
    kwargs['pin_memory'] = True
  data_loader = DataLoader(dataset, batch_size=args.batch_size,
      shuffle=True, drop_last=True, **kwargs)

  # set reg params based on dataset, using properties of singular value
  # distribution for gaussian random matrices. In particular, know
  # singular values will have a right bulk edge at:
  #     (sqrt{N_j} + sqrt{D - d}) (\sigma / sqrt{D})
  # whereas the data singular values will have a right bulk edge at:
  #     (sqrt{N_j} + sqrt{d}) sqrt{1/ d + \sigma^2 / D}
  # where we assume X = U Y + Z, Y ~ N(0, 1/d I_d), Z ~ N(0, \sigma/D I_D)
  # NOTE: data should be projected to sphere for this to make sense (?)
  Ng = dataset.N / args.model_k
  args.U_frosqr_in_lamb = ((1.0 + np.sqrt((dataset.D - args.model_d)/Ng))**2 *
      (args.sigma_hat**2 / dataset.D))
  args.U_frosqr_out_lamb = ((args.min_size / args.model_k) *
      (1.0 / args.model_d + args.sigma_hat**2 / dataset.D))
  args.z_lamb = 0.01

  # construct model
  reset_kwargs = dict(reset_patience=args.reset_patience,
      reset_try_tol=args.reset_try_tol,
      reset_max_steps=args.reset_max_steps,
      reset_accept_tol=args.reset_accept_tol,
      reset_sigma=args.reset_sigma,
      reset_cache_size=args.reset_cache_size,
      temp_scheduler=mod.GeoTempScheduler(init_temp=0.1, replicates=args.reps,
          patience=1, gamma=0.9))

  if args.serial_eval is None or args.serial_eval.lower() == 'none':
    args.serial_eval = []
  else:
    args.serial_eval = args.serial_eval.lower().strip().split(',')

  if args.init.lower() in {'pfi', 'pca'}:
    initN = 100 * int(np.ceil(
        args.model_d*args.model_k*np.log(args.model_k) / 100))
    if initN < dataset.N:
      Idx = torch.randperm(dataset.N)[:initN]
      initX = dataset.X[Idx]
    else:
      initX = dataset.X
  else:
    initX = None

  tic = time.time()
  if args.form == 'batch-alt-proj':
    reg_params = {
        'U_frosqr_in': args.U_frosqr_in_lamb,
        'U_frosqr_out': args.U_frosqr_out_lamb
    }
    model = mod.KSubspaceBatchAltProjModel(args.model_k, args.model_d,
        dataset, affine=args.affine, replicates=args.reps,
        reg_params=reg_params, serial_eval=args.serial_eval,
        svd_solver='randomized', **reset_kwargs)
  elif args.form == 'batch-alt-mf':
    reg_params = {
        'U_frosqr_in': args.U_frosqr_in_lamb / args.z_lamb,
        'U_frosqr_out': args.U_frosqr_out_lamb / args.z_lamb,
        'z': (args.z_lamb if
            max(args.U_frosqr_in_lamb, args.U_frosqr_out_lamb) > 0
            else 0.0)
    }
    model = mod.KSubspaceBatchAltMFModel(args.model_k, args.model_d,
        dataset, affine=args.affine, replicates=args.reps,
        reg_params=reg_params, serial_eval=args.serial_eval,
        svd_solver='randomized', init=args.init, initX=initX,
        initk=args.init_k, **reset_kwargs)
  elif args.form == 'proj':
    reg_params = {
        'U_frosqr_in': args.U_frosqr_in_lamb,
        'U_frosqr_out': args.U_frosqr_out_lamb
    }
    model = mod.KSubspaceProjModel(args.model_k, args.model_d, dataset.D,
        affine=args.affine, replicates=args.reps, reg_params=reg_params,
        serial_eval=args.serial_eval, **reset_kwargs)
  else:
    reg_params = {
        'U_frosqr_in': args.U_frosqr_in_lamb / args.z_lamb,
        'U_frosqr_out': args.U_frosqr_out_lamb / args.z_lamb,
        'z': (args.z_lamb if
            max(args.U_frosqr_in_lamb, args.U_frosqr_out_lamb) > 0
            else 0.0)
    }
    model = mod.KSubspaceMFModel(args.model_k, args.model_d, dataset.D,
        affine=args.affine, replicates=args.reps, reg_params=reg_params,
        serial_eval=args.serial_eval, scale_grad_freq=args.scale_grad_freq,
        init=args.init, initX=initX, initk=args.init_k, **reset_kwargs)
  init_time = time.time() - tic
  model = model.to(device)

  # optimizer
  if batch_alt_mode:
    optimizer = None
    scheduler = None
  else:
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
      patience = int(np.ceil(3 * args.reset_patience /
          (dataset.N // args.batch_size)))
      threshold = args.reset_try_tol/10
    else:
      patience = 10
      threshold = 1e-3
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=patience,
        threshold=threshold, min_lr=min_lr)

  if args.chkp_freq is None or args.chkp_freq <= 0:
    args.chkp_freq = args.epochs + 1
  if args.stop_freq is None or args.stop_freq <= 0:
    args.stop_freq = -1

  # save args
  with open('{}/args.json'.format(args.out_dir), 'w') as f:
    json.dump(args.__dict__, f, sort_keys=True, indent=4)

  tr.train_loop(model, data_loader, device, optimizer, args.out_dir,
      args.epochs, args.chkp_freq, args.stop_freq, scheduler=scheduler,
      eval_rank=args.eval_rank, reset_unused=args.reset_unused,
      save_data=args.save_large_data, init_time=init_time)
  return


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Cluster real UoS data')
  parser.add_argument('--out-dir', type=str, required=True,
                      help='Output directory.')
  parser.add_argument('--dataset', type=str, default='mnist',
                      help='Real dataset [default: mnist].',
                      choices=['mnist', 'coil100', 'coil20', 'yaleb'])
  parser.add_argument('--center', type=ut.boolarg, default=True,
                      help='Center dataset [default: 1].')
  parser.add_argument('--sv-range', nargs=2, type=int, default=None,
                      help=('Singular vector range for whitening '
                      '[default: None].'))
  parser.add_argument('--normalize', type=ut.boolarg, default=True,
                      help='Normalize dataset [default: 1].')
  # model settings
  parser.add_argument('--form', type=str, default='mf',
                      help=('Model formulation (proj, mf, batch-alt-proj, '
                      'batch-alt-mf) [default: mf]'))
  parser.add_argument('--init', type=str, default='random',
                      help=('Initialization (random, pca, pfi) '
                      '[default: random]'))
  parser.add_argument('--init-k', type=int, default=None,
                      help=('Number of clusters to initialize non-zero '
                      '[default: k]'))
  parser.add_argument('--reps', type=int, default=6,
                      help='Number of model replicates [default: 6]')
  parser.add_argument('--model-k', type=int, required=True,
                      help='Model number of subspaces')
  parser.add_argument('--model-d', type=int, required=True,
                      help='Model subspace dimension')
  parser.add_argument('--affine', type=ut.boolarg, default=False,
                      help='Affine setting [default: 0]')
  parser.add_argument('--sigma-hat', type=float, default=0.1,
                      help='Noise estimate [default: 0.1]')
  parser.add_argument('--min-size', type=float, default=0.01,
                      help=('Minimum cluster size as fraction relative to 1/k'
                      '[default: 0.01]'))
  # training settings
  parser.add_argument('--batch-size', type=int, default=100,
                      help='Input batch size for training [default: 100]')
  parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs to train [default: 50]')
  parser.add_argument('--optim', type=str, default='SGD',
                      help='Optimizer [default: SGD]')
  parser.add_argument('--init-lr', type=float, default=0.5,
                      help='Initial learning rate [default: 0.5]')
  parser.add_argument('--scale-grad-freq', type=int, default=20,
                      help=('How often to re-compute local Lipschitz for MF '
                      'formulation [default: 20]'))
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
  parser.add_argument('--reset-cache-size', type=int, default=None,
                      help=('Num samples for reset assign obj '
                      '[default: 4 k log k]'))
  parser.add_argument('--serial-eval', type=str, default=None,
                      help=('Serial evaluation mode, one of '
                      '(none, r, k, r,k) [default: None]'))
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
