"""Train and evaluate core-clustering model on variety of datasets."""

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

from corecluster import configuration as conf
from corecluster import datasets as dat
from corecluster import models as mod
from corecluster import training as tr
from corecluster import utils as ut


def eval_core_clustering(args):
  # setting some potentially undefined args depending on setting
  default_args = [('init', 'random'), ('form', 'mf'), ('miss_rate', 0.0),
      ('eval_rank', False)]
  for arg, val in default_args:
    setattr(args, arg, getattr(args, arg, val))

  # error checking, in no particular order
  if args.setting == 'synth-uos' and args.online and args.optim == 'batch-alt':
    raise ValueError(("Online mode not compatible with batch-alt "
        "optimization."))
  mc_mode = ((args.setting == 'synth-uos' and args.miss_rate > 0) or
      args.setting == 'movie-mc-uos')
  if mc_mode:
    if args.optim == 'batch-alt' or args.form == 'proj':
      raise ValueError(("batch-alt optimization and proj form not compatible "
          " with missing data."))
    if args.init != 'random':
      raise ValueError("only random initialization supported in MC setting.")
  uos_mode = args.setting != 'synth-kmeans'
  if uos_mode:
    if args.form == 'proj' and args.scale_grad_mode in {'lip', 'newton'}:
      raise ValueError("proj formulation not compatible with grad scaling.")
    if args.affine and args.scale_grad_mode in {'lip', 'newton'}:
      raise ValueError("affine setting not compatible with grad scaling.")
    if args.form == 'proj' and args.init != 'random':
      raise ValueError(("proj formulation only compatible with random "
          "initialization."))
  if not uos_mode and args.init == 'pca':
    raise ValueError("pca initialization not compatible with k-means.")
  real_data_mode = args.setting in {'img-uos', 'movie-mc-uos'}
  if real_data_mode:
    if args.model_d is None or args.model_d <= 0:
      raise ValueError("model dimension is actually required for real data.")

  use_cuda = args.cuda and torch.cuda.is_available()
  device = torch.device('cuda' if use_cuda else 'cpu')
  torch.set_num_threads(args.num_threads)

  # create output directory, deleting any existing results.
  if os.path.exists(args.out_dir):
    shutil.rmtree(args.out_dir)
  os.makedirs(args.out_dir)

  # construct/load dataset
  torch.manual_seed(args.data_seed)

  # Make N, Ng consistent, N takes precedent.
  if args.setting in {'synth-uos', 'synth-kmeans'}:
    if args.N is not None and args.N > 0:
      args.Ng = args.N // args.k
    else:
      args.N = args.Ng * args.k

  # for missing data only
  if args.setting in {'synth-uos', 'movie-mc-uos'}:
    store_dense = not (args.sparse_encode and args.sparse_decode)
    store_sparse = args.sparse_encode or args.sparse_decode

  # shuffling not necessary for online data.
  shuffle_data = not (args.setting == 'synth-uos' and args.online)

  if args.setting == 'synth-uos':
    dataset = dat.generate_synth_uos_dataset(args.k, args.d, args.D, args.Ng,
        N=args.N, affine=args.affine, sigma=args.sigma, theta=args.theta,
        miss_rate=args.miss_rate, normalize=args.normalize, online=args.online,
        miss_store_sparse=store_sparse, miss_store_dense=store_dense,
        seed=args.data_seed)
  elif args.setting == 'img-uos':
    dataset = dat.ImageUoSDataset(args.img_dataset, center=args.center,
        sv_range=args.sv_range, normalize=args.normalize)
  elif args.setting == 'movie-mc-uos':
    dataset = dat.NetflixDataset(dataset=args.mc_dataset, center=args.center,
        normalize=args.normalize, store_sparse=store_sparse,
        store_dense=store_dense)
  elif args.setting == 'synth-kmeans':
    dataset = dat.SynthKMeansDataset(args.k, args.D, args.Ng,
        separation=args.sep, seed=args.data_seed)
  else:
    raise ValueError("Invalid setting {}".format(args.setting))

  if args.setting == 'synth-kmeans' or args.optim == 'batch-alt':
    data_loader = None
  else:
    kwargs = {'num_workers': args.num_workers,
        'batch_size': args.batch_size, 'shuffle': shuffle_data,
        'drop_last': True, 'pin_memory': use_cuda}
    if mc_mode:
      kwargs['collate_fn'] = dat.missing_data_collate
    data_loader = DataLoader(dataset, **kwargs)

  # construct model
  torch.manual_seed(args.seed)
  model_tic = time.time()

  if uos_mode and (args.model_d is None or args.model_d <= 0):
    args.model_d = args.d
  if args.model_k is None or args.model_k <= 0:
    args.model_k = dataset.k

  # note auto-reg only makes sense when data are unit norm.
  if uos_mode:
    if args.auto_reg:
      if args.sigma_hat is None or args.sigma_hat < 0:
        args.sigma_hat = args.sigma
      (args.U_frosqr_in_lamb, args.U_frosqr_out_lamb,
          args.z_lamb) = ut.set_auto_reg_params(args.model_k, args.model_d,
              dataset.D, dataset.N/args.model_k, args.sigma_hat, args.min_size)
    if args.form == 'mf':
      if args.z_lamb > 0:
        args.U_frosqr_in_lamb /= args.z_lamb
        args.U_frosqr_out_lamb /= args.z_lamb
      reg_params = {'U_frosqr_in': args.U_frosqr_in_lamb,
          'U_frosqr_out': args.U_frosqr_out_lamb,
          'z': args.z_lamb}
    else:
      reg_params = {'U_frosqr_in': args.U_frosqr_in_lamb,
          'U_frosqr_out': args.U_frosqr_out_lamb}
  else:
    reg_params = {'b_frosqr_out': args.b_frosqr_out_lamb}

  if args.init in {'pfi', 'pca'}:
    initN = args.reset_cache_size
    if initN < dataset.N:
      Idx = torch.randperm(dataset.N)[:initN]
      initX = dataset.X[Idx]
    else:
      initX = dataset.X
    initX = initX.to(device)
  else:
    initX = None

  reset_kwargs = dict(reset_patience=args.reset_patience,
      reset_try_tol=args.reset_try_tol,
      reset_max_steps=args.reset_steps,
      reset_accept_tol=args.reset_accept_tol,
      reset_cache_size=args.reset_cache_size,
      temp_scheduler=mod.ConstantTempScheduler(init_temp=args.reset_temp))

  if uos_mode:
    if args.optim == 'batch-alt':
      if args.form == 'mf':
        model = mod.KSubspaceBatchAltMFModel(args.model_k, args.model_d,
            dataset, affine=args.affine, replicates=args.reps,
            reg_params=reg_params, svd_solver='randomized', init=args.init,
            **reset_kwargs)
      else:
        model = mod.KSubspaceBatchAltProjModel(args.model_k, args.model_d,
            dataset, affine=args.affine, replicates=args.reps,
            reg_params=reg_params, svd_solver='randomized', **reset_kwargs)
    else:
      if args.miss_rate > 0:
        model = mod.KSubspaceMCModel(args.model_k, args.model_d, dataset.D,
            affine=args.affine, replicates=args.reps, reg_params=reg_params,
            scale_grad_mode=args.scale_grad_mode,
            scale_grad_update_freq=args.scale_grad_update_freq,
            sparse_encode=args.sparse_encode,
            sparse_decode=args.sparse_decode, **reset_kwargs)
      elif args.form == 'mf':
        model = mod.KSubspaceMFModel(args.model_k, args.model_d, dataset.D,
            affine=args.affine, replicates=args.reps, reg_params=reg_params,
            scale_grad_mode=args.scale_grad_mode,
            scale_grad_update_freq=args.scale_grad_update_freq, init=args.init,
            **reset_kwargs)
      else:
        model = mod.KSubspaceProjModel(args.model_k, args.model_d, dataset.D,
            affine=args.affine, replicates=args.reps, reg_params=reg_params,
            **reset_kwargs)
  else:
    model = mod.KMeansBatchAltModel(args.model_k, dataset,
        replicates=args.reps, reg_params=reg_params, init=args.init,
        kpp_n_trials=args.kpp_n_trials, **reset_kwargs)
  model = model.to(device)
  if initX is not None:
    model.reset_parameters(initX=initX)
    initX = None
  else:
    model.reset_parameters()
  init_time = time.time() - model_tic

  # optimizer
  if not uos_mode or args.epoch_size is None or args.epoch_size <= 0:
    args.epoch_size = len(dataset)
  args.epoch_steps = args.epoch_size // args.batch_size

  if not uos_mode or args.optim == 'batch-alt':
    optimizer = None
    scheduler = None
  else:
    if args.scale_grad_mode == 'newton':
      optimizer = torch.optim.SGD(model.parameters(), lr=args.init_lr,
          momentum=0.0)
    elif args.optim == 'sgd':
      optimizer = torch.optim.SGD(model.parameters(), lr=args.init_lr,
          momentum=0.9, nesterov=True)
    elif args.optim == 'adam':
      optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr,
          betas=(0.9, 0.9), amsgrad=True)
    else:
      raise ValueError("Invalid optimizer {}.".format(args.optim))

    min_lr = max(1e-8, 0.1**4 * args.init_lr)
    if args.core_reset:
      patience = int(np.ceil(10 * args.reset_patience / args.epoch_steps))
    else:
      patience = 2000 // args.epoch_steps
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=patience,
        threshold=1e-3, min_lr=min_lr)

  # no checkpointing in this case
  if args.chkp_freq is None or args.chkp_freq <= 0:
    args.chkp_freq = args.epochs + 1
  if args.stop_freq is None or args.stop_freq <= 0:
    args.stop_freq = -1

  # save args
  with open('{}/args.json'.format(args.out_dir), 'w') as f:
    json.dump(args.__dict__, f, sort_keys=True, indent=4)

  return tr.train_loop(model, data_loader, device, optimizer, args.out_dir,
      args.epochs, args.chkp_freq, args.stop_freq, scheduler=scheduler,
      epoch_steps=args.epoch_steps, eval_rank=args.eval_rank,
      core_reset=args.core_reset, save_data=args.save_large_data,
      init_time=init_time)


def generate_parser():
  """Generate command-line parser for core-clustering evaluation."""
  parser = argparse.ArgumentParser(description=('Cluster various datasets by '
      'CoRe clustering'))
  subparsers = parser.add_subparsers(title='Evaluation setting',
      dest='setting')
  subparsers.required = True
  parser_uos = subparsers.add_parser('synth-uos',
      help='Synthetic UoS setting')
  parser_img = subparsers.add_parser('img-uos',
      help='Image UoS setting')
  parser_mc = subparsers.add_parser('movie-mc-uos',
      help='Movie MC UoS setting')
  parser_km = subparsers.add_parser('synth-kmeans',
      help='Synthetic k-means setting')

  conf.add_args(parser_uos,
      ['out-dir', 'k', 'd', 'D', 'Ng', 'N', 'affine', 'sigma', 'theta',
      'miss-rate', 'online', 'normalize', 'form', 'init', 'model-k', 'model-d',
      'auto-reg', 'sigma-hat', 'min-size', 'U-frosqr-in-lamb',
      'U-frosqr-out-lamb', 'z-lamb', 'epochs', 'epoch-size', 'batch-size',
      'optim', 'init-lr', 'scale-grad-mode', 'scale-grad-update-freq',
      'sparse-encode', 'sparse-decode', 'reps', 'core-reset', 'reset-temp',
      'reset-patience', 'reset-try-tol', 'reset-steps', 'reset-accept-tol',
      'reset-cache-size', 'cuda', 'num-threads', 'num-workers', 'data-seed',
      'seed', 'eval-rank', 'chkp-freq', 'stop-freq', 'save-large-data'])

  conf.add_args(parser_img,
      ['out-dir', 'img-dataset', 'center', 'normalize', 'sv-range', 'affine',
      'form', 'init', 'model-k', 'model-d', 'auto-reg', 'sigma-hat',
      'min-size', 'U-frosqr-in-lamb', 'U-frosqr-out-lamb', 'z-lamb', 'epochs',
      'epoch-size', 'batch-size', 'optim', 'init-lr', 'scale-grad-mode',
      'scale-grad-update-freq', 'sparse-encode', 'sparse-decode',
      'reps', 'core-reset', 'reset-temp', 'reset-patience', 'reset-try-tol',
      'reset-steps', 'reset-accept-tol', 'reset-cache-size', 'cuda',
      'num-threads', 'num-workers', 'data-seed', 'seed', 'eval-rank',
      'chkp-freq', 'stop-freq', 'save-large-data'])

  conf.add_args(parser_mc,
      ['out-dir', 'mc-dataset', 'center', 'normalize', 'affine',
      'model-k', 'model-d', 'auto-reg', 'sigma-hat', 'min-size',
      'U-frosqr-in-lamb', 'U-frosqr-out-lamb', 'z-lamb', 'epochs',
      'epoch-size', 'batch-size', 'optim', 'init-lr', 'scale-grad-mode',
      'scale-grad-update-freq', 'sparse-encode', 'sparse-decode', 'reps',
      'core-reset', 'reset-temp', 'reset-patience', 'reset-try-tol',
      'reset-steps', 'reset-accept-tol', 'reset-cache-size', 'cuda',
      'num-threads', 'num-workers', 'data-seed', 'seed', 'eval-rank',
      'chkp-freq', 'stop-freq', 'save-large-data'])

  conf.add_args(parser_km,
      ['out-dir', 'k', 'D', 'Ng', 'N', 'sep', 'init', 'kpp-n-trials',
      'model-k', 'b-frosqr-out-lamb', 'epochs', 'reps', 'core-reset',
      'reset-temp', 'reset-patience', 'reset-try-tol', 'reset-steps',
      'reset-accept-tol', 'reset-cache-size', 'cuda', 'num-threads',
      'num-workers', 'data-seed', 'seed', 'chkp-freq', 'stop-freq',
      'save-large-data'])
  return parser


if __name__ == '__main__':
  parser = generate_parser()
  args = parser.parse_args()
  eval_core_clustering(args)
