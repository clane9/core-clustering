"""Train and evaluate core-clustering model on variety of datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import shutil
import json
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR

from corecluster import configuration as conf
from corecluster import datasets as dat
from corecluster import models as mod
from corecluster import training as tr
from corecluster import utils as ut


def eval_core_clustering(args):
  # setting some potentially undefined args depending on setting
  default_args = [('init', 'random'), ('form', 'mf')]
  for arg, val in default_args:
    setattr(args, arg, getattr(args, arg, val))

  uos_mode = args.setting != 'synth-kmeans'
  batch_mode = args.optim == 'batch-alt'
  online_mode = args.setting in {'synth-uos', 'synth-kmeans'} and args.online
  mc_mode = ((args.setting == 'synth-uos' and args.miss_rate > 0) or
      args.setting == 'movie-mc-uos')
  real_mode = args.setting in {'img-uos', 'movie-mc-uos'}
  scale_grad_mode = args.scale_grad_lip and not batch_mode

  # error checking, in no particular order, not too organized...
  if online_mode and batch_mode:
    raise ValueError(("Online data not compatible with batch-alt "
        "optimization."))
  if mc_mode and (batch_mode or args.form == 'proj' or args.init == 'pfi'):
    raise ValueError(("batch-alt optimization, proj form, PFI initialization "
        "not compatible with missing data."))
  if scale_grad_mode and uos_mode and (args.form == 'proj' or args.affine):
    raise ValueError(("proj form, affine setting not compatible with grad "
        "scaling."))
  if real_mode and (args.model_d is None or args.model_d <= 0):
    raise ValueError("model dimension is required for real data.")

  use_cuda = args.cuda and torch.cuda.is_available()
  device = torch.device('cuda' if use_cuda else 'cpu')
  torch.set_num_threads(args.num_threads)

  # create output directory, deleting any existing results.
  if os.path.exists(args.out_dir):
    shutil.rmtree(args.out_dir)
  os.makedirs(args.out_dir)

  # save args, before later modifications
  with open('{}/args.json'.format(args.out_dir), 'w') as f:
    json.dump(args.__dict__, f, sort_keys=True, indent=4)

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
  shuffle_data = not online_mode

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
  elif args.setting == 'synth-kmeans' and online_mode:
    dataset = dat.SynthKMeansOnlineDataset(args.k, args.D, args.N,
        separation=args.sep, seed=args.data_seed)
  elif args.setting == 'synth-kmeans':
    dataset = dat.SynthKMeansDataset(args.k, args.D, args.Ng,
        separation=args.sep, seed=args.data_seed)
  else:
    raise ValueError("Invalid data setting {}".format(args.setting))

  if batch_mode:
    data_loader, bs_scheduler = None, None
  else:
    kwargs = {'num_workers': args.num_workers,
        'batch_size': args.init_bs, 'shuffle': shuffle_data,
        'drop_last': True, 'pin_memory': use_cuda}
    if mc_mode:
      kwargs['collate_fn'] = dat.missing_data_collate
    data_loader = DataLoader(dataset, **kwargs)

    if args.bs_step_size is not None and args.bs_step_size > 0:
      bs_step_decay = ut.ClampDecay(args.init_bs, args.bs_step_size,
          args.bs_gamma, max=args.bs_max)
      bs_scheduler = ut.LambdaBS(dataset, kwargs, bs_step_decay)
    else:
      bs_scheduler = None

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

  reset_kwargs = dict(reset_patience=args.reset_patience,
      reset_try_tol=args.reset_try_tol,
      reset_max_steps=args.reset_steps,
      reset_accept_tol=args.reset_accept_tol,
      reset_cache_size=args.reset_cache_size)

  if uos_mode and batch_mode and args.form == 'mf':
    model = mod.KSubspaceBatchAltMFModel(k=args.model_k, d=args.model_d,
        dataset=dataset, affine=args.affine, replicates=args.reps,
        reg_params=reg_params, svd_solver='randomized', **reset_kwargs)
  elif uos_mode and batch_mode and args.form == 'proj':
    model = mod.KSubspaceBatchAltProjModel(k=args.model_k, d=args.model_d,
        dataset=dataset, affine=args.affine, replicates=args.reps,
        reg_params=reg_params, svd_solver='randomized', **reset_kwargs)
  elif uos_mode and mc_mode:
    model = mod.KSubspaceMCModel(k=args.model_k, d=args.model_d, D=dataset.D,
        affine=args.affine, replicates=args.reps, reg_params=reg_params,
        scale_grad_lip=args.scale_grad_lip, sparse_encode=args.sparse_encode,
        sparse_decode=args.sparse_decode, **reset_kwargs)
  elif uos_mode and args.form == 'mf':
    model = mod.KSubspaceMFModel(k=args.model_k, d=args.model_d, D=dataset.D,
        affine=args.affine, replicates=args.reps, reg_params=reg_params,
        scale_grad_lip=args.scale_grad_lip, **reset_kwargs)
  elif uos_mode and args.form == 'proj':
    model = mod.KSubspaceProjModel(k=args.model_k, d=args.model_d, D=dataset.D,
        affine=args.affine, replicates=args.reps, reg_params=reg_params,
        **reset_kwargs)
  elif not uos_mode and batch_mode:
    model = mod.KMeansBatchAltModel(k=args.model_k, dataset=dataset,
        replicates=args.reps, reg_params=reg_params, **reset_kwargs)
  elif not uos_mode:
    model = mod.KMeansModel(k=args.model_k, D=dataset.D, replicates=args.reps,
        reg_params=reg_params, scale_grad_lip=args.scale_grad_lip,
        **reset_kwargs)
  else:
    raise ValueError("Invalid model setting.")

  model = model.to(device)

  # PFI initialization
  if args.init == 'pfi':
    # construct pfi initialization dataset
    if args.pfi_init_size is None or args.pfi_init_size <= 0:
      args.pfi_init_size = dataset.N
    args.pfi_init_size = min(args.pfi_init_size, dataset.N)

    if data_loader is not None:
      initX = torch.zeros(args.pfi_init_size, dataset.D)
      init_head = 0
      for x, _ in data_loader:
        append_size = min(args.pfi_init_size - init_head, x.shape[0])
        initX[init_head: (init_head+append_size), :] = x[:append_size, :]
        init_head += append_size
        if init_head >= args.pfi_init_size:
          break
    else:
      Idx = torch.randperm(dataset.N)[:args.pfi_init_size]
      initX = dataset.X[Idx, :]
    initX = initX.to(device)

    if uos_mode:
      # normalize data
      initX = initX.div(torch.norm(initX, p=2, dim=1, keepdim=True).add(1e-8))
      fit_kwargs = {'nn_q': int(np.ceil(0.1*args.model_d)), 'normalize': False}
    else:
      fit_kwargs = dict()

    model.pfi_init(initX, pfi_n_cands=args.pfi_n_cands, fit_kwargs=fit_kwargs)

  init_time = time.time() - model_tic

  # optimizer
  if args.epoch_size is None or args.epoch_size <= 0:
    args.epoch_size = len(dataset)

  if batch_mode:
    optimizer, lr_scheduler = None, None
  else:
    if args.optim == 'sgd':
      optimizer = torch.optim.SGD(model.parameters(), lr=args.init_lr,
          momentum=0.9, nesterov=True)
    elif args.optim == 'adam':
      optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr,
          betas=(0.9, 0.999), amsgrad=True)
    else:
      raise ValueError("Invalid optimizer {}.".format(args.optim))

    if args.lr_step_size is not None and args.lr_step_size > 0:
      lr_step_decay = ut.ClampDecay(args.init_lr, args.lr_step_size,
          args.lr_gamma, min=args.lr_min)
      lr_scheduler = LambdaLR(dataset, kwargs, lr_step_decay)
    else:
      lr_scheduler = None

  # no checkpointing in this case
  if args.chkp_freq is None or args.chkp_freq <= 0:
    args.chkp_freq = args.epochs + 1
  if args.stop_freq is None or args.stop_freq <= 0:
    args.stop_freq = -1

  return tr.train_loop(model, data_loader, device, optimizer, args.out_dir,
      args.epochs, args.chkp_freq, args.stop_freq, lr_scheduler=lr_scheduler,
      bs_scheduler=bs_scheduler, epoch_size=args.epoch_size,
      core_reset=args.core_reset, eval_rank=args.eval_rank,
      save_data=args.save_large_data, init_time=init_time)


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

  uos_model_args = ['form', 'model-k', 'model-d', 'auto-reg', 'sigma-hat',
      'min-size', 'U-frosqr-in-lamb', 'U-frosqr-out-lamb', 'z-lamb']
  opt_args = ['init', 'pfi-init-size', 'pfi-n-cands', 'epochs', 'epoch-size',
      'optim', 'init-lr', 'lr-step-size', 'lr-gamma', 'lr-min', 'init-bs',
      'bs-step-size', 'bs-gamma', 'bs-max', 'scale-grad-lip']
  sparse_args = ['sparse-encode', 'sparse-decode']
  reset_args = ['reps', 'core-reset', 'reset-patience', 'reset-try-tol',
      'reset-steps', 'reset-accept-tol', 'reset-cache-size']
  generic_args = ['cuda', 'num-threads', 'num-workers', 'data-seed', 'seed',
      'eval-rank', 'chkp-freq', 'stop-freq', 'save-large-data', 'config']

  conf.add_args(parser_uos,
      ['out-dir', 'k', 'd', 'D', 'Ng', 'N', 'affine', 'sigma', 'theta',
      'miss-rate', 'online', 'normalize'] + uos_model_args + opt_args +
      sparse_args + reset_args + generic_args)

  conf.add_args(parser_img,
      ['out-dir', 'img-dataset', 'center', 'normalize', 'sv-range', 'affine'] +
      uos_model_args + opt_args + reset_args + generic_args)

  conf.add_args(parser_mc,
      ['out-dir', 'mc-dataset', 'center', 'normalize', 'affine'] +
      uos_model_args[1:] + opt_args[3:] + sparse_args + reset_args +
      generic_args)

  conf.add_args(parser_km,
      ['out-dir', 'k', 'D', 'Ng', 'N', 'sep', 'online', 'model-k',
      'b-frosqr-out-lamb'] + opt_args + reset_args + generic_args)
  return parser


if __name__ == '__main__':
  parser = generate_parser()
  args = parser.parse_args()

  # update unspecified args with values from config json
  if args.config is not None:
    with open(args.config, 'r') as f:
      config = json.load(f)
    # assumes all passed argument flags start with '--'
    passed_args = set([arg[2:].replace('-', '_')
        for arg in sys.argv[1:] if arg[:2] == '--'])
    update_args = set(config.keys()).difference(passed_args)
    for arg in update_args:
      setattr(args, arg, config[arg])

  eval_core_clustering(args)
