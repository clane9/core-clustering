"""Train and evaluate online k-subspace clustering on range of real
datasets."""

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

import datasets as dat
import models as mod
import training as tr


def main():
  use_cuda = args.cuda and torch.cuda.is_available()
  device = torch.device('cuda' if use_cuda else 'cpu')
  torch.set_num_threads(args.num_threads)

  if args.dataset not in {'mnist_sc_pca', 'coil100'}:
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
  dataset = dat.YouCVPR16ImageUoS(args.dataset)
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
  Ng = dataset.N / args.model_n
  args.U_frosqr_in_lamb = ((1.0 + np.sqrt((dataset.D - args.model_d)/Ng))**2 *
      (args.sigma_hat**2 / dataset.D))
  args.U_frosqr_out_lamb = ((args.min_size / args.model_n) *
      (1.0 / args.model_d + args.sigma_hat**2 / dataset.D))

  args.U_fro_out_lamb = 0.0
  args.U_gram_fro_out_lamb = 0.0
  args.z_lamb = 0.01

  # construct model
  args.reset_metric = args.reset_metric.lower()
  reset_metric = 'value' if args.reset_metric == 'none' else args.reset_metric
  if args.reset_patience is None or args.reset_patience < 0:
    args.reset_patience = (2 if batch_alt_mode else
        int(np.ceil(dataset.N / args.batch_size)))
  reset_warmup = args.reset_patience
  reset_obj = 'full'
  if args.form == 'batch-alt-proj':
    reg_params = {
        'U_frosqr_in': args.U_frosqr_in_lamb,
        'U_frosqr_out': args.U_frosqr_out_lamb,
        'U_gram_fro_out': args.U_gram_fro_out_lamb
    }
    model = mod.KSubspaceBatchAltProjModel(args.model_n, args.model_d,
        dataset, args.affine, args.reps, reg_params=reg_params,
        reset_metric=reset_metric, unused_thr=args.unused_thr,
        reset_patience=args.reset_patience, reset_warmup=reset_warmup,
        reset_obj=reset_obj, reset_decr_tol=args.reset_decr_tol,
        reset_sigma=args.reset_sigma, svd_solver='randomized')
  elif args.form == 'batch-alt-mf':
    reg_params = {
        'U_frosqr_in': args.U_frosqr_in_lamb / args.z_lamb,
        'U_frosqr_out': args.U_frosqr_out_lamb / args.z_lamb,
        'z': (args.z_lamb if
            max(args.U_frosqr_in_lamb, args.U_frosqr_out_lamb) > 0
            else 0.0)
    }
    model = mod.KSubspaceBatchAltMFModel(args.model_n, args.model_d,
        dataset, args.affine, args.reps, reg_params=reg_params,
        reset_metric=reset_metric, unused_thr=args.unused_thr,
        reset_patience=args.reset_patience, reset_warmup=reset_warmup,
        reset_obj=reset_obj, reset_decr_tol=args.reset_decr_tol,
        reset_sigma=args.reset_sigma, svd_solver='randomized')
  elif args.form == 'mf':
    reg_params = {
        'U_frosqr_in': args.U_frosqr_in_lamb / args.z_lamb,
        'U_frosqr_out': args.U_frosqr_out_lamb / args.z_lamb,
        'U_fro_out': args.U_fro_out_lamb,
        'U_gram_fro_out': args.U_gram_fro_out_lamb,
        'z': (args.z_lamb if
            max(args.U_frosqr_in_lamb, args.U_frosqr_out_lamb) > 0
            else 0.0)
    }
    model = mod.KSubspaceMFModel(args.model_n, args.model_d,
        dataset.D, args.affine, args.reps, reg_params=reg_params,
        reset_metric=reset_metric, unused_thr=args.unused_thr,
        reset_patience=args.reset_patience, reset_warmup=reset_warmup,
        reset_obj=reset_obj, reset_decr_tol=args.reset_decr_tol,
        reset_sigma=args.reset_sigma)
  else:
    reg_params = {
        'U_frosqr_in': args.U_frosqr_in_lamb,
        'U_frosqr_out': args.U_frosqr_out_lamb,
        'U_fro_out': args.U_fro_out_lamb,
        'U_gram_fro_out': args.U_gram_fro_out_lamb
    }
    model = mod.KSubspaceProjModel(args.model_n, args.model_d, dataset.D,
        args.affine, args.reps, reg_params=reg_params,
        reset_metric=reset_metric, unused_thr=args.unused_thr,
        reset_patience=args.reset_patience, reset_warmup=reset_warmup,
        reset_obj=reset_obj, reset_decr_tol=args.reset_decr_tol,
        reset_sigma=args.reset_sigma)
  model = model.to(device)

  # optimizer
  if batch_alt_mode:
    optimizer = None
  else:
    if args.optim == 'SGD':
      optimizer = torch.optim.SGD(model.parameters(), lr=args.init_lr,
          momentum=0.9, nesterov=True)
    elif args.optim == 'Adam':
      optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr,
          betas=(0.9, 0.9), amsgrad=True)
    else:
      raise ValueError("Invalid optimizer {}.".format(args.optim))

  if args.chkp_freq is None or args.chkp_freq <= 0:
    args.chkp_freq = args.epochs
  if args.stop_freq is None or args.stop_freq <= 0:
    args.stop_freq = -1

  # save args
  with open('{}/args.json'.format(args.out_dir), 'w') as f:
    json.dump(args.__dict__, f, sort_keys=True, indent=4)

  tr.train_loop(model, data_loader, device, optimizer, args.out_dir,
      args.epochs, args.chkp_freq, args.stop_freq, scheduler=None,
      eval_rank=args.eval_rank, reset_unused=(args.reset_metric != 'none'))
  return


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Cluster real UoS data')
  parser.add_argument('--out-dir', type=str, required=True,
                      help='Output directory.')
  parser.add_argument('--dataset', type=str, default='mnist_sc_pca',
                      help='Real dataset [default: mnist_sc_pca].',
                      choices=['mnist_sc_pca', 'coil100'])
  # model settings
  parser.add_argument('--form', type=str, required=True,
                      help=('Model formulation (proj, mf, batch-alt-proj, '
                      'batch-alt-mf)'))
  parser.add_argument('--model-n', type=int, required=True,
                      help='Model number of subspaces')
  parser.add_argument('--model-d', type=int, required=True,
                      help='Model subspace dimension')
  parser.add_argument('--affine', action='store_true',
                      help='Affine setting')
  parser.add_argument('--reps', type=int, default=5,
                      help='Number of model replicates [default: 5]')
  parser.add_argument('--sigma-hat', type=float, default=0.4,
                      help='Noise estimate [default: 0.4]')
  parser.add_argument('--min-size', type=float, default=0.01,
                      help=('Minimum cluster size as fraction relative to 1/n'
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
  parser.add_argument('--eval-rank', action='store_true', default=False,
                      help='Evaluate ranks of subspace models')
  parser.add_argument('--seed', type=int, default=2018,
                      help='Training random seed [default: 2018]')
  parser.add_argument('--chkp-freq', type=int, default=None,
                      help='How often to save checkpoints [default: None]')
  parser.add_argument('--stop-freq', type=int, default=None,
                      help='How often to stop in ipdb [default: None]')
  args = parser.parse_args()

  main()
