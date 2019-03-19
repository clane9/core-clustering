"""Train and evaluate k-subspace model on synthetic union of subspaces."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import pickle

import torch
from torch.utils.data import DataLoader

import datasets as dat
import models as mod
import training as tr


def main():
  use_cuda = args.cuda and torch.cuda.is_available()
  torch.set_num_threads(args.num_threads)

  # create output directory, deleting any existing results.
  if os.path.exists(args.out_dir):
    shutil.rmtree(args.out_dir)
  os.mkdir(args.out_dir)
  # save args
  with open('{}/args.pkl'.format(args.out_dir), 'wb') as f:
    pickle.dump(args, f)

  device = torch.device('cuda' if use_cuda else 'cpu')

  # construct dataset
  torch.manual_seed(args.data_seed)
  if args.N is None:
    args.N = args.n * args.Ng
  if args.Ng is None:
    args.Ng = args.N // args.n
  if args.online:
    synth_dataset = dat.SynthUoSOnlineDataset(args.n, args.d, args.D,
        args.N, args.affine, args.sigma, args.data_seed)
    shuffle_data = False
  else:
    synth_dataset = dat.SynthUoSDataset(args.n, args.d, args.D, args.Ng,
        args.affine, args.sigma, args.data_seed)
    shuffle_data = True
  kwargs = {'num_workers': args.num_workers}
  if use_cuda:
    kwargs['pin_memory'] = True
  synth_data_loader = DataLoader(synth_dataset, batch_size=args.batch_size,
      shuffle=shuffle_data, drop_last=True, **kwargs)

  # construct model
  torch.manual_seed(args.seed)
  if args.model_d is None or args.model_d <= 0:
    args.model_d = args.d
  if args.model_n is None or args.model_n <= 0:
    args.model_n = args.n
  if args.proj_form:
    reg_params = {
        'U_frosqr_in': args.U_frosqr_in_lamb,
        'U_frosqr_out': args.U_frosqr_out_lamb,
    }
    model = mod.KSubspaceProjModel(args.model_n, args.model_d, args.D,
        args.affine, args.symmetric, args.reps, reg_params=reg_params,
        reset_patience=args.reset_patience, reset_decr_tol=args.reset_decr_tol,
        reset_sigma=args.reset_sigma)
  else:
    reg_params = {
        'U_frosqr_in': args.U_frosqr_in_lamb,
        'U_frosqr_out': args.U_frosqr_out_lamb,
        'U_fro_out': args.U_fro_out_lamb,
        'z': args.z_lamb
    }
    model = mod.KSubspaceModel(args.model_n, args.model_d, args.D, args.affine,
        args.reps, reg_params=reg_params, reset_patience=args.reset_patience,
        reset_decr_tol=args.reset_decr_tol, reset_sigma=args.reset_sigma)
  model = model.to(device)

  # optimizer
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
  tr.train_loop(model, synth_data_loader, device, optimizer, args.out_dir,
      args.epochs, args.chkp_freq, args.stop_freq, scheduler=None,
      eval_rank=args.eval_rank, reset_unused=args.reset_unused)
  return


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
  parser.add_argument('--proj-form', action='store_true', default=False,
                      help='Use projection matrix formulation')
  parser.add_argument('--reps', type=int, default=5,
                      help='Number of model replicates [default: 5]')
  parser.add_argument('--model-n', type=int, default=None,
                      help='Model number of subspaces [default: n]')
  parser.add_argument('--model-d', type=int, default=None,
                      help='Model subspace dimension [default: d]')
  parser.add_argument('--symmetric', action='store_true',
                      help='Projection matrix is U^T')
  parser.add_argument('--U-frosqr-in-lamb', type=float, default=0.01,
                      help=('Frobenius squared U reg parameter, '
                      'inside assignment [default: 0.01]'))
  parser.add_argument('--U-frosqr-out-lamb', type=float, default=1e-4,
                      help=('Frobenius squared U reg parameter, '
                      'outside assignment [default: 1e-4]'))
  parser.add_argument('--U-fro-out-lamb', type=float, default=0.0,
                      help=('Frobenius U reg parameter, '
                      'outside assignment [default: 0.0]'))
  parser.add_argument('--z-lamb', type=float, default=0.01,
                      help=('L2 squared coefficient reg parameter, '
                      'inside assignment [default: 0.01]'))
  # training settings
  parser.add_argument('--batch-size', type=int, default=100,
                      help='Input batch size for training [default: 100]')
  parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs to train [default: 50]')
  parser.add_argument('--optim', type=str, default='SGD',
                      help='Optimizer [default: SGD]')
  parser.add_argument('--init-lr', type=float, default=0.5,
                      help='Initial learning rate [default: 0.5]')
  parser.add_argument('--reset-unused', action='store_true', default=False,
                      help='Whether to reset unused clusters')
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
  parser.add_argument('--eval-rank', action='store_true', default=False,
                      help='Evaluate ranks of subspace models')
  parser.add_argument('--chkp-freq', type=int, default=None,
                      help='How often to save checkpoints [default: None]')
  parser.add_argument('--stop-freq', type=int, default=None,
                      help='How often to stop in ipdb [default: None]')
  args = parser.parse_args()

  main()
