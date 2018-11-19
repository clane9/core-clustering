"""Train and evaluate factorized manifold clustering model on synthetic union
of smooth manifolds."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import pickle

import numpy as np
import torch

import datasets as dat
import models as mod
import optimizers as opt
import training as tr

# import ipdb

CHKP_FREQ = 50
STOP_FREQ = 10


def main():
  use_cuda = args.cuda and torch.cuda.is_available()
  device = torch.device('cuda' if use_cuda else 'cpu')
  torch.set_num_threads(args.num_threads)

  # construct dataset
  torch.manual_seed(args.data_seed)
  np.random.seed(args.data_seed)
  synth_dataset = dat.SynthUoMDataset(args.n, args.d, args.D, args.Ng,
       args.H, args.nonsmooth, args.sigma, args.data_seed)
  N = args.n*args.Ng
  batch_size = args.batch_size
  if args.batch_size <= 0 or args.batch_size > N:
    batch_size = N
  kwargs = {'num_workers': 0}
  if use_cuda:
    kwargs['pin_memory'] = True
  synth_data_loader = torch.utils.data.DataLoader(synth_dataset,
      batch_size=batch_size, shuffle=(batch_size != N), **kwargs)

  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # construct model
  H_ = int(np.ceil(args.H*args.over_param))
  if args.auto_enc:
    group_models = [mod.ResidualManifoldAEModel(args.d, args.D, H_,
        args.drop_p, args.lamb) for _ in range(args.n)]
    model = mod.KManifoldAEClusterModel(args.n, args.d, args.D, N,
        batch_size, group_models)
  else:
    group_models = [mod.ResidualManifoldModel(args.d, args.D, H_,
        args.drop_p, args.lamb) for _ in range(args.n)]
    model = mod.KManifoldClusterModel(args.n, args.d, args.D, N,
        batch_size, group_models, use_cuda)

  # optimizer & lr schedule
  if args.auto_enc:
    optimizer = opt.KManifoldAESGD(model, lr=args.init_lr,
        lamb=args.lamb_U, momentum=args.momentum, nesterov=args.nesterov,
        soft_assign=args.soft_assign)
  else:
    if args.alt_opt:
      optimizer = opt.KManifoldAltSGD(model, lr=args.init_lr,
          lamb_U=args.lamb_U, lamb_V=args.lamb_V, momentum=args.momentum,
          nesterov=args.nesterov, soft_assign=args.soft_assign)
    else:
      optimizer = opt.KManifoldSGD(model, lr=args.init_lr,
          lamb_U=args.lamb_U, lamb_V=args.lamb_V, momentum=args.momentum,
          nesterov=args.nesterov, soft_assign=args.soft_assign)

  tr.train_loop(model, synth_data_loader, device, optimizer,
      args.out_dir, args.epochs, CHKP_FREQ, STOP_FREQ)
  return


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Cluster synthetic UoM data')
  parser.add_argument('--out-dir', type=str, required=True,
                      help='Output directory.')
  # data settings
  parser.add_argument('--n', type=int, default=4,
                      help='Number of subspaces [default: 4]')
  parser.add_argument('--d', type=int, default=2,
                      help='Subspace dimension [default: 2]')
  parser.add_argument('--D', type=int, default=100,
                      help='Ambient dimension [default: 100]')
  parser.add_argument('--Ng', type=int, default=1000,
                      help='Points per group [default: 1000]')
  parser.add_argument('--H', type=int, default=100,
                      help=('Size of hidden layer in manifold generator '
                          '[default: 100]'))
  parser.add_argument('--nonsmooth', type=float, default=1.0,
                      help=('Nonsmoothness level (0=affine subspace) '
                          '[default: 1.0]'))
  parser.add_argument('--sigma', type=float, default=0.01,
                      help='Data noise sigma [default: 0.01]')
  parser.add_argument('--data-seed', type=int, default=1904,
                      help='Data random seed [default: 1904]')
  # model settings
  parser.add_argument('--auto-enc', action='store_true', default=False,
                      help='use auto-encoder formulation')
  parser.add_argument('--over-param', type=float, default=1.0,
                      help=('Over-parameterization in manifold models '
                      '[default: 1.0]'))
  parser.add_argument('--drop-p', type=float, default=0.0,
                      help='Dropout in manifold models [default: 0.0]')
  parser.add_argument('--lamb-U', type=float, default=1e-4,
                      help='U reg parameter [default: 1e-4]')
  parser.add_argument('--lamb-V', type=float, default=0.1,
                      help='V reg parameter [default: 0.1]')
  parser.add_argument('--soft-assign', type=float, default=0.1,
                      help='soft assignment parameter [default: 0.1]')
  # training settings
  parser.add_argument('--alt-opt', action='store_true', default=False,
                      help='Use alternating optimization method')
  parser.add_argument('--batch-size', type=int, default=100,
                      help='Input batch size for training [default: 100]')
  parser.add_argument('--epochs', type=int, default=1000,
                      help='Number of epochs to train [default: 1000]')
  parser.add_argument('--init-lr', type=float, default=0.5,
                      help='Initial learning rate [default: 0.5]')
  parser.add_argument('--momentum', type=float, default=0.9,
                      help='Initial learning rate [default: 0.9]')
  parser.add_argument('--nesterov', action='store_true', default=False,
                      help='Use nesterov form of acceleration')
  parser.add_argument('--maxit-V', type=int, default=20,
                      help='Number of iterations for V update [default: 20]')
  parser.add_argument('--cuda', action='store_true', default=False,
                      help='Enables CUDA training')
  parser.add_argument('--num-threads', type=int, default=1,
                      help='Number of parallel threads to use [default: 1]')
  parser.add_argument('--seed', type=int, default=2018,
                      help='Training random seed [default: 2018]')
  args = parser.parse_args()

  # create output directory, deleting any existing results.
  if os.path.exists(args.out_dir):
    shutil.rmtree(args.out_dir)
  os.mkdir(args.out_dir)
  # save args
  with open('{}/args.pkl'.format(args.out_dir), 'wb') as f:
    pickle.dump(args, f)
  main()
