"""Train and evaluate factorized manifold clustering model on synthetic union
of subspaces."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import pickle

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

import datasets as dat
import models as mod
import optimizers as opt
import training as tr
import utils as ut

CHKP_FREQ = 50
STOP_FREQ = 10


def main():
  use_cuda = args.cuda and torch.cuda.is_available()
  if use_cuda and args.dist:
    raise ValueError("Cannot use cuda in distributed mode")

  device = torch.device('cuda' if use_cuda else 'cpu')
  torch.set_num_threads(args.num_threads)

  # construct dataset
  synth_dataset = dat.SynthUoSOnlineDataset(args.n, args.d, args.D, args.N,
      args.affine, args.sigma, args.data_seed)
  if args.dist:
    # separate online sampling seeds per process
    torch.manual_seed(args.data_seed + 1957*dist.get_rank())
  else:
    torch.manual_seed(args.data_seed)
  kwargs = {'num_workers': args.num_workers}
  if use_cuda:
    kwargs['pin_memory'] = True
  if args.batch_size <= 0 or args.batch_size > args.N:
    raise ValueError("Invalid batch size")
  synth_data_loader = DataLoader(synth_dataset, batch_size=args.batch_size,
      shuffle=False, drop_last=True, **kwargs)

  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # construct model
  model_d = args.model_d if args.model_d > 0 else args.d
  if args.auto_enc:
    group_models = [mod.SubspaceAEModel(model_d, args.D, args.affine,
        reg=args.reg_U) for _ in range(args.n)]
    model = mod.KManifoldAEClusterModel(args.n, model_d, args.N,
        args.batch_size, group_models)
  else:
    group_models = [mod.SubspaceModel(model_d, args.D, args.affine,
        reg=args.reg_U) for _ in range(args.n)]
    model = mod.KManifoldClusterModel(args.n, model_d, args.N, args.batch_size,
        group_models, use_cuda, store_C_V=False)
  model = model.to(device)

  # optimizer & lr schedule
  if args.auto_enc:
    optimizer = opt.KManifoldAESGD(model, lr=args.init_lr,
        lamb=args.lamb_U, momentum=args.momentum, nesterov=args.nesterov,
        soft_assign=args.soft_assign, dist_mode=args.dist,
        size_scale=args.size_scale)
  else:
    if args.prox_reg_U:
      prox_U = (ut.prox_grp_sprs if args.reg_U == 'grp_sprs'
          else ut.prox_fro_sqr)
      optimizer = opt.KSubspaceAltProxSGD(model, lr=args.init_lr,
          lamb_U=args.lamb_U, lamb_V=args.lamb_V, momentum=args.momentum,
          prox_U=prox_U, soft_assign=args.soft_assign,
          dist_mode=args.dist, size_scale=args.size_scale)
    else:
      optimizer = opt.KSubspaceAltSGD(model, lr=args.init_lr,
          lamb_U=args.lamb_U, lamb_V=args.lamb_V, momentum=args.momentum,
          nesterov=args.nesterov, soft_assign=args.soft_assign,
          dist_mode=args.dist, size_scale=args.size_scale)

  tr.train_loop(model, synth_data_loader, device, optimizer,
      args.out_dir, args.epochs, CHKP_FREQ, STOP_FREQ, args.dist,
      eval_rank=True)
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
  parser.add_argument('--N', type=int, default=100000,
                      help='Total data points [default: 10^5]')

  parser.add_argument('--affine', action='store_true',
                      help='Affine setting')
  parser.add_argument('--sigma', type=float, default=0.01,
                      help='Data noise sigma [default: 0.01]')
  parser.add_argument('--data-seed', type=int, default=1904,
                      help='Data random seed [default: 1904]')
  # model settings
  parser.add_argument('--model-d', type=int, default=-1,
                      help='Model subspace dimension [default: d]')
  parser.add_argument('--auto-enc', action='store_true', default=False,
                      help='use auto-encoder formulation')
  parser.add_argument('--reg-U', type=str, default='fro_sqr',
                      help=("U reg function (one of 'fro_sqr', 'grp_sprs') "
                          "[default: fro_sqr]"))
  parser.add_argument('--prox-reg-U', action='store_true', default=False,
                      help='update U by proximal gradient')
  parser.add_argument('--lamb-U', type=float, default=1e-4,
                      help='U reg parameter [default: 1e-4]')
  parser.add_argument('--lamb-V', type=float, default=0.1,
                      help='V reg parameter [default: 0.1]')
  parser.add_argument('--soft-assign', type=float, default=0.1,
                      help='soft assignment parameter [default: 0.1]')
  # training settings
  parser.add_argument('--batch-size', type=int, default=100,
                      help='Input batch size for training [default: 100]')
  parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs to train [default: 50]')
  parser.add_argument('--init-lr', type=float, default=0.5,
                      help='Initial learning rate [default: 0.5]')
  parser.add_argument('--momentum', type=float, default=0.9,
                      help='Initial learning rate [default: 0.9]')
  parser.add_argument('--nesterov', action='store_true', default=False,
                      help='Use nesterov form of acceleration')
  parser.add_argument('--dist', action='store_true', default=False,
                      help='Enables distributed training')
  parser.add_argument('--size-scale', action='store_true', default=False,
                      help=('Scale objective wrt U to compensate for '
                      'size imbalance'))
  parser.add_argument('--cuda', action='store_true', default=False,
                      help='Enables CUDA training')
  parser.add_argument('--num-threads', type=int, default=1,
                      help='Number of parallel threads to use [default: 1]')
  parser.add_argument('--num-workers', type=int, default=1,
                      help='Number of workers for data loading [default: 1]')
  parser.add_argument('--seed', type=int, default=2018,
                      help='Training random seed [default: 2018]')
  args = parser.parse_args()

  if args.dist:
    dist.init_process_group(backend="mpi")

  is_logging = (not args.dist) or (dist.get_rank() == 0)
  # create output directory, deleting any existing results.
  if is_logging:
    if os.path.exists(args.out_dir):
      shutil.rmtree(args.out_dir)
    os.mkdir(args.out_dir)
    # save args
    with open('{}/args.pkl'.format(args.out_dir), 'wb') as f:
      pickle.dump(args, f)
  main()