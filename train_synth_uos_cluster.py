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

import datasets as dat
import models as mod
import optimizers as opt
import training as tr

import ipdb

CHKP_FREQ = 50
STOP_FREQ = 10


def main():
  use_cuda = args.cuda and torch.cuda.is_available()
  device = torch.device('cuda' if use_cuda else 'cpu')
  torch.set_num_threads(args.num_threads)

  # construct dataset
  torch.manual_seed(args.data_seed)
  np.random.seed(args.data_seed)
  synth_dataset = dat.SynthUoSDataset(args.n, args.d, args.D, args.Ng,
      args.affine, args.sigma, args.data_seed)
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
  ipdb.set_trace()
  group_models = [mod.SubspaceAEModel(args.d, args.D, args.affine)
      for _ in range(args.n)]
  model = mod.KManifoldAEClusterModel(args.n, args.d, args.D, N,
      batch_size, group_models)
  model = model.to(device)

  # optimizer & lr schedule
  optimizer = opt.KManifoldAESGD(model, lr=args.init_lr, lamb=args.lamb,
      momentum=0.9, nesterov=True, soft_assign=args.soft_assign)

  tr.train_loop(model, synth_data_loader, device, optimizer,
      args.out_dir, args.epochs, CHKP_FREQ, STOP_FREQ)
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
  parser.add_argument('--affine', action='store_true',
                      help='Affine setting')
  parser.add_argument('--sigma', type=float, default=0.,
                      help='Data noise sigma [default: 0.]')
  parser.add_argument('--data-seed', type=int, default=1904,
                      help='Data random seed [default: 1904]')
  # model settings
  parser.add_argument('--lamb', type=float, default=.1,
                      help='reg parameter [default: .1]')
  parser.add_argument('--soft-assign', type=float, default=0.1,
                      help='soft assignment parameter [default: 0.1]')
  parser.add_argument('--soft-assign-decay', action='store_true',
                      help='decay soft assignment parameter at a rate 1/k')
  # training settings
  parser.add_argument('--alt-opt', action='store_true', default=False,
                      help='Use alternating optimization method')
  parser.add_argument('--batch-size', type=int, default=100,
                      help='Input batch size for training [default: 100]')
  parser.add_argument('--epochs', type=int, default=1000,
                      help='Number of epochs to train [default: 1000]')
  parser.add_argument('--init-lr', type=float, default=0.5,
                      help='Initial learning rate [default: 0.5]')
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
