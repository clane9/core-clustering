"""Train and evaluate factorized manifold clustering model on MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import pickle

import numpy as np
import torch
from torch import optim
from torchvision import transforms

import datasets as dat
import models as mod
import optimizers as opt
import training as tr

CHKP_FREQ = 50
STOP_FREQ = -1 


def main():
  use_cuda = args.cuda and torch.cuda.is_available()
  device = torch.device('cuda' if use_cuda else 'cpu')
  torch.set_num_threads(args.num_threads)

  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # construct dataset
  mnist_dataset = dat.MNISTUoM(args.data_dir, train=True,
      transform=transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.1307,), (0.3081,))]),
      download=True, classes=args.classes)
  N = len(mnist_dataset)
  kwargs = {'num_workers': args.num_workers}
  if use_cuda:
    kwargs['pin_memory'] = True
  mnist_data_loader = torch.utils.data.DataLoader(mnist_dataset,
      batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)

  # construct model
  arch = args.arch.lower()
  model_n = mnist_dataset.n if args.n is None else args.n
  if arch == 'dc':
    group_models = [mod.MNISTDCManifoldModel(args.d, args.filters, args.drop_p)
        for _ in range(model_n)]
  elif arch == 'res':
    group_models = [mod.ResidualManifoldModel(args.d, (1, 28, 28),
        args.filters, args.drop_p) for _ in range(model_n)]
  elif arch == 'mlp':
    group_models = [mod.MNISTMlpAEModel(args.d, args.filters, args.drop_p)
        for _ in range(model_n)]
  else:
    raise ValueError("Invalid architecture {}".format(arch))

  if arch in ['dc', 'res']:
    model = mod.KManifoldClusterModel(model_n, args.d, N, args.batch_size,
        group_models, use_cuda)
  else:
    model = mod.KManifoldAEClusterModel(model_n, args.d, N, args.batch_size,
        group_models)
  model.to(device)

  # optimizer
  if arch in ['dc', 'res']:
    optimizer = opt.KManifoldSGD(model, lr=args.init_lr,
        lamb_U=args.lamb_U, lamb_V=args.lamb_V, momentum=args.momentum,
        nesterov=args.nesterov, soft_assign=args.soft_assign,
        size_scale=args.size_scale)
  else:
    optimizer = opt.KManifoldAEMetaOptimizer(model, optim.Adam,
        lr=args.init_lr, lamb=args.lamb_U,
        soft_assign=args.soft_assign, dist_mode=False,
        opt_kwargs={'betas': (0.9, 0.9)}) 

  tr.train_loop(model, mnist_data_loader, device, optimizer,
      args.out_dir, args.epochs, CHKP_FREQ, STOP_FREQ, eval_rank=False)
  return


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Cluster MNIST data')
  parser.add_argument('--out-dir', type=str, required=True,
                      help='Output directory.')
  parser.add_argument('--data-dir', type=str,
                      default='~/Documents/Datasets/MNIST',
                      help=('Data directory '
                          '[default: ~/Documents/Datasets/MNIST].'))
  parser.add_argument('--classes', type=int, default=None, nargs='+',
                      help='Subset of digit classes [default: all]')
  # model settings
  parser.add_argument('--arch', type=str, default='mlp',
                      help='Network architecture [default: mlp]',
                      choices=['dc', 'res', 'mlp'])
  parser.add_argument('--n', type=int, default=None,
                      help='Manifold dimension [default: # classes]')
  parser.add_argument('--d', type=int, default=2,
                      help='Manifold dimension [default: 2]')
  parser.add_argument('--filters', type=int, default=50,
                      help=('Number of hidden layer filters [default: 50]'))
  parser.add_argument('--drop-p', type=float, default=0.0,
                      help='Dropout in manifold models [default: 0.0]')
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

  # create output directory, deleting any existing results.
  if os.path.exists(args.out_dir):
    shutil.rmtree(args.out_dir)
  os.mkdir(args.out_dir)
  # save args
  with open('{}/args.pkl'.format(args.out_dir), 'wb') as f:
    pickle.dump(args, f)
  main()
