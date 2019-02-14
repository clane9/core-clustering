from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import pickle

import torch

import datasets as dat
from ksubspaces import KSubspaceBatchAltModel


def main():
  if os.path.exists(args.out_dir):
    shutil.rmtree(args.out_dir)
  os.mkdir(args.out_dir)
  # save args
  with open('{}/args.pkl'.format(args.out_dir), 'wb') as f:
    pickle.dump(args, f)

  use_cuda = args.cuda and torch.cuda.is_available()
  torch.set_num_threads(args.num_threads)
  device = torch.device('cuda' if use_cuda else 'cpu')

  # construct dataset
  dataset = dat.SynthUoSDataset(args.n, args.d, args.D, args.Ng,
      args.affine, args.sigma, args.data_seed)

  # construct model
  torch.manual_seed(args.seed)
  if args.model_d is None or args.model_d <= 0:
    args.model_d = args.d
  if args.model_n is None or args.model_n <= 0:
    args.model_n = args.n
  model = KSubspaceBatchAltModel(args.model_n, args.model_d, dataset,
      affine=args.affine, svd_solver=args.svd_solver)
  model.to(device)

  model.fit(steps=args.steps, out_dir=args.out_dir,
      reset_unused=args.reset_unused)
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
  parser.add_argument('--sigma', type=float, default=0.01,
                      help='Data noise sigma [default: 0.01]')
  parser.add_argument('--data-seed', type=int, default=1904,
                      help='Data random seed [default: 1904]')
  # model settings
  parser.add_argument('--model-n', type=int, default=None,
                      help='Model number of subspaces [default: n]')
  parser.add_argument('--model-d', type=int, default=None,
                      help='Model subspace dimension [default: d]')
  # training settings
  parser.add_argument('--steps', type=int, default=None,
                      help='Max number of steps [default: inf]')
  parser.add_argument('--svd-solver', type=str, default='randomized',
                      help=('SVD solver (randomized, svds, svd) '
                          '[default: randomized]'))
  parser.add_argument('--reset-unused', action='store_true', default=False,
                      help='Whether to reset unused clusters')
  parser.add_argument('--cuda', action='store_true', default=False,
                      help='Enables CUDA training')
  parser.add_argument('--num-threads', type=int, default=1,
                      help='Number of parallel threads to use [default: 1]')
  parser.add_argument('--seed', type=int, default=2018,
                      help='Training random seed [default: 2018]')
  args = parser.parse_args()

  main()
