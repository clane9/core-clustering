"""Train and evaluate k-means model on synthetic data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import json
import time

import torch

import datasets as dat
import models as mod
import training as tr
import utils as ut


def train_synth_kmeans_cluster(args):
  use_cuda = args.cuda and torch.cuda.is_available()
  device = torch.device('cuda' if use_cuda else 'cpu')
  torch.set_num_threads(args.num_threads)

  # create output directory, deleting any existing results.
  if os.path.exists(args.out_dir):
    shutil.rmtree(args.out_dir)
  os.mkdir(args.out_dir)

  # construct dataset
  torch.manual_seed(args.data_seed)
  synth_dataset = dat.SynthKMeansDataset(args.k, args.D, args.Ng,
      separation=args.sep, seed=args.data_seed)
  synth_data_loader = None

  # construct model
  torch.manual_seed(args.seed)
  if args.model_k is None or args.model_k <= 0:
    args.model_k = args.k
  reg_params = {
      'b_frosqr_out': args.b_frosqr_out_lamb
  }
  if args.reset_cache_size is None or args.reset_cache_size <= 0:
    args.reset_cache_size = synth_dataset.N

  tic = time.time()
  model = mod.KMeansBatchAltModel(args.model_k, synth_dataset, init=args.init,
      replicates=args.reps, reg_params=reg_params,
      reset_patience=args.reset_patience, reset_jitter=args.reset_jitter,
      reset_try_tol=args.reset_try_tol, reset_accept_tol=args.reset_accept_tol,
      reset_stochastic=args.reset_stochastic,
      reset_low_value=args.reset_low_value,
      reset_value_thr=args.reset_value_thr, reset_sigma=args.reset_sigma,
      reset_cache_size=args.reset_cache_size, kpp_n_trials=args.kpp_n_trials)

  init_time = time.time() - tic
  model = model.to(device)

  if args.chkp_freq is None or args.chkp_freq <= 0:
    args.chkp_freq = args.epochs
  if args.stop_freq is None or args.stop_freq <= 0:
    args.stop_freq = -1

  # save args
  with open('{}/args.json'.format(args.out_dir), 'w') as f:
    json.dump(args.__dict__, f, sort_keys=True, indent=4)

  optimizer = None
  tr.train_loop(model, synth_data_loader, device, optimizer, args.out_dir,
      args.epochs, args.chkp_freq, args.stop_freq, scheduler=None,
      eval_rank=False, reset_unused=args.reset_unused, init_time=init_time)
  return


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Cluster synthetic kmeans data')
  parser.add_argument('--out-dir', type=str, required=True,
                      help='Output directory.')
  # data settings
  parser.add_argument('--k', type=int, default=10,
                      help='Number of clusters [default: 10]')
  parser.add_argument('--D', type=int, default=100,
                      help='Ambient dimension [default: 100]')
  parser.add_argument('--Ng', type=int, default=1000,
                      help='Points per group [default: 1000]')
  parser.add_argument('--sep', type=float, default=2.0,
                      help=('means separation in gaussian radius units '
                      '[default: 2.0]'))
  parser.add_argument('--data-seed', type=int, default=1904,
                      help='Data random seed [default: 1904]')
  # model settings
  parser.add_argument('--init', type=str, default='random',
                      help='Init mode (random, k-means++ [default: random]')
  parser.add_argument('--kpp-n-trials', type=int, default=None,
                      help='Number of k-means++ samples [default: 2 log n]')
  parser.add_argument('--reps', type=int, default=6,
                      help='Number of model replicates [default: 6]')
  parser.add_argument('--model-k', type=int, default=None,
                      help='Model number of clusters [default: k]')
  parser.add_argument('--b-frosqr-out-lamb', type=float, default=0.0,
                      help=('Frobenius squared b reg parameter, '
                      'outside assignment [default: 0.0]'))
  # training settings
  parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs to train [default: 50]')
  parser.add_argument('--reset-unused', type=ut.boolarg, default=False,
                      help='Reset nearly unused clusters [default: 0]')
  parser.add_argument('--reset-patience', type=int, default=2,
                      help=('Epochs to wait without obj decrease '
                      'before trying to reset [default: 2]'))
  parser.add_argument('--reset-jitter', type=ut.boolarg, default=True,
                      help='Jitter reset patience [default: 1]')
  parser.add_argument('--reset-low-value', type=ut.boolarg, default=False,
                      help='Choose low value cluster to reset [default: 0]')
  parser.add_argument('--reset-value-thr', type=float, default=0.1,
                      help=('Value threshold for low value clusters '
                      '[default: 0.1]'))
  parser.add_argument('--reset-stochastic', type=ut.boolarg, default=True,
                      help=('Choose reset substitution stochastically '
                      '[default: 1]'))
  parser.add_argument('--reset-try-tol', type=float, default=0.01,
                      help=('Objective decrease tolerance for deciding'
                      'when to reset [default: 0.01]'))
  parser.add_argument('--reset-accept-tol', type=float, default=0.01,
                      help=('Objective decrease tolerance for accepting'
                      'a reset [default: 1e-3]'))
  parser.add_argument('--reset-sigma', type=float, default=0.05,
                      help=('Scale of perturbation to add after reset '
                      '[default: 0.05]'))
  parser.add_argument('--reset-cache-size', type=int, default=None,
                      help='Num samples for reset assign obj [default: N]')
  parser.add_argument('--cuda', type=ut.boolarg, default=False,
                      help='Enable CUDA training [default: 0]')
  parser.add_argument('--num-threads', type=int, default=1,
                      help='Number of parallel threads to use [default: 1]')
  parser.add_argument('--seed', type=int, default=2018,
                      help='Training random seed [default: 2018]')
  parser.add_argument('--chkp-freq', type=int, default=None,
                      help='How often to save checkpoints [default: None]')
  parser.add_argument('--stop-freq', type=int, default=None,
                      help='How often to stop in ipdb [default: None]')
  args = parser.parse_args()

  train_synth_kmeans_cluster(args)
