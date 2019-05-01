"""Train and evaluate k-means model on synthetic data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import json

import torch

import datasets as dat
import models as mod
import training as tr


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
  synth_dataset = dat.SynthKMeansDataset(args.n, args.D, args.Ng,
      separation=args.sep, seed=args.data_seed)
  synth_data_loader = None

  # construct model
  torch.manual_seed(args.seed)
  if args.model_n is None or args.model_n <= 0:
    args.model_n = args.n
  args.reset_metric = args.reset_metric.lower()
  reset_metric = 'value' if args.reset_metric == 'none' else args.reset_metric
  if args.reset_patience is None or args.reset_patience < 0:
    args.reset_patience = 2
  if args.reset_warmup is None or args.reset_warmup < 0:
    args.reset_warmup = args.reset_patience
  reg_params = {
      'b_frosqr_out': args.b_frosqr_out_lamb
  }
  model = mod.KMeansBatchAltModel(args.model_n, synth_dataset, args.init,
      args.reps, reg_params=reg_params, reset_metric=reset_metric,
      unused_thr=args.unused_thr, reset_patience=args.reset_patience,
      reset_warmup=args.reset_warmup, reset_obj=args.reset_obj,
      reset_decr_tol=args.reset_decr_tol, reset_sigma=0.0,
      reset_batch_size=args.reset_batch_size)
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
      eval_rank=False, reset_unused=(args.reset_metric != 'none'))
  return


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Cluster synthetic kmeans data')
  parser.add_argument('--out-dir', type=str, required=True,
                      help='Output directory.')
  # data settings
  parser.add_argument('--n', type=int, default=10,
                      help='Number of subspaces [default: 10]')
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
                      help='Init mode (random, k-means++) [default: random]')
  parser.add_argument('--reps', type=int, default=5,
                      help='Number of model replicates [default: 5]')
  parser.add_argument('--model-n', type=int, default=None,
                      help='Model number of clusters [default: n]')
  parser.add_argument('--b-frosqr-out-lamb', type=float, default=0.0,
                      help=('Frobenius squared b reg parameter, '
                      'outside assignment [default: 0.0]'))
  # training settings
  parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs to train [default: 50]')
  parser.add_argument('--reset-metric', type=str, default='value',
                      help='Reset metric (value, size, none) [default: value]')
  parser.add_argument('--reset-obj', type=str, default='full',
                      help='Reset objective (assign, full) [default: full]')
  parser.add_argument('--unused-thr', type=float, default=0.25,
                      help=('Threshold for identifying reinit candidates, '
                      'relative to observed max [default: .25]'))
  parser.add_argument('--reset-patience', type=int, default=None,
                      help='Steps to wait between resets [default: 2 epochs]')
  parser.add_argument('--reset-warmup', type=int, default=None,
                      help='Extra steps to wait at start [default: 2 epochs]')
  parser.add_argument('--reset-decr-tol', type=float, default=1e-4,
                      help=('Relative objective decrease tolerance to reset '
                      '[default: 1e-4]'))
  parser.add_argument('--reset-batch-size', type=int, default=None,
                      help='Num samples for reset assign obj [default: N]')
  parser.add_argument('--cuda', action='store_true', default=False,
                      help='Enables CUDA training')
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
