"""Train and evaluate online k-subspace clustering on range of real
datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import pickle

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import datasets as dat
import models as mod
import training as tr


def main():
  use_cuda = args.cuda and torch.cuda.is_available()
  torch.set_num_threads(args.num_threads)

  if args.dist:
    if use_cuda:
      dist.init_process_group(backend="nccl")
    else:
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

  device = torch.device('cuda' if use_cuda else 'cpu')
  if use_cuda:
    cuda_devices = map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if len(cuda_devices) != dist.get_world_size():
      raise RuntimeError("Visible cuda devices must equal world size")
    device_id = cuda_devices[dist.get_rank()]
    torch.cuda.set_device(device_id)

  # determines data sampling, initialization
  torch.manual_seed(args.seed)

  # load dataset
  dataset = dat.YouCVPR16ImageUoS(args.dataset)
  kwargs = {'num_workers': args.num_workers}
  if use_cuda:
    kwargs['pin_memory'] = True
  if args.dist:
    sampler = DistributedSampler(dataset, dist.get_world_size(),
        dist.get_rank())
  else:
    sampler = DistributedSampler(dataset, 1, 0)
  data_loader = DataLoader(dataset, batch_size=args.batch_size,
      sampler=sampler, drop_last=True, **kwargs)

  # construct model
  if args.proj_form:
    model = mod.KSubspaceProjModel(args.n, args.model_d, dataset.D,
        args.affine, args.symmetric, lamb=args.lamb,
        soft_assign=args.soft_assign, c_sigma=args.c_sigma,
        size_scale=args.size_scale)
  else:
    model = mod.KSubspaceModel(args.n, args.model_d, dataset.D, args.affine,
        U_lamb=args.lamb, z_lamb=args.z_lamb, soft_assign=args.soft_assign,
        c_sigma=args.c_sigma, size_scale=args.size_scale)
  model = model.to(device)
  if args.dist:
    if use_cuda:
      model = torch.nn.parallel.DistributedDataParallel(model,
          device_ids=[device_id], output_device=device_id)
    else:
      model = torch.nn.parallel.DistributedDataParallelCPU(model)

  # optimizer
  optimizer = torch.optim.SGD(model.parameters(), lr=args.init_lr,
      momentum=args.momentum, nesterov=args.nesterov)

  chkp_freq = args.chkp_freq
  if chkp_freq is None or chkp_freq <= 0:
    chkp_freq = args.epochs
  stop_freq = args.stop_freq
  if stop_freq is None or stop_freq <= 0:
    stop_freq = -1
  tr.train_loop(model, data_loader, device, optimizer,
      args.out_dir, args.epochs, chkp_freq, stop_freq, scheduler=None,
      dist_mode=args.dist, eval_rank=args.eval_rank)
  return


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Cluster real UoS data')
  parser.add_argument('--out-dir', type=str, required=True,
                      help='Output directory.')
  parser.add_argument('--dataset', type=str, default='mnist_sc_pca',
                      help='Real dataset [default: mnist_sc_pca].',
                      choices=['mnist_sc_pca', 'coil100'])
  # model settings
  parser.add_argument('--proj-form', action='store_true', default=False,
                      help='Use projection matrix formulation')
  parser.add_argument('--affine', action='store_true',
                      help='Affine setting')
  parser.add_argument('--symmetric', action='store_true',
                      help='Projection matrix is U^T')
  parser.add_argument('--model-d', type=int, default=10,
                      help='Model subspace dimension [default: 10]')
  parser.add_argument('--lamb', type=float, default=1e-4,
                      help='Subspace reg parameter [default: 1e-4]')
  parser.add_argument('--z-lamb', type=float, default=0.1,
                      help='Coefficient reg parameter [default: 0.1]')
  parser.add_argument('--soft-assign', type=float, default=0.1,
                      help='Soft assignment parameter [default: 0.1]')
  parser.add_argument('--c-sigma', type=float, default=0.01,
                      help='Assignment noise parameter [default: 0.01]')
  parser.add_argument('--size-scale', action='store_true',
                      help=('Scale gradients to compensate for cluster '
                          'size imbalance'))
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
  parser.add_argument('--chkp-freq', type=int, default=10,
                      help='How often to save checkpoints [default: 10]')
  parser.add_argument('--stop-freq', type=int, default=None,
                      help='How often to stop in ipdb [default: None]')
  args = parser.parse_args()

  main()
