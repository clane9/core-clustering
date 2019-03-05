"""Train and evaluate k-subspace model on synthetic union of subspaces."""

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
  if args.dist and use_cuda:
    cuda_devices = map(int, os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if len(cuda_devices) != dist.get_world_size():
      raise RuntimeError("Visible cuda devices must equal world size")
    device_id = cuda_devices[dist.get_rank()]
    torch.cuda.set_device(device_id)

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
  synth_data_loader = DataLoader(synth_dataset, batch_size=args.batch_size,
      shuffle=False, drop_last=True, **kwargs)

  # construct model
  torch.manual_seed(args.seed)
  if args.model_d is None or args.model_d <= 0:
    args.model_d = args.d
  if args.model_n is None or args.model_n <= 0:
    args.model_n = args.n
  args.assign_reg_terms = args.assign_reg_terms.split(',')
  if 'U' not in args.assign_reg_terms:
    args.U_lamb /= args.model_n
  if args.proj_form:
    model = mod.KSubspaceProjModel(args.model_n, args.model_d, args.D,
        args.affine, args.symmetric, U_lamb=args.U_lamb,
        coh_gamma=args.coh_gamma, soft_assign=args.soft_assign,
        c_sigma=args.c_sigma, assign_reg_terms=args.assign_reg_terms,
        size_scale=args.size_scale)
  else:
    model = mod.KSubspaceModel(args.model_n, args.model_d, args.D, args.affine,
        U_lamb=args.U_lamb, z_lamb=args.z_lamb, coh_gamma=args.coh_gamma,
        soft_assign=args.soft_assign, c_sigma=args.c_sigma,
        assign_reg_terms=args.assign_reg_terms, size_scale=args.size_scale)
  model = model.to(device)
  if args.dist:
    if use_cuda:
      model = torch.nn.parallel.DistributedDataParallel(model,
          device_ids=[device_id], output_device=device_id)
    else:
      model = torch.nn.parallel.DistributedDataParallelCPU(model)

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
  reset_kwargs = {'split_sigma': args.reset_split_sigma,
      'sample_p': args.reset_sample_p}
  tr.train_loop(model, synth_data_loader, device, optimizer,
      args.out_dir, args.epochs, args.chkp_freq, args.stop_freq,
      scheduler=None, dist_mode=args.dist, eval_rank=args.eval_rank,
      reset_unused=args.reset_unused, reset_kwargs=reset_kwargs)
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
  parser.add_argument('--proj-form', action='store_true', default=False,
                      help='Use projection matrix formulation')
  parser.add_argument('--model-n', type=int, default=None,
                      help='Model number of subspaces [default: n]')
  parser.add_argument('--model-d', type=int, default=None,
                      help='Model subspace dimension [default: d]')
  parser.add_argument('--symmetric', action='store_true',
                      help='Projection matrix is U^T')
  parser.add_argument('--U-lamb', type=float, default=1e-4,
                      help='Subspace reg parameter [default: 1e-4]')
  parser.add_argument('--z-lamb', type=float, default=0.1,
                      help='Coefficient reg parameter [default: 0.1]')
  parser.add_argument('--coh-gamma', type=float, default=0.0,
                      help='Coherence reg parameter [default: 0.0]')
  parser.add_argument('--soft-assign', type=float, default=0.1,
                      help='Soft assignment parameter [default: 0.1]')
  parser.add_argument('--c-sigma', type=float, default=0.01,
                      help='Assignment noise parameter [default: 0.01]')
  parser.add_argument('--assign-reg-terms', type=str, default='U,z',
                      help=('Assignment reg terms, subset of U,z,coh '
                          '[default: U,z]'))
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
  parser.add_argument('--optim', type=str, default='SGD',
                      help='Optimizer [default: SGD]')
  parser.add_argument('--reset-unused', action='store_true', default=False,
                      help='Whether to reset unused clusters')
  parser.add_argument('--reset-split-sigma', type=float, default=0.1,
                      help='Reset split perturbation parameter [default: 0.1]')
  parser.add_argument('--reset-sample-p', type=float, default=None,
                      help='Reset (over-)sampling parameter [default: None]')
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
