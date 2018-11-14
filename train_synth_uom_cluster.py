"""Train and evaluate factorized manifold clustering model on synthetic union
of smooth manifolds."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import pickle
import time

import numpy as np
import torch
import torch.optim as optim

import datasets as dat
import models as mod
import optimizers as opt
import utils as ut

import ipdb

CHKP_FREQ = 50
STOP_FREQ = 10


def main():
  use_cuda = args.cuda and torch.cuda.is_available()
  device = torch.device('cuda' if use_cuda else 'cpu')
  torch.set_num_threads(args.num_threads)

  # construct dataset
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

  # construct model
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  H_ = int(np.ceil(args.H*args.over_param))
  group_models = [mod.ResidualManifoldModel(args.d, args.D, H_,
      args.drop_p, args.lamb) for _ in range(args.n)]
  model = mod.KManifoldClusterModel(args.n, args.d, args.D, N,
      batch_size, group_models)

  # optimizer & lr schedule
  optimizer = opt.KManifoldSGD(model, lr=args.init_lr, lamb_U=args.lamb,
      lamb_V=args.lamb, momentum=0.9, nesterov=True,
      soft_assign=args.soft_assign)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
      factor=0.8, patience=20, threshold=1e-3)

  printformstr = ('(epoch {:d}/{:d}) lr={:.3e} err={:.4f} obj={:.3e} '
      'loss={:.3e} reg(U)(V)={:.3e},{:.3e},{:.3e} sprs={:.2f} '
      '|x_|={:.3e} samp/s={:.0f}')
  logheader = ('Epoch,LR,Err,Obj,Loss,Reg,U.reg,V.reg,'
      'Sprs,Norm.x_,Samp.s')
  logformstr = ('{:d},{:.9e},{:.9f},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},'
      '{:.9e},{:.9e},{:.0f}')
  val_logf = '{}/val_log.csv'.format(args.out_dir)
  with open(val_logf, 'w') as f:
    print(logheader, file=f)

  # training loop
  best_obj = float('inf')
  model.train()
  for epoch in range(1, args.epochs+1):
    metrics = train_epoch(synth_data_loader, device, optimizer)
    cluster_error, _ = ut.eval_cluster_error(model.get_groups(),
        synth_dataset.groups)
    lr = ut.get_learning_rate(optimizer)

    with open(val_logf, 'a') as f:
      print(logformstr.format(epoch, lr, cluster_error, *metrics), file=f)
    print(printformstr.format(epoch, args.epochs, lr, cluster_error, *metrics))

    obj = metrics[0]
    is_best = obj < best_obj
    best_obj = min(obj, best_obj)
    if epoch == 1 or epoch % CHKP_FREQ == 0:
      ut.save_checkpoint({
          'epoch': epoch,
          'model': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'err': cluster_error,
          'obj': obj},
          is_best,
          filename='{}/checkpoint{}.pth.tar'.format(args.out_dir, epoch),
          best_filename='{}/model_best.pth.tar'.format(args.out_dir))

    scheduler.step(obj)
    if args.soft_assign_decay and args.soft_assign > 0:
      optimizer.set_soft_assign(args.soft_assign/(epoch+1))

    if STOP_FREQ > 0 and epoch % STOP_FREQ == 0:
      ipdb.set_trace()
  return


def train_epoch(data_loader, device, optimizer):
  """train model for one epoch and record convergence measures."""
  (obj, loss, reg, Ureg, Vreg,
      sprs, norm_x_, sampsec) = [ut.AverageMeter() for _ in range(8)]
  tic = time.time()
  for kk, (ii, x) in enumerate(data_loader):
    # forward
    ii, x = ii.to(device), x.to(device)
    (batch_obj, batch_loss, batch_reg, batch_Ureg, batch_Vreg,
        batch_sprs, batch_norm_x_) = optimizer.step(ii, x)
    batch_size = x.size(0)

    obj.update(batch_obj, batch_size)
    loss.update(batch_loss, batch_size)
    reg.update(batch_reg, batch_size)
    Ureg.update(batch_Ureg, batch_size)
    Vreg.update(batch_Vreg, batch_size)
    sprs.update(batch_sprs, batch_size)
    norm_x_.update(batch_norm_x_, batch_size)

    batch_time = time.time() - tic
    sampsec.update(batch_size/batch_time, batch_size)
    tic = time.time()

    if torch.isnan(batch_obj):
      raise RuntimeError('Divergence! NaN objective.')
  return (obj.avg, loss.avg, reg.avg,
      Ureg.avg, Vreg.avg, sprs.avg, norm_x_.avg, sampsec.avg)


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Cluster synthetic UoS data')
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
  parser.add_argument('--sigma', type=float, default=0.,
                      help='Data noise sigma [default: 0.]')
  parser.add_argument('--data-seed', type=int, default=1904,
                      help='Data random seed [default: 1904]')
  # model settings
  parser.add_argument('--over-param', type=float, default=1.0,
                      help=('Over-parameterization in manifold models '
                      '[default: 1.0]'))
  parser.add_argument('--drop-p', type=float, default=0.0,
                      help='Dropout in manifold models [default: 0.0]')
  parser.add_argument('--lamb', type=float, default=.1,
                      help='reg parameter [default: .1]')
  parser.add_argument('--soft-assign', type=float, default=0.0,
                      help='soft assignment parameter [default: 0.0]')
  parser.add_argument('--soft-assign-decay', action='store_true',
                      help='decay soft assignment parameter at a rate 1/k')
  # training settings
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
