"""Train and evaluate factorized manifold clustering model on synthetic union
of subspaces."""

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

from datasets import SynthUoSDataset
from models import GSManifoldClusterModel, ManifoldClusterModel, SubspaceModel
import utils as ut

import ipdb

CHKP_FREQ = 10
STOP_FREQ = 500
LR_INCR_FREQ = 100


def main():
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device('cuda' if use_cuda else 'cpu')

  # construct dataset
  synth_dataset = SynthUoSDataset(args.n, args.d, args.D, args.Ng,
      args.affine, args.sigma, args.data_seed)
  N = args.n*args.Ng
  batch_size = args.batch_size
  if args.batch_size <= 0 or args.batch_size > N:
    batch_size = N
  kwargs = {'num_workers': 1}
  if use_cuda:
    kwargs['pin_memory'] = True
  synth_data_loader = torch.utils.data.DataLoader(synth_dataset,
      batch_size=batch_size, shuffle=(batch_size != N), **kwargs)

  # construct model
  group_models = [SubspaceModel(args.d, args.D, args.affine)
      for _ in range(args.n)]
  # model = ManifoldClusterModel(args.n, args.d, args.D, N,
  #     args.C_sigma, group_models)
  model = GSManifoldClusterModel(args.n, args.d, args.D, N, group_models)

  # objective function
  def objfun(model, ii, x):
    x_ = model(ii)
    # measure whether reconstruction x_ is close in size to x.
    norm_x = torch.norm(x.detach(), 2, dim=1)
    norm_x_ = torch.norm(x_.detach(), 2, dim=1)
    norm_x_ = torch.mean(norm_x_ / (norm_x + 1e-8))
    # loss, reg, obj
    loss = torch.mean(torch.sum((x - x_)**2, dim=1))
    # Ureg, Vreg, Creg = model.reg(ii)
    # reg = args.UV_lamb*(Ureg + Vreg) + args.C_gamma*Creg
    reg = args.UV_lamb*model.reg(ii)
    obj = loss + reg
    return obj, loss, reg, norm_x_

  # optimizer & lr schedule
  optimizer = optim.SGD(model.parameters(), lr=args.init_lr,
      momentum=0.9, nesterov=True)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
      factor=0.5, patience=50, threshold=1e-4)

  printformstr = ('(epoch {:d}/{:d}) lr={:.3e} err={:.4f} obj={:.3e} '
      'loss={:.3e} reg={:.3e} |x_|={:.3e} samp/s={:.0f}')
  logheader = 'Epoch,LR,Err,Obj,Loss,Reg,Norm.x_,Samp.s'
  logformstr = '{:d},{:.9e},{:.9f},{:.9e},{:.9e},{:.9e},{:.9e},{:.0f}'
  val_logf = '{}/val_log.csv'.format(args.out_dir)
  with open(val_logf, 'w') as f:
    print(logheader, file=f)

  # training loop
  best_obj = float('inf')
  for epoch in range(args.epochs):
    obj, loss, reg, norm_x_, sampsec = train_epoch(model, objfun,
        synth_data_loader, device, optimizer)
    cluster_error, groups = ut.eval_cluster_error(model.get_groups(),
        synth_dataset.groups)
    lr = ut.get_learning_rate(optimizer)

    with open(val_logf, 'a') as f:
      print(logformstr.format(epoch, lr, cluster_error, obj, loss, reg,
          norm_x_, sampsec), file=f)
    print(printformstr.format(epoch, args.epochs, lr, cluster_error, obj, loss,
        reg, norm_x_, sampsec))

    is_best = obj < best_obj
    best_obj = min(obj, best_obj)
    if is_best or epoch % CHKP_FREQ == 0:
      ut.save_checkpoint({
          'epoch': epoch+1,
          'model': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'err': cluster_error,
          'obj': obj},
          is_best,
          filename='{}/checkpoint.pth.tar'.format(args.out_dir),
          best_filename='{}/model_best.pth.tar'.format(args.out_dir))

    scheduler.step(obj)

    if epoch % STOP_FREQ == 0:
      ipdb.set_trace()

    # if epoch % LR_INCR_FREQ == 0:
    #   ut.adjust_learning_rate(optimizer, 1.1*lr)
  return


def train_epoch(model, objfun, data_loader, device, optimizer):
  model.train()
  obj, loss, reg, norm_x_, sampsec = [ut.AverageMeter() for _ in range(5)]
  tic = time.time()
  for kk, (ii, x) in enumerate(data_loader):
    # forward
    ii, x = ii.to(device), x.to(device)
    batch_obj, batch_loss, batch_reg, batch_norm_x_ = objfun(model, ii, x)
    batch_size = x.size(0)

    # backward
    optimizer.zero_grad()
    batch_obj.backward()
    optimizer.step()

    obj.update(batch_obj, batch_size)
    loss.update(batch_loss, batch_size)
    reg.update(batch_reg, batch_size)
    norm_x_.update(batch_norm_x_, batch_size)

    batch_time = time.time() - tic
    sampsec.update(batch_size/batch_time, batch_size)
    tic = time.time()

    divergence = torch.isnan(batch_obj)
    if divergence:
      raise RuntimeError('Divergence! NaN objective.')
  return obj.avg, loss.avg, reg.avg, norm_x_.avg, sampsec.avg


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
  parser.add_argument('--Ng', type=int, default=100,
                      help='Points per group [default: 100]')
  parser.add_argument('--affine', action='store_true',
                      help='Affine setting')
  parser.add_argument('--sigma', type=float, default=0.,
                      help='Data noise sigma [default: 0.]')
  parser.add_argument('--data-seed', type=int, default=1904,
                      help='Data random seed [default: 1904]')
  # model settings
  parser.add_argument('--UV-lamb', type=float, default=.01,
                      help='U, V l2 reg parameter [default: .01]')
  parser.add_argument('--C-gamma', type=float, default=.01,
                      help='C l1 reg parameter [default: .01]')
  parser.add_argument('--C-sigma', type=float, default=0.,
                      help='C noise sigma [default: 0.]')
  # training settings
  parser.add_argument('--batch-size', type=int, default=-1,
                      help='Input batch size for training [default: -1]')
  parser.add_argument('--epochs', type=int, default=5000,
                      help='Number of epochs to train [default: 500]')
  parser.add_argument('--init-lr', type=float, default=0.01,
                      help='Initial learning rate [default: 0.01]')
  parser.add_argument('--no-cuda', action='store_true', default=False,
                      help='Disables CUDA training')
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
