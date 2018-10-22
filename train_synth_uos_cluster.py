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
import torch.nn.functional as F
import torch.optim as optim

from datasets import SynthUoSDataset
from models import ManifoldClusterModel, SubspaceModel
from utils import AverageMeter, get_learning_rate, save_checkpoint


def main():
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  use_cuda = not args.no_cuda and torch.cuda.is_available()
  device = torch.device('cuda' if use_cuda else 'cpu')

  # construct dataset
  synth_dataset = SynthUoSDataset(args.n, args.d, args.D, args.Ng,
      args.affine, args.sigma, args.data_seed)
  N = args.n*args.Ng
  batch_size = np.min([args.batch_size, N])
  kwargs = {'num_workers': 1}
  if use_cuda:
    kwargs['pin_memory'] = True
  synth_data_loader = torch.utils.data.DataLoader(synth_dataset,
      batch_size=batch_size, shuffle=(batch_size != N), **kwargs)

  # construct model
  group_models = [SubspaceModel(args.d, args.D, args.affine)
      for _ in range(args.n)]
  model = ManifoldClusterModel(args.n, args.d, args.D, N,
      args.C_p, args.C_sigma, group_models)

  # objective function
  def objfun(model, ii, x):
    x_ = model(ii)
    loss = F.mse_loss(x, x_)
    Ureg, Vreg, Creg = model.reg()
    # NOTE: scaling reg on U, V by 1/N
    reg = (args.UV_lamb/N)*(Ureg + Vreg) + args.C_gamma*Creg
    obj = loss + reg
    return obj, loss, reg

  # optimizer & lr schedule
  optimizer = optim.SGD(model.parameters(), lr=args.init_lr)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
      factor=0.5, patience=5, threshold=1e-4)

  printformstr = ('(epoch {:d}/{:d}) lr={:.3e} obj={:.3e} loss={:.3e} '
      'reg={:.3e} samp/s={:.0f}')
  logheader = 'Epoch,Obj,Loss,Reg,Samp.s'
  logformstr = '{:d},{:.9e},{:.9e},{:.9e},{:.9e},{:.0f}'
  val_logf = '{}/val_log.csv'.format(args.out_dir)
  with open(val_logf, 'w') as f:
    print(logheader, file=f)

  # training loop
  best_obj = float('inf')
  for epoch in range(args.epochs):
    obj, loss, reg, sampsec = train_epoch(model, objfun,
        synth_data_loader, device, optimizer)

    lr = get_learning_rate(optimizer)

    with open(val_logf, 'a') as f:
      print(logformstr.format(epoch, lr, obj, loss, reg, sampsec))
    print(printformstr.format(epoch, args.epochs, lr, obj, loss, reg, sampsec))

    is_best = obj < best_obj
    best_obj = min(obj, best_obj)
    save_checkpoint({
        'epoch': epoch+1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'obj': obj},
        is_best,
        filename='{}/checkpoint.pth.tar'.format(args.out_dir),
        best_filename='{}/model_best.pth.tar'.format(args.out_dir))

    scheduler.step(obj)
  return


def train_epoch(model, objfun, data_loader, device, optimizer):
  model.train()
  obj, loss, reg, sampsec = [AverageMeter() for _ in range(4)]
  tic = time.time()
  for kk, (ii, x) in enumerate(data_loader):
    # forward
    ii, x = ii.to(device), x.to(device)
    batch_obj, batch_loss, batch_reg = objfun(ii, x, model)
    batch_size = x.size(0)

    # backward
    optimizer.zero_grad()
    batch_obj.backward()
    optimizer.step()

    obj.update(batch_obj, batch_size)
    loss.update(batch_loss, batch_size)
    reg.update(batch_reg, batch_size)

    batch_time = time.time() - tic
    sampsec.update(batch_size/batch_time, batch_size)
    tic = time.time()

    divergence = torch.isnan(batch_obj)
    if divergence:
      raise RuntimeError('Divergence! NaN objective.')
  return obj.avg, loss.avg, reg.val, sampsec.avg


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
  parser.add_argument('--UV-lamb', type=float, default=.01,
                      help='U, V l2 reg parameter [default: .01]')
  parser.add_argument('--C-gamma', type=float, default=.01,
                      help='C l1 reg parameter [default: .01]')
  parser.add_argument('--C-p', type=int, default=1,
                      help='C activation power [default: 1]')
  parser.add_argument('--C-sigma', type=float, default=0.,
                      help='C noise sigma [default: 0.]')
  # training settings
  parser.add_argument('--batch-size', type=int, default=20,
                      help='Input batch size for training [default: 20]')
  parser.add_argument('--epochs', type=int, default=500,
                      help='Number of epochs to train [default: 500]')
  parser.add_argument('--init-lr', type=float, default=0.001,
                      help='Initial learning rate [default: 0.001]')
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
