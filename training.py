from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch
from torch import optim

import utils as ut

import ipdb


def train_loop(model, data_loader, device, optimizer, out_dir,
      epochs=200, chkp_freq=50, stop_freq=-1):
  """Train k-manifold model for series of epochs.

  Args:
    model: k-manifold ae model instance.
    data_loader: training dataset loader
    device: torch compute device id
    optimizer: k-manifold ae optimizer instance
    out_dir: path to output directory, must already exist
    epochs: number of epochs to run for (default: 200)
    chkp_freq: how often to save checkpoint (default: 50)
    stop_freq: how often to stop for debugging (default: -1)
  """

  printformstr = ('(epoch {:d}/{:d}) lr={:.3e} err={:.4f} obj={:.3e} '
      'loss={:.3e} reg={:.3e} sprs={:.2f} |x_|={:.3e} '
      'samp/s={:.0f} rtime={:.3f}')
  logheader = ('Epoch,LR,Err,Obj,Loss,Reg,Sprs,Norm.x_,Samp.s,RT')
  logformstr = ('{:d},{:.9e},{:.9f},{:.9e},{:.9e},{:.9e},'
      '{:.9e},{:.9e},{:.0f},{:.9f}')
  val_logf = '{}/val_log.csv'.format(out_dir)
  with open(val_logf, 'w') as f:
    print(logheader, file=f)

  min_lr = 1e-6*ut.get_learning_rate(optimizer)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
      factor=0.5, patience=10, threshold=1e-4, min_lr=min_lr)

  # training loop
  best_obj = float('inf')
  lr = float('inf')
  model.train()
  for epoch in range(1, epochs+1):
    try:
      metrics = train_epoch(data_loader, optimizer, device)
      lr = ut.get_learning_rate(optimizer)

      with open(val_logf, 'a') as f:
        print(logformstr.format(epoch, lr, *metrics), file=f)
      print(printformstr.format(epoch, epochs, lr, *metrics))

      cluster_error, obj = metrics[:2]
      is_conv = lr <= min_lr or epoch == epochs
      scheduler.step(obj)
    except Exception as err:
      print('Error: {}'.format(err))
      with open('{}/error'.format(out_dir), 'w') as f:
        print(err, file=f)
      obj = float('inf')
      cluster_error = float('inf')
      is_conv = True

    is_best = obj < best_obj
    best_obj = min(obj, best_obj)
    if epoch == 1 or epoch % chkp_freq == 0 or is_conv:
      ut.save_checkpoint({
          'epoch': epoch,
          'model': model.state_dict(),
          'optimizer': optimizer.state_dict(),
          'err': cluster_error,
          'obj': obj},
          is_best,
          filename='{}/checkpoint{}.pth.tar'.format(out_dir, epoch),
          best_filename='{}/model_best.pth.tar'.format(out_dir))

    if stop_freq > 0 and epoch % stop_freq == 0:
      ipdb.set_trace()

    if is_conv:
      break
  return


def train_epoch(data_loader, optimizer, device):
  """train model for one epoch and record convergence measures."""
  (obj, loss, reg, sprs, norm_x_,
      conf_mat, sampsec) = [ut.AverageMeter() for _ in range(7)]
  epoch_tic = time.time()
  for _, x, groups in data_loader:
    # forward
    tic = time.time()
    x = x.to(device)
    (batch_obj, batch_loss, batch_reg, batch_sprs,
        batch_norm_x_, batch_conf_mat) = optimizer.step(x, groups)
    batch_size = x.size(0)

    obj.update(batch_obj, batch_size)
    loss.update(batch_loss, batch_size)
    reg.update(batch_reg, batch_size)
    sprs.update(batch_sprs, batch_size)
    norm_x_.update(batch_norm_x_, batch_size)
    conf_mat.update(batch_conf_mat, 1)

    batch_time = time.time() - tic
    sampsec.update(batch_size/batch_time, batch_size)

    if torch.isnan(batch_obj):
      raise RuntimeError('Divergence! NaN objective.')
  rtime = time.time() - epoch_tic
  cluster_error = ut.eval_cluster_error(conf_mat.sum)
  return (cluster_error, obj.avg, loss.avg, reg.avg,
      sprs.avg, norm_x_.avg, sampsec.avg, rtime)
