from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import ipdb
import numpy as np
import torch
from torch import optim
import torch.distributed as dist
from datasets import DistributedSampler
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

import utils as ut


def train_loop(model, data_loader, device, optimizer, out_dir,
      epochs=200, chkp_freq=50, stop_freq=-1, dist_mode=False,
      eval_rank=False):
  """Train k-manifold model for series of epochs.

  Args:
    model: k-manifold model instance.
    data_loader: training dataset loader
    device: torch compute device id
    optimizer: k-manifold optimizer instance
    out_dir: path to output directory, must already exist
    epochs: number of epochs to run for (default: 200)
    chkp_freq: how often to save checkpoint (default: 50)
    stop_freq: how often to stop for debugging (default: -1)
    dist_mode: whether in mpi distributed mode (default: False)
    eval_rank: evaluate ranks of group models, only implemented for subspace
      models (default: False)
  """
  printformstr = ('(epoch {:d}/{:d}) lr={:.3e} err={:.4f} obj={:.3e} '
      'loss={:.3e} reg(U)(V)={:.3e},{:.3e},{:.3e} ')
  if eval_rank:
    printformstr += 'rank(min)(max)={:.0f},{:.0f},{:.0f} '
  printformstr += 'sprs={:.2f} |x_|={:.3e} samp/s={:.0f} rtime={:.3f}'
  logheader = 'Epoch,LR,Err,Obj,Loss,Reg,U.reg,V.reg,'
  if eval_rank:
    logheader += 'Rank.med,Rank.min,Rank.max,'
  logheader += 'Sprs,Norm.x_,Samp.s,RT'
  logformstr = '{:d},{:.9e},{:.9f},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},'
  if eval_rank:
    logformstr += '{:.9f},{:.0f},{:.0f},'
  logformstr += '{:.9e},{:.9e},{:.0f},{:.9f}'
  val_logf = '{}/val_log.csv'.format(out_dir)

  if dist_mode and not (dist.is_initialized()):
    raise RuntimeError("Distributed package not initialized")
  is_logging = (not dist_mode) or (dist.get_rank() == 0)

  if is_logging:
    with open(val_logf, 'w') as f:
      print(logheader, file=f)
  conf_mats = np.zeros((epochs, model.k, data_loader.dataset.classes.size),
      dtype=np.int64)
  if eval_rank:
    svs = np.zeros((epochs, model.k, model.d), dtype=np.float32)
  else:
    svs = None

  min_lr = 1e-6*ut.get_learning_rate(optimizer)
  scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
      factor=0.5, patience=10, threshold=1e-3, min_lr=min_lr)

  # training loop
  err = None
  lr = float('inf')
  model.train()
  try:
    for epoch in range(1, epochs+1):
      # deterministic data shuffling by epoch for this sampler
      if isinstance(data_loader.sampler, DistributedSampler):
        data_loader.sampler.set_epoch(epoch)

      metrics, conf_mats[epoch-1, :] = train_epoch(model, data_loader,
          optimizer, device, dist_mode)
      lr = ut.get_learning_rate(optimizer)

      if eval_rank:
        ranks, epoch_svs = zip(*[gm.rank() for gm in model.group_models])
        svs[epoch-1, :] = torch.stack(epoch_svs).cpu().numpy()

        ranks = torch.stack(ranks).cpu().numpy()
        rank_stats = np.median(ranks), ranks.min(), ranks.max()
        metrics = metrics[:6] + rank_stats + metrics[6:]

      if is_logging:
        with open(val_logf, 'a') as f:
          print(logformstr.format(epoch, lr, *metrics), file=f)
        print(printformstr.format(epoch, epochs, lr, *metrics))

      cluster_error, obj = metrics[:2]
      is_conv = lr <= min_lr or epoch == epochs
      scheduler.step(obj)

      if is_logging and (epoch == 1 or epoch % chkp_freq == 0 or is_conv):
        ut.save_checkpoint({
            'epoch': epoch,
            'model': model.state_dict(),
            'err': cluster_error,
            'obj': obj},
            is_best=False,
            filename='{}/checkpoint{}.pth.tar'.format(out_dir, epoch))

      if (not dist_mode) and stop_freq > 0 and epoch % stop_freq == 0:
        ipdb.set_trace()

      if is_conv:
        break

      # test whether all processes are error free and safe to continue. Is
      # there a better way?
      if dist_mode:
        err_count = torch.tensor(0.)
        dist.all_reduce(err_count, op=dist.reduce_op.SUM)
        if err_count > 0:
          raise RuntimeError("dist error")

  except Exception as e:
    err = e
    if str(err) != "dist error":
      print('{}: {}'.format(type(err), err))
      with open("{}/error".format(out_dir), "a") as f:
        print('{}: {}'.format(type(err), err), file=f)
      if dist_mode:
        err_count = torch.tensor(1.)
        dist.all_reduce(err_count, op=dist.reduce_op.SUM)

  finally:
    if is_logging:
      with open('{}/conf_mats.npz'.format(out_dir), 'wb') as f:
        np.savez(f, conf_mats=conf_mats[:epoch, :])
      if eval_rank:
        with open('{}/svs.npz'.format(out_dir), 'wb') as f:
          np.savez(f, svs=svs[:epoch, :])
    if err is not None:
      raise err

  return conf_mats, svs


def train_epoch(model, data_loader, optimizer, device, dist_mode=False):
  """train model for one epoch and record convergence measures."""
  (obj, loss, reg, Ureg, Vreg, sprs, norm_x_,
      conf_mat, sampsec) = [ut.AverageMeter() for _ in range(9)]
  epoch_tic = time.time()
  for ii, x, groups in data_loader:
    # opt step
    tic = time.time()
    ii, x = ii.to(device), x.to(device)
    # metrics: obj, loss, reg, Ureg, Vreg, sprs, norm_x_
    batch_metrics = optimizer.step(ii, x)

    # eval batch cluster confusion
    batch_conf_mat = ut.eval_confusion(model.get_groups(), groups, k=model.k,
        true_classes=data_loader.dataset.classes)
    batch_metrics += (batch_conf_mat,)
    if dist_mode:
      coalesced = _flatten_dense_tensors(batch_metrics)
      dist.all_reduce(coalesced, op=dist.reduce_op.SUM)
      coalesced /= dist.get_world_size()
      batch_metrics = _unflatten_dense_tensors(coalesced, batch_metrics)
      # only conf_mat should not be averaged.
      batch_metrics[-1] *= dist.get_world_size()

    batch_size = x.size(0)
    if dist_mode:
      batch_size *= dist.get_world_size()

    for met, batch_met in zip([obj, loss, reg, Ureg, Vreg, sprs, norm_x_],
          batch_metrics[:-1]):
      met.update(batch_met, batch_size)
    conf_mat.update(batch_metrics[-1], 1)

    batch_time = time.time() - tic
    sampsec.update(batch_size/batch_time, batch_size)

    if torch.isnan(batch_metrics[0]):
      raise RuntimeError('Divergence! NaN objective.')

  rtime = time.time() - epoch_tic
  cluster_error, conf_mat = ut.eval_cluster_error(conf_mat.sum)
  return ((cluster_error, obj.avg, loss.avg, reg.avg, Ureg.avg, Vreg.avg,
      sprs.avg, norm_x_.avg, sampsec.avg, rtime), conf_mat)
