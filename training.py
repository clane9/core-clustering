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


# catalog of metric_printformstrs and metric_logheaders
print_prefix = 'err={:.4f} obj={:.3e} loss={:.3e} reg={:.3e}'
print_adv_reg_str = 'adv_reg={:.3e} D_x={:.3f} D_G_z={:.3f}'
print_rank_str = 'rank(min)(max)={:.0f},{:.0f},{:.0f}'
print_suffix = 'sprs={:.2f} |x_|={:.3e} samp/s={:.0f} rtime={:.3f}'

metric_printformstrs = {
    'default': ' '.join([print_prefix, print_suffix]),
    'adv_reg': ' '.join([print_prefix, print_adv_reg_str, print_suffix]),
    'default_rk': ' '.join([print_prefix, print_rank_str, print_suffix]),
    'adv_reg_rk': ' '.join([print_prefix, print_adv_reg_str, print_rank_str,
        print_suffix])}

log_prefix = 'Err,Obj,Loss,Reg'
log_adv_reg_str = 'Adv.reg,D.x,D.G.z'
log_rank_str = 'Rank.med,Rank.min,Rank.max'
log_suffix = 'Sprs,Norm.x_,Samp.s,RT'

metric_logheaders = {
    'default': ','.join([log_prefix, log_suffix]),
    'adv_reg': ','.join([log_prefix, log_adv_reg_str, log_suffix]),
    'default_rk': ','.join([log_prefix, log_rank_str, log_suffix]),
    'adv_reg_rk': ','.join([log_prefix, log_adv_reg_str, log_rank_str,
        log_suffix])}


def train_loop(model, data_loader, device, optimizer, metric_printformstr,
      out_dir=None, metric_logheader=None, epochs=20, chkp_freq=10,
      stop_freq=-1, scheduler=None, dist_mode=False, eval_rank=False):
  """Train k-manifold model for series of epochs.

  Args:
    model: k-manifold model instance.
    data_loader: training dataset loader
    device: torch compute device id
    optimizer: k-manifold optimizer instance
    metric_printformstr: format string for optimizer metrics
    out_dir: path to output directory, must already exist (default: None)
    metric_logheader: log header for optimizer metrics (default: None)
    epochs: number of epochs to run for (default: 20)
    chkp_freq: how often to save checkpoint (default: 10)
    stop_freq: how often to stop for debugging (default: -1)
    scheduler: lr scheduler. If None, use reduce on plateau (default: None)
    dist_mode: whether in mpi distributed mode (default: False)
    eval_rank: evaluate ranks of subspaces (default: False)
  """
  printformstr = '(epoch {:d}/{:d}) lr={:.3e} ' + metric_printformstr
  is_logging = (((not dist_mode) or (dist.get_rank() == 0)) and
      out_dir is not None)
  if is_logging and metric_logheader is not None:
    logheader = 'Epoch,LR' + metric_logheader
    logformstr = ','.join(logheader.count(',')*['{:.12e}'])
    val_logf = '{}/val_log.csv'.format(out_dir)
    with open(val_logf, 'w') as f:
      print(logheader, file=f)
  else:
    val_logf = None

  if dist_mode and not (dist.is_initialized()):
    raise RuntimeError("Distributed package not initialized")

  conf_mats = np.zeros((epochs, model.k, data_loader.dataset.classes.size),
      dtype=np.int64)
  if eval_rank:
    svs = np.zeros((epochs, model.k, model.d), dtype=np.float32)
  else:
    svs = None

  if scheduler is None:
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

      metrics, conf_mats[epoch-1, :], epoch_svs = train_epoch(model,
          data_loader, optimizer, device, dist_mode)
      lr = ut.get_learning_rate(optimizer)

      if eval_rank:
        # otherwise epoch_svs is None
        svs[epoch-1, :] = epoch_svs

      if val_logf is not None:
        with open(val_logf, 'a') as f:
          print(logformstr.format(epoch, lr, *metrics), file=f)
      print(printformstr.format(epoch, epochs, lr, *metrics))

      # cluster_error, obj must always be first two metrics
      cluster_error, obj = metrics[:2]
      is_conv = lr <= min_lr or epoch == epochs
      scheduler.step(obj)

      if is_logging and (epoch % chkp_freq == 0 or is_conv):
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
      if out_dir is not None:
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


def train_epoch(model, data_loader, optimizer, device, dist_mode=False,
      eval_rank=False):
  """train model for one epoch and record convergence measures."""
  metrics = None
  conf_mat, sampsec = ut.AverageMeter(), ut.AverageMeter()
  epoch_tic = time.time()
  for _, x, groups in data_loader:
    # opt step
    tic = time.time()
    x = x.to(device)
    # metrics depend on optimizer, but typically:
    # (obj, loss, reg, lossD, lossG, sprs, norm_x_)
    batch_metrics = optimizer.step(x)

    # eval batch cluster confusion
    batch_conf_mat = ut.eval_confusion(model.groups, groups, k=model.k,
        true_classes=data_loader.dataset.classes)
    if dist_mode:
      bufs = batch_metrics + (batch_conf_mat,)
      coalesced = _flatten_dense_tensors(bufs)
      dist.all_reduce(coalesced, op=dist.reduce_op.SUM)
      coalesced /= dist.get_world_size()
      bufs = _unflatten_dense_tensors(coalesced, bufs)
      batch_metrics, batch_conf_mat = bufs[:-1], bufs[-1]
      batch_conf_mat *= dist.get_world_size()

    batch_size = x.size(0)
    if dist_mode:
      batch_size *= dist.get_world_size()

    if metrics is None:
      metrics = [ut.AverageMeter() for _ in range(len(batch_metrics))]
    for kk in range(len(batch_metrics)):
      metrics[kk].update(batch_metrics[kk], batch_size)
    conf_mat.update(batch_conf_mat, 1)

    batch_time = time.time() - tic
    sampsec.update(batch_size/batch_time, batch_size)

    # batch_metrics[0] must be objective
    if torch.isnan(batch_metrics[0]):
      raise RuntimeError('Divergence! NaN objective.')

  rtime = time.time() - epoch_tic
  cluster_error, conf_mat = ut.eval_cluster_error(conf_mat.sum)
  if eval_rank:
    ranks, svs = model.eval_rank()
    ranks = torch.stack(ranks).cpu().numpy()
    svs = torch.stack(svs).cpu().numpy()
    rank_stats = np.median(ranks), ranks.min(), ranks.max()
  else:
    rank_stats = []
    svs = None
  metrics = [met.avg for met in metrics]
  metrics = [cluster_error] + metrics + rank_stats + [sampsec.avg, rtime]
  return metrics, conf_mat, svs
