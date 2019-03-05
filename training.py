from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import ipdb
import numpy as np
import torch
from torch import optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

import utils as ut


def train_loop(model, data_loader, device, optimizer, out_dir=None, epochs=200,
      chkp_freq=50, stop_freq=-1, scheduler=None, dist_mode=False,
      eval_rank=False, reset_unused=False, reset_kwargs={}):
  """Train k-subspace model for series of epochs.

  Args:
    model: k-manifold model instance.
    data_loader: training dataset loader
    device: torch compute device id
    optimizer: k-manifold optimizer instance
    out_dir: path to output directory (default: None)
    epochs: number of epochs to run for (default: 200)
    chkp_freq: how often to save checkpoint (default: 50)
    stop_freq: how often to stop for debugging (default: -1)
    scheduler: lr scheduler. If None, use reduce on plateau (default: None)
    dist_mode: whether doing distributed training (default: False)
    eval_rank: evaluate ranks of group models, only implemented for subspace
      models (default: False)
    reset_unused: whether to reset unused clusters (default: False)
    reset_kwargs: kwargs to pass to reset_unused (default: {})
  """
  printformstr = ('(epoch {:d}/{:d}) lr={:.3e} err={:.4f} obj={:.3e} '
      'loss={:.3e} reg={:.3e} sprs={:.2f} |x_|={:.3e} ')
  if eval_rank:
    printformstr += 'rank(min)(max)={:.0f},{:.0f},{:.0f} '
  printformstr += 'resets={:d} samp/s={:.0f} rtime={:.3f}'

  if dist_mode and not (dist.is_initialized()):
    raise RuntimeError("Distributed package not initialized")
  is_logging = (not dist_mode) or (dist.get_rank() == 0)

  if is_logging and out_dir is not None:
    logheader = 'Epoch,LR,Err,Obj,Loss,Reg,Sprs,Norm.x_,'
    if eval_rank:
      logheader += 'Rank.med,Rank.min,Rank.max,'
    logheader += 'Resets,Samp.s,RT'
    logformstr = '{:d},{:.9e},{:.9f},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},'
    if eval_rank:
      logformstr += '{:.9f},{:.0f},{:.0f},'
    logformstr += '{:d},{:.0f},{:.9f}'
    val_logf = '{}/val_log.csv'.format(out_dir)
    with open(val_logf, 'w') as f:
      print(logheader, file=f)
  else:
    val_logf = None

  conf_mats = np.zeros((epochs, model.k, data_loader.dataset.classes.size),
      dtype=np.int64)
  svs = (np.zeros((epochs, model.k, model.d), dtype=np.float32)
      if eval_rank else None)
  resets = [] if reset_unused else None

  if scheduler is None:
    min_lr = 1e-6*ut.get_learning_rate(optimizer)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        factor=0.5, patience=5, threshold=1e-3, min_lr=min_lr)

  # training loop
  err = None
  lr = float('inf')
  model.train()
  try:
    for epoch in range(1, epochs+1):
      # deterministic data shuffling by epoch for this sampler
      if isinstance(data_loader.sampler, DistributedSampler):
        data_loader.sampler.set_epoch(epoch)

      (metrics, conf_mats[epoch-1, :], epoch_svs,
          epoch_resets) = train_epoch(model, data_loader, optimizer, device,
              dist_mode, eval_rank, reset_unused, reset_kwargs)
      lr = ut.get_learning_rate(optimizer)

      if eval_rank:
        svs[epoch-1, :] = epoch_svs
      if reset_unused:
        epoch_resets = np.insert(epoch_resets, 0, epoch, axis=1)
        resets.append(epoch_resets)

      if is_logging:
        if val_logf is not None:
          with open(val_logf, 'a') as f:
            print(logformstr.format(epoch, lr, *metrics), file=f)
        print(printformstr.format(epoch, epochs, lr, *metrics))

      cluster_error, obj = metrics[:2]
      is_conv = lr <= min_lr or epoch == epochs
      scheduler.step(obj)

      save_chkp = (is_logging and out_dir is not None and
          (epoch % chkp_freq == 0 or is_conv))
      if save_chkp:
        ut.save_checkpoint({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
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
    if reset_unused:
      resets = (np.concatenate(resets, axis=0) if len(resets) > 0
          else np.zeros((0, 4), dtype=np.int64))
    if is_logging and out_dir is not None:
      with open('{}/conf_mats.npz'.format(out_dir), 'wb') as f:
        np.savez(f, conf_mats=conf_mats[:epoch, :])
      if eval_rank:
        with open('{}/svs.npz'.format(out_dir), 'wb') as f:
          np.savez(f, svs=svs[:epoch, :])
      if reset_unused:
        with open('{}/resets.npz'.format(out_dir), 'wb') as f:
          np.savez(f, resets=resets)
    if err is not None:
      raise err
  return conf_mats, svs, resets


def train_epoch(model, data_loader, optimizer, device, dist_mode=False,
      eval_rank=False, reset_unused=False, reset_kwargs={}):
  """train model for one epoch and record convergence measures."""
  epoch_tic = time.time()
  metrics = None
  conf_mat, sampsec = [ut.AverageMeter() for _ in range(2)]
  resets = []
  for x, groups in data_loader:
    tic = time.time()
    x = x.to(device)

    # opt step
    optimizer.zero_grad()
    (batch_obj, batch_scale_obj, batch_loss,
        batch_reg, x_) = model.objective(x)
    batch_scale_obj.backward()
    optimizer.step()

    batch_sprs = model.eval_sprs()
    batch_norm_x_ = model.eval_shrink(x, x_)
    batch_metrics = (batch_obj, batch_loss, batch_reg, batch_sprs,
        batch_norm_x_)

    # eval batch cluster confusion
    batch_conf_mat = ut.eval_confusion(model.groups.data.cpu(), groups,
        k=model.k, true_classes=data_loader.dataset.classes)

    if dist_mode:
      bufs = batch_metrics + (batch_conf_mat,)
      coalesced = _flatten_dense_tensors(bufs)
      dist.all_reduce(coalesced, op=dist.reduce_op.SUM)
      coalesced /= dist.get_world_size()
      bufs = _unflatten_dense_tensors(coalesced, bufs)
      batch_metrics, batch_conf_mat = bufs[:-1], bufs[-1]
      # only conf mat not averaged
      batch_conf_mat *= dist.get_world_size()

    batch_size = x.size(0)
    if dist_mode:
      batch_size *= dist.get_world_size()

    if metrics is None:
      metrics = [ut.AverageMeter() for _ in range(len(batch_metrics))]
    for kk in range(len(batch_metrics)):
      metrics[kk].update(batch_metrics[kk].item(), batch_size)
    conf_mat.update(batch_conf_mat, 1)

    if reset_unused:
      reset_ids, split_ids, split_ranks = model.reset_unused(**reset_kwargs)
      if reset_ids.size > 0:
        # copy optimizer states
        for p in [model.Us, model.bs]:
          state = optimizer.state[p]
          for key, val in state.items():
            if isinstance(val, torch.Tensor) and val.shape == p.shape:
              val[reset_ids, :] = val[split_ids, :]
        # append to reset summary
        resets.append(np.stack([reset_ids, split_ids, split_ranks], axis=1))

    batch_time = time.time() - tic
    sampsec.update(batch_size/batch_time, batch_size)

    if torch.isnan(batch_metrics[0]):
      raise RuntimeError('Divergence! NaN objective.')

  rtime = time.time() - epoch_tic
  cluster_error, conf_mat = ut.eval_cluster_error(conf_mat.sum,
      sort_conf_mat=True)

  if eval_rank:
    ranks, svs = model.eval_rank()
    ranks = torch.stack(ranks).cpu().numpy()
    svs = torch.stack(svs).cpu().numpy()
    rank_stats = [np.median(ranks), ranks.min(), ranks.max()]
  else:
    rank_stats, svs = [], None

  if reset_unused:
    resets = (np.concatenate(resets, axis=0) if len(resets) > 0
        else np.zeros((0, 3), dtype=np.int64))
    reset_count = resets.shape[0]
  else:
    reset_count = 0

  metrics = [met.avg for met in metrics]
  metrics = ([cluster_error] + metrics +
      rank_stats + [reset_count] + [sampsec.avg, rtime])
  return metrics, conf_mat, svs, resets
