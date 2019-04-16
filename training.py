from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import ipdb
import numpy as np
import torch
from torch import optim

import utils as ut
from models import KSubspaceBatchAltProjModel

EPS = 1e-8


def train_loop(model, data_loader, device, optimizer, out_dir=None, epochs=200,
      chkp_freq=50, stop_freq=-1, scheduler=None, eval_rank=False,
      reset_unused=False):
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
    eval_rank: evaluate ranks of group models, only implemented for subspace
      models (default: False)
    reset_unused: whether to reset unused clusters (default: False)
  """
  printformstr = ('(epoch {:d}/{:d}) lr={:.1e} '
      'err={:.3f},{:.3f},{:.3f} obj={:.2e},{:.2e},{:.2e} '
      'loss={:.1e},{:.1e},{:.1e} reg.in={:.1e},{:.1e},{:.1e} '
      'reg.out={:.1e},{:.1e},{:.1e} |x_|={:.1e},{:.1e},{:.1e} ')
  if eval_rank:
    printformstr += 'rank={:.0f},{:.0f},{:.0f} '
  printformstr += 'resets={:d}/{:d} samp/s={:.0f} rtime={:.2f}'

  if out_dir is not None:
    logheader = ('Epoch,LR,' + ','.join(['{}.{}'.format(met, meas)
        for met in ['Err', 'Obj', 'Loss', 'Reg.in', 'Reg.out', 'Norm.x_']
        for meas in ['min', 'med', 'max']]) + ',')
    if eval_rank:
      logheader += 'Rank.min,Rank.med,Rank.max,'
    logheader += 'Resets,Reset.attempts,Samp.s,RT'

    logformstr = ('{:d},{:.9e},{:.9f},{:.9f},{:.9f},'
        '{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},'
        '{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},'
        '{:.9e},{:.9e},{:.9e},')
    if eval_rank:
      logformstr += '{:.0f},{:.9f},{:.0f},'
    logformstr += '{:d},{:d},{:.0f},{:.9f}'

    val_logf = '{}/val_log.csv'.format(out_dir)
    with open(val_logf, 'w') as f:
      print(logheader, file=f)
  else:
    val_logf = None

  batch_alt_mode = isinstance(model, KSubspaceBatchAltProjModel)

  # dim 1: err, obj, loss, reg.in, reg.out, |x_|
  metrics = np.zeros((epochs, 6, model.r), dtype=np.float32)
  true_n = (model.true_classes.size if batch_alt_mode
      else data_loader.dataset.classes.size)
  conf_mats = np.zeros((epochs, model.r, model.k, true_n), dtype=np.int64)
  svs = (np.zeros((epochs, model.r, model.k, model.d), dtype=np.float32)
      if eval_rank else None)
  resets = [] if reset_unused else None

  if not batch_alt_mode and scheduler is None:
    min_lr = 1e-6*ut.get_learning_rate(optimizer)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
        factor=0.5, patience=5, threshold=1e-3, min_lr=min_lr)

  # training loop
  err = None
  lr = float('inf')
  batch_alt_conv_steps = 0
  model.train()
  try:
    for epoch in range(1, epochs+1):
      if batch_alt_mode:
        (metrics_summary, metrics[epoch-1, :], conf_mats[epoch-1, :],
            epoch_svs, epoch_resets) = batch_alt_step(model,
                eval_rank, reset_unused)
      else:
        (metrics_summary, metrics[epoch-1, :], conf_mats[epoch-1, :],
            epoch_svs, epoch_resets) = train_epoch(model, data_loader,
                optimizer, device, eval_rank, reset_unused)
        lr = ut.get_learning_rate(optimizer)

      if eval_rank:
        svs[epoch-1, :] = epoch_svs
      if reset_unused and epoch_resets.shape[0] > 0:
        epoch_resets = np.insert(epoch_resets, 0, epoch, axis=1)
        resets.append(epoch_resets)

      if val_logf is not None:
        with open(val_logf, 'a') as f:
          print(logformstr.format(epoch, lr, *metrics_summary), file=f)
      print(printformstr.format(epoch, epochs, lr, *metrics_summary))

      cluster_error, min_obj, max_obj = [metrics_summary[ii]
          for ii in [0, 3, 5]]
      if batch_alt_mode:
        if model._updates == 0:
          batch_alt_conv_steps += 1
        else:
          batch_alt_conv_steps = 0
        is_conv = (batch_alt_conv_steps >= 2*model.reset_patience or
            epoch == epochs)
      else:
        is_conv = lr <= min_lr or epoch == epochs

      save_chkp = (out_dir is not None and (epoch % chkp_freq == 0 or is_conv))
      if save_chkp:
        ut.save_checkpoint({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': (optimizer.state_dict()
                if not batch_alt_mode else None),
            'err': cluster_error,
            'obj': min_obj},
            is_best=False,
            filename='{}/checkpoint{}.pth.tar'.format(out_dir, epoch))

      if not batch_alt_mode:
        scheduler.step(max_obj)

      if stop_freq > 0 and epoch % stop_freq == 0:
        ipdb.set_trace()

      if is_conv:
        break

  except Exception as e:
    err = e
    print('{}: {}'.format(type(err), err))
    if out_dir is not None:
      with open("{}/error".format(out_dir), "a") as f:
        print('{}: {}'.format(type(err), err), file=f)

  finally:
    if reset_unused:
      resets = (np.concatenate(resets, axis=0) if len(resets) > 0
          else np.zeros((0, 10), dtype=object))
    if out_dir is not None:
      with open('{}/metrics.npz'.format(out_dir), 'wb') as f:
        np.savez(f, metrics=metrics[:epoch, :])
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


def train_epoch(model, data_loader, optimizer, device, eval_rank=False,
      reset_unused=False):
  """train model for one epoch and record convergence measures."""
  epoch_tic = time.time()
  metrics = None
  resets = []
  conf_mats, sampsec, reset_attempts = [ut.AverageMeter() for _ in range(3)]
  for x, groups in data_loader:
    tic = time.time()
    x = x.to(device)

    # opt step
    optimizer.zero_grad()
    (batch_obj_mean, batch_obj, batch_loss,
        batch_reg_in, batch_reg_out, x_) = model.objective(x)
    batch_obj_mean.backward()
    optimizer.step()
    model.step()  # increment step counter

    # zero out near zero bases to improve numerical performance
    model.zero()

    batch_norm_x_ = model.eval_shrink(x, x_)
    batch_metrics = [batch_obj, batch_loss, batch_reg_in, batch_reg_out,
        batch_norm_x_]

    # eval batch cluster confusion
    batch_conf_mats = torch.stack([
        ut.eval_confusion(model.groups[:, ii], groups, k=model.k,
            true_classes=data_loader.dataset.classes)
        for ii in range(model.replicates)])

    batch_size = x.size(0)
    if metrics is None:
      metrics = [ut.AverageMeter() for _ in range(len(batch_metrics))]
    for kk in range(len(batch_metrics)):
      metrics[kk].update(batch_metrics[kk], batch_size)
    conf_mats.update(batch_conf_mats, 1)

    if reset_unused:
      # cols (reset_ridx, reset_cidx, reset_metric, cand_ridx, cand_cidx,
      # reset_success, obj_decr, cand_c_mean, cand_value)
      batch_resets = model.reset_unused()
      if batch_resets.shape[0] > 0:
        rIdx = batch_resets[:, 0].astype(np.int64)
        cIdx = batch_resets[:, 1].astype(np.int64)
        # zero optimizer states
        for p in model.parameters():
          if not p.requires_grad:
            continue
          state = optimizer.state[p]
          for key, val in state.items():
            if isinstance(val, torch.Tensor) and val.shape == p.shape:
              # assuming all parameters have (r, k) as first dims.
              val[rIdx, cIdx, :] = 0.0
        resets.append(batch_resets)

    batch_time = time.time() - tic
    sampsec.update(batch_size/batch_time, batch_size)
    if torch.isnan(batch_obj_mean):
      raise RuntimeError('Divergence! NaN objective.')

  rtime = time.time() - epoch_tic

  errors = torch.tensor([ut.eval_cluster_error(conf_mats.sum[ii, :])[0]
      for ii in range(model.replicates)])
  conf_mats = conf_mats.sum.numpy()

  if eval_rank:
    ranks, svs = model.eval_rank()
    rank_stats = [ranks.min().item(), ranks.median().item(),
        ranks.max().item()]
    svs = svs.cpu().numpy()
  else:
    rank_stats, svs = [], None

  if reset_unused:
    resets = (np.concatenate(resets) if len(resets) > 0
        else np.zeros((0, 9), dtype=object))
    reset_attempts = resets.shape[0]
    reset_count = int(resets[:, 5].sum())
  else:
    reset_attempts, reset_count = 0, 0

  metrics = torch.stack([errors] + [met.avg for met in metrics])
  metrics_summary = torch.stack([metrics.min(dim=1)[0],
      metrics.median(dim=1)[0], metrics.max(dim=1)[0]], dim=1)
  metrics = metrics.cpu().numpy()
  metrics_summary = metrics_summary.view(-1).numpy().tolist()
  metrics_summary = (metrics_summary + rank_stats +
      [reset_count, reset_attempts] + [sampsec.avg, rtime])
  return metrics_summary, metrics, conf_mats, svs, resets


def batch_alt_step(model, eval_rank=False, reset_unused=False):
  """Take one full batch alt min step and record convergence measures."""
  epoch_tic = time.time()
  _, obj, loss, reg_in, reg_out, x_ = model.objective()
  model.step()

  norm_x_ = model.eval_shrink(x_)
  # eval cluster confusion
  conf_mats = torch.stack([
      ut.eval_confusion(model.groups[:, ii], model.true_groups, k=model.k,
          true_classes=model.true_classes)
      for ii in range(model.replicates)])

  if reset_unused:
    # cols (reset_ridx, reset_cidx, reset_metric, cand_ridx, cand_cidx,
    # reset_success, obj_decr, cand_c_mean, cand_value)
    resets = model.reset_unused()
    reset_attempts = resets.shape[0]
    reset_count = int(resets[:, 5].sum())
  else:
    resets = np.zeros((0, 9), dtype=object)
    reset_count, reset_attempts = 0, 0

  rtime = time.time() - epoch_tic
  sampsec = model.N / rtime

  errors = torch.tensor([ut.eval_cluster_error(conf_mats[ii, :])[0]
      for ii in range(model.replicates)])
  conf_mats = conf_mats.numpy()

  if eval_rank:
    ranks, svs = model.eval_rank()
    rank_stats = [ranks.min().item(), ranks.median().item(),
        ranks.max().item()]
    svs = svs.cpu().numpy()
  else:
    rank_stats, svs = [], None

  metrics = torch.stack([errors, obj, loss, reg_in, reg_out, norm_x_])
  metrics_summary = torch.stack([metrics.min(dim=1)[0],
      metrics.median(dim=1)[0], metrics.max(dim=1)[0]], dim=1)
  metrics = metrics.cpu().numpy()
  metrics_summary = metrics_summary.view(-1).numpy().tolist()
  metrics_summary = (metrics_summary + rank_stats +
      [reset_count, reset_attempts] + [sampsec, rtime])
  return metrics_summary, metrics, conf_mats, svs, resets
