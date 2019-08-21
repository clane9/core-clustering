from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import ipdb
import numpy as np
import torch

from . import utils as ut
from .models import KSubspaceBatchAltBaseModel, KSubspaceMCModel
from .core_reset import RESET_NCOL

EPS = 1e-8


def train_loop(model, data_loader, device, optimizer, out_dir=None, epochs=200,
      chkp_freq=50, stop_freq=-1, lr_scheduler=None, bs_scheduler=None,
      epoch_size=None, eval_rank=False, core_reset=False, save_data=True,
      init_time=0.0):
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
    lr_scheduler: lr scheduler (default: None)
    bs_scheduler: batch size scheduler (default: None)
    epoch_size: maximum number of samples per "epoch" (default: len(dataset))
    eval_rank: evaluate ranks of group models, only implemented for subspace
      models (default: False)
    core_reset: whether to perform cooperative re-initialization
      (default: False)
    save_data: whether to save bigger data, ie conf_mats, svs (default: True)
    init_time: initialization time to add to first epoch (default: 0.0)
  """
  printformstr = ('(epoch {:d}/{:d}) lr={:.1e} bs={:d} '
      'err={:.3f},{:.3f},{:.3f} obj={:.2e},{:.2e},{:.2e} '
      'loss={:.1e},{:.1e},{:.1e} reg.in={:.1e},{:.1e},{:.1e} '
      'reg.out={:.1e},{:.1e},{:.1e} ')
  mc_mode = isinstance(model, KSubspaceMCModel)
  if mc_mode:
    printformstr += 'comp.err={:.2f},{:.2f},{:.2f} '
  if eval_rank:
    printformstr += 'rank={:.0f},{:.0f},{:.0f} '
  printformstr += ('resets={:d} samp/s={:.0f} rtime={:.2f} '
      'data.rt={:.2f} reset.rt={:.2f}')

  if out_dir is not None:
    logheader = ('Epoch,LR,BS,' + ','.join(['{}.{}'.format(met, meas)
        for met in ['Err', 'Obj', 'Loss', 'Reg.in', 'Reg.out']
        for meas in ['min', 'med', 'max']]) + ',')
    if mc_mode:
      logheader += 'Comp.err.min,Comp.err.med,Comp.err.max,'
    if eval_rank:
      logheader += 'Rank.min,Rank.med,Rank.max,'
    logheader += 'Resets,Samp.s,RT,Data.RT,Reset.RT'

    logformstr = ('{:d},{:.9e},{:d},{:.9f},{:.9f},{:.9f},'
        '{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},'
        '{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},')
    if mc_mode:
      logformstr += '{:.9f},{:.9f},{:.9f},'
    if eval_rank:
      logformstr += '{:.0f},{:.9f},{:.0f},'
    logformstr += '{:d},{:.0f},{:.9f},{:.9f},{:.9f}'

    val_logf = '{}/val_log.csv'.format(out_dir)
    with open(val_logf, 'w') as f:
      print(logheader, file=f)
  else:
    val_logf = None

  batch_alt_mode = isinstance(model, KSubspaceBatchAltBaseModel)
  if batch_alt_mode:
    eval_cluster_error = model.true_classes is not None
  else:
    eval_cluster_error = data_loader.dataset.classes is not None

  # dim 1: err, obj, loss, reg.in, reg.out, (comp.err), resets
  nmet = 7 if mc_mode else 6
  metrics = np.zeros((epochs, nmet, model.r), dtype=np.float32)
  if eval_cluster_error:
    true_n = (model.true_classes.size if batch_alt_mode
        else data_loader.dataset.classes.size)
    conf_mats = np.zeros((epochs, model.r, model.k, true_n), dtype=np.int64)
  else:
    conf_mats = None
  svs = (np.zeros((epochs, model.r, model.k, model.d), dtype=np.float32)
      if eval_rank else None)
  resets = [] if core_reset else None

  # training loop
  err = None
  lr = float('inf')
  data_iter = None
  model.train()
  try:
    for epoch in range(1, epochs+1):
      if batch_alt_mode:
        (metrics_summary, metrics[epoch-1, :], epoch_conf_mats, epoch_svs,
            epoch_resets) = batch_alt_step(model, eval_rank=eval_rank,
                core_reset=core_reset,
                eval_cluster_error=eval_cluster_error)
      else:
        (metrics_summary, metrics[epoch-1, :], epoch_conf_mats, epoch_svs,
            epoch_resets, data_iter) = train_epoch(model, data_loader,
                data_iter, optimizer, device, epoch_size=epoch_size,
                eval_rank=eval_rank, core_reset=core_reset,
                mc_mode=mc_mode, eval_cluster_error=eval_cluster_error)
        lr = ut.get_learning_rate(optimizer)
        bs = data_loader.batch_size

      if epoch == 1:
        metrics_summary[-3] += init_time

      if eval_cluster_error:
        conf_mats[epoch-1, :] = epoch_conf_mats
      if eval_rank:
        svs[epoch-1, :] = epoch_svs
      if core_reset and epoch_resets.shape[0] > 0:
        epoch_resets = np.insert(epoch_resets, 0, epoch, axis=1)
        resets.append(epoch_resets)

      if val_logf is not None:
        with open(val_logf, 'a') as f:
          print(logformstr.format(epoch, lr, bs, *metrics_summary), file=f)
      print(printformstr.format(epoch, epochs, lr, bs, *metrics_summary))

      cluster_error, min_obj, max_obj = [metrics_summary[ii]
          for ii in [0, 3, 5]]

      is_conv = model._updates == 0 if batch_alt_mode else epoch == epochs
      save_chkp = (out_dir is not None and
          (epoch % chkp_freq == 0 or (is_conv and chkp_freq <= epochs)))
      if save_chkp:
        ut.save_checkpoint({
            'epoch': epoch,
            'model': model.state_dict(),
            'err': cluster_error,
            'obj': min_obj},
            is_best=False,
            filename='{}/checkpoint{}.pth.tar'.format(out_dir, epoch))

      if not batch_alt_mode:
        if lr_scheduler is not None:
          lr_scheduler.step()
        if bs_scheduler is not None and bs_scheduler.step():
          data_loader = bs_scheduler.new_data_loader()
          data_iter = None

      if stop_freq > 0 and epoch % stop_freq == 0:
        ipdb.set_trace()

      if is_conv:
        break

  except Exception as e:
    epoch = max(0, epoch-1)  # go back to last successful epoch
    err = e
    print('{}: {}'.format(type(err), err))
    if out_dir is not None:
      with open("{}/error".format(out_dir), "a") as f:
        print('{}: {}'.format(type(err), err), file=f)

  finally:
    metrics = metrics[:epoch, :]
    conf_mats = conf_mats[:epoch, :] if eval_cluster_error else None
    svs = svs[:epoch, :] if eval_rank else None

    if core_reset:
      resets = (np.concatenate(resets, axis=0) if len(resets) > 0
          else np.zeros((0, RESET_NCOL+2), dtype=object))
      reset_summary = ut.aggregate_resets(resets)
    else:
      resets = reset_summary = None

    if out_dir is not None:
      with open('{}/metrics.npz'.format(out_dir), 'wb') as f:
        np.savez(f, metrics=metrics)
      if eval_cluster_error:
        save_conf_mats = conf_mats if save_data else conf_mats[[epoch-1], :]
        with open('{}/conf_mats.npz'.format(out_dir), 'wb') as f:
          np.savez(f, conf_mats=save_conf_mats)
      if eval_rank:
        save_svs = svs if save_data else svs[[epoch-1], :]
        with open('{}/svs.npz'.format(out_dir), 'wb') as f:
          np.savez(f, svs=save_svs)
      if core_reset:
        if save_data:
          with open('{}/resets.npz'.format(out_dir), 'wb') as f:
            np.savez(f, resets=resets)
        reset_summary.to_csv('{}/reset_summary.csv'.format(out_dir),
            index=False)

    if err is not None:
      raise err
  return metrics, conf_mats, svs, resets, reset_summary


def train_epoch(model, data_loader, data_iter, optimizer, device,
      epoch_size=None, eval_rank=False, core_reset=False, mc_mode=False,
      eval_cluster_error=True):
  """train model for one epoch and record convergence measures."""
  data_tic = epoch_tic = time.time()
  data_rtime = 0.0
  reset_rtime = 0.0
  metrics = None
  resets = []
  conf_mats, comp_err = ut.AverageMeter(), ut.AverageMeter()
  itr, epochN = 1, 0
  epoch_stop = False
  if data_iter is None:
    data_iter = iter(data_loader)
  model.epoch_init()
  while not epoch_stop:
    try:
      data_tup = next(data_iter)
    except StopIteration:
      if epoch_size is None or epochN >= epoch_size:
        data_iter = None
        epoch_stop = True
        break
      else:
        data_iter = iter(data_loader)
        data_tup = next(data_iter)
    if epochN >= epoch_size:
      epoch_stop = True

    if len(data_tup) == 3:
      x, groups, x0 = data_tup
      if x0 is not None:
        x0 = x0.to(device)
    else:
      x, groups = data_tup
      x0 = None
    x = x.to(device)
    batch_size = x.shape[0]
    epochN += batch_size
    data_rtime += time.time() - data_tic

    # opt step
    optimizer.zero_grad()
    (batch_obj_mean, batch_obj, batch_loss,
        batch_reg_in, batch_reg_out) = model.objective(x)

    if torch.isnan(batch_obj_mean.data):
      raise RuntimeError('Divergence! NaN objective.')

    batch_obj_mean.backward()
    optimizer.step()

    batch_metrics = [batch_obj, batch_loss, batch_reg_in, batch_reg_out]

    # eval batch cluster confusion
    if eval_cluster_error:
      batch_conf_mats = torch.stack([
          torch.from_numpy(ut.eval_confusion(model.groups[:, ii], groups,
              model.k, true_classes=data_loader.dataset.classes))
          for ii in range(model.replicates)])

    # eval batch completion if in missing data setting
    if mc_mode and x0 is not None:
      batch_comp_err = model.eval_comp_error(x0).cpu()

    if metrics is None:
      metrics = [ut.AverageMeter() for _ in range(len(batch_metrics))]
    for kk in range(len(batch_metrics)):
      metrics[kk].update(batch_metrics[kk].cpu(), batch_size)
    if eval_cluster_error:
      conf_mats.update(batch_conf_mats, 1)
    if mc_mode:
      comp_err.update(batch_comp_err, batch_size)

    if core_reset:
      reset_tic = time.time()
      batch_resets = model.core_reset()
      if batch_resets.shape[0] > 0:
        rIdx = np.unique(batch_resets[:, 0].astype(np.int64))
        ut.reset_optimizer_state(model, optimizer, rIdx)
        batch_resets = np.insert(batch_resets, 0, itr, axis=1)
        resets.append(batch_resets)
      reset_rtime += time.time() - reset_tic

    itr += 1
    data_tic = time.time()

  if eval_cluster_error:
    errors = torch.tensor([ut.eval_cluster_error(conf_mats.sum[ii, :])[0]
        for ii in range(model.replicates)])
    conf_mats = conf_mats.sum.numpy()
  else:
    errors = torch.ones(model.replicates).mul_(np.nan)
    conf_mats = None

  if eval_rank:
    ranks, svs = model.eval_rank()
    rank_stats = [ranks.min().item(), ranks.median().item(),
        ranks.max().item()]
    svs = svs.cpu().numpy()
  else:
    rank_stats, svs = [], None

  if core_reset:
    resets = (np.concatenate(resets) if len(resets) > 0
        else np.zeros((0, RESET_NCOL+1), dtype=object))
    reset_count = resets.shape[0]
    rep_reset_counts = np.zeros((1, model.r))
    if reset_count > 0:
      rIdx, success_counts = np.unique(resets[:, 1].astype(np.int64),
          return_counts=True)
      rep_reset_counts[0, rIdx] = success_counts
  else:
    reset_count = 0
    rep_reset_counts = np.zeros((1, model.r))

  if mc_mode:
    comp_err = comp_err.avg.view(1, -1)
    comp_err_stats = [comp_err.min().item(), comp_err.median().item(),
        comp_err.max().item()]
    comp_err = comp_err.numpy()
  else:
    comp_err_stats = []
    comp_err = np.nan * np.ones((model.replicates,))

  rtime = time.time() - epoch_tic
  sampsec = epochN / rtime

  metrics = torch.stack([errors] + [met.avg for met in metrics])
  metrics_summary = torch.stack([metrics.min(dim=1)[0],
      metrics.median(dim=1)[0], metrics.max(dim=1)[0]], dim=1)
  metrics = metrics.numpy()
  metrics = ((metrics, comp_err, rep_reset_counts) if mc_mode
      else (metrics, rep_reset_counts))
  metrics = np.concatenate(metrics)
  metrics_summary = metrics_summary.view(-1).numpy().tolist()
  metrics_summary = (metrics_summary + comp_err_stats + rank_stats +
      [reset_count, sampsec, rtime, data_rtime, reset_rtime])
  return metrics_summary, metrics, conf_mats, svs, resets, data_iter


def batch_alt_step(model, eval_rank=False, core_reset=False,
      eval_cluster_error=True):
  """Take one full batch alt min step and record convergence measures."""
  epoch_tic = time.time()
  _, obj, loss, reg_in, reg_out = model.objective()
  model.step()

  # eval cluster confusion
  if eval_cluster_error:
    conf_mats = torch.stack([
        torch.from_numpy(ut.eval_confusion(model.groups[:, ii],
            model.true_groups, model.k))
        for ii in range(model.replicates)])
  else:
    conf_mats = None

  if core_reset:
    reset_tic = time.time()
    resets = model.core_reset()
    reset_count = resets.shape[0]
    rep_reset_counts = np.zeros((1, model.r))
    if reset_count > 0:
      resets = np.insert(resets, 0, 1, axis=1)
      rIdx, success_counts = np.unique(resets[:, 1].astype(np.int64),
          return_counts=True)
      rep_reset_counts[0, rIdx] = success_counts
    reset_rtime = time.time() - reset_tic
  else:
    resets = np.zeros((0, RESET_NCOL+1), dtype=object)
    rep_reset_counts = np.zeros((1, model.r))
    reset_count, reset_rtime = 0, 0.0

  if eval_cluster_error:
    errors = torch.tensor([ut.eval_cluster_error(conf_mats.sum[ii, :])[0]
        for ii in range(model.replicates)])
    conf_mats = conf_mats.sum.numpy()
  else:
    errors = torch.ones(model.replicates).mul_(np.nan)

  if eval_rank:
    ranks, svs = model.eval_rank()
    rank_stats = [ranks.min().item(), ranks.median().item(),
        ranks.max().item()]
    svs = svs.cpu().numpy()
  else:
    rank_stats, svs = [], None

  rtime = time.time() - epoch_tic
  sampsec = model.N / rtime
  data_rtime = 0.0

  metrics = torch.stack([errors, obj.cpu(), loss.cpu(), reg_in.cpu(),
      reg_out.cpu()])
  metrics_summary = torch.stack([metrics.min(dim=1)[0],
      metrics.median(dim=1)[0], metrics.max(dim=1)[0]], dim=1)
  metrics = metrics.numpy()
  metrics = np.concatenate((metrics, rep_reset_counts))
  metrics_summary = metrics_summary.view(-1).numpy().tolist()
  metrics_summary = (metrics_summary + rank_stats +
      [reset_count, sampsec, rtime, data_rtime, reset_rtime])
  return metrics_summary, metrics, conf_mats, svs, resets
