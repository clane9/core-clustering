from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import ipdb
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

import utils as ut
from models import KSubspaceBatchAltBaseModel, KSubspaceMCModel

EPS = 1e-8
# ridx,  cidx, cand_ridx, cand_cidx, success, obj_decr, cumu_obj_decr, temp
RESET_NCOL = 8


def train_loop(model, data_loader, device, optimizer, out_dir=None, epochs=200,
      chkp_freq=50, stop_freq=-1, scheduler=None, eval_rank=False,
      reset_unused=False, save_data=True, init_time=0.0):
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
    save_data: whether to save bigger data, ie conf_mats, svs (default: True)
    init_time: initialization time to add to first epoch (default: 0.0)
  """
  printformstr = ('(epoch {:d}/{:d}) lr={:.1e} '
      'err={:.3f},{:.3f},{:.3f} obj={:.2e},{:.2e},{:.2e} '
      'loss={:.1e},{:.1e},{:.1e} reg.in={:.1e},{:.1e},{:.1e} '
      'reg.out={:.1e},{:.1e},{:.1e} ')
  mc_mode = isinstance(model, KSubspaceMCModel)
  if mc_mode:
    printformstr += 'comp.err={:.2f},{:.2f},{:.2f} '
  if eval_rank:
    printformstr += 'rank={:.0f},{:.0f},{:.0f} '
  printformstr += 'resets={:d} samp/s={:.0f} rtime={:.2f}'

  if out_dir is not None:
    logheader = ('Epoch,LR,' + ','.join(['{}.{}'.format(met, meas)
        for met in ['Err', 'Obj', 'Loss', 'Reg.in', 'Reg.out']
        for meas in ['min', 'med', 'max']]) + ',')
    if mc_mode:
      logheader += 'Comp.err.min,Comp.err.med,Comp.err.max,'
    if eval_rank:
      logheader += 'Rank.min,Rank.med,Rank.max,'
    logheader += 'Resets,Samp.s,RT'

    logformstr = ('{:d},{:.9e},{:.9f},{:.9f},{:.9f},'
        '{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},'
        '{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},{:.9e},')
    if mc_mode:
      logformstr += '{:.9f},{:.9f},{:.9f},'
    if eval_rank:
      logformstr += '{:.0f},{:.9f},{:.0f},'
    logformstr += '{:d},{:.0f},{:.9f}'

    val_logf = '{}/val_log.csv'.format(out_dir)
    with open(val_logf, 'w') as f:
      print(logheader, file=f)
  else:
    val_logf = None

  batch_alt_mode = isinstance(model, KSubspaceBatchAltBaseModel)

  # dim 1: err, obj, loss, reg.in, reg.out, (comp.err), resets
  nmet = 7 if mc_mode else 6
  metrics = np.zeros((epochs, nmet, model.r), dtype=np.float32)
  true_n = (model.true_classes.size if batch_alt_mode
      else data_loader.dataset.classes.size)
  conf_mats = np.zeros((epochs, model.r, model.k, true_n), dtype=np.int64)
  svs = (np.zeros((epochs, model.r, model.k, model.d), dtype=np.float32)
      if eval_rank else None)
  resets = [] if reset_unused else None

  if not batch_alt_mode and scheduler is None:
    max_lr = ut.get_learning_rate(optimizer)
    min_lr = max(EPS, 0.5**10 * max_lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10,
        threshold=(model.reset_try_tol/10), min_lr=min_lr)
  elif scheduler is not None:
    min_lr = scheduler.min_lrs[0]

  # training loop
  err = None
  lr = float('inf')
  model.train()
  try:
    for epoch in range(1, epochs+1):
      if batch_alt_mode:
        (metrics_summary, metrics[epoch-1, :], conf_mats[epoch-1, :],
            epoch_svs, epoch_resets) = batch_alt_step(model, eval_rank,
                reset_unused)
      else:
        (metrics_summary, metrics[epoch-1, :], conf_mats[epoch-1, :],
            epoch_svs, epoch_resets) = train_epoch(model, data_loader,
                optimizer, device, eval_rank, reset_unused)
        lr = ut.get_learning_rate(optimizer)

      if epoch == 1:
        metrics_summary[-1] += init_time

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

      waiting_to_reset = (reset_unused and
          (model.num_bad_steps[1, :] <= 3*model.reset_patience).any().item())
      is_conv = (((model._updates == 0 if batch_alt_mode else lr <= min_lr) and
          (not waiting_to_reset)) or epoch == epochs)

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
        scheduler.step(max_obj)

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
    if reset_unused:
      resets = (np.concatenate(resets, axis=0) if len(resets) > 0
          else np.zeros((0, RESET_NCOL+2), dtype=object))
    if out_dir is not None:
      with open('{}/metrics.npz'.format(out_dir), 'wb') as f:
        np.savez(f, metrics=metrics[:epoch, :])
      if save_data:
        save_conf_mats = conf_mats[:epoch, :]
        save_svs = svs[:epoch, :] if eval_rank else None
      else:
        save_conf_mats = conf_mats[[epoch-1], :]
        save_svs = svs[[epoch-1], :] if eval_rank else None
      with open('{}/conf_mats.npz'.format(out_dir), 'wb') as f:
        np.savez(f, conf_mats=save_conf_mats)
      if eval_rank:
        with open('{}/svs.npz'.format(out_dir), 'wb') as f:
          np.savez(f, svs=save_svs)
      if reset_unused:
        if save_data:
          with open('{}/resets.npz'.format(out_dir), 'wb') as f:
            np.savez(f, resets=resets)
        reset_summary = ut.aggregate_resets(resets)
        reset_summary.to_csv('{}/reset_summary.csv'.format(out_dir),
            index=False)

    if err is not None:
      raise err
  return conf_mats, svs, resets


def train_epoch(model, data_loader, optimizer, device, eval_rank=False,
      reset_unused=False):
  """train model for one epoch and record convergence measures."""
  epoch_tic = time.time()
  metrics = None
  resets = []
  mc_mode = isinstance(model, KSubspaceMCModel)
  conf_mats, sampsec, comp_err = [ut.AverageMeter() for _ in range(3)]
  itr = 1
  tic = time.time()
  for data_tup in data_loader:
    if len(data_tup) == 3:
      x, groups, x0 = data_tup
      if x0 is not None:
        x0 = x0.to(device)
    else:
      x, groups = data_tup
      x0 = None
    x = x.to(device)

    # opt step
    optimizer.zero_grad()
    (batch_obj_mean, batch_obj, batch_loss,
        batch_reg_in, batch_reg_out) = model.objective(x)
    batch_obj_mean.backward()
    optimizer.step()

    # zero out near zero bases to improve numerical performance
    model.zero()

    batch_metrics = [batch_obj, batch_loss, batch_reg_in, batch_reg_out]

    # eval batch cluster confusion
    batch_conf_mats = torch.stack([
        torch.from_numpy(ut.eval_confusion(model.groups[:, ii], groups,
            model.k, true_classes=data_loader.dataset.classes))
        for ii in range(model.replicates)])

    # eval batch completion if in missing data setting
    if mc_mode and x0 is not None:
      batch_comp_err = model.eval_comp_error(x0).cpu()
    else:
      batch_comp_err = torch.ones(model.r) * np.nan

    batch_size = x.size(0)
    if metrics is None:
      metrics = [ut.AverageMeter() for _ in range(len(batch_metrics))]
    for kk in range(len(batch_metrics)):
      metrics[kk].update(batch_metrics[kk].cpu(), batch_size)
    conf_mats.update(batch_conf_mats, 1)
    comp_err.update(batch_comp_err, batch_size)

    if reset_unused:
      batch_resets = model.reset_unused()
      success_mask = batch_resets[:, 4] == 1
      if success_mask.sum() > 0:
        rIdx = np.unique(batch_resets[success_mask, 0].astype(np.int64))
        ut.reset_optimizer_state(model, optimizer, rIdx)
      if batch_resets.shape[0] > 0:
        batch_resets = np.insert(batch_resets, 0, itr, axis=1)
        resets.append(batch_resets)

    itr += 1
    toc = time.time()
    batch_time = toc - tic
    tic = toc
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
        else np.zeros((0, RESET_NCOL+1), dtype=object))
    success_mask = resets[:, 5] == 1
    reset_count = success_mask.sum()
    rep_reset_counts = np.zeros((1, model.r))
    if reset_count > 0:
      reset_rids, success_counts = np.unique(
          resets[success_mask, 1].astype(np.int64), return_counts=True)
      rep_reset_counts[0, reset_rids] = success_counts
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

  metrics = torch.stack([errors] + [met.avg for met in metrics])
  metrics_summary = torch.stack([metrics.min(dim=1)[0],
      metrics.median(dim=1)[0], metrics.max(dim=1)[0]], dim=1)
  metrics = metrics.numpy()
  metrics = ((metrics, comp_err, rep_reset_counts) if mc_mode
      else (metrics, rep_reset_counts))
  metrics = np.concatenate(metrics)
  metrics_summary = metrics_summary.view(-1).numpy().tolist()
  metrics_summary = (metrics_summary + comp_err_stats + rank_stats +
      [reset_count, sampsec.avg, rtime])
  return metrics_summary, metrics, conf_mats, svs, resets


def batch_alt_step(model, eval_rank=False, reset_unused=False):
  """Take one full batch alt min step and record convergence measures."""
  epoch_tic = time.time()
  _, obj, loss, reg_in, reg_out = model.objective()
  model.step()

  # eval cluster confusion
  conf_mats = torch.stack([
      torch.from_numpy(ut.eval_confusion(model.groups[:, ii],
          model.true_groups, model.k))
      for ii in range(model.replicates)])

  if reset_unused:
    resets = model.reset_unused()
    success_mask = resets[:, 4] == 1
    reset_count = success_mask.sum()
    rep_reset_counts = np.zeros((1, model.r))
    if reset_count > 0:
      reset_rids, success_counts = np.unique(
          resets[success_mask, 0].astype(np.int64), return_counts=True)
      rep_reset_counts[0, reset_rids] = success_counts
    resets = np.insert(resets, 0, 1, axis=1)
  else:
    resets = np.zeros((0, RESET_NCOL+1), dtype=object)
    reset_count = 0
    rep_reset_counts = np.zeros((1, model.r))

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

  metrics = torch.stack([errors, obj.cpu(), loss.cpu(), reg_in.cpu(),
      reg_out.cpu()])
  metrics_summary = torch.stack([metrics.min(dim=1)[0],
      metrics.median(dim=1)[0], metrics.max(dim=1)[0]], dim=1)
  metrics = metrics.numpy()
  metrics = np.concatenate((metrics, rep_reset_counts))
  metrics_summary = metrics_summary.view(-1).numpy().tolist()
  metrics_summary = (metrics_summary + rank_stats +
      [reset_count, sampsec, rtime])
  return metrics_summary, metrics, conf_mats, svs, resets
