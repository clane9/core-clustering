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
      epoch_size=None, core_reset=False, eval_rank=False, save_data=True,
      init_time=0.0):
  """Train k-subspace model

  Args:
    model: k-manifold model instance.
    data_loader: training dataset loader
    device: torch compute device id
    optimizer: optimizer instance
    out_dir: path to output directory (default: None)
    epochs: number of epochs to run for (default: 200)
    chkp_freq: how often to save checkpoint (default: 50)
    stop_freq: how often to stop for debugging (default: -1)
    lr_scheduler: lr scheduler (default: None)
    bs_scheduler: batch size scheduler (default: None)
    epoch_size: maximum number of samples per "epoch" (default: len(dataset))
    core_reset: do cooperative re-initialization (default: False)
    eval_rank: evaluate subspace ranks (default: False)
    save_data: save bigger data, ie conf_mats, svs (default: True)
    init_time: initialization time to add to first epoch (default: 0.0)

  Returns:
    metrics: per epoch summary metrics, shape (epochs, metrics, replicates).
      metrics are: cluster error, objective, loss, inside reg, outside reg,
      resets, comp error, lipschitz constant.
    conf_mats: confusion matrics, shape (epochs, replicates, model k, k).
    resets, reset_summary: complete log and summary of cluster
      re-initializations.
    svs: subspace singular values, shape (epochs, replicates, model k, d).
  """
  batch_mode = isinstance(model, KSubspaceBatchAltBaseModel)
  if batch_mode:
    eval_cluster_error = model.true_classes is not None
  else:
    eval_cluster_error = data_loader.dataset.classes is not None
  mc_mode = isinstance(model, KSubspaceMCModel)
  lip_mode = getattr(model, 'scale_grad_lip', False)

  printformstr = '(epoch {:d}/{:d}) lr={:.1e} bs={:d} '
  if eval_cluster_error:
    printformstr += 'err={:.3f},{:.3f},{:.3f} '
  printformstr += ('obj={:.2e},{:.2e},{:.2e} loss={:.1e},{:.1e},{:.1e} '
      'reg.in={:.1e},{:.1e},{:.1e} reg.out={:.1e},{:.1e},{:.1e} resets={:d} ')
  if eval_rank:
    printformstr += 'rank={:.0f},{:.0f},{:.0f} '
  if mc_mode:
    printformstr += 'comp.err={:.2f},{:.2f},{:.2f} '
  if lip_mode:
    printformstr += 'lip={:.2e},{:.2e},{:.2e} lip.err={:.2f},{:.2f},{:.2f} '
  printformstr += 'samp/s={:.0f} rtime={:.2f} data.rt={:.2f} reset.rt={:.2f}'

  if out_dir is not None:
    logheader = 'Epoch,LR,BS,'
    logformstr = '{:d},{:.9e},{:d},'
    if eval_cluster_error:
      logheader += 'Err.min,Err.med,Err.max,'
      logformstr += 3*'{:.9f},'
    logheader += (','.join(['{}.{}'.format(met, meas)
        for met in ['Obj', 'Loss', 'Reg.in', 'Reg.out']
        for meas in ['min', 'med', 'max']]) + ',Resets,')
    logformstr += 12*'{:.9e},' + '{:d},'
    if eval_rank:
      logheader += 'Rank.min,Rank.med,Rank.max,'
      logformstr += '{:.0f},{:.9f},{:.0f},'
    if mc_mode:
      logheader += 'Comp.err.min,Comp.err.med,Comp.err.max,'
      logformstr += 3*'{:.9f},'
    if lip_mode:
      logheader += ('Lip.min,Lip.med,Lip.max,'
          'Lip.err.min,Lip.err.med,Lip.err.max,')
      logformstr += 3*'{:.9e},' + 3*'{:.9f},'
    logheader += 'Samp.s,RT,Data.RT,Reset.RT'
    logformstr += '{:.0f},{:.9f},{:.9f},{:.9f}'

    val_logf = '{}/val_log.csv'.format(out_dir)
    with open(val_logf, 'w') as f:
      print(logheader, file=f)
  else:
    val_logf = None

  # metrics: err, obj, loss, reg.in, reg.out, resets, comp.err, lip
  nmet = 8
  metrics = np.zeros((epochs, nmet, model.r), dtype=np.float32)
  if eval_cluster_error:
    true_n = (model.true_classes.size if batch_mode
        else data_loader.dataset.classes.size)
    conf_mats = np.zeros((epochs, model.r, model.k, true_n), dtype=np.int64)
  else:
    conf_mats = None
  svs = (np.zeros((epochs, model.r, model.k, model.d), dtype=np.float32)
      if eval_rank else None)
  resets = [] if core_reset else None

  if batch_mode:
    lr, bs = 1.0, len(model.dataset)

  # training loop
  err = None
  data_iter = None
  model.train()
  try:
    for epoch in range(1, epochs+1):
      if batch_mode:
        (metrics_summary, metrics[epoch-1, :], epoch_conf_mats, epoch_resets,
            epoch_svs) = batch_alt_step(model,
                eval_cluster_error=eval_cluster_error, core_reset=core_reset,
                eval_rank=eval_rank)
      else:
        (metrics_summary, metrics[epoch-1, :], epoch_conf_mats, epoch_resets,
            epoch_svs, data_iter) = train_epoch(model, data_loader, data_iter,
                optimizer, device, epoch_size=epoch_size,
                eval_cluster_error=eval_cluster_error, core_reset=core_reset,
                eval_rank=eval_rank, mc_mode=mc_mode, lip_mode=lip_mode)
        lr = ut.get_learning_rate(optimizer)
        bs = data_loader.batch_size

      if epoch == 1:
        metrics_summary[-3] += init_time

      if eval_cluster_error:
        conf_mats[epoch-1, :] = epoch_conf_mats
      if core_reset and epoch_resets.shape[0] > 0:
        epoch_resets = np.insert(epoch_resets, 0, epoch, axis=1)
        resets.append(epoch_resets)
      if eval_rank:
        svs[epoch-1, :] = epoch_svs

      if val_logf is not None:
        with open(val_logf, 'a') as f:
          print(logformstr.format(epoch, lr, bs, *metrics_summary), file=f)
      print(printformstr.format(epoch, epochs, lr, bs, *metrics_summary))

      waiting_to_reset = (core_reset and
          (model.num_bad_steps[1, :] <= 3*model.reset_patience).any().item())
      is_conv = ((model._updates == 0 and not waiting_to_reset) if batch_mode
          else epoch == epochs)
      save_chkp = (out_dir is not None and
          (epoch % chkp_freq == 0 or (is_conv and chkp_freq <= epochs)))
      if save_chkp:
        ut.save_checkpoint({
            'epoch': epoch,
            'model': model.state_dict()},
            is_best=False,
            filename='{}/checkpoint{}.pth.tar'.format(out_dir, epoch))

      if not batch_mode:
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
    if core_reset:
      resets = (np.concatenate(resets, axis=0) if len(resets) > 0
          else np.zeros((0, RESET_NCOL+2), dtype=object))
      reset_summary = ut.aggregate_resets(resets)
    else:
      resets = reset_summary = None
    svs = svs[:epoch, :] if eval_rank else None

    if out_dir is not None:
      with open('{}/metrics.npz'.format(out_dir), 'wb') as f:
        np.savez(f, metrics=metrics)
      if eval_cluster_error:
        save_conf_mats = conf_mats if save_data else conf_mats[[epoch-1], :]
        with open('{}/conf_mats.npz'.format(out_dir), 'wb') as f:
          np.savez(f, conf_mats=save_conf_mats)
      if core_reset:
        if save_data:
          with open('{}/resets.npz'.format(out_dir), 'wb') as f:
            np.savez(f, resets=resets)
        reset_summary.to_csv('{}/reset_summary.csv'.format(out_dir),
            index=False)
      if eval_rank:
        save_svs = svs if save_data else svs[[epoch-1], :]
        with open('{}/svs.npz'.format(out_dir), 'wb') as f:
          np.savez(f, svs=save_svs)

    if err is not None:
      raise err
  return metrics, conf_mats, resets, reset_summary, svs


def train_epoch(model, data_loader, data_iter, optimizer, device,
      epoch_size=None, eval_cluster_error=True, core_reset=False,
      eval_rank=False, mc_mode=False, lip_mode=False):
  """train model for one epoch and record convergence measures."""
  data_tic = epoch_tic = time.time()
  data_rtime, reset_rtime = 0.0, 0.0
  metrics = None
  conf_mats = ut.AverageMeter() if eval_cluster_error else None
  resets = [] if core_reset else None
  comp_err = ut.AverageMeter() if mc_mode else None
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
        data_iter, epoch_stop = None, True
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
    if metrics is None:
      metrics = [ut.AverageMeter() for _ in range(len(batch_metrics))]
    for kk in range(len(batch_metrics)):
      metrics[kk].update(batch_metrics[kk].cpu(), batch_size)

    # eval batch cluster confusion
    if eval_cluster_error:
      batch_conf_mats = torch.stack([
          torch.from_numpy(ut.eval_confusion(model.groups[:, ii], groups,
              model.k, true_classes=data_loader.dataset.classes))
          for ii in range(model.replicates)])
      conf_mats.update(batch_conf_mats, 1)

    # eval batch completion if in missing data setting
    if mc_mode and x0 is not None:
      batch_comp_err = model.eval_comp_error(x0)
      comp_err.update(batch_comp_err.cpu(), batch_size)

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

  # evaluate summary metrics
  metrics = torch.stack([met.avg for met in metrics])
  conf_mats, errors, error_stats = _cluster_error_summary(eval_cluster_error,
      conf_mats, model)
  resets, reset_count, rep_reset_counts = _resets_summary(core_reset, resets,
      model)
  svs, rank_stats = _rank_summary(eval_rank, model)
  comp_err, comp_err_stats = _comp_err_summary(mc_mode, comp_err, model)
  lip, lip_stats = _lip_summary(lip_mode, model)

  rtime = time.time() - epoch_tic
  sampsec = epochN / rtime

  metrics, metrics_summary = _all_metrics_summary(metrics, errors, error_stats,
      reset_count, rep_reset_counts, rank_stats, comp_err, comp_err_stats, lip,
      lip_stats, sampsec, rtime, data_rtime, reset_rtime)
  return metrics_summary, metrics, conf_mats, resets, svs, data_iter


def batch_alt_step(model, eval_cluster_error=True, core_reset=False,
      eval_rank=False):
  """Take one full batch alt min step and record convergence measures."""
  epoch_tic = time.time()
  _, obj, loss, reg_in, reg_out = model.objective()
  model.step()

  metrics = torch.stack([obj.cpu(), loss.cpu(), reg_in.cpu(), reg_out.cpu()])

  conf_mats, errors, error_stats = _cluster_error_summary(eval_cluster_error,
      None, model)

  if core_reset:
    reset_tic = time.time()
    resets = model.core_reset()
    if resets.shape[0] > 0:
      resets = np.insert(resets, 0, 1, axis=1)
    reset_rtime = time.time() - reset_tic
  else:
    resets, reset_rtime = None, 0.0
  resets, reset_count, rep_reset_counts = _resets_summary(core_reset, resets,
      model)

  svs, rank_stats = _rank_summary(eval_rank, model)

  # these modes not supported in batch setting, but include placeholder null
  # results for consistency.
  comp_err, comp_err_stats = _comp_err_summary(False, None, model)
  lip, lip_stats = _lip_summary(False, model)

  rtime = time.time() - epoch_tic
  sampsec = model.N / rtime
  data_rtime = 0.0

  metrics, metrics_summary = _all_metrics_summary(metrics, errors, error_stats,
      reset_count, rep_reset_counts, rank_stats, comp_err, comp_err_stats, lip,
      lip_stats, sampsec, rtime, data_rtime, reset_rtime)
  return metrics_summary, metrics, conf_mats, resets, svs


def _cluster_error_summary(eval_cluster_error, conf_mats, model):
  if eval_cluster_error:
    if conf_mats is None:
      # batch setting, compute confusion matrices
      conf_mats = torch.stack([
          torch.from_numpy(ut.eval_confusion(model.groups[:, ii],
              model.true_groups, model.k))
          for ii in range(model.replicates)])
    else:
      # stochastic setting, use sum over iterations
      conf_mats = conf_mats.sum

    errors = np.array([ut.eval_cluster_error(conf_mats[ii, :])[0]
        for ii in range(model.replicates)]).reshape(1, -1)
    error_stats = ut.min_med_max(torch.from_numpy(errors))
    conf_mats = conf_mats.numpy()
  else:
    conf_mats = None
    errors = np.nan * np.ones((model.replicates,))
    error_stats = []
  return conf_mats, errors, error_stats


def _resets_summary(core_reset, resets, model):
  if core_reset:
    if type(resets) is list:
      resets = (np.concatenate(resets) if len(resets) > 0
          else np.zeros((0, RESET_NCOL+1), dtype=object))
    reset_count = resets.shape[0]
    rep_reset_counts = np.zeros((1, model.replicates))
    if reset_count > 0:
      rIdx, success_counts = np.unique(resets[:, 1].astype(np.int64),
          return_counts=True)
      rep_reset_counts[0, rIdx] = success_counts
  else:
    resets = None
    reset_count = 0
    rep_reset_counts = np.zeros((1, model.replicates))
  return resets, reset_count, rep_reset_counts


def _rank_summary(eval_rank, model):
  if eval_rank:
    ranks, svs = model.eval_rank()
    svs = svs.cpu().numpy()
    rank_stats = ut.min_med_max(ranks)
  else:
    svs, rank_stats = None, []
  return svs, rank_stats


def _comp_err_summary(mc_mode, comp_err, model):
  if mc_mode:
    comp_err = comp_err.avg.view(1, -1)
    comp_err_stats = ut.min_med_max(comp_err)
    comp_err = comp_err.cpu().numpy()
  else:
    comp_err = np.nan * np.ones((1, model.replicates))
    comp_err_stats = []
  return comp_err, comp_err_stats


def _lip_summary(lip_mode, model):
  if lip_mode:
    lip = model.Lip.cpu().numpy()
    lip_stats = ut.min_med_max(model.Lip)

    if hasattr(model, 'Hess'):
      Hess = model.Hess.cpu().numpy()
      true_lip = np.linalg.norm(Hess, ord=2, axis=(2, 3)).max(axis=1)
      lip_err = lip / true_lip
    else:
      lip_err = np.ones_like(lip)
    lip_stats += ut.min_med_max(torch.from_numpy(lip_err))

    lip = lip.reshape(1, -1)
  else:
    lip_stats = []
    lip = np.nan * np.ones((1, model.replicates))
  return lip, lip_stats


def _all_metrics_summary(metrics, errors, error_stats, reset_count,
      rep_reset_counts, rank_stats, comp_err, comp_err_stats, lip, lip_stats,
      sampsec, rtime, data_rtime, reset_rtime):
  metrics_summary = torch.stack([metrics.min(dim=1)[0],
      metrics.median(dim=1)[0], metrics.max(dim=1)[0]], dim=1)
  metrics_summary = metrics_summary.view(-1).numpy().tolist()
  metrics_summary = (
      error_stats + metrics_summary + [reset_count] + rank_stats +
      comp_err_stats + lip_stats + [sampsec, rtime, data_rtime, reset_rtime])
  metrics = metrics.numpy()
  metrics = np.concatenate((errors, metrics, rep_reset_counts, comp_err, lip))
  return metrics, metrics_summary
