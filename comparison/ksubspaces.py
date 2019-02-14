from __future__ import print_function
from __future__ import division

import time

import numpy as np
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd
import torch
import torch.nn as nn

import utils as ut


class KSubspaceBatchAltModel(nn.Module):
  def __init__(self, k, d, dataset, affine=False, svd_solver='randomized'):
    if svd_solver not in ('randomized', 'svds', 'svd'):
      raise ValueError("Invalid svd solver {}".format(svd_solver))

    super(KSubspaceBatchAltModel, self).__init__()
    self.k = k
    self.d = d
    self.dataset = dataset
    self.register_buffer('X', dataset.X)
    self.true_groups = dataset.groups
    self.true_classes = dataset.classes
    # X assumed to be N x D.
    self.N, self.D = self.X.shape
    self.affine = affine
    self.svd_solver = svd_solver

    # group assignment
    self.C = nn.Parameter(torch.zeros(k, self.N, dtype=torch.uint8),
        requires_grad=False)
    self.groups = None
    self.c_mean = None
    # subspace bases and coefficients
    self.Uts = nn.Parameter(torch.Tensor(k, d, self.D), requires_grad=False)
    if affine:
      self.bs = nn.Parameter(torch.Tensor(k, self.D), requires_grad=False)
    else:
      self.register_parameter('bs', None)

    self.reset_parameters()
    return

  def reset_parameters(self):
    """Initialize bases Uts by svd on random gaussian matrices,
        bs ~ N(0, 0.1/sqrt(D))"""
    Xnoise = torch.randn(self.d, self.D,
        device=self.Uts.device).div(np.sqrt(self.D))
    for ii in range(self.k):
      # initialize each orthonormal basis uniformly at random
      Ut, b, s = self._pca(Xnoise, solver='svd')
      self.Uts.data[ii, :] = Ut
      Xnoise.normal_(0., 1./np.sqrt(self.D))
    if self.affine:
      self.bs.data.normal_(0., 0.1/np.sqrt(self.D))
    return

  def fit(self, steps=None, out_dir=None, reset_unused=False):
    """Train k-subspace model.

    Args:
      steps: max optimization steps (default: inf)
      out_dir: output dir (default: None)
      reset_unused: reset unused clusters by splitting large clusters
        (default: False)

    Returns:
      cluster_err, conf_mats
    """
    printformstr = ('(step {:d}) err={:.4f} loss={:.3e} updates={:d} '
        'resets={:d} rtime={:.3f}')
    if out_dir is not None:
      logheader = 'Step,Err,Loss,Updates,Resets,RT'
      logformstr = '{:d},{:.9f},{:.9e},{:d},{:d},{:.9f}'
      val_logf = '{}/val_log.csv'.format(out_dir)
      with open(val_logf, 'w') as f:
        print(logheader, file=f)
    else:
      val_logf = None

    self.reset_parameters()
    conf_mats = []
    kk, is_conv = 0, False
    resets = 0
    while not is_conv:
      tic = time.time()
      loss, updates = self.assign_step()
      if reset_unused:
        reset_ids, split_ids = self.reset_unused()
        resets = reset_ids.numel()
      self.pca_step()

      cluster_err, conf_mat = ut.eval_cluster_error(self.groups.cpu(),
          self.true_groups, sort_conf_mat=True, k=self.k,
          true_classes=self.true_classes)
      conf_mats.append(conf_mat)
      rtime = time.time() - tic

      metrics = [cluster_err, loss, updates, resets, rtime]
      if val_logf is not None:
        with open(val_logf, 'a') as f:
          print(logformstr.format(kk+1, *metrics), file=f)
      print(printformstr.format(kk+1, *metrics))

      kk += 1
      is_conv = updates == 0 or kk == steps

    conf_mats = np.stack(conf_mats)
    groups = self.groups.cpu().numpy()

    if out_dir is not None:
      with open('{}/results.npz'.format(out_dir), 'wb') as f:
        np.savez(f, conf_mats=conf_mats, groups=groups)
    return cluster_err, conf_mats

  def loss(self):
    """Evaluate least-squares reconstruction loss.

    Returns:
      loss: shape (N, k)
    """
    # z = U^T (x - b)
    if self.affine:
      # (k, N, D)
      Xshift = self.X.sub(self.bs.unsqueeze(1))
    else:
      # (1, N, D)
      Xshift = self.X.unsqueeze(0)
    # (k, N, d)
    Z = torch.matmul(Xshift, self.Uts.transpose(1, 2))
    # since U orthonormal can compute reconstruction loss as:
    # 0.5 || (x - b) - UU^T(x - b)||_2^2 = 0.5 ||(U^c)^T (x - b)||_2^2
    #       = 0.5 ||x - b||_2^2 - 0.5 ||U^T (x - b) ||_2^2
    # where U^c is orth complement to U
    # (k, N)
    Xnormsqr = Xshift.pow(2).sum(dim=2).mul(0.5)
    Znormsqr = Z.pow(2).sum(dim=2).mul(0.5)
    loss = Xnormsqr.sub(Znormsqr)
    return loss

  def assign_step(self):
    """Update cluster assignment."""
    loss = self.loss()

    groups_prev = self.groups
    self.groups = torch.argmin(loss, dim=0)
    self.C.zero_().scatter_(0, self.groups.view(1, -1), 1)
    self.c_mean = self.C.sum(dim=1, keepdim=True).float().div_(self.N)
    updates = ((self.groups != groups_prev).sum()
        if groups_prev is not None else self.N)

    # reduce
    loss = loss.gather(0, self.groups.view(1, -1)).mean()
    return loss, updates

  def _pca(self, X, solver='randomized'):
    """Solve pca for data subset X.

    Args:
      X: N_i x D data subset.
    """
    if self.affine:
      b = X.mean(dim=0)
      X = X.sub(b)
    else:
      b = None

    if solver == 'svd':
      _, s, U = torch.svd(X)
      s, Ut = s[:self.d], U[:, :self.d].t()
    else:
      X = X.cpu().numpy()
      if solver == 'svds':
        _, s, Ut = svds(X, self.d)
      else:
        _, s, Ut = randomized_svd(X, self.d)
      s, Ut = [torch.from_numpy(z).to(self.Uts.device) for z in (s, Ut)]
    return Ut, b, s

  def pca_step(self):
    """Update subspace bases and coefficients by pca"""
    for ii in range(self.k):
      if self.c_mean[ii] > 0:
        Ut, b, s = self._pca(self.X[self.C[ii, :], :], solver=self.svd_solver)
        self.Uts.data[ii, :] = Ut
        if self.affine:
          self.bs.data[ii, :] = b
    return

  def reset_unused(self, split_metric=None, sample_p=None, reset_thr=.01):
    """Reset (nearly) unused clusters by duplicating clusters likely to contain
    >1 group. By default, choose to duplicate largest clusters.

    Args:
      split_metric: shape (k,) metric used to rank groups for splitting
        (largest values chosen) (default: c_mean).
      sample_p: probability of splitting group j is proportional to
        split_metric[j]**sample_p (default: round(log_2(k))).
      reset_thr: size threshold for nearly empty clusters in 1/k units
        (default: .01).

    Returns:
      reset_ids, split_ids
    """
    if split_metric is None:
      split_metric = self.c_mean
    split_metric = split_metric.view(-1)
    if sample_p is None:
      # default chosen so that a cluster twice as "big" has k times prob of
      # getting split
      sample_p = int(np.round(np.log2(self.k)))
    if split_metric.shape != (self.k,):
      raise ValueError("Invalid splitting metric")
    if sample_p < 0:
      raise ValueError("Invalid sample power {}".format(sample_p))
    if reset_thr < 0:
      raise ValueError("Invalid reset threshold {}".format(reset_thr))

    reset_ids = torch.nonzero(self.c_mean.view(-1) <=
        reset_thr/self.k).view(-1)
    reset_count = reset_ids.size(0)
    if reset_count > 0:
      # can't reset more that k/2 clusters.
      if reset_count > self.k / 2:
        reset_count = (self.k - reset_count)
        reset_ids = reset_ids[:reset_count]

      split_metric = split_metric.pow(sample_p)
      split_metric[reset_ids] = 0.0
      split_ids = torch.multinomial(split_metric, reset_count)

      # "split" clusters by re-assigning ~50% of points
      for rid, sid in zip(reset_ids, split_ids):
        split_mask = self.C[sid, :]
        split_mask = split_mask.float().mul(0.5).sub(torch.rand(self.N)).gt(0)
        self.groups[split_mask] = rid

      self.C.zero_().scatter_(0, self.groups.view(1, -1), 1)
      self.c_mean = self.C.sum(dim=1, keepdim=True).float().div_(self.N)
    else:
      split_ids = torch.clone(reset_ids)
    return reset_ids, split_ids
