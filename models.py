from __future__ import print_function
from __future__ import division

import numpy as np
import torch
import torch.nn as nn

import utils as ut

EPS = 1e-8
EMA_DECAY = 0.99


class KSubspaceBaseModel(nn.Module):
  """Base K-subspace class."""
  default_reg_params = dict()
  assign_reg_terms = set()

  def __init__(self, k, d, D, affine=False, replicates=5, reg_params={},
        reset_patience=100, reset_jitter=True, reset_try_tol=0.01,
        reset_max_steps=None, reset_accept_tol=1e-4, reset_stochastic=True,
        reset_low_value=False, reset_value_thr=0.1, reset_sigma=0.05,
        reset_cache_size=None, cache_subsample_rate=None):

    if replicates < 1:
      raise ValueError("Invalid replicates parameter {}".format(replicates))
    for key in self.default_reg_params:
      val = reg_params.get(key, None)
      if val is None:
        reg_params[key] = self.default_reg_params[key]
      elif val < 0:
        raise ValueError("Invalid {} reg parameter {}".format(k, val))
    if not set(reg_params).issubset(set(self.default_reg_params)):
      raise ValueError("Invalid reg parameter keys")
    if reset_patience <= 0:
      raise ValueError("Invalid reset_patience parameter {}".format(
          reset_patience))
    if reset_try_tol <= 0:
      raise ValueError("Invalid reset_try_tol parameter {}".format(
          reset_try_tol))
    if reset_sigma < 0:
      raise ValueError("Invalid reset_sigma parameter {}".format(
          reset_sigma))

    super().__init__()
    self.k = k  # number of groups
    self.d = d  # dimension of manifold
    self.D = D  # number of data points
    self.affine = affine
    self.replicates = self.r = replicates
    self.reg_params = reg_params

    self.reset_patience = reset_patience
    self.reset_jitter = reset_jitter
    self.reset_try_tol = reset_try_tol
    if reset_max_steps is None or reset_max_steps <= 0:
      reset_max_steps = k
    self.reset_max_steps = reset_max_steps
    self.reset_accept_tol = reset_accept_tol
    self.reset_stochastic = reset_stochastic
    self.reset_low_value = reset_low_value
    self.reset_value_thr = reset_value_thr
    self.n_reset_cands = 1 if reset_low_value else k
    self.reset_sigma = reset_sigma
    if reset_cache_size is None or reset_cache_size <= 0:
      reset_cache_size = 100 * int(np.ceil(4*k*np.log(k) / 100))
    self.reset_cache_size = reset_cache_size
    if cache_subsample_rate is None or cache_subsample_rate <= 0:
      cache_subsample_rate = 1.0 / k
    self.cache_subsample_rate = cache_subsample_rate
    self.cache_subsample_size = min(reset_cache_size,
        int(np.ceil(cache_subsample_rate * reset_cache_size)))

    # group assignment, ultimate shape (batch_size, r, k)
    self.c = None
    self.groups = None
    # subspace coefficients, ultimate shape (batch_size, r, k, d)
    self.z = None

    self.Us = nn.Parameter(torch.Tensor(self.r, k, D, d))
    if affine:
      self.bs = nn.Parameter(torch.Tensor(self.r, k, D))
    else:
      self.register_parameter('bs', None)

    self.register_buffer('c_mean', torch.ones(self.r, k).mul_(np.nan))
    self.register_buffer('value', torch.ones(self.r, k).mul_(np.nan))
    self.register_buffer('obj', torch.ones(self.r).mul_(np.nan))
    self.register_buffer('best_obj', torch.ones(self.r).mul_(np.nan))
    if reset_jitter:
      init_jitter = torch.randint(-reset_patience//2, reset_patience//2 + 1,
          (self.r,), dtype=torch.float32)
    else:
      init_jitter = torch.zeros(self.r)
    self.register_buffer('jitter', init_jitter)
    # 1st row counts since last reset, 2nd since last *successful* reset
    self.register_buffer('num_bad_steps', torch.zeros(2, self.r))
    self.register_buffer('cooldown_counter',
        torch.ones(self.r).mul_(reset_patience//2))

    self.register_buffer('cache_assign_obj',
        torch.ones(reset_cache_size, self.r, k).mul_(np.nan))
    self.cache_assign_obj_head = 0
    self._reset_buffers = None
    return

  def reset_parameters(self):
    """Initialize Us, bs with entries drawn from normal with std
    0.1/sqrt(D)."""
    std = .1 / np.sqrt(self.D)
    self.Us.data.normal_(0., std)
    if self.affine:
      self.bs.data.zero_()
    return

  def prob_farthest_insert(self, X, nn_q=0):
    """Initialize Us, bs by probabilistic farthest insertion (single trial)

    Args:
      X: Data matrix from which to select bases, shape (N, D)
      nn_q: Extra nearest neighbors (default: 0).
    """
    nn_k = self.d + nn_q
    if self.affine:
      nn_k += 1
    N = X.shape[0]
    X = ut.unit_normalize(X, p=2, dim=1)

    self.Us.data.zero_()
    if self.affine:
      self.bs.data.zero_()

    # choose first basis randomly
    Idx = torch.randint(0, N, (self.r,), dtype=torch.int64)
    self._insert_next(0, X[Idx, :], X, nn_k)

    for jj in range(1, self.k):
      # (N, r, k')
      loss = self._prob_insert_loss(X, jj)
      # (N, r)
      min_loss = torch.min(loss, dim=2)[0]
      # (r,)
      Idx = torch.multinomial(min_loss.t(), 1).view(-1)
      self._insert_next(jj, X[Idx, :], X, nn_k)
    return

  def _prob_insert_loss(self, X, jj):
    """Evaluate loss wrt X for current initialized subspaces up to but not
    including jj. Assumes Us are orthogonal and x_i unit norm. Does not
    consider bs.

    Computes:
      1/2 || x - U U^T x ||_2^2
      1/2 || x ||_2^2 - x^T U U^T x + 1/2 x^T U U^T U U^T x
      1/2 || x ||_2^2 - 1/2 || U^T x ||_2^2
      1/2 (1 - || U^T x ||_2^2)
    """
    # shape (r, k', D, d), assumed to be orthogonal
    Us = self.Us.data[:, :jj, :]
    # (r, k', N, d)
    z = torch.matmul(X, Us)
    # (r, k', N)
    znormsqr = z.pow(2).sum(dim=3)
    # (N, r, k')
    # assumes x_i are normalized
    loss = (1.0 - znormsqr).mul(0.5).permute(2, 0, 1)
    return loss

  def _insert_next(self, jj, y, X, nn_k):
    """Insert next basis (jj) centered on sampled points y from X."""
    y_knn = ut.cos_knn(y, X, nn_k)
    for ii in range(self.r):
      U, _ = ut.reg_pca(y_knn[ii, :], self.d)
      self.Us.data[ii, jj, :] = U
    return

  def forward(self, x):
    """Compute representation of x wrt each subspace.

    Input:
      x: data, shape (batch_size, D)

    Returns:
      x_: reconstruction, shape (r, k, batch_size, D)
    """
    z = self.encode(x)
    self.z = z.data
    x_ = self.decode(z)
    return x_

  def decode(self, z):
    """Embed low-dim code z into ambient space.

    Input:
      z: latent code, shape (r, k, batch_size, d)

    Returns:
      x_: reconstruction, shape (r, k, batch_size, D)
    """
    assert(z.dim() == 4 and
        (z.size(0), z.size(1), z.size(3)) == (self.r, self.k, self.d))

    # x_ = U z + b
    # shape (r, k, batch_size, D)
    x_ = torch.matmul(z, self.Us.transpose(2, 3))
    if self.affine:
      x_ = x_.add(self.bs.unsqueeze(2))
    return x_

  def objective(self, x):
    """Evaluate objective function.

    Input:
      x: data, shape (batch_size, D)

    Returns:
      obj_mean: average objective across replicates
      obj, loss, reg_in, reg_out: metrics per replicate, shape (r,)
      x_: reconstruction, shape (r, k, batch_size, D)
    """
    x_ = self(x)
    loss = self.loss(x, x_)
    reg = self.reg()

    # split regs into assign and outside terms
    reg_in = torch.zeros(self.r, self.k, dtype=loss.dtype, device=loss.device)
    reg_out = torch.zeros(self.r, self.k, dtype=loss.dtype, device=loss.device)
    for key, val in reg.items():
      if key in self.assign_reg_terms:
        reg_in = reg_in + val
      else:
        reg_out = reg_out + val

    # update assignment c
    assign_obj = loss.data + reg_in.data
    self.set_assign(assign_obj)
    loss = self.c * loss
    reg_in = self.c * reg_in

    # cache assign_obj and outside reg values
    self._batch_reg_out = reg_out.data

    # reduce and compute objective, shape (r,)
    loss = loss.sum(dim=2).mean(dim=0)
    reg_in = reg_in.sum(dim=2).mean(dim=0)
    reg_out = reg_out.sum(dim=1)
    obj = loss + reg_in + reg_out
    obj_mean = obj.mean()

    # ema objective, shape (r,)
    nan_mask = torch.isnan(self.obj)
    self.obj[nan_mask] = obj.data[nan_mask]
    self.obj.mul_(EMA_DECAY).add_(1-EMA_DECAY, obj.data)
    return obj_mean, obj.data, loss.data, reg_in.data, reg_out.data, x_.data

  def loss(self, x, x_):
    """Evaluate reconstruction loss

    Inputs:
      x: data, shape (batch_size, D)
      x_: reconstruction, shape (r, k, batch_size, D)

    Returns:
      loss: shape (batch_size, r, k)
    """
    loss = torch.sum((x - x_)**2, dim=3).mul(0.5).permute(2, 0, 1)
    return loss

  def set_assign(self, assign_obj):
    """Compute cluster assignment.

    Inputs:
      assign_obj: shape (batch_size, r, k)
    """
    if self.c is None or self.c.shape[0] != assign_obj.shape[0]:
      self.c = torch.zeros_like(assign_obj.data)
    else:
      self.c.zero_()

    top2obj, top2idx = torch.topk(assign_obj, 2, dim=2,
        largest=False, sorted=True)
    self.groups = top2idx[:, :, 0]
    self.c.scatter_(2, self.groups.unsqueeze(2), 1)

    self._batch_c_mean = torch.mean(self.c, dim=0)
    self._batch_value = torch.mean(self.c *
        (top2obj[:, :, [1]] - top2obj[:, :, [0]]), dim=0)

    nan_mask = torch.isnan(self.c_mean)
    if nan_mask.sum() > 0:
      self.c_mean[nan_mask] = self._batch_c_mean[nan_mask]
      self.value[nan_mask] = self._batch_value[nan_mask]

    self.c_mean.mul_(EMA_DECAY).add_(1-EMA_DECAY, self._batch_c_mean)
    self.value.mul_(EMA_DECAY).add_(1-EMA_DECAY, self._batch_value)

    self.groups = self.groups.cpu()
    self._update_assign_obj_cache(assign_obj)
    return

  def _update_assign_obj_cache(self, assign_obj):
    """Add batch assign obj to a recent cache. Used when resetting unused
    clusters."""
    batch_size = assign_obj.shape[0]
    if self.reset_cache_size < batch_size:
      Idx = torch.randperm(batch_size)[:self.reset_cache_size]
      self.cache_assign_obj.copy_(assign_obj[Idx, :])
      self.cache_assign_obj_head = 0
    elif self.reset_cache_size == batch_size:
      self.cache_assign_obj.copy_(assign_obj)
      self.cache_assign_obj_head = 0
    else:
      Idx = (torch.arange(self.cache_assign_obj_head,
          self.cache_assign_obj_head + batch_size) % self.reset_cache_size)
      self.cache_assign_obj[Idx, :] = assign_obj
      self.cache_assign_obj_head = ((Idx[-1] + 1) %
          self.reset_cache_size).item()
    return

  def eval_shrink(self, x, x_):
    """measure shrinkage of reconstruction wrt data.

    Inputs:
      x: data, shape (batch_size, D)
      x_: reconstruction, shape (r, k, batch_size, D)

    Returns:
      norm_x_: average norm of x_ relative to x, shape (r,)
    """
    # (batch_size, r, k)
    norm_x_ = x_.data.pow(2).sum(dim=3).sqrt().permute(2, 0, 1)
    # (batch_size, r)
    norm_x_ = norm_x_.mul_(self.c.data).sum(dim=2)
    # (batch_size, 1)
    norm_x = torch.norm(x.data, dim=1, keepdim=True)
    # (r,)
    norm_x_ = norm_x_.div(norm_x.add(EPS)).mean(dim=0)
    return norm_x_

  def eval_rank(self, tol=.01):
    """Compute rank and singular values of subspace bases.

    Inputs:
      tol: rank tolerance (default: 0.01)

    Returns:
      ranks: numerical ranks of each basis (r, k)
      svs: singular values (r, k, d)
    """
    # svd is ~1000x faster on cpu for small matrices, e.g. (100, 10).
    Us = self.Us.data.cpu() if self.D <= 1000 else self.Us.data
    # ideally there would be a batched svd instead of this loop, but at least
    # k, r should be "small"
    # (rk, d)
    svs = torch.stack([torch.svd(Us[ii, jj, :])[1] for ii in range(self.r)
        for jj in range(self.k)])
    svs = svs.view(self.r, self.k, self.d)
    # (r, 1, 1)
    tol = svs[:, :, 0].max(dim=1)[0].mul(tol).view(self.r, 1, 1)
    # (r, k)
    ranks = (svs > tol).sum(dim=2)
    return ranks, svs

  def reset_unused(self):
    """Reset replicates' whose progress has slowed by doing several iterations
    of greedy hill-climbing over the graph of all (rk choose k) subsets of
    clusters.

    A very naive and brute-force approach with overall complexity potentially
    O(r^2 k^3) if you reset all replicates for k steps each. But serial part is
    at most O(r k).

    Returns:
      resets: np array log of resets, shape (n_resets, 6). Columns
        are (reset rep idx, reset cluster idx, candidate rep idx,
        candidate cluster idx, reset success, obj decrease).
    """
    # (ridx, cidx, cand_ridx, cand_cidx, success, obj_decr)
    empty_output = np.zeros((0, 6), dtype=object)

    # reset not possible if only one replicate
    # NOTE: should probably give warning
    if self.r == 1:
      return empty_output

    # identify clusters to reset based on objective decrease
    reset_rids = self._reset_obj_criterion()
    if reset_rids.shape[0] == 0:
      return empty_output

    # check that cache is full and calculate current cache objective
    cache_not_full = torch.any(torch.isnan(self.cache_assign_obj))
    if cache_not_full:
      return empty_output

    # allocate buffers to use during resets
    if self._reset_buffers is None:
      # NOTE: will cause late OOM error if cache_size, k too large
      device = self.cache_assign_obj.device
      self._reset_buffers = (
          # drop_assign_obj
          torch.zeros((self.cache_subsample_size, self.n_reset_cands, self.k),
              device=device),
          # alt_assign_obj
          torch.zeros((self.cache_subsample_size, self.n_reset_cands,
              self.r*self.k, 2), device=device),
          # alt_obj
          torch.zeros((self.cache_subsample_size, self.n_reset_cands,
              self.r*self.k), device=device),
          # tmp_Idx
          torch.zeros((self.cache_subsample_size, self.n_reset_cands,
              self.r*self.k), dtype=torch.int64, device=device)
      )

    # attempt to re-initialize replicates by greedy hill-climbing
    resets = []
    for ridx in reset_rids:
      rep_resets = self._reset_replicate(ridx.item())
      resets.extend(rep_resets)

    if len(resets) == 0:
      resets = empty_output
    else:
      resets = np.array(resets, dtype=object)

    # perturb all bases slightly in replicates with resets to make sure we
    # always have some diversity across replicates
    success_mask = resets[:, 4] == 1
    if self.reset_sigma > 0 and success_mask.sum() > 0:
      rIdx = np.unique(resets[success_mask, 0].astype(np.int64))
      reset_r = rIdx.shape[0]
      # (reset_r, k, 1, 1)
      Unorms = self.Us.data[rIdx, :].pow(2).sum(
          dim=(2, 3), keepdim=True).sqrt()
      # scale of perturbation is relative to each basis' norm
      reset_Sigma = Unorms.mul(
          self.reset_sigma / np.sqrt(self.D*self.d))
      Z = torch.randn(reset_r, self.k, self.D, self.d,
          dtype=Unorms.dtype, device=Unorms.device).mul_(reset_Sigma)
      self.Us.data[rIdx, :] = self.Us.data[rIdx, :] + Z
    return resets

  def _reset_obj_criterion(self):
    """Identify replicates to reset based on relative objective decrease."""
    nan_mask = torch.isnan(self.best_obj)
    self.best_obj[nan_mask] = self.obj[nan_mask]

    cooldown_mask = self.cooldown_counter > 0
    self.cooldown_counter[cooldown_mask] -= 1.0

    better_or_cooldown = torch.max(cooldown_mask,
        (self.obj < (1.0 - self.reset_try_tol)*self.best_obj))
    self.best_obj[better_or_cooldown] = self.obj[better_or_cooldown]
    self.num_bad_steps[:, better_or_cooldown] = 0.0
    self.num_bad_steps[:, better_or_cooldown == 0] += 1.0

    reset_mask = self.num_bad_steps[0, :] > (self.reset_patience + self.jitter)
    reset_rids = reset_mask.nonzero().view(-1)
    return reset_rids

  def _reset_replicate(self, ridx):
    """Re-initialize a replicate by executing several steps of greedy
    hill-climbing over the graph of (rk choose k) subsets of bases."""
    cids = np.arange(self.k)
    attempt_cids = []
    resets = []
    successes = 0

    for _ in range(self.reset_max_steps):
      value, rep_min_assign_obj = self._eval_value(ridx)

      if self.reset_low_value:
        # find lowest value cluster among those we haven't tried to reset
        avail_cids = np.setdiff1d(cids, attempt_cids)
        cidx = avail_cids[np.argmin(value[avail_cids])]
        # don't try to reset if it's too valuable
        if value[cidx] > self.reset_value_thr:
          break
        attempt_cids.append(cidx)
      else:
        cidx = None

      if self.cache_subsample_size is not None:
        # restrict to worst represented data points
        sub_Idx = torch.sort(rep_min_assign_obj, descending=True)[1]
        sub_Idx = sub_Idx[:self.cache_subsample_size]
        old_obj = (rep_min_assign_obj[sub_Idx].mean() +
            self._batch_reg_out[ridx].sum())
      else:
        sub_Idx = None
        old_obj = (rep_min_assign_obj.mean() + self._batch_reg_out[ridx].sum())

      # evaluate all alternative objectives
      # either shape (k, rk), or (1, rk) if resetting low value
      alt_obj = self._eval_alt_obj(ridx, cidx=cidx, sub_Idx=sub_Idx)
      alt_obj = alt_obj.view(-1)

      # choose substitution either probabilistically, weighted by relative
      # objective decrease, or deterministically.
      obj_decr = 1.0 - alt_obj / old_obj
      if obj_decr.max() > self.reset_accept_tol:
        if self.reset_stochastic:
          sample_prob = (obj_decr - self.reset_accept_tol).clamp_(min=0)
          idx = torch.multinomial(sample_prob, 1)[0].item()
        else:
          idx = obj_decr.view(-1).argmax()
      else:
        # only for logging purpose
        idx = obj_decr.argmax().item()

      obj_decr = obj_decr[idx].item()
      if self.reset_low_value:
        cand_ridx, cand_cidx = np.unravel_index(idx,
            (self.r, self.k))
      else:
        cidx, cand_ridx, cand_cidx = np.unravel_index(idx,
            (self.k, self.r, self.k))
      success = obj_decr > self.reset_accept_tol
      resets.append([ridx, cidx, cand_ridx, cand_cidx, int(success), obj_decr])

      if success:
        # reset parameters and caches
        self.Us.data[ridx, cidx, :] = self.Us.data[cand_ridx, cand_cidx, :]
        if self.affine:
          self.bs.data[ridx, cidx, :] = self.bs.data[
              cand_ridx, cand_cidx, :]
        self.cache_assign_obj[:, ridx, cidx] = self.cache_assign_obj[
            :, cand_ridx, cand_cidx]
        self._batch_reg_out[ridx, cidx] = self._batch_reg_out[
            cand_ridx, cand_cidx]

        successes += 1
      elif not self.reset_low_value:
        # we know we've converged if doing full search
        break

    # reset some state
    if successes > 0:
      self.c_mean[ridx, :] = np.nan
      self.value[ridx, :] = np.nan
      self.obj[ridx] = self.best_obj[ridx] = np.nan
      self.num_bad_steps[1, ridx] = 0
      self.cooldown_counter[ridx] = self.reset_patience // 2

    self.num_bad_steps[0, ridx] = 0
    if self.reset_jitter:
      self.jitter[ridx].random_(-self.reset_patience//2,
          self.reset_patience//2 + 1)
    return resets

  def _eval_value(self, ridx):
    """Evaluate current value of every cluster based on cache objective in given
    replicate."""
    rep_assign_obj = self.cache_assign_obj[:, ridx, :]
    top2obj, top2idx = torch.topk(rep_assign_obj, 2, dim=1,
        largest=False, sorted=True)
    rep_groups = top2idx[:, [0]]
    value = torch.zeros_like(rep_assign_obj)
    value = value.scatter_(1, rep_groups,
        (top2obj[:, [1]] - top2obj[:, [0]])).mean(dim=0)
    value.sub_(self._batch_reg_out[ridx, :])
    value.div_(max(value.max(), EPS))
    return value, top2obj[:, 0]

  def _eval_alt_obj(self, ridx, cidx=None, sub_Idx=None):
    """Find the best candidate swap for each of a given replicate's
    clusters."""
    # (cache_subsample_size, n_cand, k), (cache_subsample_size, n_cand, rk, 2),
    # (cache_subsample_size, n_cand, rk), (cache_subsample_size, n_cand, rk)
    drop_assign_obj, alt_assign_obj, alt_obj, tmp_Idx = self._reset_buffers

    # full search: construct (cache_subsample_size, k) matrix of min assign
    # objs, where col j has cluster j dropped.
    # cidx given: (cache_subsample_size, 1) of min assign objs with cidx
    # dropped.
    if sub_Idx is None:
      drop_assign_obj[:] = self.cache_assign_obj[:, [ridx], :]
    else:
      drop_assign_obj[:] = self.cache_assign_obj[sub_Idx, :][:, [ridx], :]
    if cidx is None:
      Idx = torch.arange(self.k)
      # (cache_subsample_size, k, k)
      drop_assign_obj[:, Idx, Idx] = np.inf
    else:
      drop_assign_obj[:, 0, cidx] = np.inf
    drop_assign_obj = drop_assign_obj.min(dim=2)[0]

    # full search: contstruct (k,) vector of reg values, one for each dropped
    # cluster
    # cidx given: scalar reg value for single dropped cluster cidx
    drop_reg_out = self._batch_reg_out[ridx, :]
    if cidx is None:
      drop_reg_out = drop_reg_out.sum() - drop_reg_out
    else:
      drop_reg_out = drop_reg_out.sum() - drop_reg_out[cidx]

    # (cache_subsample_size, n_cand, rk, 2)
    alt_assign_obj[:, :, :, 0] = drop_assign_obj.unsqueeze(2)
    if sub_Idx is None:
      alt_assign_obj[:, :, :, 1] = self.cache_assign_obj.view(-1,
          self.r*self.k).unsqueeze(1)
    else:
      alt_assign_obj[:, :, :, 1] = self.cache_assign_obj[sub_Idx, :].view(
          -1, self.r*self.k).unsqueeze(1)

    # (cache_subsample_size, n_cand, rk)
    alt_obj, _ = torch.min(alt_assign_obj, dim=3, out=(alt_obj, tmp_Idx))

    # average over cache and add regularization, (n_cand, rk)
    alt_obj = alt_obj.mean(dim=0)
    alt_obj.add_(drop_reg_out.view(-1, 1))
    alt_obj.add_(self._batch_reg_out.view(-1))
    return alt_obj

  def zero(self):
    """Zero out near zero bases.

    Numerically near zero values slows matmul performance up to 6x.
    """
    Unorms = self.Us.data.pow(2).sum(dim=(2, 3)).sqrt()
    self.Us.data[(Unorms < EPS*max(1, Unorms.max())), :, :] = 0.0
    return


class KSubspaceMFModel(KSubspaceBaseModel):
  """K-subspace model where low-dim coefficients are computed in closed
  form."""
  default_reg_params = {
      'U_frosqr_in': 0.01,
      'U_frosqr_out': 1e-4,
      'U_fro_out': 0.0,
      'U_gram_fro_out': 0.0,
      'z': 0.01
  }
  assign_reg_terms = {'U_frosqr_in', 'z'}

  def __init__(self, k, d, D, affine=False, replicates=5, reg_params={},
        reset_patience=100, reset_jitter=True, reset_try_tol=0.01,
        reset_max_steps=None, reset_accept_tol=1e-4, reset_stochastic=True,
        reset_low_value=False, reset_value_thr=0.1, reset_sigma=0.05,
        reset_cache_size=None, cache_subsample_rate=None, scale_grad_freq=100):

    super().__init__(k, d, D, affine, replicates, reg_params,
        reset_patience, reset_jitter, reset_try_tol, reset_max_steps,
        reset_accept_tol, reset_stochastic, reset_low_value, reset_value_thr,
        reset_sigma, reset_cache_size, cache_subsample_rate)

    if scale_grad_freq is not None and scale_grad_freq >= 0:
      self.scale_grad_freq = scale_grad_freq
      self.Lip = None
      self.steps = 0
      self.Us.register_hook(lambda UGrad: self._scale_grad(UGrad))
      if self.affine:
        self.bs.register_hook(lambda bGrad: self._scale_grad(bGrad,
            update=False))

    self.reset_parameters()
    return

  def encode(self, x):
    """Compute subspace coefficients for x in closed form, but computing
    batched solution to normal equations.

      min_z 1/2 || x - (Uz + b) ||_2^2 + \lambda/2 ||z||_2^2
      (U^T U + \lambda I) z* = U^T (x - b)

    Input:
      x: shape (batch_size, D)

    Returns:
      z: latent low-dimensional codes (r, k, batch_size, d)
    """
    assert(x.dim() == 2 and x.size(1) == self.D)
    batch_size = x.size(0)

    # shape (r, k, d, D)
    Uts = self.Us.data.transpose(2, 3)
    # (r, k, d, d)
    A = torch.matmul(Uts, self.Us.data)
    if self.reg_params['z'] > 0:
      # (d, d)
      lambeye = torch.eye(self.d,
          dtype=A.dtype, device=A.device).mul_(self.reg_params['z'])
      # (r, k, d, d)
      A.add_(lambeye)

    # (1, 1, D, batch_size)
    B = x.data.t().view(1, 1, self.D, batch_size)
    if self.affine:
      # bs shape (r, k, D)
      B = B.sub(self.bs.data.unsqueeze(3))
    # (r, k, d, batch_size)
    B = torch.matmul(Uts, B)

    # (r, k, d, batch_size)
    z, _ = torch.gesv(B, A)
    # (r, k, batch_size, d)
    z = z.transpose(2, 3)
    return z

  def reg(self):
    """Evaluate subspace regularization."""
    regs = dict()
    # U regularization, each is shape (r, k)
    if max([self.reg_params[key] for key in
          ('U_frosqr_in', 'U_frosqr_out', 'U_fro_out')]) > 0:
      U_frosqr = torch.sum(self.Us.pow(2), dim=(2, 3))

      if self.reg_params['U_frosqr_in'] > 0:
        regs['U_frosqr_in'] = U_frosqr.mul(
            self.reg_params['U_frosqr_in']*0.5)

      if self.reg_params['U_frosqr_out'] > 0:
        regs['U_frosqr_out'] = U_frosqr.mul(
            self.reg_params['U_frosqr_out']*0.5)

      if self.reg_params['U_fro_out'] > 0:
        U_fro = U_frosqr.sqrt()
        regs['U_fro_out'] = U_fro.mul(self.reg_params['U_fro_out'])

    if self.reg_params['U_gram_fro_out'] > 0:
      UtUs = torch.matmul(self.Us.transpose(2, 3), self.Us)
      UtUs_fro = torch.sum(UtUs.pow(2), dim=(2, 3)).sqrt()
      regs['U_gram_fro_out'] = UtUs_fro.mul(
          self.reg_params['U_gram_fro_out'])

    # z regularization, shape is (batch_size, r, k)
    # does not affect gradients, only included to ensure objective value
    # is accurate
    if self.reg_params['z'] > 0:
      z_frosqr = torch.sum(self.z.data.pow(2), dim=3).permute(2, 0, 1)
      regs['z'] = z_frosqr.mul(self.reg_params['z']*0.5)
    return regs

  def _scale_grad(self, Grad, update=True):
    """Divide gradients by an estimate of the Lipschitz constant, per
    replicate."""
    if self.Lip is None or (update and self.steps % self.scale_grad_freq == 0):
      device = self.z.device
      batch_size = self.c.shape[0]
      z_zero = self.c.permute(1, 2, 0).unsqueeze(3) * self.z
      Hess = torch.matmul(z_zero.transpose(2, 3), z_zero).div_(batch_size)

      # (r, k)
      batch_cluster_sizes = self.c.sum(dim=0)
      lamb = ((self.reg_params['U_frosqr_in']/batch_size) *
          batch_cluster_sizes + self.reg_params['U_frosqr_out']).to(device)
      # (r, k, d, d)
      Id = torch.eye(self.d, device=device).mul(
          lamb.view(self.r, self.k, 1, 1))
      Hess.add_(Id)

      # (r, k)
      Lip = np.linalg.norm(Hess.cpu().numpy(), ord=2, axis=(2, 3))
      Lip = torch.from_numpy(Lip).to(device)
      # (r,)
      self.Lip = Lip.max(dim=1)[0].view(self.r, 1, 1, 1)
    Grad_scale = Grad.div(self.Lip)
    if update:
      self.steps += 1
    return Grad_scale


class KSubspaceProjModel(KSubspaceBaseModel):
  """K-subspace model where low-dim coefficients are computed by a projection
  matrix."""
  default_reg_params = {
      'U_frosqr_in': 0.01,
      'U_frosqr_out': 1e-4,
      'U_fro_out': 0.0,
      'U_gram_fro_out': 0.0
  }
  assign_reg_terms = {'U_frosqr_in'}

  def __init__(self, k, d, D, affine=False, replicates=5, reg_params={},
        reset_patience=100, reset_jitter=True, reset_try_tol=0.01,
        reset_max_steps=None, reset_accept_tol=1e-4, reset_stochastic=True,
        reset_low_value=False, reset_value_thr=0.1, reset_sigma=0.05,
        reset_cache_size=None, cache_subsample_rate=None):

    super().__init__(k, d, D, affine, replicates, reg_params,
        reset_patience, reset_jitter, reset_try_tol, reset_max_steps,
        reset_accept_tol, reset_stochastic, reset_low_value, reset_value_thr,
        reset_sigma, reset_cache_size, cache_subsample_rate)

    self.reset_parameters()
    return

  def encode(self, x):
    """Project x onto each of k low-dimensional spaces.

    Input:
      x: shape (batch_size, D)

    Returns:
      z: latent low-dimensional codes (k, batch_size, d)
    """
    assert(x.dim() == 2 and x.size(1) == self.D)
    batch_size = x.size(0)

    # z = U^T (x - b) or z = V (x - b)
    if self.affine:
      # (r, k, batch_size, D)
      x = x.sub(self.bs.unsqueeze(2))
    else:
      x = x.view(1, 1, batch_size, self.D)

    # (r, k, batch_size, d)
    z = torch.matmul(x, self.Us)
    return z

  def reg(self):
    """Evaluate subspace regularization."""
    regs = dict()
    # U regularization, each is shape (r, k)
    if max([self.reg_params[key] for key in
          ('U_frosqr_in', 'U_frosqr_out', 'U_fro_out')]) > 0:
      U_frosqr = torch.sum(self.Us.pow(2), dim=(2, 3))

      if self.reg_params['U_frosqr_in'] > 0:
        regs['U_frosqr_in'] = U_frosqr.mul(
            self.reg_params['U_frosqr_in'])

      if self.reg_params['U_frosqr_out'] > 0:
        regs['U_frosqr_out'] = U_frosqr.mul(
            self.reg_params['U_frosqr_out'])

      if self.reg_params['U_fro_out'] > 0:
        regs['U_fro_out'] = U_frosqr.sqrt().mul(
            self.reg_params['U_fro_out'])

    if self.reg_params['U_gram_fro_out'] > 0:
      UtUs = torch.matmul(self.Us.transpose(2, 3), self.Us)
      UtUs_fro = torch.sum(UtUs.pow(2), dim=(2, 3)).sqrt()
      regs['U_gram_fro_out'] = UtUs_fro.mul(
          self.reg_params['U_gram_fro_out'])
    return regs


class KSubspaceBatchAltBaseModel(KSubspaceBaseModel):
  default_reg_params = dict()
  assign_reg_terms = dict()

  def __init__(self, k, d, dataset, affine=False, replicates=5, reg_params={},
        reset_patience=2, reset_jitter=True, reset_try_tol=0.01,
        reset_max_steps=None, reset_accept_tol=1e-4, reset_stochastic=True,
        reset_low_value=False, reset_value_thr=0.1, reset_sigma=0.05,
        reset_cache_size=None, cache_subsample_rate=None,
        svd_solver='randomized'):

    if svd_solver not in ('randomized', 'svds', 'svd'):
      raise ValueError("Invalid svd solver {}".format(svd_solver))

    # X assumed to be N x D.
    D = dataset.X.shape[1]
    super().__init__(k, d, D, affine, replicates, reg_params, reset_patience,
        reset_jitter, reset_try_tol, reset_max_steps, reset_accept_tol,
        reset_stochastic, reset_low_value, reset_value_thr, reset_sigma,
        reset_cache_size, cache_subsample_rate)

    self.dataset = dataset
    self.register_buffer('X', dataset.X)
    self.true_groups = dataset.groups
    self.true_classes = dataset.classes
    self.N = self.X.shape[0]
    self.svd_solver = svd_solver

    self.Us.requires_grad_(False)
    if affine:
      self.bs.requires_grad_(False)
    self.c = nn.Parameter(torch.Tensor(self.N, self.r, k), requires_grad=False)
    return

  def objective(self):
    """Evaluate objective function.

    Returns:
      obj_mean: average objective across replicates
      obj, loss, reg_in, reg_out: metrics per replicate, shape (r,)
      x_: reconstruction, shape (r, k, batch_size, D)
    """
    ret_vals = super().objective(self.X)
    # objective is exact for batch algorithm, override ema.
    self.obj = ret_vals[1].clone()
    return ret_vals

  def set_assign(self, assign_obj):
    """Compute cluster assignment.

    Inputs:
      assign_obj: shape (N, r, k)
    """
    groups_prev = self.groups
    top2obj, top2idx = torch.topk(assign_obj, 2, dim=2,
        largest=False, sorted=True)
    self.groups = top2idx[:, :, 0]
    self.c.zero_().scatter_(2, self.groups.unsqueeze(2), 1)

    self.c_mean = self._batch_c_mean = self.c.mean(dim=0)
    self.value = self._batch_value = torch.mean(self.c *
        (top2obj[:, :, [1]] - top2obj[:, :, [0]]), dim=0)

    self.groups = self.groups.cpu()
    self._updates = ((self.groups != groups_prev).sum()
        if groups_prev is not None else self.N)
    self._update_assign_obj_cache(assign_obj)
    return

  def eval_shrink(self, x_):
    """measure shrinkage of reconstruction wrt data.

    Inputs:
      x_: reconstruction, shape (r, k, N, D)

    Returns:
      norm_x_: average norm of x_ relative to x, shape (r,)
    """
    return super().eval_shrink(self.X, x_)


class KSubspaceBatchAltProjModel(KSubspaceBatchAltBaseModel,
      KSubspaceProjModel):
  default_reg_params = {
      'U_frosqr_in': 0.01,
      'U_frosqr_out': 1e-4,
      'U_fro_out': 0.0,
      'U_gram_fro_out': 0.0
  }
  assign_reg_terms = {'U_frosqr_in'}

  def __init__(self, k, d, dataset, affine=False, replicates=5, reg_params={},
        reset_patience=2, reset_jitter=True, reset_try_tol=0.01,
        reset_max_steps=None, reset_accept_tol=1e-4, reset_stochastic=True,
        reset_low_value=False, reset_value_thr=0.1, reset_sigma=0.05,
        reset_cache_size=None, cache_subsample_rate=None,
        svd_solver='randomized'):

    super().__init__(k, d, dataset, affine, replicates, reg_params,
        reset_patience, reset_jitter, reset_try_tol, reset_max_steps,
        reset_accept_tol, reset_stochastic, reset_low_value, reset_value_thr,
        reset_sigma, reset_cache_size, cache_subsample_rate, svd_solver)

    # must be zero, not supported
    self.reg_params['U_fro_out'] = 0.0

    self.reset_parameters()
    return

  def step(self):
    """Update subspace bases by regularized pca (and increment step count)."""
    gamma = self.N*self.reg_params['U_gram_fro_out']

    for ii in range(self.r):
      for jj in range(self.k):
        if self.c_mean[ii, jj] > 0:
          Xj = self.X[self.c[:, ii, jj] == 1, :]
          Nj = Xj.shape[0]
          lamb = (Nj*self.reg_params['U_frosqr_in'] +
              self.N*self.reg_params['U_frosqr_out'])
          U, b = ut.reg_pca(Xj, self.d, form='proj', lamb=lamb, gamma=gamma,
              affine=self.affine, solver=self.svd_solver)

          self.Us.data[ii, jj, :] = U
          if self.affine:
            self.bs.data[ii, jj, :] = b
    return


class KSubspaceBatchAltMFModel(KSubspaceBatchAltBaseModel, KSubspaceMFModel):
  default_reg_params = {
      'U_frosqr_in': 0.01,
      'U_frosqr_out': 1e-4,
      'U_fro_out': 0.0,
      'U_gram_fro_out': 0.0,
      'z': 0.01
  }
  assign_reg_terms = {'U_frosqr_in', 'z'}

  def __init__(self, k, d, dataset, affine=False, replicates=5, reg_params={},
        reset_patience=2, reset_jitter=True, reset_try_tol=0.01,
        reset_max_steps=None, reset_accept_tol=1e-4, reset_stochastic=True,
        reset_low_value=False, reset_value_thr=0.1, reset_sigma=0.05,
        reset_cache_size=None, cache_subsample_rate=None,
        svd_solver='randomized'):

    super().__init__(k, d, dataset, affine, replicates, reg_params,
        reset_patience, reset_jitter, reset_try_tol, reset_max_steps,
        reset_accept_tol, reset_stochastic, reset_low_value, reset_value_thr,
        reset_sigma, reset_cache_size, cache_subsample_rate, svd_solver)

    # must be zero, not supported
    self.reg_params['U_fro_out'] = 0.0
    self.reg_params['U_gram_fro_out'] = 0.0

    self.reset_parameters()
    return

  def reset_parameters(self):
    """Initialize Us, bs with entries drawn from normal with std
    0.1/sqrt(D)."""
    std = .1 / np.sqrt(self.D)
    self.Us.data.normal_(0., std)
    if self.affine:
      self.bs.data.normal_(0., std)
    for ii in range(self.r):
      for jj in range(self.k):
        P, S, _ = torch.svd(self.Us.data[ii, jj, :])
        self.Us.data[ii, jj, :] = P.mul_(S)
    return

  def encode(self, x):
    """Compute subspace coefficients for x in closed form, by computing
    batched solution to normal equations. Use fact that colums of U are
    orthogonal.

      min_z 1/2 || x - (Uz + b) ||_2^2 + \lambda/2 ||z||_2^2
      (U^T U + \lambda I) z* = U^T (x - b)

    Input:
      x: shape (batch_size, D)

    Returns:
      z: latent low-dimensional codes (r, k, batch_size, d)
    """
    assert(x.dim() == 2 and x.size(1) == self.D)
    batch_size = x.size(0)

    if self.affine:
      # (r, k, batch_size, D)
      x = x.sub(self.bs.unsqueeze(2))
    else:
      x = x.view(1, 1, batch_size, self.D)

    # (r, k, batch_size, d)
    b = torch.matmul(x, self.Us.data)

    # (r, k, 1, d)
    S = torch.norm(self.Us.data, dim=2, keepdim=True)
    lambeye = torch.ones_like(S).mul_(self.reg_params['z'])

    # (r, k, batch_size, d)
    z = b.div_(S.pow_(2).add_(lambeye))
    return z

  def step(self):
    """Update subspace bases by regularized pca (and increment step count)."""
    for ii in range(self.r):
      for jj in range(self.k):
        if self.c_mean[ii, jj] > 0:
          Xj = self.X[self.c[:, ii, jj] == 1, :]
          Nj = Xj.shape[0]
          lamb = np.sqrt((Nj*self.reg_params['U_frosqr_in'] +
              self.N*self.reg_params['U_frosqr_out']) *
              self.reg_params['z'])
          U, b = ut.reg_pca(Xj, self.d, form='mf', lamb=lamb, gamma=0.0,
              affine=self.affine, solver=self.svd_solver)

          # re-scale U
          if lamb > 0:
            alpha = (np.sqrt(self.reg_params['z']) /
                np.sqrt(Nj*self.reg_params['U_frosqr_in'] +
                self.N*self.reg_params['U_frosqr_out'])) ** 0.5
            U.mul_(alpha)

          self.Us.data[ii, jj, :] = U
          if self.affine:
            self.bs.data[ii, jj, :] = b
    return


class KMeansBatchAltModel(KSubspaceBatchAltBaseModel):
  """K-means model as a special case of K-subspaces."""
  default_reg_params = {
      'b_frosqr_out': 0.0
  }
  assign_reg_terms = {}

  def __init__(self, k, dataset, init='random', replicates=5, reg_params={},
        reset_patience=2, reset_jitter=True, reset_try_tol=0.01,
        reset_max_steps=None, reset_accept_tol=1e-4, reset_stochastic=True,
        reset_low_value=False, reset_value_thr=0.1, reset_sigma=0.05,
        reset_cache_size=None, cache_subsample_rate=None, kpp_n_trials=None):

    if init not in {'k-means++', 'random'}:
      raise ValueError("Invalid init parameter {}".format(init))

    d = 1  # Us not used, but retained for consistency/out of laziness
    affine = True

    super().__init__(k, d, dataset, affine, replicates, reg_params,
        reset_patience, reset_jitter, reset_try_tol, reset_max_steps,
        reset_accept_tol, reset_stochastic, reset_low_value, reset_value_thr,
        reset_sigma, reset_cache_size, cache_subsample_rate)

    self.init = init
    # Number of candidate means to test. Will add the one that reduces the loss
    # the most. This procedure is described briefly in original paper, and also
    # implemented in scikit-learn k_means.
    if kpp_n_trials is None or kpp_n_trials <= 0:
      kpp_n_trials = int(np.ceil(2 * np.log(self.k)))
    self.kpp_n_trials = kpp_n_trials
    self.Xsqrnorms = self.X.pow(2).sum(dim=1).mul(0.5)
    self.XT = self.X.t().contiguous()

    self.reset_parameters()
    return

  def reset_parameters(self):
    """Initialize bs with random data points."""
    self.Us.data.zero_()
    self.bs.data.zero_()
    if self.init == 'k-means++':
      # pick first randomly
      Idx = torch.randint(0, self.N, (self.r,), dtype=torch.int64)
      self.bs.data[:, 0, :] = self.X[Idx, :]
      for ii in range(1, self.k):
        # evaluate loss for current centers
        # (batch_size, r, ii)
        loss = self.loss(None, None, self.bs.data[:, :ii, :])
        # (batch_size, r)
        min_loss = loss.min(dim=2)[0]

        # pick candidates by probabilistic farthest selection
        # (r, 2 * log k)
        cand_Idx = torch.multinomial(min_loss.t(), self.kpp_n_trials)
        # (r, 2 * logk, D)
        cand_bs = self.X[cand_Idx, :]

        # select the best, i.e. the one that will reduce loss the most
        # (batch_size, r, 2 * logk, 1)
        cand_loss = self.loss(None, None, cand_bs).unsqueeze(3)
        rep_min_loss = min_loss.view(min_loss.shape + (1, 1)).repeat(
            1, 1, self.kpp_n_trials, 1)
        # (batch_size, r, 2*log k, 2)
        cand_loss = torch.cat((rep_min_loss, cand_loss), dim=3)
        # (r, 2*log k)
        cand_loss = (cand_loss.min(dim=3)[0]).mean(dim=0)
        Idx = cand_loss.min(dim=1)[1]
        Idx = cand_Idx[np.arange(self.r), Idx]

        self.bs.data[:, ii, :] = self.X[Idx, :]
    else:
      # pick random data points
      Idx = torch.randint(0, self.N, (self.r, self.k), dtype=torch.int64)
      self.bs.data.copy_(self.X[Idx, :])
    return

  def forward(self, x):
    """Null placeholder forward method

    Input:
      x: data, shape (batch_size, D)

    Returns:
      x_: None
    """
    x_ = torch.zeros((), dtype=x.dtype, device=x.device)
    return x_

  def loss(self, x, x_, bs=None):
    """Evaluate reconstruction loss

    Inputs:
      x: data, shape (batch_size, D) (not used)
      x_: reconstruction, shape (r, k, batch_size, D) (not used)
      bs: k means (default: None).

    Returns:
      loss: shape (batch_size, r, k)
    """
    bs = bs if bs is not None else self.bs
    bsqrnorms = bs.pow(2).sum(dim=2).mul(0.5)

    # X (batch_size, D)
    # b (r, k, D)
    # XTb (batch_size, r, k)
    XTb = torch.matmul(bs, self.XT).permute(2, 0, 1)
    # (batch_size, r, k)
    loss = self.Xsqrnorms.view(-1, 1, 1).sub(XTb).add(bsqrnorms.unsqueeze(0))
    return loss.clamp_(min=0.0)

  def encode(self, x):
    raise NotImplementedError("encode not implemented.")

  def decode(self, x):
    raise NotImplementedError("decode not implemented.")

  def reg(self):
    """Evaluate regularization."""
    regs = dict()
    if self.reg_params['b_frosqr_out'] > 0:
      regs['b_frosqr_out'] = torch.sum(self.bs.pow(2), dim=2).mul(
          self.reg_params['b_frosqr_out']*0.5)
    return regs

  def eval_shrink(self, x_):
    """measure shrinkage of reconstruction wrt data.

    Inputs:
      x_: reconstruction, shape (r, k, batch_size, D) (not used)

    Returns:
      norm_x_: vector of -1, null placeholder.
    """
    norm_x_ = torch.ones(self.r, dtype=x_.dtype, device=x_.device).mul(-1)
    return norm_x_

  def step(self):
    """Update k means (and increment step count).

    min_{{b_j}} 1/N \sum_{i=1}^N [\sum_{j=1}^k 1/2 c_{i j} || b_j - x_i ||_2^2
                                  + lambda/2 || b_j ||_2^2 ]
    grad b_j = N_j b_j - \sum_{i in C_j} x_i + N lambda b_j
    b_j = (1/(N_j + lambda N)) \sum_{i in C_j} x_i
    """
    for ii in range(self.r):
      for jj in range(self.k):
        if self.c_mean[ii, jj] > 0:
          Xj = self.X[self.c[:, ii, jj] == 1, :]
          b = Xj.sum(dim=0).div_(Xj.shape[0] +
              self.reg_params['b_frosqr_out']*self.N)
        else:
          b = torch.zeros(self.D, dtype=self.bs.dtype, device=self.bs.device)
        self.bs.data[ii, jj, :] = b
    return


class KSubspaceMFCorruptBaseModel(KSubspaceBaseModel):
  """K-subspace model where low-dim coefficients are computed in closed
  form. Adapted to handle corrupted data."""
  default_reg_params = {
      'U_frosqr_in': 0.01,
      'U_frosqr_out': 1e-4,
      'z': 0.01,
      'e': 0.0
  }
  assign_reg_terms = {'U_frosqr_in', 'z', 'e'}

  def __init__(self, k, d, D, affine=False, replicates=5, reg_params={},
        reset_patience=100, reset_jitter=True, reset_try_tol=0.01,
        reset_max_steps=None, reset_accept_tol=1e-4, reset_stochastic=True,
        reset_low_value=False, reset_value_thr=0.1, reset_sigma=0.05,
        reset_cache_size=None, cache_subsample_rate=None, encode_max_iter=20,
        encode_tol=1e-3):

    if encode_max_iter <= 0:
      raise ValueError(("Invalid encode_max_iter parameter {}").format(
          encode_max_iter))

    super().__init__(k, d, D, affine, replicates, reg_params, reset_patience,
        reset_jitter, reset_try_tol, reset_max_steps, reset_accept_tol,
        reset_stochastic, reset_low_value, reset_value_thr, reset_sigma,
        reset_cache_size, cache_subsample_rate)

    # when e reg parameter is set to zero we disable the corruption term,
    # although strictly speaking it should give the trivial solution e = full
    # residual.
    if self.reg_params['e'] == 0:
      encode_max_iter = 0
    self.encode_max_iter = encode_max_iter
    self.encode_tol = encode_tol
    self.encode_steps = None
    self.encode_update = None

    # corruption term
    self.e = None

    # cached SVDs for Us.
    # NOTE: left singular vectors must be stored transposed, due to output
    # format of torch.svd.
    self.register_buffer('Us_P', torch.zeros(self.r, k, d, D).transpose(2, 3))
    self.register_buffer('Us_s', torch.zeros(self.r, k, d))
    self.register_buffer('Us_Q', torch.zeros(self.r, k, d, d))
    return

  def forward(self, x):
    """Compute subspace coefficients and reconstruction for x by alternating
    least squares and proximal operator on e.

      min_{z,e} 1/2 || Uz - (x - b) - e ||_2^2 +
          \lambda/2 ||z||_2^2 + \gamma ||e||

    Input:
      x: shape (batch_size, D)

    Returns:
      x_: reconstruction, shape (r, k, batch_size, D)
    """
    assert(x.dim() == 2 and x.size(1) == self.D)
    batch_size = x.size(0)
    xnorm = max(x.abs().max(), 1.0)

    if self.affine:
      # (r, k, batch_size, D)
      x = x.sub(self.bs.unsqueeze(2))
    else:
      x = x.view(1, 1, batch_size, self.D)

    ut.batch_svd(self.Us.data, out=(self.Us_P, self.Us_s, self.Us_Q))
    UsT = self.Us.data.transpose(2, 3).contiguous()

    e = torch.zeros((), dtype=x.dtype, device=x.device)
    for kk in range(self.encode_max_iter):
      e_old = e
      z = self._least_squares(x + e)
      x_ = torch.matmul(z, UsT)
      e = self._prox_e(x_ - x)

      update = (e - e_old).abs().max()
      if update <= self.encode_tol * xnorm:
        break

    if self.encode_max_iter > 0:
      self.encode_steps = kk + 1
      self.encode_update = update
    else:
      self.encode_steps = 0.0
      self.encode_update = 0.0

    # one more time for tracking gradients.
    z = self._least_squares(x + e)
    x_ = torch.matmul(z, self.Us.transpose(2, 3))
    e = self._prox_e(x_.data - x)
    if self.affine:
      x_ = x_.add(self.bs.unsqueeze(2))

    self.z = z
    self.e = e
    return x_

  def encode(self, x):
    raise NotImplementedError("encode not implemented.")

  def decode(self, x):
    raise NotImplementedError("decode not implemented.")

  def _least_squares(self, y):
    """Solve regularized least squares using cached SVD

    min_z 1/2 || Uz - y ||_2^2 + \lambda/2 ||z||_2^2
              || diag(s) (Q^Tz) - P^T y ||_2^2 + \lambda/2 || (Q^Tz) ||_2^2

              (diag(s)^2 + \lambda I) (Q^Tz) = diag(s) P^T y
              z = Q(diag(s)/(diag(s)^2 + \lambda I)) P^T y
    Input:
      y: least squares target, shape (r, k, batch_size, D)

    Returns:
      z: solution (r, k, batch_size, d)
    """
    # y shape (r, k, batch_size, D)
    # (r, k, batch_size, d)
    PTy = torch.matmul(y, self.Us_P)
    lamb = self.reg_params['z']
    if lamb == 0:
      lamb = EPS
    # (r, k, 1, d)
    s_div_sqr = (self.Us_s / (self.Us_s ** 2 + lamb)).unsqueeze(2)
    # (r, k, batch_size, d)
    Qtz = s_div_sqr * PTy
    z = torch.matmul(Qtz, self.Us_Q.transpose(2, 3))
    return z

  def _prox_e(self, w):
    """Compute proximal operator of e regularizer wrt w."""
    return torch.zeros((), dtype=w.dtype, device=w.device)

  def loss(self, x, x_):
    """Evaluate reconstruction loss

    Inputs:
      x: data, shape (batch_size, D)
      x_: reconstruction, shape (r, k, batch_size, D)

    Returns:
      loss: shape (batch_size, r, k)
    """
    loss = torch.sum((x.add(self.e) - x_)**2, dim=3).mul(0.5).permute(2, 0, 1)
    return loss

  def reg(self):
    """Evaluate subspace regularization."""
    regs = dict()
    # U regularization, each is shape (r, k)
    if max([self.reg_params[key] for key in
          ('U_frosqr_in', 'U_frosqr_out')]) > 0:
      U_frosqr = torch.sum(self.Us.pow(2), dim=(2, 3))

      if self.reg_params['U_frosqr_in'] > 0:
        regs['U_frosqr_in'] = U_frosqr.mul(
            self.reg_params['U_frosqr_in']*0.5)

      if self.reg_params['U_frosqr_out'] > 0:
        regs['U_frosqr_out'] = U_frosqr.mul(
            self.reg_params['U_frosqr_out']*0.5)

    # z and e regularization, shapes are (batch_size, r, k)
    # does not affect gradients, only included to ensure objective value
    # is accurate
    if self.reg_params['z'] > 0:
      z_frosqr = torch.sum(self.z.data.pow(2), dim=3).permute(2, 0, 1)
      regs['z'] = z_frosqr.mul(self.reg_params['z']*0.5)

    if self.reg_params['e'] > 0:
      regs['e'] = self.reg_e() * self.reg_params['e']
    return regs

  def reg_e(self):
    """Compute regularizer wrt e."""
    return 0.0


class KSubspaceMCModel(KSubspaceMFCorruptBaseModel):
  """K-subspace model where low-dim coefficients are computed in closed
  form. Adapted to handle missing data."""
  default_reg_params = {
      'U_frosqr_in': 0.01,
      'U_frosqr_out': 1e-4,
      'z': 0.01,
      'e': 1.0  # not meaningful in this case
  }
  assign_reg_terms = {'U_frosqr_in', 'z', 'e'}

  def __init__(self, k, d, D, affine=False, replicates=5, reg_params={},
        reset_patience=100, reset_jitter=True, reset_try_tol=0.01,
        reset_max_steps=None, reset_accept_tol=1e-4, reset_stochastic=True,
        reset_low_value=False, reset_value_thr=0.1, reset_sigma=0.05,
        reset_cache_size=None, cache_subsample_rate=None, encode_max_iter=20,
        encode_tol=1e-3):

    super().__init__(k, d, D, affine, replicates, reg_params, reset_patience,
        reset_jitter, reset_try_tol, reset_max_steps, reset_accept_tol,
        reset_stochastic, reset_low_value, reset_value_thr, reset_sigma,
        reset_cache_size, cache_subsample_rate, encode_max_iter, encode_tol)

    self.omega = None
    self.reset_parameters()
    return

  def objective(self, x_omega):
    """Evaluate objective function.

    Input:
      x_omega: data with missing mask, shape (batch_size, 2, D).
        x_omega[:, 0, :] is assumed to be data, x_omega[:, 1, :] the mask.

    Returns:
      obj_mean: average objective across replicates
      obj, loss, reg_in, reg_out: metrics per replicate, shape (r,)
      x_: reconstruction, shape (r, k, batch_size, D)
    """
    x = self.parse_x_omega(x_omega)
    return super().objective(x)

  def eval_shrink(self, x_omega, x_):
    """measure shrinkage of reconstruction wrt data.

    Inputs:
      x_omega: data with missing mask, shape (batch_size, 2, D).
        x_omega[:, 0, :] is assumed to be data, x_omega[:, 1, :] the mask.
      x_: reconstruction, shape (r, k, batch_size, D)

    Returns:
      norm_x_: average norm of x_ relative to x, shape (r,)
    """
    x = self.parse_x_omega(x_omega, update_omega=False)
    return super().eval_shrink(x, x_)

  def parse_x_omega(self, x_omega, update_omega=True):
    """Parse stacked (x, omega) input

    Input:
      x_omega: data with missing mask, shape (batch_size, 2, D).
        x_omega[:, 0, :] is assumed to be data, x_omega[:, 1, :] the mask.
      update_omega: whether to update omega attribute (default: True).

    Updates:
      omega: updated with current mask in place.

    Returns:
      x: masked data x, shape (batch_size, D).
    """
    if update_omega:
      self.omega = x_omega[:, 1, :] > 0
    # NOTE: omega assumed to be binary mask.
    x = x_omega[:, 0, :] * x_omega[:, 1, :]
    return x

  def _prox_e(self, w):
    """Compute projection onto *unobserved* entries. Note, operates on w in
    place."""
    # w shape (r, k, batch_size, D)
    w[:, :, self.omega] = 0.0
    return w
