from __future__ import print_function
from __future__ import division

import numpy as np
import torch
import torch.nn as nn

import utils as ut

EPS = 1e-8
EMA_DECAY = 0.99
RESET_NCOL = 8
EVAL_RANK_CPU = True


class KSubspaceBaseModel(nn.Module):
  """Base K-subspace class."""
  default_reg_params = dict()
  assign_reg_terms = set()

  def __init__(self, k, d, D, affine=False, replicates=5, reg_params={},
        serial_eval={}, reset_patience=100, reset_try_tol=0.01,
        reset_max_steps=50, reset_accept_tol=1e-3, reset_sigma=0.0,
        reset_cache_size=500, temp_scheduler=None):

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
    if not set(serial_eval).issubset({'r', 'k'}):
      raise ValueError("Invalid serial eval mode.".format(serial_eval))
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
    self.serial_eval = set(serial_eval)

    self.reset_patience = reset_patience
    self.reset_try_tol = reset_try_tol
    self.reset_max_steps = reset_max_steps
    self.reset_accept_tol = reset_accept_tol
    self.reset_sigma = reset_sigma
    self.reset_cache_size = reset_cache_size
    if temp_scheduler is None:
      temp_scheduler = GeoTempScheduler(init_temp=0.1, replicates=self.r,
          patience=1, gamma=0.9)
    elif not isinstance(temp_scheduler, TempScheduler):
      raise ValueError("Invalid temperature scheduler")
    self.temp_scheduler = temp_scheduler

    # group assignment, ultimate shape (batch_size, r, k)
    self.c = None
    self.groups = None
    # subspace coefficient norms, shape (r, k, batch_size)
    self.z_frosqr = None
    # subspace coefficients, shape (r, k, batch_size, d)
    # stored only if cache_z is True, since can be a large memory burden
    self.z = None
    self.cache_z = False

    self.Us = nn.Parameter(torch.Tensor(self.r, k, D, d))
    if affine:
      self.bs = nn.Parameter(torch.Tensor(self.r, k, D))
    else:
      self.register_parameter('bs', None)

    self.register_buffer('c_mean', torch.ones(self.r, k).mul_(np.nan))
    self.register_buffer('value', torch.ones(self.r, k).mul_(np.nan))
    self.register_buffer('obj', torch.ones(self.r).mul_(np.nan))
    self.register_buffer('best_obj', torch.ones(self.r).mul_(np.nan))
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

  def forward(self, x, ii=None, jj=None):
    """Compute representation of x wrt each subspace.

    Input:
      x: data, shape (batch_size, D)
      ii, jj: rep, cluster indices. Either int index or None.

    Returns:
      x_: reconstruction, shape (r, k, batch_size, D)
    """
    z = self.encode(x, ii, jj)
    x_ = self.decode(z, ii, jj)
    self._update_forward_cache(ii, jj, z)
    return x_

  def decode(self, z, ii=None, jj=None):
    """Embed low-dim code z into ambient space.

    Input:
      z: latent code, shape (r, k, batch_size, d)
      ii, jj: rep, cluster indices. Either int index or None.

    Returns:
      x_: reconstruction, shape (r, k, batch_size, D)
    """
    Us, bs = self._slice_Us_bs(ii, jj, no_grad=False)
    assert(z.dim() == 4 and
        (z.size(0), z.size(1)) == (Us.size(0), Us.size(1)))

    # x_ = U z + b
    # shape (r, k, batch_size, D)
    x_ = torch.matmul(z, Us.transpose(2, 3))
    if self.affine:
      x_ = x_.add(self.bs.unsqueeze(2))
    return x_

  def _slice_Us_bs(self, ii, jj, no_grad=True):
    """Extract slice of Us, bs. Limited slicing supported, ii, jj must be index
    or slice(None)."""
    # don't lose dims
    ii, jj = self._parse_slice(ii, jj)
    if no_grad:
      Us = self.Us.data
      bs = self.bs.data if self.affine else None
    else:
      Us = self.Us
      bs = self.bs if self.affine else None
    if not (ii == slice(None) and jj == slice(None)):
      Us = Us[ii, jj, :]
      bs = bs[ii, jj, :] if self.affine else None
    return Us, bs

  def _parse_slice(self, ii, jj):
    """Parse ii jj slice into Us, bs."""
    # this is probably dumb but too tired..
    if type(ii) is int and type(jj) is int:
      ii = np.array([[ii]])
      jj = np.array([[jj]])
    elif type(ii) is int:
      ii = np.array([ii])
      jj = slice(None)
    elif type(jj) is int:
      ii = slice(None)
      jj = np.array([jj])
    else:
      ii, jj = slice(None), slice(None)
    return ii, jj

  def _update_forward_cache(self, ii, jj, z):
    """Update any cached values from forward call."""
    batch_size = z.shape[2]
    device = z.device

    if self.z_frosqr is None or self.z_frosqr.shape[2] != batch_size:
      self.z_frosqr = torch.zeros(self.r, self.k, batch_size, device=device)
      if self.cache_z:
        self.z = torch.zeros(self.r, self.k, batch_size, self.d, device=device)

    ii, jj = self._parse_slice(ii, jj)
    self.z_frosqr[ii, jj, :] = z.data.pow(2).sum(dim=-1)
    if self.cache_z:
      self.z[ii, jj, :] = z
    return

  def objective(self, x):
    """Evaluate objective function.

    Input:
      x: data, shape (batch_size, D)

    Returns:
      obj_mean: average objective across replicates
      obj, loss, reg_in, reg_out: metrics per replicate, shape (r,)
    """
    loss = self.loss(x)
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
    return obj_mean, obj.data, loss.data, reg_in.data, reg_out.data

  def loss(self, x):
    """Evaluate reconstruction loss

    Inputs:
      x: data, shape (batch_size, D)

    Returns:
      loss: shape (batch_size, r, k)
    """
    # Note that X[slice(None)] acts the same as X[:]
    rep_range = range(self.r) if 'r' in self.serial_eval else [None]
    cluster_range = range(self.k) if 'k' in self.serial_eval else [None]

    losses = []
    for ii in rep_range:
      rep_losses = []
      for jj in cluster_range:
        x_ = self.forward(x, ii, jj)
        rep_losses.append(torch.sum((x_ - x)**2, dim=-1).mul(0.5))
      losses.append(torch.cat(rep_losses, dim=1))
    loss = torch.cat(losses, dim=0)
    loss = loss.permute(2, 0, 1)
    return loss

  def set_assign(self, assign_obj):
    """Compute cluster assignment.

    Inputs:
      assign_obj: shape (batch_size, r, k)
    """
    (self.groups, _, self.c, self._batch_c_mean,
        self._batch_value) = self._assign_and_value(assign_obj)

    nan_mask = torch.isnan(self.c_mean)
    if nan_mask.sum() > 0:
      self.c_mean[nan_mask] = self._batch_c_mean[nan_mask]
      self.value[nan_mask] = self._batch_value[nan_mask]

    self.c_mean.mul_(EMA_DECAY).add_(1-EMA_DECAY, self._batch_c_mean)
    self.value.mul_(EMA_DECAY).add_(1-EMA_DECAY, self._batch_value)

    self.groups = self.groups.cpu()
    self._update_assign_obj_cache(assign_obj)
    return

  def _assign_and_value(self, assign_obj):
    """Compute assignments and cluster size & values."""
    batch_size, tmpr = assign_obj.shape[:2]
    device = assign_obj.device

    if self.k > 1:
      top2obj, top2idx = torch.topk(assign_obj, 2, dim=2, largest=False,
          sorted=True)
      groups = top2idx[:, :, 0]
      min_assign_obj = top2obj[:, :, 0]
    else:
      groups = torch.zeros(batch_size, tmpr, device=device, dtype=torch.int64)
      min_assign_obj = assign_obj.squeeze(2)

    c = torch.zeros_like(assign_obj)
    c.scatter_(2, groups.unsqueeze(2), 1)
    c_mean = c.mean(dim=0)

    if self.k > 1:
      value = torch.zeros_like(assign_obj)
      value = value.scatter_(2, groups.unsqueeze(2),
          (top2obj[:, :, [1]] - top2obj[:, :, [0]])).mean(dim=0)
    else:
      value = torch.ones(tmpr, self.k, device=device)
    return groups, min_assign_obj, c, c_mean, value

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
    Us = self.Us.data.cpu() if EVAL_RANK_CPU else self.Us.data
    # (r, k, d)
    svs = ut.batch_svd(Us)[1]
    tol = svs[:, :, 0].max(dim=1)[0].mul(tol).view(self.r, 1, 1)
    # (r, k)
    ranks = (svs > tol).sum(dim=2)
    return ranks, svs

  def reset_unused(self):
    """Reset replicates whose progress has slowed by doing several iterations
    of simulated annealing over the graph of all (rk choose k) subsets of
    clusters.

    Returns:
      resets: np array log of resets, shape (n_resets, 8). Columns
        are (reset rep idx, reset cluster idx, candidate rep idx,
        candidate cluster idx, reset success, obj decrease, cumulative obj
        decrease, temperature).
    """
    empty_output = np.zeros((0, RESET_NCOL), dtype=object)

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

    # attempt to re-initialize replicates by greedy hill-climbing
    resets = []
    for ridx in reset_rids:
      rep_resets = self._reset_replicate(ridx.item())
      resets.append(rep_resets)
    resets = np.concatenate(resets, axis=0)

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

    reset_mask = self.num_bad_steps[0, :] > self.reset_patience
    reset_rids = reset_mask.nonzero().view(-1)
    return reset_rids

  def _reset_replicate(self, ridx):
    """Re-initialize a replicate by executing several steps of simulated
    annealing over the graph of (rk choose k) subsets of bases."""
    device = self.Us.data.device
    resets = []
    # fixed temp for this inner loop
    temp = self.temp_scheduler.step(ridx)

    value, min_assign_obj = self._eval_value()
    sample_prob = (value.unsqueeze(0).clamp(min=0) /
        value[ridx, :].view(self.k, 1, 1).clamp(min=EPS)).cpu()
    sample_prob[:, ridx, :] = 0.0
    rep_obj = (min_assign_obj[:, ridx].mean() +
        self._batch_reg_out[ridx, :].sum()).item()
    best_obj = old_obj = rep_obj

    # first dim: (tmp, current, best)
    rep_assign_obj = torch.zeros(3, self.reset_cache_size, self.k,
        device=device).copy_(self.cache_assign_obj[:, ridx, :].unsqueeze(0))
    rep_reg_out = torch.zeros(3, self.k, device=device).copy_(
        self._batch_reg_out[[ridx], :])

    for _ in range(self.reset_max_steps):
      idx = torch.multinomial(sample_prob.view(-1), 1)[0].item()
      cidx, cand_ridx, cand_cidx = np.unravel_index(idx,
          (self.k, self.r, self.k))
      rep_assign_obj[0, :, cidx] = self.cache_assign_obj[:,
          cand_ridx, cand_cidx]
      rep_reg_out[0, cidx] = self._batch_reg_out[cand_ridx, cand_cidx]

      cand_obj = (rep_assign_obj[0, :].min(dim=1)[0].mean() +
          rep_reg_out[0, :].sum()).item()
      obj_decr = (rep_obj - cand_obj) / rep_obj
      accept_prob = np.exp(min(obj_decr - self.reset_accept_tol, 0) / temp)
      success = torch.rand(1)[0].item() <= accept_prob

      if success:
        # update current bases and assign obj
        rep_assign_obj[1, :, cidx] = rep_assign_obj[0, :, cidx]
        rep_reg_out[1, cidx] = rep_reg_out[0, cidx]

        # update sample prob and replicate obj
        rep_value, rep_min_assign_obj = self._eval_value(
            assign_obj=rep_assign_obj[1, :], reg_out=rep_reg_out[1, :])
        sample_prob.copy_(value.unsqueeze(0).clamp(min=0) /
            rep_value.view(self.k, 1, 1).clamp(min=EPS))
        sample_prob[:, ridx, :] = 0.0
        rep_obj = (rep_min_assign_obj.mean() + rep_reg_out[1, :].sum()).item()

        # update best assign obj
        if rep_obj < best_obj:
          best_obj = rep_obj
          rep_assign_obj[2, :] = rep_assign_obj[1, :]
          rep_reg_out[2, :] = rep_reg_out[1, :]
      else:
        rep_assign_obj[0, :, cidx] = rep_assign_obj[1, :, cidx]
        rep_reg_out[0, cidx] = rep_reg_out[1, cidx]

      cumu_obj_decr = (old_obj - rep_obj) / old_obj
      resets.append([ridx, cidx, cand_ridx, cand_cidx,
          int(success), obj_decr, cumu_obj_decr, temp])

    resets = np.array(resets, dtype=object)
    self.num_bad_steps[0, ridx] = 0

    # final updates if objective sufficiently improved
    if (old_obj - best_obj) / old_obj >= self.reset_accept_tol:
      bestitr = np.argmax(resets[:, 6])
      resets[bestitr+1:, 4] = 0

      reset_ids = resets[resets[:, 4] == 1, 1:4].astype(np.int64)
      uniqIdx = ut.unique_resets(reset_ids[:, 0])
      reset_ids = reset_ids[uniqIdx, :]
      cIdx, cand_rIdx, cand_cIdx = [reset_ids[:, ii] for ii in range(3)]
      self.Us.data[ridx, cIdx, :] = self.Us.data[cand_rIdx, cand_cIdx, :]
      if self.affine:
        self.bs.data[ridx, cIdx, :] = self.bs.data[cand_rIdx, cand_cIdx, :]

      self.cache_assign_obj[:, ridx, :] = rep_assign_obj[2, :]
      self._batch_reg_out[ridx, :] = rep_reg_out[2, :]

      self.c_mean[ridx, :] = np.nan
      self.value[ridx, :] = np.nan
      self.obj[ridx] = self.best_obj[ridx] = np.nan
      self.num_bad_steps[1, ridx] = 0
      self.cooldown_counter[ridx] = self.reset_patience // 2
    else:
      resets[:, 4] = 0
    return resets

  def _eval_value(self, assign_obj=None, reg_out=None):
    """Evaluate value of every replicate, cluster based on cache objective."""
    assert((assign_obj is None) == (reg_out is None))
    if assign_obj is None:
      assign_obj = self.cache_assign_obj
    if reg_out is None:
      reg_out = self._batch_reg_out
    assert(reg_out.dim() == assign_obj.dim() - 1)
    if assign_obj.dim() == 2:
      assign_obj = assign_obj.unsqueeze(1)
      reg_out = reg_out.unsqueeze(0)

    _, min_assign_obj, _, _, value = self._assign_and_value(assign_obj)
    value.sub_(reg_out)
    return value.squeeze(0), min_assign_obj.squeeze(1)

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
      'z': 0.01
  }
  assign_reg_terms = {'U_frosqr_in', 'z'}

  def __init__(self, k, d, D, affine=False, replicates=5, reg_params={},
        serial_eval={}, reset_patience=100, reset_try_tol=0.01,
        reset_max_steps=50, reset_accept_tol=1e-3, reset_sigma=0.0,
        reset_cache_size=500, temp_scheduler=None, scale_grad_freq=100,
        init='random', initX=None, initk=None):

    if init.lower() not in {'random', 'pfi', 'pca'}:
      raise ValueError("Invalid init mode {}".format(init))
    if init.lower in {'pfi', 'pca'}:
      if initX is None:
        raise ValueError(("Data subset initX required for PFI or PCA "
            "initialization"))
      elif not (torch.is_tensor(initX) and
            initX.dim() == 2 and initX.shape[1] == D):
        raise ValueError("Invalid data subset initX ")

    super().__init__(k, d, D,
        affine=affine, replicates=replicates, reg_params=reg_params,
        serial_eval=serial_eval, reset_patience=reset_patience,
        reset_try_tol=reset_try_tol, reset_max_steps=reset_max_steps,
        reset_accept_tol=reset_accept_tol, reset_sigma=reset_sigma,
        reset_cache_size=reset_cache_size, temp_scheduler=temp_scheduler)

    if scale_grad_freq is not None and scale_grad_freq >= 0:
      self.scale_grad_freq = scale_grad_freq
      self.Lip = None
      self.steps = 0
      self.cache_z = True
      self.Us.register_hook(lambda UGrad: self._scale_grad(UGrad))
      if self.affine:
        self.bs.register_hook(lambda bGrad: self._scale_grad(bGrad,
            update=False))

    self.init = init.lower()
    self.initX = initX
    if initk is None or initk <= 0:
      initk = k
    self.initk = initk
    self.reset_parameters()
    return

  def reset_parameters(self):
    """Initialize bases either with small random entries, by probabilistic
    farthest insertion, or PCA."""
    if self.init == 'random':
      super().reset_parameters()
      if self.initk < self.k:
        self.Us.data[:, self.initk:, :] = 0.0
    elif self.init == 'pca':
      self._pca_init(self.initX)
    else:
      self._prob_farthest_insert_init(self.initX,
          nn_q=int(np.ceil(0.1*self.d)))
    return

  def _pca_init(self, X, sigma=0.2):
    """Initialize bases with perturbed singular vectors of X."""
    N = X.shape[0]
    lamb = np.sqrt(N*self.reg_params['U_frosqr_in'] *
        self.reg_params['z'])
    U, b = ut.reg_pca(X, self.d, form='mf', lamb=lamb, gamma=0.0,
        affine=self.affine, solver='randomized')

    # re-scale U
    if lamb > 0:
      alpha = (np.sqrt(self.reg_params['z']) /
          np.sqrt(N*self.reg_params['U_frosqr_in'])) ** 0.5
      U.mul_(alpha)

    ZU = torch.randn(self.r, self.initk, self.D, self.d,
        device=U.device).mul_(sigma*torch.norm(U) / np.sqrt(self.D*self.d))
    self.Us.data.zero_()
    self.Us.data[:, :self.initk, :] = U + ZU
    if self.affine:
      self.bs.data.zero_()
      zb = torch.randn(self.r, self.k, self.D,
          device=U.device).mul_(sigma*torch.norm(b) / np.sqrt(self.D))
      self.bs.data[:, :self.initk, :] = b + zb
    return

  def _prob_farthest_insert_init(self, X, nn_q=0):
    """Initialize Us, bs by probabilistic farthest insertion (single trial)

    Args:
      X: Data matrix from which to select bases, shape (N, D)
      nn_q: Extra nearest neighbors (default: 0).
    """
    nn_k = self.d + nn_q
    if self.affine:
      nn_k += 1
    N = X.shape[0]

    self.Us.data.zero_()
    if self.affine:
      self.bs.data.zero_()

    # normalization needed for easier nearest-neigbor calculation
    # but possibly not the best in affine case
    X = ut.unit_normalize(X, dim=1)

    # choose first basis randomly
    Idx = torch.randint(0, N, (self.r,), dtype=torch.int64)
    self._insert_next(0, X[Idx, :], X, nn_k)

    for cidx in range(1, self.initk):
      # (N, r, k')
      obj = self._prob_insert_objective(X, cidx)
      # (N, r)
      min_obj = torch.min(obj, dim=2)[0]
      # (r,)
      Idx = torch.multinomial(min_obj.t(), 1).view(-1)
      self._insert_next(cidx, X[Idx, :], X, nn_k)
    return

  def _prob_insert_objective(self, X, cidx):
    """Evaluate objective wrt X for current initialized subspaces up to but not
    including cidx.
    """
    # NOTE: this mostly duplicates code from objective, loss. Purpose is to
    # avoid unnecessary computation. But still, could probably simplify.
    device = X.device
    batch_size = X.shape[0]
    loss = torch.zeros(self.r, cidx, batch_size, device=device)

    with torch.no_grad():
      for ii in range(self.r):
        for jj in range(cidx):
          X_ = self.forward(X, ii, jj)
          slci, slcj = self._parse_slice(ii, jj)
          loss[slci, slcj, :] = torch.sum((X_ - X)**2, dim=-1).mul(0.5)
      loss = loss.permute(2, 0, 1)

      reg = self.reg()
      reg_in = torch.zeros(self.r, self.k, device=device)
      for key, val in reg.items():
        if key in self.assign_reg_terms:
          reg_in = reg_in + val
      if reg_in.dim() == 2:
        reg_in = reg_in.unsqueeze(0)
      reg_in = reg_in[:, :, :cidx]

      assign_obj = loss + reg_in
    return assign_obj

  def _insert_next(self, cidx, y, X, nn_k):
    """Insert next basis (cidx) centered on sampled points y from X."""
    y_knn = ut.cos_knn(y, X, nn_k)
    for ii in range(self.r):
      Xj = y_knn[ii, :]
      Nj = Xj.shape[0]
      lamb = np.sqrt(Nj*self.reg_params['U_frosqr_in'] *
          self.reg_params['z'])
      U, b = ut.reg_pca(Xj, self.d, form='mf', lamb=lamb, gamma=0.0,
          affine=self.affine, solver='randomized')

      # re-scale U
      if lamb > 0:
        alpha = (np.sqrt(self.reg_params['z']) /
            np.sqrt(Nj*self.reg_params['U_frosqr_in'])) ** 0.5
        U.mul_(alpha)

      self.Us.data[ii, cidx, :] = U
      if self.affine:
        self.bs.data[ii, cidx, :] = b
    return

  def encode(self, x, ii=None, jj=None):
    """Compute subspace coefficients for x in closed form, by computing
    batched solution to normal equations.

      min_z 1/2 || x - (Uz + b) ||_2^2 + \lambda/2 ||z||_2^2
      (U^T U + \lambda I) z* = U^T (x - b)

    Input:
      x: shape (batch_size, D)
      ii, jj: rep, cluster indices. Either int index or None.

    Returns:
      z: latent low-dimensional codes (r, k, batch_size, d)
    """
    # Us shape (r, k, D, d)
    # bs shape (r, k, D)
    Us, bs = self._slice_Us_bs(ii, jj, no_grad=True)
    batch_size = x.size(0)

    # (1, 1, D, batch_size)
    x = x.data.t().view(1, 1, self.D, batch_size)
    if self.affine:
      # (r, k, D, batch_size)
      x = x.sub(bs.unsqueeze(3))
    # (r, k, d, batch_size)
    z = ut.batch_ridge(x, Us, lamb=self.reg_params['z'])

    # (r, k, batch_size, d)
    z = z.transpose(2, 3)
    return z

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

    # z regularization, shape is (batch_size, r, k)
    # does not affect gradients, only included to ensure objective value
    # is accurate
    if self.reg_params['z'] > 0:
      regs['z'] = self.z_frosqr.permute(2, 0, 1).mul(self.reg_params['z']*0.5)
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
      'U_frosqr_out': 1e-4
  }
  assign_reg_terms = {'U_frosqr_in'}

  def __init__(self, k, d, D, affine=False, replicates=5, reg_params={},
        serial_eval={}, reset_patience=100, reset_try_tol=0.01,
        reset_max_steps=50, reset_accept_tol=1e-3, reset_sigma=0.0,
        reset_cache_size=500, temp_scheduler=None):

    super().__init__(k, d, D,
        affine=affine, replicates=replicates, reg_params=reg_params,
        serial_eval=serial_eval, reset_patience=reset_patience,
        reset_try_tol=reset_try_tol, reset_max_steps=reset_max_steps,
        reset_accept_tol=reset_accept_tol, reset_sigma=reset_sigma,
        reset_cache_size=reset_cache_size, temp_scheduler=temp_scheduler)

    self.reset_parameters()
    return

  def encode(self, x, ii=None, jj=None):
    """Project x onto each of k low-dimensional spaces.

    Input:
      x: shape (batch_size, D)
      ii, jj: rep, cluster indices. Either int index or None.

    Returns:
      z: latent low-dimensional codes (r, k, batch_size, d)
    """
    Us, bs = self._slice_Us_bs(ii, jj, no_grad=False)
    batch_size = x.size(0)

    # z = U^T (x - b) or z = V (x - b)
    if self.affine:
      # (r, k, batch_size, D)
      x = x.sub(bs.unsqueeze(2))
    else:
      x = x.view(1, 1, batch_size, self.D)

    # (r, k, batch_size, d)
    z = torch.matmul(x, Us)
    return z

  def reg(self):
    """Evaluate subspace regularization."""
    regs = dict()
    # U regularization, each is shape (r, k)
    if max([self.reg_params[key] for key in
          ('U_frosqr_in', 'U_frosqr_out')]) > 0:
      U_frosqr = torch.sum(self.Us.pow(2), dim=(2, 3))

      if self.reg_params['U_frosqr_in'] > 0:
        regs['U_frosqr_in'] = U_frosqr.mul(
            self.reg_params['U_frosqr_in'])

      if self.reg_params['U_frosqr_out'] > 0:
        regs['U_frosqr_out'] = U_frosqr.mul(
            self.reg_params['U_frosqr_out'])
    return regs


class KSubspaceBatchAltBaseModel(KSubspaceBaseModel):
  default_reg_params = dict()
  assign_reg_terms = dict()

  def __init__(self, k, d, dataset, affine=False, replicates=5, reg_params={},
        serial_eval={}, reset_patience=2, reset_try_tol=0.01,
        reset_max_steps=50, reset_accept_tol=1e-3, reset_sigma=0.0,
        reset_cache_size=500, temp_scheduler=None, svd_solver='randomized',
        **kwargs):

    if svd_solver not in ('randomized', 'svds', 'svd'):
      raise ValueError("Invalid svd solver {}".format(svd_solver))

    # X assumed to be N x D.
    D = dataset.X.shape[1]
    super().__init__(k, d, D,
        affine=affine, replicates=replicates, reg_params=reg_params,
        serial_eval=serial_eval, reset_patience=reset_patience,
        reset_try_tol=reset_try_tol, reset_max_steps=reset_max_steps,
        reset_accept_tol=reset_accept_tol, reset_sigma=reset_sigma,
        reset_cache_size=reset_cache_size, temp_scheduler=temp_scheduler,
        **kwargs)

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
    (self.groups, _, self.c, self._batch_c_mean,
        self._batch_value) = self._assign_and_value(assign_obj)
    self.c_mean = self._batch_c_mean
    self.value = self._batch_value

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
      'U_frosqr_out': 1e-4
  }
  assign_reg_terms = {'U_frosqr_in'}

  def __init__(self, k, d, dataset, affine=False, replicates=5, reg_params={},
        serial_eval={}, reset_patience=2, reset_try_tol=0.01,
        reset_max_steps=50, reset_accept_tol=1e-3, reset_sigma=0.0,
        reset_cache_size=500, temp_scheduler=None, svd_solver='randomized'):

    super().__init__(k, d, dataset,
        affine=affine, replicates=replicates, reg_params=reg_params,
        serial_eval=serial_eval, reset_patience=reset_patience,
        reset_try_tol=reset_try_tol, reset_max_steps=reset_max_steps,
        reset_accept_tol=reset_accept_tol, reset_sigma=reset_sigma,
        reset_cache_size=reset_cache_size, temp_scheduler=temp_scheduler,
        svd_solver=svd_solver)
    return

  def step(self):
    """Update subspace bases by regularized pca (and increment step count)."""
    for ii in range(self.r):
      for jj in range(self.k):
        if self.c_mean[ii, jj] > 0:
          Xj = self.X[self.c[:, ii, jj] == 1, :]
          Nj = Xj.shape[0]
          lamb = (Nj*self.reg_params['U_frosqr_in'] +
              self.N*self.reg_params['U_frosqr_out'])
          U, b = ut.reg_pca(Xj, self.d, form='proj', lamb=lamb,
              affine=self.affine, solver=self.svd_solver)

          self.Us.data[ii, jj, :] = U
          if self.affine:
            self.bs.data[ii, jj, :] = b
    return


class KSubspaceBatchAltMFModel(KSubspaceBatchAltBaseModel, KSubspaceMFModel):
  default_reg_params = {
      'U_frosqr_in': 0.01,
      'U_frosqr_out': 1e-4,
      'z': 0.01
  }
  assign_reg_terms = {'U_frosqr_in', 'z'}

  def __init__(self, k, d, dataset, affine=False, replicates=5, reg_params={},
        serial_eval={}, reset_patience=2, reset_try_tol=0.01,
        reset_max_steps=50, reset_accept_tol=1e-3, reset_sigma=0.0,
        reset_cache_size=500, temp_scheduler=None, init='random', initX=None,
        initk=None, svd_solver='randomized'):

    mf_kwargs = {'scale_grad_freq': None, 'init': init, 'initX': initX,
        'initk': initk}
    super().__init__(k, d, dataset,
        affine=affine, replicates=replicates, reg_params=reg_params,
        serial_eval=serial_eval, reset_patience=reset_patience,
        reset_try_tol=reset_try_tol, reset_max_steps=reset_max_steps,
        reset_accept_tol=reset_accept_tol, reset_sigma=reset_sigma,
        reset_cache_size=reset_cache_size, temp_scheduler=temp_scheduler,
        svd_solver=svd_solver, **mf_kwargs)

    # bases initialized in KSubspaceMFModel
    # ensure columns are orthogonal
    for ii in range(self.r):
      for jj in range(self.k):
        P, S, _ = torch.svd(self.Us.data[ii, jj, :])
        self.Us.data[ii, jj, :] = P.mul_(S)
    return

  def encode(self, x, ii=None, jj=None):
    """Compute subspace coefficients for x in closed form, by computing
    batched solution to normal equations. Use fact that colums of U are
    orthogonal (althought not necessarily unit norm).

      min_z 1/2 || x - (Uz + b) ||_2^2 + \lambda/2 ||z||_2^2
      (U^T U + \lambda I) z* = U^T (x - b)

    Input:
      x: shape (batch_size, D)
      ii, jj: rep, cluster indices. Either int index or None.

    Returns:
      z: latent low-dimensional codes (r, k, batch_size, d)
    """
    Us, bs = self._slice_Us_bs(ii, jj, no_grad=True)
    batch_size = x.size(0)

    if self.affine:
      # (r, k, batch_size, D)
      x = x.sub(bs.unsqueeze(2))
    else:
      x = x.view(1, 1, batch_size, self.D)

    # (r, k, batch_size, d)
    b = torch.matmul(x, Us)

    # (r, k, 1, d)
    S = torch.norm(Us, dim=2, keepdim=True)
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
        serial_eval={}, reset_patience=2, reset_try_tol=0.01,
        reset_max_steps=50, reset_accept_tol=1e-3, reset_sigma=0.0,
        reset_cache_size=500, temp_scheduler=None, kpp_n_trials=None):

    if init not in {'k-means++', 'random'}:
      raise ValueError("Invalid init parameter {}".format(init))

    d = 1  # Us not used, but retained for consistency/out of laziness
    affine = True

    super().__init__(k, d, dataset,
        affine=affine, replicates=replicates, reg_params=reg_params,
        serial_eval=serial_eval, reset_patience=reset_patience,
        reset_try_tol=reset_try_tol, reset_max_steps=reset_max_steps,
        reset_accept_tol=reset_accept_tol, reset_sigma=reset_sigma,
        reset_cache_size=reset_cache_size, temp_scheduler=temp_scheduler)

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

  def forward(self, x, ii=None, jj=None):
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

  def encode(self, x, ii=None, jj=None):
    raise NotImplementedError("encode not implemented.")

  def decode(self, x, ii=None, jj=None):
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
    norm_x_ = torch.ones(x_.shape[0], dtype=x_.dtype, device=x_.device).mul(-1)
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


class KSubspaceMCModel(KSubspaceMFModel):
  """K-subspace model where low-dim coefficients are computed in closed
  form. Adapted to handle missing data."""
  default_reg_params = {
      'U_frosqr_in': 0.01,
      'U_frosqr_out': 1e-4,
      'z': 0.01
  }
  assign_reg_terms = {'U_frosqr_in', 'z', 'e'}

  def __init__(self, k, d, D, affine=False, replicates=5, reg_params={},
        reset_patience=100, reset_try_tol=0.01, reset_max_steps=50,
        reset_accept_tol=1e-3, reset_sigma=0.0, reset_cache_size=500,
        temp_scheduler=None, scale_grad_freq=100, sparse_encode=True,
        sparse_decode=False, norm_comp_error=True):

    super().__init__(k, d, D,
        affine=affine, replicates=replicates, reg_params=reg_params,
        reset_patience=reset_patience, reset_try_tol=reset_try_tol,
        reset_max_steps=reset_max_steps, reset_accept_tol=reset_accept_tol,
        reset_sigma=reset_sigma, reset_cache_size=reset_cache_size,
        temp_scheduler=temp_scheduler, scale_grad_freq=scale_grad_freq)

    self.sparse_encode = sparse_encode
    self.sparse_decode = sparse_decode
    self.norm_comp_error = norm_comp_error
    self.repUs = None
    self.repbs = None

    self.cache_z = True
    self.z = None
    self.cache_x_ = not sparse_decode
    self.x_ = None
    return

  def loss(self, x):
    """Evaluate reconstruction loss

    Inputs:
      x: either sparse format (list of sparse tensors), or dense format
        (tensor with missing elements coded as nan, shape (batch_size, D)).

    Returns:
      loss: shape (batch_size, r, k)
    """
    x_ = self(x)
    # loss (r, k, batch_size)
    if self.sparse_decode:
      # x_ assumed (r, k, batch_size, pad_nnz)
      loss = torch.sum(((x_ * x.omega_float) - x.values)**2,
          dim=-1).mul(0.5)
    else:
      # x_ assumed (r, k, batch_size, D)
      loss = torch.sum(((x_ * x.omega_float_dense) - x.values_dense)**2,
          dim=-1).mul(0.5)
    # (batch_size, r, k)
    loss = loss.permute(2, 0, 1)
    return loss

  def forward(self, x):
    """Compute representation of x wrt each subspace.

    Input:
      x: data, shape (batch_size, D)

    Returns:
      x_: reconstruction, shape (r, k, batch_size, D)
    """
    z = self.encode(x)
    x_ = self.decode(z, x, compute_repUs=False)
    self._update_forward_cache(z, x_)
    return x_

  def encode(self, x):
    """Encode x by vectorized (hence parallelizable) (r*k*batch_size)
    O(pad_nnz) least-squares problems."""
    # Us shape (r, k, D, d)
    # bs shape (r, k, D)
    batch_size = x.shape[0]

    if self.sparse_encode:
      # (r, k, batch_size, pad_nnz, d)
      # cache before zeroing out
      self.repUs = self.Us[:, :, x.indices, :]
      self.repUs = self.repUs.mul(x.omega_float.unsqueeze(2))
    else:
      # (r, k, batch_size, D, d)
      if (self.repUs is None or self.repUs.shape[2] != batch_size):
        self.repUs = torch.zeros(self.r, self.k, batch_size, self.D, self.d,
            device=self.Us.device)
      self.repUs.copy_(self.Us.data.unsqueeze(2))
      self.repUs.mul_(x.omega_float_dense.unsqueeze(2))

    # (batch_size, *)
    xval = x.values if self.sparse_encode else x.values_dense
    if self.affine:
      if self.sparse_encode:
        # (r, k, batch_size, pad_nnz)
        self.repbs = self.bs[:, :, x.indices]
        # (r, k, batch_size, pad_nnz)
        xval = xval.sub(self.repbs.detach())
      else:
        # (r, k, batch_size, D)
        xval = xval.sub(self.bs.unsqueeze(2))
    # (r, k, batch_size, *, 1)
    xval = xval.unsqueeze(-1)

    # (r, k, batch_size, d)
    z = ut.batch_ridge(xval, self.repUs.detach(),
        lamb=self.reg_params['z']).squeeze(4)
    return z

  def decode(self, z, x, compute_repUs=True):
    """Encode x by vectorized (hence parallelizable) (r*k*batch_size)
    O(pad_nnz) mat-vec products."""
    # x_ = U z + b
    # Us shape (r, k, D, d)
    # bs shape (r, k, D)
    if self.sparse_decode:
      # repUs (r, k, batch_size, pad_nnz, d)
      if compute_repUs:
        self.repUs = self.Us[:, :, x.indices, :]
      # (r, k, batch_size, pad_nnz)
      x_ = torch.matmul(self.repUs, z.unsqueeze(4)).squeeze(4)

      if self.affine:
        # repbs (r, k, batch_size, pad_nnz)
        if compute_repUs:
          self.repbs = self.bs[:, :, x.indices]
        x_ = x_.add(self.repbs)
    else:
      # (r, k, batch_size, D)
      x_ = torch.matmul(z, self.Us.transpose(2, 3))
      if self.affine:
        x_ = x_.add(self.bs.unsqueeze(2))
    return x_

  def _update_forward_cache(self, z, x_):
    """Update any cached values from forward call."""
    self.z_frosqr = z.data.pow(2).sum(dim=-1)
    if self.cache_z:
      self.z = z.data
    if self.cache_x_:
      self.x_ = x_.data
    return

  def eval_shrink(self, x, x_):
    raise NotImplementedError("eval_shrink not implemented.")

  def eval_comp_error(self, x0):
    """Evaluate completion error over observed entries in x0.

    Input:
      x0: either sparse format (list of sparse tensors), or dense format
        (tensor with missing elements coded as nan, shape (batch_size, D)).

    Returns:
      comp_err: rmse relative to mean squared magnitude of x0.
    """
    batch_size = x0.shape[0]
    assert(batch_size == self.groups.shape[0])
    if x0.max_nnz == 0:
      return torch.zeros(self.r, device=x0.device)

    with torch.no_grad():
      if self.sparse_decode:
        # (r, k, batch_size, max_nnz)
        x0_ = self.decode(self.z, x0)
      else:
        # (r, k, batch_size, D)
        x0_ = self.x_

      # (r, batch_size, *)
      rIdx = torch.arange(self.r).view(-1, 1)
      batchIdx = torch.arange(batch_size).view(1, -1)
      x0_ = x0_[rIdx, self.groups.t(), batchIdx, :]
      # (r, total_nnz)
      x0_ = x0_[:, x0.omega] if self.sparse_decode else x0_[:, x0.omega_dense]
      # (total_nnz,)
      if x0.store_sparse:
        x0 = x0.values[x0.omega]
      else:
        x0 = x0.values_dense[x0.omega_dense]

      # (r, )
      denom = x0.pow(2).mean() if self.norm_comp_error else 1.0
      comp_err = ((x0_ - x0).pow(2).mean(dim=1) / denom).sqrt()
    return comp_err


class KSubspaceMFCorruptModel(KSubspaceMFModel):
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
        serial_eval={}, reset_patience=100, reset_try_tol=0.01,
        reset_max_steps=50, reset_accept_tol=1e-3, reset_sigma=0.0,
        reset_cache_size=500, temp_scheduler=None, scale_grad_freq=100,
        encode_max_iter=20, encode_tol=1e-3):

    if encode_max_iter <= 0:
      raise ValueError(("Invalid encode_max_iter parameter {}").format(
          encode_max_iter))

    super().__init__(k, d, D,
        affine=affine, replicates=replicates, reg_params=reg_params,
        serial_eval=serial_eval, reset_patience=reset_patience,
        reset_try_tol=reset_try_tol, reset_max_steps=reset_max_steps,
        reset_accept_tol=reset_accept_tol, reset_sigma=reset_sigma,
        reset_cache_size=reset_cache_size, temp_scheduler=temp_scheduler,
        scale_grad_freq=scale_grad_freq)

    # when e reg parameter is set to zero we disable the corruption term,
    # although strictly speaking it should give the trivial solution e = full
    # residual.
    if self.reg_params['e'] == 0:
      encode_max_iter = 0
    self.encode_max_iter = encode_max_iter
    self.encode_tol = encode_tol
    self.encode_steps = None
    self.encode_update = None
    self.cache_z = True

    # denoised x
    self.cache_x0_ = True
    self.x0_ = None
    # corruption
    self.cache_e = False
    self.e = None
    return

  def forward(self, x, ii=None, jj=None):
    """Compute subspace coefficients and reconstruction for x by alternating
    least squares and proximal operator on e.

      min_{z,e} 1/2 || x - (Uz + b + e) ||_2^2
          \lambda/2 ||z||_2^2 + \gamma ||e||

    Input:
      x: shape (batch_size, D)
      ii, jj: rep, cluster indices. Either int index or None.

    Returns:
      x_: reconstruction, shape (r, k, batch_size, D)
    """
    Us, bs = self._slice_Us_bs(ii, jj, no_grad=False)
    batch_size = x.size(0)
    xnorm = max(x.abs().max(), 1.0)

    if self.affine:
      # (r, k, batch_size, D)
      x = x.sub(bs.data.unsqueeze(2))
    else:
      x = x.view(1, 1, batch_size, self.D)

    Us_P, Us_s, Us_Q = ut.batch_svd(Us.data)
    UsT = Us.data.transpose(2, 3).contiguous()

    e = torch.zeros((), dtype=x.dtype, device=x.device)
    for kk in range(self.encode_max_iter):
      e_old = e
      z = self._least_squares(x - e, Us_P, Us_s, Us_Q)
      x_ = torch.matmul(z, UsT)
      e = self._prox_e(x - x_)

      update = (e - e_old).abs().max()
      if update <= self.encode_tol * xnorm:
        break

    if self.encode_max_iter > 0:
      if self.encode_steps is None:
        self.encode_steps = kk+1
        self.encode_update = update
      else:
        self.encode_steps = (EMA_DECAY*self.encode_steps +
            (1-EMA_DECAY)*(kk+1))
        self.encode_update = (EMA_DECAY*self.encode_update +
            (1-EMA_DECAY)*update)

    # one more time for tracking gradients.
    # z = self._least_squares(x - e, Us_P, Us_s, Us_Q)
    x_ = torch.matmul(z, Us.transpose(2, 3))
    if self.affine:
      x_ = x_.add(bs.unsqueeze(2))
    x0_ = x_.data.clone()
    x_ = x_.add(e)

    self._update_forward_cache(ii, jj, z, e, x0_)
    return x_

  def encode(self, x, ii=None, jj=None):
    raise NotImplementedError("encode not implemented.")

  def decode(self, x, ii=None, jj=None):
    raise NotImplementedError("decode not implemented.")

  def _least_squares(self, y, P, s, Q):
    """Solve regularized least squares using cached SVD

    min_z 1/2 || Uz - y ||_2^2 + \lambda/2 ||z||_2^2
              || diag(s) (Q^Tz) - P^T y ||_2^2 + \lambda/2 || (Q^Tz) ||_2^2

              (diag(s)^2 + \lambda I) (Q^Tz) = diag(s) P^T y
              z = Q(diag(s)/(diag(s)^2 + \lambda I)) P^T y
    Input:
      y: least squares target, shape (r, k, batch_size, D)
      P, s, Q: Us SVD.

    Returns:
      z: solution (r, k, batch_size, d)
    """
    # y shape (r, k, batch_size, D)
    # (r, k, batch_size, d)
    PTy = torch.matmul(y, P)
    lamb = self.reg_params['z']
    if lamb == 0:
      lamb = EPS
    # (r, k, 1, d)
    s_div_sqr = (s / (s ** 2 + lamb)).unsqueeze(2)
    # (r, k, batch_size, d)
    Qtz = s_div_sqr * PTy
    z = torch.matmul(Qtz, Q.transpose(2, 3))
    return z

  def _prox_e(self, w):
    """Compute proximal operator of e regularizer wrt w."""
    return torch.zeros((), dtype=w.dtype, device=w.device)

  def _update_forward_cache(self, ii, jj, z, e, x0_):
    """Update any cached values from forward call."""
    batch_size = z.shape[2]
    device = z.device
    ii, jj = self._parse_slice(ii, jj)

    if self.z_frosqr is None or self.z_frosqr.shape[2] != batch_size:
      self.z_frosqr = torch.zeros(self.r, self.k, batch_size, device=device)
      if self.cache_z:
        self.z = torch.zeros(self.r, self.k, batch_size, self.d, device=device)

    self.z_frosqr[ii, jj, :] = z.data.pow(2).sum(dim=-1)
    if self.cache_z:
      self.z[ii, jj, :] = z

    if self.cache_x0_:
      if self.x0_ is None or self.x0_.shape[2] != batch_size:
        self.x0_ = torch.zeros(self.r, self.k, batch_size, self.D,
            device=device)
      self.x0_[ii, jj, :] = x0_

    if self.cache_e:
      if self.e is None or self.e.shape[2] != batch_size:
        self.e = torch.zeros(self.r, self.k, batch_size, self.D,
            device=device)
      self.e[ii, jj, :] = e
    return

  def reg(self):
    """Evaluate subspace regularization."""
    regs = super().reg()

    if self.reg_params['e'] > 0:
      regs['e'] = self.reg_e() * self.reg_params['e']
    return regs

  def reg_e(self):
    """Compute regularizer wrt e."""
    return 0.0


class KSubspaceMFOutlierModel(KSubspaceMFCorruptModel):
  """K-subspace model where low-dim coefficients are computed in closed
  form. Adapted to handle outliers."""
  default_reg_params = {
      'U_frosqr_in': 0.01,
      'U_frosqr_out': 1e-4,
      'z': 0.01,
      'e': 0.4
  }
  assign_reg_terms = {'U_frosqr_in', 'z', 'e'}

  def __init__(self, k, d, D, affine=False, replicates=5, reg_params={},
        serial_eval={}, reset_patience=100, reset_try_tol=0.01,
        reset_max_steps=50, reset_accept_tol=1e-3, reset_sigma=0.0,
        reset_cache_size=500, temp_scheduler=None,
        encode_max_iter=20, encode_tol=1e-3, scale_grad_freq=100):

    super().__init__(k, d, D,
        affine=affine, replicates=replicates, reg_params=reg_params,
        serial_eval=serial_eval, reset_patience=reset_patience,
        reset_try_tol=reset_try_tol, reset_max_steps=reset_max_steps,
        reset_accept_tol=reset_accept_tol, reset_sigma=reset_sigma,
        reset_cache_size=reset_cache_size, temp_scheduler=temp_scheduler,
        encode_max_iter=encode_max_iter, encode_tol=encode_tol,
        scale_grad_freq=scale_grad_freq)

    self.cache_e = True
    return

  def _prox_e(self, w):
    """Compute proximal operator of ||.||_2."""
    # w shape (r, k, batch_size, D)
    wnorm = w.pow(2).sum(dim=3, keepdim=True).sqrt()
    beta = (wnorm - self.reg_params['e']).clamp_(min=0).div_(wnorm)
    e = beta * w
    return e

  def reg_e(self):
    """Compute regularizer wrt e."""
    return torch.norm(self.e, dim=3).permute(2, 0, 1)


class TempScheduler(object):
  def __init__(self, init_temp=1.0, replicates=None):
    self.init_temp = init_temp
    self.replicates = self.r = replicates
    self.reset()
    return

  def reset(self):
    self.steps = 0 if self.r is None else np.zeros((self.r,), dtype=np.int64)
    return

  def step(self, ridx=None):
    if ridx is None:
      self.steps += 1
    else:
      self.steps[ridx] += 1
    temp = self.get_temp()
    if ridx is not None:
      temp = temp[ridx]
    return temp

  def get_temp(self):
    raise NotImplementedError


class GeoTempScheduler(TempScheduler):
  def __init__(self, init_temp=1.0, replicates=None, patience=100, gamma=0.5):
    super().__init__(init_temp, replicates)
    self.patience = patience
    self.gamma = gamma
    return

  def get_temp(self):
    return self.init_temp * self.gamma ** (self.steps // self.patience)


class FastTempScheduler(TempScheduler):
  def __init__(self, init_temp=1.0, replicates=None):
    super().__init__(init_temp, replicates)
    return

  def get_temp(self):
    return self.init_temp / self.steps


class BoltzTempScheduler(TempScheduler):
  def __init__(self, init_temp=1.0, replicates=None):
    super().__init__(init_temp, replicates)
    return

  def get_temp(self):
    return self.init_temp / np.log(self.steps+1)
