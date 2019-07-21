from __future__ import print_function
from __future__ import division

import numpy as np
import torch
import torch.nn as nn

from . import utils as ut
from .core_reset import reset_replicate

EPS = 1e-8
EMA_DECAY = 0.99
RESET_NCOL = 8
RESET_CPU = True
EVAL_RANK_CPU = True


class KSubspaceBaseModel(nn.Module):
  """Base K-subspace class."""
  default_reg_params = dict()
  assign_reg_terms = set()

  def __init__(self, k, d, D, affine=False, replicates=5, reg_params={},
        reset_patience=100, reset_try_tol=0.01, reset_cand_metric='obj_decr',
        reset_max_steps=50, reset_accept_tol=1e-3, reset_cache_size=500,
        temp_scheduler=None):

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
    if reset_cand_metric not in {'obj_decr', 'value'}:
      raise ValueError("Invalid reset_cand_metric parameter {}".format(
          reset_cand_metric))

    super().__init__()
    self.k = k  # number of groups
    self.d = d  # dimension of manifold
    self.D = D  # number of data points
    self.affine = affine
    self.replicates = self.r = replicates
    self.reg_params = reg_params

    self.reset_patience = reset_patience
    # no threshold on objective plateau
    if reset_try_tol <= 0:
      reset_try_tol = None
    self.reset_try_tol = reset_try_tol
    self.reset_cand_metric = reset_cand_metric
    self.reset_max_steps = reset_max_steps
    self.reset_accept_tol = reset_accept_tol
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
    self._z_frosqr = None
    # subspace coefficients, shape (r, k, batch_size, d)
    # stored only if cache_z is True, since can be a large memory burden
    self._z = None
    self.cache_z = False

    self.Us = nn.Parameter(torch.Tensor(self.r, k, D, d))
    if affine:
      self.bs = nn.Parameter(torch.Tensor(self.r, k, D))
    else:
      self.register_parameter('bs', None)

    self.register_buffer('c_mean', torch.ones(self.r, k).mul_(np.nan))
    self.register_buffer('value', torch.ones(self.r, k).mul_(np.nan))
    self.register_buffer('obj_ema', torch.ones(self.r).mul_(np.nan))
    self.register_buffer('best_obj', torch.ones(self.r).mul_(np.nan))
    # 1st row counts since last reset, 2nd since last successful reset
    self.register_buffer('num_bad_steps', torch.zeros(2, self.r))
    if reset_try_tol is None:
      # need to jitter patience when try tol is none to avoid many simultaneous
      # resets.
      init_jitter = torch.randint(-reset_patience//2, reset_patience//2 + 1,
          (self.r,), dtype=torch.float32)
      init_cooldown_counter = torch.zeros(self.r)
    else:
      init_jitter = torch.zeros(self.r)
      init_cooldown_counter = torch.ones(self.r).mul_(reset_patience//2)
    self.register_buffer('_jitter', init_jitter)
    self.register_buffer('_cooldown_counter', init_cooldown_counter)

    self.register_buffer('_cache_assign_obj',
        torch.ones(reset_cache_size, self.r, k).mul_(np.nan))
    self._cache_assign_obj_head = 0
    return

  def reset_parameters(self):
    """Initialize Us, bs with entries drawn from normal with std
    0.1/sqrt(D)."""
    std = .1 / np.sqrt(self.D)
    self.Us.data.normal_(0., std)
    if self.affine:
      self.bs.data.zero_()
    return

  def forward(self, x):
    """Compute union of subspace embedding of x.

    Inputs:
      x: data, shape (batch_size, D)

    Returns:
      x_: reconstruction, shape (r, batch_size, D)
    """
    x_ = self.embed(x)
    loss = self.loss(x, x_)
    # reg_in shape (batch_size, r, k) or (r, k)
    # reg_out shape (r, k)
    reg_in, reg_out = self.reg()
    # update assignment c, shape (batch_size, r, k)
    assign_c = self.set_assign(loss.data + reg_in.data)
    # select among k subspace reconstructions, (r, batch_size, D)
    x_ = x_.mul(assign_c.permute(1, 2, 0).unsqueeze(3)).sum(dim=1)
    self._update_forward_cache(assign_c, loss, reg_in, reg_out)
    return x_

  def embed(self, x):
    """Compute embedding of x wrt each subspace

    Inputs:
      x: data, shape (batch_size, D)

    Returns:
      x_: reconstruction, shape (r, k, batch_size, D)
    """
    z = self.encode(x)
    x_ = self.decode(z)
    self._update_embed_cache(z)
    return x_

  def decode(self, z):
    """Embed low-dim code z into ambient space.

    Inputs:
      z: latent code, shape (r, k, batch_size, d)

    Returns:
      x_: reconstruction, shape (r, k, batch_size, D)
    """
    # z shape (r, k, batch_size, d)
    # Us shape (r, k, D, d)
    # bs shape (r, k, D)
    assert(z.dim() == 4 and z.shape[:2] == self.Us.shape[:2])

    # x_ = U z + b
    # shape (r, k, batch_size, D)
    x_ = torch.matmul(z, self.Us.transpose(2, 3))
    if self.affine:
      x_ = x_.add(self.bs.unsqueeze(2))
    return x_

  def _update_embed_cache(self, z):
    """Cache values computed during embedding."""
    self._z_frosqr = z.data.pow(2).sum(dim=-1)
    self._z = z.data if self.cache_z else None
    return

  def loss(self, x, x_):
    """Evaluate l2 squared reconstruction loss between x, x_.

    Inputs:
      x: data, shape (batch_size, D)
      x_: reconstructions for each replicate and cluster, shape
        (r, k, batch_size, D)

    Returns:
      loss: l2 squared loss, shape (batch_size, r, k).
    """
    assert(x_.dim() == 4 and x_.shape[:2] == (self.r, self.k) and
        x.shape == x_.shape[-2:])

    # (r, k, batch_size)
    loss = torch.sum((x_ - x)**2, dim=-1).mul(0.5)
    # (batch_size, r, k)
    loss = loss.permute(2, 0, 1)
    return loss

  def _parse_reg(self, reg):
    """Split reg into assign and outside terms."""
    reg_in = torch.zeros(self.r, self.k, device=self.Us.device)
    reg_out = torch.zeros(self.r, self.k, device=self.Us.device)
    for key, val in reg.items():
      if key in self.assign_reg_terms:
        reg_in = reg_in + val
      else:
        reg_out = reg_out + val
    return reg_in, reg_out

  def set_assign(self, assign_obj):
    """Compute cluster assignment.

    Inputs:
      assign_obj: shape (batch_size, r, k)

    Returns:
      assign_c: assignment indicator matrix, (batch_size, r, k)
    """
    (self.groups, _, self.c, batch_c_mean,
        batch_value) = ut.assign_and_value(assign_obj)

    nan_mask = torch.isnan(self.c_mean)
    self.c_mean[nan_mask] = batch_c_mean[nan_mask]
    self.value[nan_mask] = batch_value[nan_mask]

    self.c_mean.mul_(EMA_DECAY).add_(1-EMA_DECAY, batch_c_mean)
    self.value.mul_(EMA_DECAY).add_(1-EMA_DECAY, batch_value)

    self.groups = self.groups.cpu()
    self._update_assign_obj_cache(assign_obj)
    return self.c

  def _update_assign_obj_cache(self, assign_obj):
    """Add batch assign obj to a recent cache. Used when resetting unused
    clusters."""
    batch_size = assign_obj.shape[0]
    if self.reset_cache_size < batch_size:
      Idx = torch.randperm(batch_size)[:self.reset_cache_size]
      self._cache_assign_obj.copy_(assign_obj[Idx, :])
      self._cache_assign_obj_head = 0
    elif self.reset_cache_size == batch_size:
      self._cache_assign_obj.copy_(assign_obj)
      self._cache_assign_obj_head = 0
    else:
      Idx = (torch.arange(self._cache_assign_obj_head,
          self._cache_assign_obj_head + batch_size) % self.reset_cache_size)
      self._cache_assign_obj[Idx, :] = assign_obj
      self._cache_assign_obj_head = ((Idx[-1] + 1) %
          self.reset_cache_size).item()
    return

  def _update_forward_cache(self, assign_c, loss, reg_in, reg_out):
    """Cache loss, reg, obj values computed during forward."""
    self._reg_out_per_cluster = reg_out.data

    # reduce and compute objective, shape (r,)
    loss = loss.mul(assign_c).sum(dim=2).mean(dim=0)
    reg_in = reg_in.mul(assign_c).sum(dim=2).mean(dim=0)
    reg_out = reg_out.sum(dim=1)
    obj = loss + reg_in + reg_out
    obj_mean = obj.mean()

    # per replicate loss, reg shape (r,)
    self._loss = loss.data
    self._reg_in = reg_in.data
    self._reg_out = reg_out.data
    self._obj = obj.data

    # scalar average loss with grad tracking
    self._obj_mean = obj_mean

    # ema objective, shape (r,)
    nan_mask = torch.isnan(self.obj_ema)
    self.obj_ema[nan_mask] = obj.data[nan_mask]
    self.obj_ema.mul_(EMA_DECAY).add_(1-EMA_DECAY, obj.data)
    return

  def objective(self, x=None):
    """Evaluate objective function.

    Inputs:
      x: data, shape (batch_size, D) (default: None)

    Returns:
      obj_mean: average objective across replicates
      obj, loss, reg_in, reg_out: metrics per replicate, shape (r,)
    """
    if x is not None:
      self(x)
    return self._obj_mean, self._obj, self._loss, self._reg_in, self._reg_out

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

  def core_reset(self):
    """Reset replicates whose progress has slowed by cooperative
    re-initialization (CoRe).

    Returns:
      resets: log of resets, shape (n_resets, 8). Columns are (reset rep idx,
        reset cluster idx, candidate rep idx, candidate cluster idx, reset
        success, obj decrease, cumulative obj decrease, temperature).
    """
    empty_output = np.zeros((0, RESET_NCOL), dtype=object)

    # reset not possible if only one replicate
    # NOTE: should probably give warning
    if self.r == 1:
      return empty_output

    # identify clusters to reset based on objective decrease
    reset_rids = self._reset_criterion()
    if reset_rids.shape[0] == 0:
      return empty_output

    # check that cache is full and calculate current cache objective
    cache_not_full = torch.any(torch.isnan(self._cache_assign_obj))
    if cache_not_full:
      return empty_output

    assign_obj = self._cache_assign_obj
    reg_out = self._reg_out_per_cluster
    if RESET_CPU:
      assign_obj, reg_out = assign_obj.cpu(), reg_out.cpu()

    resets = []
    for ridx in reset_rids:
      ridx = ridx.item()
      temp = self.temp_scheduler.step(ridx)
      success, rep_resets, rep_assign_obj, rep_reg_out = reset_replicate(
          ridx, assign_obj, reg_out, temp, max_steps=self.reset_max_steps,
          accept_tol=self.reset_accept_tol, cand_metric=self.reset_cand_metric)
      if success:
        rep_reset_ids = rep_resets[rep_resets[:, 4] == 1, 1:4].astype(np.int64)
        self._post_reset_updates(ridx, rep_reset_ids, rep_assign_obj,
            rep_reg_out)
      self.num_bad_steps[0, ridx] = 0
      resets.append(rep_resets)

    resets = np.concatenate(resets, axis=0)
    return resets

  def _reset_criterion(self):
    """Identify replicates to reset based on relative objective decrease."""
    nan_mask = torch.isnan(self.best_obj)
    self.best_obj[nan_mask] = self.obj_ema[nan_mask]

    if self.reset_try_tol is not None:
      cooldown_mask = self._cooldown_counter > 0
      self._cooldown_counter[cooldown_mask] -= 1.0

      better_or_cooldown = torch.max(cooldown_mask,
          (self.obj_ema < (1.0 - self.reset_try_tol)*self.best_obj))
      self.best_obj[better_or_cooldown] = self.obj_ema[better_or_cooldown]
      self.num_bad_steps[:, better_or_cooldown] = 0.0
      self.num_bad_steps[:, better_or_cooldown == 0] += 1.0
    else:
      # no threshold on objective plateau
      better_mask = self.obj_ema < self.best_obj
      self.best_obj[better_mask] = self.obj_ema[better_mask]
      self.num_bad_steps += 1.0

    reset_mask = (self.num_bad_steps[0, :] > (self.reset_patience +
        self._jitter))
    reset_rids = reset_mask.nonzero().view(-1)
    return reset_rids

  def _post_reset_updates(self, ridx, reset_ids, rep_assign_obj, rep_reg_out):
    """Duplicate bases from (cand_rIdx, cand_cIdx) to (rIdx, cIdx), and other
    re-initializations."""
    # reset_ids = cIdx, cand_rIdx, cand_cIdx
    uniqIdx = ut.unique_resets(reset_ids[:, 0])
    reset_ids = reset_ids[uniqIdx, :]
    cIdx, cand_rIdx, cand_cIdx = [reset_ids[:, ii] for ii in range(3)]

    self.Us.data[ridx, cIdx, :] = self.Us.data[cand_rIdx, cand_cIdx, :]
    if self.affine:
      self.bs.data[ridx, cIdx, :] = self.bs.data[cand_rIdx, cand_cIdx, :]

    self._cache_assign_obj[:, ridx, :] = rep_assign_obj
    self._reg_out_per_cluster[ridx, :] = rep_reg_out

    self.c_mean[ridx, :] = np.nan
    self.value[ridx, :] = np.nan
    self.obj_ema[ridx] = self.best_obj[ridx] = np.nan
    self.num_bad_steps[1, ridx] = 0
    if self.reset_try_tol is None:
      self._jitter[ridx].random_(-self.reset_patience//2,
          self.reset_patience//2 + 1)
    else:
      self._cooldown_counter[ridx] = self.reset_patience // 2
    return

  def zero(self):
    """Zero out near zero bases.

    Numerically near zero values slows matmul performance up to 6x.
    """
    Unorms = self.Us.data.pow(2).sum(dim=(2, 3)).sqrt()
    self.Us.data[(Unorms < EPS*max(1, Unorms.max())), :, :] = 0.0
    return

  def epoch_init(self):
    """Initialization at the start of an epoch."""
    self.zero()
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
        reset_patience=100, reset_try_tol=0.01, reset_cand_metric='obj_decr',
        reset_max_steps=50, reset_accept_tol=1e-3, reset_cache_size=500,
        temp_scheduler=None, scale_grad_mode=None, scale_grad_update_freq=20,
        init='random', initX=None, initk=None):

    if not (scale_grad_mode is None or scale_grad_mode == 'none'):
      if affine:
        raise ValueError("gradient scaling not supported in affine setting.")
      if scale_grad_mode not in {'lip', 'newton'}:
        raise ValueError("Invalid scale grad mode {}.".format(scale_grad_mode))

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
        reset_patience=reset_patience, reset_try_tol=reset_try_tol,
        reset_cand_metric=reset_cand_metric, reset_max_steps=reset_max_steps,
        reset_accept_tol=reset_accept_tol, reset_cache_size=reset_cache_size,
        temp_scheduler=temp_scheduler)

    if scale_grad_mode == 'none':
      scale_grad_mode = None
    self.scale_grad_mode = scale_grad_mode
    self.scale_grad_update_freq = scale_grad_update_freq
    if scale_grad_mode is not None:
      # ultimate shape (2, r, k, d, d)
      self.Hess = None
      self.cache_z = True
      self.Us.register_hook(lambda UGrad: self._scale_grad(UGrad))

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

  def encode(self, x):
    """Compute subspace coefficients for x in closed form, by computing
    batched solution to normal equations.

      min_z 1/2 || x - (Uz + b) ||_2^2 + \lambda/2 ||z||_2^2
      (U^T U + \lambda I) z* = U^T (x - b)

    Inputs:
      x: shape (batch_size, D)

    Returns:
      z: latent low-dimensional codes (r, k, batch_size, d)
    """
    # Us shape (r, k, D, d)
    # bs shape (r, k, D)
    batch_size = x.size(0)

    # (1, 1, D, batch_size)
    x = x.data.t().view(1, 1, self.D, batch_size)
    if self.affine:
      # (r, k, D, batch_size)
      x = x.sub(self.bs.data.unsqueeze(3))
    # (r, k, d, batch_size)
    z = ut.batch_ridge(x, self.Us.data, lamb=self.reg_params['z'])

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
      regs['z'] = self._z_frosqr.permute(2, 0, 1).mul(self.reg_params['z']*0.5)
    return self._parse_reg(regs)

  def _scale_grad(self, Grad, update_accum=True):
    """Divide gradients by an estimate of the Lipschitz constant, per
    replicate."""
    update_grad_scale = (self.scale_grad_steps %
        self.scale_grad_update_freq == 0)

    if update_accum:
      device = self._z.device
      batch_size = self.c.shape[0]
      z_zero = self.c.permute(1, 2, 0).unsqueeze(3) * self._z
      Hess = torch.matmul(z_zero.transpose(2, 3), z_zero)

      # (r, k)
      batch_cluster_sizes = self.c.sum(dim=0)
      # 1e-3 added to avoid singularity (huge steps)
      lamb = (self.reg_params['U_frosqr_in'] * batch_cluster_sizes).add_(
          (self.reg_params['U_frosqr_out'] + 1e-3) * batch_size)
      # (r, k, d, d)
      Id = torch.eye(self.d, device=device).mul(
          lamb.view(self.r, self.k, 1, 1))
      Hess = Hess.add_(Id)
      self.Hess = self.Hess.add_(Hess)
      self.HessN += batch_size
      self.scale_grad_steps += 1

    if update_grad_scale:
      HessN = self.HessN[0, :].view(self.r, 1, 1, 1)
      Hess = (self.Hess[0, :] / HessN).cpu().numpy()

    if self.scale_grad_mode == 'lip':
      if update_grad_scale:
        # spectral norm for each rep, cluster
        # (r, k)
        Lip = np.linalg.norm(Hess, ord=2, axis=(2, 3))
        # (r,)
        self.Lip = torch.from_numpy(Lip).to(device).max(dim=1)[0]
      Grad_scale = Grad.div(self.Lip.view(
          self.Lip.shape + (Grad.dim() - self.Lip.dim()) * (1,)))
    else:
      # newton
      if update_grad_scale:
        # pseudo-inverse of hessian for each rep, cluster
        # (r, k, d, d)
        Hessinv = np.linalg.pinv(Hess)
        self.Hessinv = torch.from_numpy(Hessinv).to(device)
      Grad_scale = torch.matmul(Grad, self.Hessinv)
    return Grad_scale

  def epoch_init(self):
    """Initialization at the start of an epoch."""
    super().epoch_init()
    if self.scale_grad_mode is not None:
      if self.Hess is None:
        self.Hess = torch.eye(self.d, device=self.Us.device).repeat(
            2, self.r, self.k, 1, 1)
        self.HessN = torch.zeros((2, self.r), device=self.Us.device)
      else:
        self.Hess = self._shift_and_zero(self.Hess)
        self.HessN = self._shift_and_zero(self.HessN)
      self.scale_grad_steps = 0
    return

  def _shift_and_zero(self, buf):
    """Shift 1 -> 0, zero 1."""
    buf[0, :] = buf[1, :]
    buf[1, :] = 0.0
    return buf

  def _post_reset_updates(self, ridx, reset_ids, rep_assign_obj, rep_reg_out):
    """Duplicate bases from (cand_rIdx, cand_cIdx) to (rIdx, cIdx), and other
    re-initializations."""
    super()._post_reset_updates(ridx, reset_ids, rep_assign_obj, rep_reg_out)
    if self.scale_grad_mode is not None:
      # re-initialize hessian estimate with identity.
      # NOTE: would using the candidate's hessian be a better choice? I think
      # probably not, since there is some dependence on the "sibling clusters",
      # through the assignment.
      self.Hess[:, ridx, :] = torch.eye(self.d, device=self.Us.device)
      self.HessN[:, ridx] = 0.0
      # be sure to update grad scale on next iter
      self.scale_grad_steps = 0
    return


class KSubspaceProjModel(KSubspaceBaseModel):
  """K-subspace model where low-dim coefficients are computed by a projection
  matrix."""
  default_reg_params = {
      'U_frosqr_in': 0.01,
      'U_frosqr_out': 1e-4
  }
  assign_reg_terms = {'U_frosqr_in'}

  def __init__(self, k, d, D, affine=False, replicates=5, reg_params={},
        reset_patience=100, reset_try_tol=0.01,
        reset_cand_metric='obj_decr', reset_max_steps=50,
        reset_accept_tol=1e-3, reset_cache_size=500, temp_scheduler=None):

    super().__init__(k, d, D,
        affine=affine, replicates=replicates, reg_params=reg_params,
        reset_patience=reset_patience, reset_try_tol=reset_try_tol,
        reset_cand_metric=reset_cand_metric, reset_max_steps=reset_max_steps,
        reset_accept_tol=reset_accept_tol, reset_cache_size=reset_cache_size,
        temp_scheduler=temp_scheduler)

    self.reset_parameters()
    return

  def encode(self, x):
    """Project x onto each of k low-dimensional spaces.

    Inputs:
      x: shape (batch_size, D)

    Returns:
      z: latent low-dimensional codes (r, k, batch_size, d)
    """
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
          ('U_frosqr_in', 'U_frosqr_out')]) > 0:
      U_frosqr = torch.sum(self.Us.pow(2), dim=(2, 3))

      if self.reg_params['U_frosqr_in'] > 0:
        regs['U_frosqr_in'] = U_frosqr.mul(
            self.reg_params['U_frosqr_in'])

      if self.reg_params['U_frosqr_out'] > 0:
        regs['U_frosqr_out'] = U_frosqr.mul(
            self.reg_params['U_frosqr_out'])
    return self._parse_reg(regs)


class KSubspaceBatchAltBaseModel(KSubspaceBaseModel):
  default_reg_params = dict()
  assign_reg_terms = dict()

  def __init__(self, k, d, dataset, affine=False, replicates=5, reg_params={},
        reset_patience=2, reset_try_tol=0.01, reset_cand_metric='obj_decr',
        reset_max_steps=50, reset_accept_tol=1e-3, reset_cache_size=500,
        temp_scheduler=None, svd_solver='randomized', **kwargs):

    if svd_solver not in ('randomized', 'svds', 'svd'):
      raise ValueError("Invalid svd solver {}".format(svd_solver))

    # X assumed to be N x D.
    D = dataset.X.shape[1]
    super().__init__(k, d, D,
        affine=affine, replicates=replicates, reg_params=reg_params,
        reset_patience=reset_patience, reset_try_tol=reset_try_tol,
        reset_cand_metric=reset_cand_metric, reset_max_steps=reset_max_steps,
        reset_accept_tol=reset_accept_tol, reset_cache_size=reset_cache_size,
        temp_scheduler=temp_scheduler, **kwargs)

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
    """Evaluate objective function. Note, forward evaluation is always
    computed.

    Returns:
      obj_mean: average objective across replicates
      obj, loss, reg_in, reg_out: metrics per replicate, shape (r,)
    """
    self()
    return self._obj_mean, self._obj, self._loss, self._reg_in, self._reg_out

  def forward(self):
    """Compute union of subspace embedding of X.

    Inputs:
      x: data, shape (batch_size, D) (default: self.X)

    Returns:
      x_: reconstruction, shape (r, N, D)
    """
    # NOTE: will this only call KSubspaceBaseModel forward as intended in
    # subclasses?
    return super().forward(self.X)

  def set_assign(self, assign_obj):
    """Compute cluster assignment.

    Inputs:
      assign_obj: shape (N, r, k)

    Returns:
      assign_c: assignment indicator matrix, (batch_size, r, k)
    """
    groups_prev = self.groups
    (self.groups, _, self.c, self.c_mean,
        self.value) = ut.assign_and_value(assign_obj)

    self.groups = self.groups.cpu()
    self.updates = ((self.groups != groups_prev).sum()
        if groups_prev is not None else self.N)
    self._update_assign_obj_cache(assign_obj)
    return self.c


class KSubspaceBatchAltProjModel(KSubspaceBatchAltBaseModel,
      KSubspaceProjModel):
  default_reg_params = {
      'U_frosqr_in': 0.01,
      'U_frosqr_out': 1e-4
  }
  assign_reg_terms = {'U_frosqr_in'}

  def __init__(self, k, d, dataset, affine=False, replicates=5, reg_params={},
        reset_patience=2, reset_try_tol=0.01, reset_cand_metric='obj_decr',
        reset_max_steps=50, reset_accept_tol=1e-3, reset_cache_size=500,
        temp_scheduler=None, svd_solver='randomized'):

    super().__init__(k, d, dataset,
        affine=affine, replicates=replicates, reg_params=reg_params,
        reset_patience=reset_patience, reset_try_tol=reset_try_tol,
        reset_cand_metric=reset_cand_metric, reset_max_steps=reset_max_steps,
        reset_accept_tol=reset_accept_tol, reset_cache_size=reset_cache_size,
        temp_scheduler=temp_scheduler, svd_solver=svd_solver)
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
        reset_patience=2, reset_try_tol=0.01, reset_cand_metric='obj_decr',
        reset_max_steps=50, reset_accept_tol=1e-3, reset_cache_size=500,
        temp_scheduler=None, init='random', initX=None, initk=None,
        svd_solver='randomized'):

    mf_kwargs = {'scale_grad_mode': None, 'init': init, 'initX': initX,
        'initk': initk}
    super().__init__(k, d, dataset,
        affine=affine, replicates=replicates, reg_params=reg_params,
        reset_patience=reset_patience, reset_try_tol=reset_try_tol,
        reset_cand_metric=reset_cand_metric, reset_max_steps=reset_max_steps,
        reset_accept_tol=reset_accept_tol, reset_cache_size=reset_cache_size,
        temp_scheduler=temp_scheduler, svd_solver=svd_solver, **mf_kwargs)

    # bases initialized in KSubspaceMFModel
    # ensure columns are orthogonal
    for ii in range(self.r):
      for jj in range(self.k):
        P, S, _ = torch.svd(self.Us.data[ii, jj, :])
        self.Us.data[ii, jj, :] = P.mul_(S)
    return

  def encode(self, x):
    """Compute subspace coefficients for x in closed form, by computing
    batched solution to normal equations. Use fact that colums of U are
    orthogonal (althought not necessarily unit norm).

      min_z 1/2 || x - (Uz + b) ||_2^2 + \lambda/2 ||z||_2^2
      (U^T U + \lambda I) z* = U^T (x - b)

    Inputs:
      x: shape (batch_size, D)

    Returns:
      z: latent low-dimensional codes (r, k, batch_size, d)
    """
    batch_size = x.size(0)

    if self.affine:
      # (r, k, batch_size, D)
      x = x.sub(self.bs.data.unsqueeze(2))
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

  def __init__(self, k, dataset, replicates=5, reg_params={},
        reset_patience=2, reset_try_tol=0.01, reset_cand_metric='obj_decr',
        reset_max_steps=50, reset_accept_tol=1e-3, reset_cache_size=500,
        temp_scheduler=None, init='random', kpp_n_trials=None):

    if init not in {'pfi', 'random'}:
      raise ValueError("Invalid init parameter {}".format(init))

    d = 1  # Us not used, but retained for consistency/out of laziness
    affine = True

    super().__init__(k, d, dataset,
        affine=affine, replicates=replicates, reg_params=reg_params,
        reset_patience=reset_patience, reset_try_tol=reset_try_tol,
        reset_cand_metric=reset_cand_metric, reset_max_steps=reset_max_steps,
        reset_accept_tol=reset_accept_tol, reset_cache_size=reset_cache_size,
        temp_scheduler=temp_scheduler)

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
    if self.init == 'pfi':
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

  def forward(self, compute_x_=False):
    """Evaluate loss, compute assignment, and optionally compute k-means
    embedding.

    Inputs:
      compute_x_: Compute k-means embedding.

    Returns:
      x_: reconstruction, shape (r, N, D) if compute_x_, otherwise None.
    """
    loss = self.loss()
    # reg_in shape (batch_size, r, k) or (r, k)
    # reg_out shape (r, k)
    reg_in, reg_out = self.reg()
    # update assignment c, shape (batch_size, r, k)
    assign_c = self.set_assign(loss.data + reg_in.data)
    self._update_forward_cache(assign_c, loss, reg_in, reg_out)
    if compute_x_:
      # bs shape (r, k, D)
      # groups shape (N, r)
      rIdx = torch.arange(self.r).view(-1, 1)
      # shape (r, N, D)
      x_ = self.bs[rIdx, self.groups.t(), :]
    else:
      x_ = None
    return x_

  def loss(self, bs=None):
    """Evaluate reconstruction loss

    Inputs:
      bs: k means (default: None).

    Returns:
      loss: shape (batch_size, r, k)
    """
    if bs is None:
      bs = self.bs
    bsqrnorms = bs.pow(2).sum(dim=2).mul(0.5)

    # X (batch_size, D)
    # b (r, k, D)
    # XTb (batch_size, r, k)
    XTb = torch.matmul(bs, self.XT).permute(2, 0, 1)
    # (batch_size, r, k)
    loss = self.Xsqrnorms.view(-1, 1, 1).sub(XTb).add(bsqrnorms.unsqueeze(0))
    return loss.clamp_(min=0.0)

  def embed(self, x):
    raise NotImplementedError("embed not implemented.")

  def encode(self, x):
    raise NotImplementedError("encode not implemented.")

  def decode(self, z):
    raise NotImplementedError("decode not implemented.")

  def reg(self):
    """Evaluate regularization."""
    regs = dict()
    if self.reg_params['b_frosqr_out'] > 0:
      regs['b_frosqr_out'] = torch.sum(self.bs.pow(2), dim=2).mul(
          self.reg_params['b_frosqr_out']*0.5)
    return self._parse_reg(regs)

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
        reset_patience=100, reset_try_tol=0.01, reset_cand_metric='obj_decr',
        reset_max_steps=50, reset_accept_tol=1e-3, reset_cache_size=500,
        temp_scheduler=None, scale_grad_mode=None, scale_grad_update_freq=20,
        sparse_encode=True, sparse_decode=False, norm_comp_error=True):

    super().__init__(k, d, D,
        affine=affine, replicates=replicates, reg_params=reg_params,
        reset_patience=reset_patience, reset_try_tol=reset_try_tol,
        reset_cand_metric=reset_cand_metric, reset_max_steps=reset_max_steps,
        reset_accept_tol=reset_accept_tol, reset_cache_size=reset_cache_size,
        temp_scheduler=temp_scheduler, scale_grad_mode=scale_grad_mode,
        scale_grad_update_freq=scale_grad_update_freq)

    self.sparse_encode = sparse_encode
    self.sparse_decode = sparse_decode
    self.norm_comp_error = norm_comp_error
    self.repUs = None
    self.repbs = None

    self.cache_z = True
    self._z = None
    self.cache_x_ = not sparse_decode
    self._x_ = None
    return

  def loss(self, x, x_):
    """Evaluate reconstruction loss

    Inputs:
      x: MissingDataBatch instance, dense shape (batch_size, D).
      x_: reconstruction, shape (r, k, batch_size, M) where M = x.pad_nnz if
        sparse_decode, or M = D otherwise.

    Returns:
      loss: shape (batch_size, r, k)
    """
    # loss (r, k, batch_size)
    if self.sparse_decode:
      # x_ assumed (r, k, batch_size, pad_nnz)
      loss = torch.sum(((x_ * x.omega_float) - x.values)**2, dim=-1).mul(0.5)
    else:
      # x_ assumed (r, k, batch_size, D)
      loss = torch.sum(((x_ * x.omega_float_dense) - x.values_dense)**2,
          dim=-1).mul(0.5)
    # (batch_size, r, k)
    loss = loss.permute(2, 0, 1)
    return loss

  def embed(self, x):
    """Compute embedding of x wrt each subspace.

    Inputs:
      x: MissingDataBatch instance, dense shape (batch_size, D).

    Returns:
      x_: reconstruction, either shape (r, k, batch_size, M) where M =
        x.pad_nnz if sparse_decode, or M = D otherwise.
    """
    z = self.encode(x)
    x_ = self.decode(z, x, compute_repUs=False)
    self._update_embed_cache(z, x_)
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

  def _update_embed_cache(self, z, x_):
    """Cache values computed during embedding."""
    self._z_frosqr = z.data.pow(2).sum(dim=-1)
    self._z = z.data if self.cache_z else None
    self._x_ = x_.data if self.cache_x_ else None
    return

  def eval_comp_error(self, x0):
    """Evaluate completion error over observed entries in x0.

    Inputs:
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
        # (r, k, batch_size, pad_nnz)
        x0_ = self.decode(self._z, x0)
      else:
        # (r, k, batch_size, D)
        x0_ = self._x_

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
    if ridx is None or self.r is None:
      self.steps += 1
    else:
      self.steps[ridx] += 1
    temp = self.get_temp()
    if ridx is not None and self.r is not None:
      temp = temp[ridx]
    return temp

  def get_temp(self):
    raise NotImplementedError


class ConstantTempScheduler(TempScheduler):
  def __init__(self, init_temp=1.0, replicates=None):
    super().__init__(init_temp, replicates)
    self.temp = (init_temp if self.r is None else
        (init_temp * np.ones(self.r)))
    return

  def get_temp(self):
    return self.temp


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
