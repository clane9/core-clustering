from __future__ import print_function
from __future__ import division

import numpy as np
import torch
import torch.nn as nn

from . import utils as ut
from .core_reset import reset_replicate, RESET_NCOL

EPS = 1e-8
EMA_DECAY = 0.99
RESET_CPU = True
EVAL_RANK_CPU = True


class KSubspaceBaseModel(nn.Module):
  """Base K-subspace class."""
  default_reg_params = dict()
  assign_reg_terms = set()

  def __init__(self, k=10, d=10, D=100, affine=False, replicates=5,
        reg_params={}, reset_patience=100, reset_try_tol=0.01,
        reset_max_steps=50, reset_accept_tol=1e-3, reset_cache_size=500,
        **kwargs):

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
    if (min(reset_patience, reset_max_steps, reset_accept_tol,
          reset_cache_size) <= 0):
      raise ValueError(("Reset patience, max steps, accept tol, "
          "cache size must all be > 0"))

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
    self.reset_max_steps = reset_max_steps
    self.reset_accept_tol = reset_accept_tol
    self.reset_cache_size = reset_cache_size

    # group assignment, ultimate shape (batch_size, r, k)
    self.c = None
    self.groups = None
    # subspace coefficients, shape (r, k, batch_size, d)
    # stored only if cache_z is True, since can be a large memory burden
    self._z, self.cache_z = None, False
    # similarly reconstructions, shape (r, k, batch_size, D)
    self._x_, self.cache_x_ = None, False

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

    # initialize randomnly, possibly will be overridden later.
    self.reset_parameters()
    return

  def reset_parameters(self):
    """Initialize Us with entries drawn from normal with std 0.1/sqrt(D), bs
    with zeros."""
    std = .1 / np.sqrt(self.D)
    self.Us.data.normal_(0., std)
    if self.affine:
      self.bs.data.zero_()
    return

  def pfi_init(self, X, pfi_n_cands=None, fit_kwargs=None):
    """Initialize by probabilistic farthest insertion.

    Args:
      X: Data to use for initialization, shape (init_size, D).
      pfi_n_cands: Number of candidate insertion centers to generate on each
        iteration (default: 2 log k).
      fit_kwargs: keyword arguments to pass through to _pfi_fit_cluster
        (default: None).
    """
    if pfi_n_cands is None:
      pfi_n_cands = int(np.ceil(2 * np.log(self.k)))
    if fit_kwargs is None:
      fit_kwargs = dict()
    init_size = X.shape[0]
    rIdx = torch.arange(self.r)

    # initialize large tensors used in computing alternate objectives.
    alt_assign_obj = torch.zeros(init_size, self.r, pfi_n_cands, 2,
        device=X.device)
    alt_min_assign_obj = torch.zeros(init_size, self.r, pfi_n_cands,
        device=X.device)
    alt_min_assign_Idx = torch.zeros(init_size, self.r, pfi_n_cands,
        dtype=torch.int64, device=X.device)

    self.Us.data.zero_()
    if self.affine:
      self.bs.data.zero_()

    # choose first basis randomly
    Idx = torch.randint(0, init_size, (self.r,), dtype=torch.int64)
    U0, b0 = self._pfi_fit_cluster(X[Idx, :], X, **fit_kwargs)
    self.Us.data[:, 0, :] = U0
    if self.affine:
      self.bs.data[:, 0, :] = b0

    with torch.no_grad():
      for cidx in range(1, self.k):
        Us = self.Us[:, :cidx, :]
        bs = self.bs[:, :cidx, :] if self.affine else None

        # compute assign objective based on current clusters
        self.forward(X, Us=Us, bs=bs, update_cache=False)
        # (init_size, r, cidx)
        assign_obj = self._loss + self._reg_in
        # (init_size, r)
        min_assign_obj = assign_obj.min(dim=2)[0]

        # sample poorly represented data points to serve as candidate cluster
        # centers.
        # (r*pfi_n_cands,)
        Idx = torch.multinomial(min_assign_obj.t(), pfi_n_cands).view(-1)
        Uj, bj = self._pfi_fit_cluster(X[Idx, :], X, **fit_kwargs)
        Uj = Uj.view(self.r, pfi_n_cands, self.D, self.d)
        if self.affine:
          bj = bj.view(self.r, pfi_n_cands, self.D)

        # evaluate assignment objectives for these candidates
        self.forward(X, Us=Uj, bs=bj, update_cache=False)
        # (init_size, r, pfi_n_cands)
        cand_assign_obj = self._loss + self._reg_in

        # compute alternative objectives for each candidate.
        alt_assign_obj[:, :, :, 0] = min_assign_obj.unsqueeze(2)
        alt_assign_obj[:, :, :, 1] = cand_assign_obj
        alt_min_assign_obj, _ = torch.min(alt_assign_obj, dim=3,
            out=(alt_min_assign_obj, alt_min_assign_Idx))
        # (r, pfi_n_cands)
        alt_obj = alt_min_assign_obj.mean(dim=0)
        alt_obj = alt_obj.add_(self._reg_out)

        # insert the candidate cluster that gives the lowest objective.
        insertIdx = alt_obj.argmin(dim=1)
        self.Us.data[:, cidx, :] = Uj[rIdx, insertIdx, :]
        if self.affine:
          self.bs.data[:, cidx, :] = bj[rIdx, insertIdx, :]

    torch.cuda.empty_cache()
    return

  def _pfi_fit_cluster(xcenter, x, **kwargs):
    raise NotImplementedError

  def forward(self, x, Us=None, bs=None, update_cache=True):
    """Compute union of subspace embedding of x.

    Inputs:
      x: data, shape (batch_size, D)
      Us: bases, shape (r, k, D, d) (default: None)
      bs: centers, shape (r, k, D) (default: None)
      update_cache: (default: True)

    Returns:
      x_: reconstruction, shape (r, batch_size, D)
    """
    # (r, k, batch_size, d)
    z = self.encode(x, Us=Us, bs=bs)
    # (r, k, batch_size, D)
    x_ = self.decode(z, Us=Us, bs=bs)

    # (batch_size, r, k)
    loss = self.loss(x, x_)
    # reg_in (batch_size, r, k) or (r, k)
    # reg_out (r, k)
    reg_in, reg_out = self.reg(z, Us=Us, bs=bs)

    assign_obj = loss.data + reg_in.data
    # groups (batch_size, r)
    # assign_c (batch_size, r, k)
    # c_mean, value (r, k)
    groups, _, assign_c, c_mean, value = ut.assign_and_value(assign_obj)

    # reduce based on assignment
    x_ = x_.mul(assign_c.permute(1, 2, 0).unsqueeze(3)).sum(dim=1)

    # cache quantities needed to compute objective later
    self._loss, self._reg_in, self._reg_out = loss, reg_in, reg_out
    self.c = assign_c

    if update_cache:
      self.groups = groups.cpu()
      self._update_forward_cache(z, x_, reg_out, assign_obj, c_mean, value)
    return x_

  def encode(self, x, Us=None, bs=None):
    raise NotImplementedError

  def decode(self, z, Us=None, bs=None):
    """Embed low-dim code z into ambient space.

    Inputs:
      z: latent code, shape (r, k, batch_size, d)
      Us: bases, shape (r, k, D, d) (default: None)
      bs: centers, shape (r, k, D) (default: None)

    Returns:
      x_: reconstruction, shape (r, k, batch_size, D)
    """
    Us = self.Us if Us is None else Us
    bs = self.bs if bs is None else bs

    # z shape (r, k, batch_size, d)
    # Us shape (r, k, D, d)
    # bs shape (r, k, D)
    assert(z.dim() == 4 and z.shape[:2] == Us.shape[:2])

    # x_ = U z + b
    # shape (r, k, batch_size, D)
    x_ = torch.matmul(z, Us.transpose(2, 3))
    if self.affine:
      x_ = x_.add(bs.unsqueeze(2))
    return x_

  def loss(self, x, x_):
    """Evaluate l2 squared reconstruction loss between x, x_.

    Inputs:
      x: data, shape (batch_size, D)
      x_: reconstructions for each replicate and cluster, shape
        (r, k, batch_size, D)

    Returns:
      loss: l2 squared loss, shape (batch_size, r, k).
    """
    # (r, k, batch_size)
    loss = torch.sum((x_ - x)**2, dim=-1).mul(0.5)
    # (batch_size, r, k)
    loss = loss.permute(2, 0, 1)
    return loss

  def reg(self, z=None, Us=None, bs=None):
    """Evaluate subspace regularization."""
    return self._parse_reg(dict())

  def _parse_reg(self, reg, r, k):
    """Split reg into assign and outside terms."""
    reg_in = torch.zeros(r, k, device=self.Us.device)
    reg_out = torch.zeros(r, k, device=self.Us.device)
    for key, val in reg.items():
      if key in self.assign_reg_terms:
        reg_in = reg_in + val
      else:
        reg_out = reg_out + val
    return reg_in, reg_out

  def _update_forward_cache(self, z, x_, reg_out, assign_obj, c_mean, value):
    """Cache loss, reg, obj values computed during forward."""
    self._z = z.data if self.cache_z else None
    self._x_ = x_.data if self.cache_x_ else None
    self._reg_out_per_cluster = reg_out.data

    # cached assignment objectives used in CoRe
    self._update_assign_obj_cache(assign_obj)

    # cluster size, value, shape (r,)
    ut.update_ema_metric(c_mean, self.c_mean, ema_decay=EMA_DECAY,
        inplace=True)
    ut.update_ema_metric(value, self.value, ema_decay=EMA_DECAY,
        inplace=True)
    return

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

  def objective(self, x=None, Us=None, bs=None, update_cache=True):
    """Evaluate objective function.

    Inputs:
      x: data, shape (batch_size, D) (default: None)
      Us: bases, shape (r, k, D, d) (default: None)
      bs: centers, shape (r, k, D) (default: None)
      update_cache: (default: True)

    Returns:
      obj_mean: average objective across replicates
      obj, loss, reg_in, reg_out: metrics per replicate, shape (r,)
    """
    if x is not None:
      self(x, Us=Us, bs=bs, update_cache=update_cache)

    # extract cached quantities (with grad tracking)
    loss, reg_in, reg_out = self._loss, self._reg_in, self._reg_out

    # reduce across batch, cluster based on assignment
    loss = loss.mul(self.c).sum(dim=2).mean(dim=0)
    reg_in = reg_in.mul(self.c).sum(dim=2).mean(dim=0)
    reg_out = reg_out.sum(dim=1)
    obj = loss + reg_in + reg_out
    obj_mean = obj.mean()

    # ema objective, shape (r,)
    if update_cache:
      ut.update_ema_metric(obj.data, self.obj_ema, ema_decay=EMA_DECAY,
          inplace=True)

    self._loss, self._reg_in, self._reg_out = None, None, None
    return obj_mean, obj.data, loss.data, reg_in.data, reg_out.data

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
    re-initialization (CoRe). On each CoRe iteration, we first find the best
    candidate from the set of all rk clusters to add to the current replicate.
    We then find the best of the replicate's clusters to drop. Both decisions
    made based on objective value. The relative objective decrease offered by
    the swap must pass a tolerance to be accepted.

    Returns:
      resets: log of swap updates, shape (n_resets, 7). Columns are (reset rep
        idx, iteration, reset cluster idx, candidate rep idx, candidate cluster
        idx, obj decrease, cumulative obj decrease).
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

    # check that assign obj cache is full for all reps
    cache_not_full = torch.isnan(self._cache_assign_obj[
        self._cache_assign_obj_head, 0, 0])
    if cache_not_full:
      return empty_output

    assign_obj = self._cache_assign_obj
    reg_out = self._reg_out_per_cluster
    if RESET_CPU:
      assign_obj, reg_out = assign_obj.cpu(), reg_out.cpu()

    resets = []
    for ridx in reset_rids:
      ridx = ridx.item()
      rep_resets = reset_replicate(ridx, assign_obj, reg_out,
          max_steps=self.reset_max_steps, accept_tol=self.reset_accept_tol)
      if rep_resets.shape[0] > 0:
        rep_reset_ids = rep_resets[:, 2:5].astype(np.int64)
        self._post_reset_updates(ridx, rep_reset_ids)
      self.num_bad_steps[0, ridx] = 0
      resets.append(rep_resets)
    resets = np.concatenate(resets, axis=0)

    # empty cache after re-initialization since clusters are likely to move
    # around.
    if resets.shape[0] > 0:
      self._cache_assign_obj.mul_(np.nan)
      self._cache_assign_obj_head = 0
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

  def _post_reset_updates(self, ridx, reset_ids):
    """Duplicate bases from (cand_rIdx, cand_cIdx) to (rIdx, cIdx), and other
    re-initializations."""
    # reset_ids = cIdx, cand_rIdx, cand_cIdx
    uniqIdx = ut.unique_resets(reset_ids[:, 0])
    reset_ids = reset_ids[uniqIdx, :]
    cIdx, cand_rIdx, cand_cIdx = [reset_ids[:, ii] for ii in range(3)]
    if np.any(cand_rIdx == ridx):
      raise RuntimeError(("Shouldn't swap with a cluster from the same "
          "replica."))

    self.Us.data[ridx, cIdx, :] = self.Us.data[cand_rIdx, cand_cIdx, :]
    if self.affine:
      self.bs.data[ridx, cIdx, :] = self.bs.data[cand_rIdx, cand_cIdx, :]
    self._cache_assign_obj[:, ridx, cIdx] = self._cache_assign_obj[:,
        cand_rIdx, cand_cIdx]

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

  def epoch_init(self):
    """Initialization at the start of an epoch."""
    self._zero()
    return

  def _zero(self):
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

  def __init__(self, k=10, d=10, D=100, affine=False, replicates=5,
        reg_params={}, reset_patience=100, reset_try_tol=0.01,
        reset_max_steps=50, reset_accept_tol=1e-3, reset_cache_size=500,
        scale_grad_lip=False, **kwargs):

    if scale_grad_lip and affine:
      raise ValueError("gradient scaling not supported in affine setting.")

    super().__init__(k=k, d=d, D=D, affine=affine, replicates=replicates,
        reg_params=reg_params, reset_patience=reset_patience,
        reset_try_tol=reset_try_tol, reset_max_steps=reset_max_steps,
        reset_accept_tol=reset_accept_tol, reset_cache_size=reset_cache_size,
        **kwargs)

    self.scale_grad_lip = scale_grad_lip
    if scale_grad_lip:
      # ultimate shape (2, r, k, d, d), (2, r)
      self.HessUN, self.epochN = None, None
      self.cache_z = True
      self.Us.register_hook(lambda UGrad: self._scale_grad(UGrad))
    return

  def _pfi_fit_cluster(self, xcenter, X, nn_q=None, normalize=False):
    """Fit subspaces to neighborhoods of candidate cluster centers.

    Args:
      xcenter: candidate centers, shape (pfi_n_cand, D).
      X: initialization data set, shape (init_size, D).
      nn_q: extra nearest neighbors (default: int(ceil(0.1*d))).
      normalize: normalize data before computing cosine distances (otherwise
        assumed to already be normalized) (default: False).

    Returns:
      Ucand, bcand: candidate bases, centers.
    """
    if nn_q is None:
      nn_q = int(np.ceil(0.1*self.d))

    ncand = xcenter.shape[0]
    Ucand = torch.zeros(ncand, self.D, self.d, device=X.device)
    bcand = (torch.zeros(ncand, self.D, device=X.device)
        if self.affine else None)

    nn_k = self.d + nn_q
    lamb = np.sqrt(nn_k*self.reg_params['U_frosqr_in'] *
        self.reg_params['z'])
    if lamb > 0:
      alpha = (np.sqrt(self.reg_params['z']) /
          np.sqrt(nn_k*self.reg_params['U_frosqr_in'])) ** 0.5
    else:
      alpha = None

    knnIdx = ut.cos_knn(xcenter, X, nn_k, normalize=normalize)

    for jj in range(ncand):
      Xj = X[knnIdx[jj, :], :]
      U, b = ut.reg_pca(Xj, self.d, form='mf', lamb=lamb, gamma=0.0,
          affine=self.affine, solver='randomized')

      # re-scale U
      if alpha is not None:
        U.mul_(alpha)

      Ucand[jj, :] = U
      if self.affine:
        bcand[jj, :] = b
    return Ucand, bcand

  def encode(self, x, Us=None, bs=None):
    """Compute subspace coefficients for x in closed form, by computing
    batched solution to normal equations.

      min_z 1/2 || x - (Uz + b) ||_2^2 + \lambda/2 ||z||_2^2
      (U^T U + \lambda I) z* = U^T (x - b)

    Inputs:
      x: shape (batch_size, D)
      Us: bases, shape (r, k, D, d) (default: None)
      bs: centers, shape (r, k, D) (default: None)

    Returns:
      z: latent low-dimensional codes (r, k, batch_size, d)
    """
    Us = self.Us if Us is None else Us
    bs = self.bs if bs is None else bs

    # Us shape (r, k, D, d)
    # bs shape (r, k, D)
    batch_size = x.size(0)

    # (1, 1, D, batch_size)
    x = x.data.t().view(1, 1, self.D, batch_size)
    if self.affine:
      # (r, k, D, batch_size)
      x = x.sub(bs.data.unsqueeze(3))
    # (r, k, d, batch_size)
    z = ut.batch_ridge(x, Us.data, lamb=self.reg_params['z'])

    # (r, k, batch_size, d)
    z = z.transpose(2, 3)
    return z

  def reg(self, z, Us=None, bs=None):
    """Evaluate subspace regularization."""
    Us = self.Us if Us is None else Us
    bs = self.bs if bs is None else bs
    r, k = Us.shape[:2]

    regs = dict()
    # U regularization, each is shape (r, k)
    if max([self.reg_params[key] for key in
          ('U_frosqr_in', 'U_frosqr_out')]) > 0:
      U_frosqr = torch.sum(Us.pow(2), dim=(2, 3))

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
      z_frosqr = z.data.pow(2).sum(dim=-1)
      regs['z'] = z_frosqr.permute(2, 0, 1).mul(self.reg_params['z']*0.5)
    return self._parse_reg(regs, r, k)

  def _scale_grad(self, UGrad):
    """Divide gradients by an estimate of the Lipschitz constant, per
    replicate."""
    device = self._z.device

    # compute un-normalized batch Hessian per cluster, replicate.
    batch_size = self.c.shape[0]
    z_zero = self.c.permute(1, 2, 0).unsqueeze(3) * self._z
    HessUN = torch.matmul(z_zero.transpose(2, 3), z_zero)

    # (r, k)
    batch_cluster_sizes = self.c.sum(dim=0)
    # 1e-3 added to avoid singularity (huge steps)
    lamb = (self.reg_params['U_frosqr_in'] * batch_cluster_sizes).add_(
        (self.reg_params['U_frosqr_out'] + 1e-3) * batch_size)
    # (r, k, d, d)
    Id = torch.eye(self.d, device=device).mul(
        lamb.view(self.r, self.k, 1, 1))
    self.HessUN = self.HessUN.add_(HessUN + Id)
    self.epochN += batch_size

    # full epoch Hessian
    self.Hess = self.HessUN[0, :] / self.epochN[0, :].view(self.r, 1, 1, 1)

    # approximate Lipschitz constant (max diagonal entry across clusters).
    dIdx = torch.arange(self.d)
    Hessdiag = self.Hess[:, :, dIdx, dIdx]
    self.Lip = torch.max(Hessdiag.view(self.r, -1), dim=1)[0]

    # scale gradient by 1/L
    UGrad_scale = UGrad.div(self.Lip.view(self.r, 1, 1, 1))
    return UGrad_scale

  def epoch_init(self):
    """Initialization at the start of an epoch."""
    super().epoch_init()
    if self.scale_grad_lip:
      if self.HessUN is None:
        Id = torch.diag(torch.ones(self.d).mul_(1e-3))
        self.HessUN = Id.repeat(2, self.r, self.k, 1, 1).to(self.Us.device)
        self.epochN = torch.zeros((2, self.r), device=self.Us.device)
      else:
        self.HessUN = ut.shift_and_zero(self.HessUN)
        self.epochN = ut.shift_and_zero(self.epochN)
    return

  def _post_reset_updates(self, ridx, reset_ids):
    """Duplicate bases from (cand_rIdx, cand_cIdx) to (rIdx, cIdx), and other
    re-initializations."""
    super()._post_reset_updates(ridx, reset_ids)
    if self.scale_grad_lip:
      # re-initialize hessian estimate with identity.
      self.HessUN[:, ridx, :] = torch.diag(
          torch.ones(self.d, device=self.Us.device).mul_(1e-3))
      self.epochN[:, ridx] = 0.0
    return


class KSubspaceProjModel(KSubspaceBaseModel):
  """K-subspace model where low-dim coefficients are computed by a projection
  matrix."""
  default_reg_params = {
      'U_frosqr_in': 0.01,
      'U_frosqr_out': 1e-4
  }
  assign_reg_terms = {'U_frosqr_in'}

  def __init__(self, k=10, d=10, D=100, affine=False, replicates=5,
        reg_params={}, reset_patience=100, reset_try_tol=0.01,
        reset_max_steps=50, reset_accept_tol=1e-3, reset_cache_size=500,
        **kwargs):

    super().__init__(k=k, d=d, D=D, affine=affine, replicates=replicates,
        reg_params=reg_params, reset_patience=reset_patience,
        reset_try_tol=reset_try_tol, reset_max_steps=reset_max_steps,
        reset_accept_tol=reset_accept_tol, reset_cache_size=reset_cache_size,
        **kwargs)
    return

  def _pfi_fit_cluster(self, xcenter, X, nn_q=None, normalize=False):
    """Fit subspaces to neighborhoods of candidate cluster centers.

    Args:
      xcenter: candidate centers, shape (pfi_n_cand, D).
      X: initialization data set, shape (init_size, D).
      nn_q: extra nearest neighbors (default: int(ceil(0.1*d))).
      normalize: normalize data before computing cosine distances (otherwise
        assumed to already be normalized) (default: False).

    Returns:
      Ucand, bcand: candidate bases, centers.
    """
    if nn_q is None:
      nn_q = int(np.ceil(0.1*self.d))

    ncand = xcenter.shape[0]
    Ucand = torch.zeros(ncand, self.D, self.d, device=X.device)
    bcand = (torch.zeros(ncand, self.D, device=X.device)
        if self.affine else None)

    nn_k = self.d + nn_q
    lamb = nn_k*self.reg_params['U_frosqr_in']

    knnIdx = ut.cos_knn(xcenter, X, nn_k, normalize=normalize)

    for jj in range(ncand):
      Xj = X[knnIdx[jj, :], :]
      U, b = ut.reg_pca(Xj, self.d, form='proj', lamb=lamb,
          affine=self.affine, solver='randomized')

      Ucand[jj, :] = U
      if self.affine:
        bcand[jj, :] = b
    return Ucand, bcand

  def encode(self, x, Us=None, bs=None):
    """Project x onto each of k low-dimensional spaces.

    Inputs:
      x: shape (batch_size, D)
      Us: bases, shape (r, k, D, d) (default: None)
      bs: centers, shape (r, k, D) (default: None)

    Returns:
      z: latent low-dimensional codes (r, k, batch_size, d)
    """
    Us = self.Us if Us is None else Us
    bs = self.bs if bs is None else bs
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

  def reg(self, _, Us=None, bs=None):
    """Evaluate subspace regularization."""
    Us = self.Us if Us is None else Us
    bs = self.bs if bs is None else bs
    r, k = Us.shape[:2]

    regs = dict()
    # U regularization, each is shape (r, k)
    if max([self.reg_params[key] for key in
          ('U_frosqr_in', 'U_frosqr_out')]) > 0:
      U_frosqr = torch.sum(Us.pow(2), dim=(2, 3))

      if self.reg_params['U_frosqr_in'] > 0:
        regs['U_frosqr_in'] = U_frosqr.mul(
            self.reg_params['U_frosqr_in'])

      if self.reg_params['U_frosqr_out'] > 0:
        regs['U_frosqr_out'] = U_frosqr.mul(
            self.reg_params['U_frosqr_out'])
    return self._parse_reg(regs, r, k)


class KSubspaceBatchAltBaseModel(KSubspaceBaseModel):
  default_reg_params = dict()
  assign_reg_terms = dict()

  def __init__(self, k=10, d=10, dataset=None, affine=False, replicates=5,
        reg_params={}, reset_patience=2, reset_try_tol=0.01,
        reset_max_steps=50, reset_accept_tol=1e-3, reset_cache_size=500,
        svd_solver='randomized', **kwargs):

    if dataset is None:
      raise ValueError("dataset is required")
    if svd_solver not in ('randomized', 'svds', 'svd'):
      raise ValueError("Invalid svd solver {}".format(svd_solver))

    # X assumed to be N x D.
    kwargs['D'] = dataset.X.shape[1]
    super().__init__(k=k, d=d, affine=affine, replicates=replicates,
        reg_params=reg_params, reset_patience=reset_patience,
        reset_try_tol=reset_try_tol, reset_max_steps=reset_max_steps,
        reset_accept_tol=reset_accept_tol, reset_cache_size=reset_cache_size,
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
    return

  def objective(self, Us=None, bs=None, update_cache=True):
    """Evaluate objective function. Note, forward evaluation is always
    computed.

    Args:
      Us: bases, shape (r, k, D, d) (default: None)
      bs: centers, shape (r, k, D) (default: None)
      update_cache: (default: True)

    Returns:
      obj_mean: average objective across replicates
      obj, loss, reg_in, reg_out: metrics per replicate, shape (r,)
    """
    groups_prev = self.groups

    obj_vals = super().objective(self.X, Us=Us, bs=bs,
        update_cache=update_cache)

    if update_cache:
      self._updates = ((self.groups != groups_prev).sum()
          if groups_prev is not None else self.N)
    return obj_vals


class KSubspaceBatchAltProjModel(KSubspaceBatchAltBaseModel,
      KSubspaceProjModel):
  default_reg_params = {
      'U_frosqr_in': 0.01,
      'U_frosqr_out': 1e-4
  }
  assign_reg_terms = {'U_frosqr_in'}

  def __init__(self, k=10, d=10, dataset=None, affine=False, replicates=5,
        reg_params={}, reset_patience=2, reset_try_tol=0.01,
        reset_max_steps=50, reset_accept_tol=1e-3, reset_cache_size=500,
        svd_solver='randomized', **kwargs):

    super().__init__(k=k, d=d, dataset=dataset, affine=affine,
        replicates=replicates, reg_params=reg_params,
        reset_patience=reset_patience, reset_try_tol=reset_try_tol,
        reset_max_steps=reset_max_steps, reset_accept_tol=reset_accept_tol,
        reset_cache_size=reset_cache_size, svd_solver=svd_solver, **kwargs)
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

  def __init__(self, k=10, d=10, dataset=None, affine=False, replicates=5,
        reg_params={}, reset_patience=2, reset_try_tol=0.01,
        reset_max_steps=50, reset_accept_tol=1e-3, reset_cache_size=500,
        init='random', svd_solver='randomized', **kwargs):

    super().__init__(k=k, d=d, dataset=dataset, affine=affine,
        replicates=replicates, reg_params=reg_params,
        reset_patience=reset_patience, reset_try_tol=reset_try_tol,
        reset_max_steps=reset_max_steps, reset_accept_tol=reset_accept_tol,
        reset_cache_size=reset_cache_size, svd_solver=svd_solver, **kwargs)

    # bases initialized in KSubspaceMFModel
    # ensure columns are orthogonal
    for ii in range(self.r):
      for jj in range(self.k):
        P, S, _ = torch.svd(self.Us.data[ii, jj, :])
        self.Us.data[ii, jj, :] = P.mul_(S)
    return

  def encode(self, x, Us=None, bs=None):
    """Compute subspace coefficients for x in closed form, by computing
    batched solution to normal equations. Use fact that colums of U are
    orthogonal (althought not necessarily unit norm).

      min_z 1/2 || x - (Uz + b) ||_2^2 + \lambda/2 ||z||_2^2
      (U^T U + \lambda I) z* = U^T (x - b)

    Inputs:
      x: shape (batch_size, D)
      Us: bases, shape (r, k, D, d) (default: None)
      bs: centers, shape (r, k, D) (default: None)

    Returns:
      z: latent low-dimensional codes (r, k, batch_size, d)
    """
    Us = self.Us if Us is None else Us
    bs = self.bs if bs is None else bs
    batch_size = x.size(0)

    if self.affine:
      # (r, k, batch_size, D)
      x = x.sub(bs.data.unsqueeze(2))
    else:
      x = x.view(1, 1, batch_size, self.D)

    # (r, k, batch_size, d)
    b = torch.matmul(x, Us.data)

    # (r, k, 1, d)
    S = torch.norm(Us.data, dim=2, keepdim=True)
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


class KMeansModel(KSubspaceBaseModel):
  """K-means model as a special case of K-subspaces."""
  default_reg_params = {
      'b_frosqr_out': 0.0
  }
  assign_reg_terms = {}

  def __init__(self, k=10, D=100, replicates=5, reg_params={},
        reset_patience=2, reset_try_tol=0.01, reset_max_steps=50,
        reset_accept_tol=1e-3, reset_cache_size=500, scale_grad_lip=False,
        **kwargs):

    kwargs['d'] = 1  # Us not used, but retained for consistency
    kwargs['affine'] = True

    super().__init__(k=k, D=D, replicates=replicates, reg_params=reg_params,
        reset_patience=reset_patience, reset_try_tol=reset_try_tol,
        reset_max_steps=reset_max_steps, reset_accept_tol=reset_accept_tol,
        reset_cache_size=reset_cache_size, **kwargs)

    self.scale_grad_lip = scale_grad_lip
    if scale_grad_lip:
      self.LipUN, self.Lip, self.epochN = None, None, None
      self.bs.register_hook(lambda bGrad: self._scale_grad(bGrad))
    return

  def reset_parameters(self):
    """Initialize Us with zeros and bs with entries drawn from normal with std
    0.1/sqrt(D)."""
    std = .1 / np.sqrt(self.D)
    self.Us.data.normal_(0., std)
    self.bs.data.normal_(0., std)
    return

  def _pfi_fit_cluster(self, xcenter, X, **kwargs):
    """Fit subspaces to neighborhoods of candidate cluster centers.

    Args:
      xcenter: candidate centers, shape (pfi_n_cand, D).
      X: initialization data set, not used.

    Returns:
      Ucand, bcand: candidate bases, centers.
    """
    Ucand = torch.zeros(xcenter.shape[0], self.D, self.d, device=X.device)
    bcand = xcenter.clone()
    return Ucand, bcand

  def encode(self, x, Us=None, bs=None):
    """Placeholder, since no latent codes in k-means case."""
    return None

  def decode(self, _, Us=None, bs=None):
    """Returns cluster centers as embeddings.

    Inputs:
      z: latent code, shape (r, k, batch_size, d)
      Us: bases, shape (r, k, D, d) (default: None)
      bs: centers, shape (r, k, D) (default: None)

    Returns:
      bs: centers posing as reconstruction, shape (r, k, 1, D)
    """
    bs = self.bs if bs is None else bs
    return bs.clone().unsqueeze(2)

  def loss(self, x, x_):
    """Evaluate reconstruction loss

    Inputs:
      x: data, shape (batch_size, D)
      x_: k means posing as reconstruction, shape (r, k, 1, D)

    Returns:
      loss: shape (batch_size, r, k)
    """
    bs = x_.squeeze(2)
    bsqrnorms = bs.pow(2).sum(dim=2).mul(0.5)
    xsqrnorms = x.pow(2).sum(dim=1).mul(0.5)

    # x (batch_size, D)
    # b (r, k, D)
    # xTb (batch_size, r, k)
    xTb = torch.matmul(bs, x.t()).permute(2, 0, 1)
    # (batch_size, r, k)
    loss = xsqrnorms.view(-1, 1, 1).sub(xTb).add(bsqrnorms.unsqueeze(0))
    loss = loss.clamp(min=0.0)
    return loss

  def reg(self, _, Us=None, bs=None):
    """Evaluate regularization on cluster centers."""
    bs = self.bs if bs is None else bs
    r, k = bs.shape[:2]

    regs = dict()
    if self.reg_params['b_frosqr_out'] > 0:
      regs['b_frosqr_out'] = torch.sum(bs.pow(2), dim=2).mul(
          self.reg_params['b_frosqr_out']*0.5)
    return self._parse_reg(regs, r, k)

  def _scale_grad(self, bGrad):
    """Divide gradients by an estimate of the Lipschitz constant, per
    replicate."""
    batch_size = self.c.shape[0]
    # (r, k)
    batch_cluster_sizes = self.c.sum(dim=0)

    # un-normalized Lipschitz constant for each cluster
    # 1e-3 added to avoid singularity (huge steps)
    lamb = batch_cluster_sizes.add_(batch_size *
        (self.reg_params['b_frosqr_out'] + 1e-3))
    # (r, k)
    self.LipUN = self.LipUN.add_(lamb)
    self.epochN += batch_size

    self.Lip = self.LipUN[0, :].max(dim=1)[0] / self.epochN[0, :]
    bGrad_scale = bGrad.div(self.Lip.view(self.r, 1, 1))
    return bGrad_scale

  def epoch_init(self):
    """Initialization at the start of an epoch."""
    super().epoch_init()
    if self.scale_grad_lip:
      if self.LipUN is None:
        self.LipUN = torch.ones(2, self.r, self.k,
            device=self.bs.device).mul_(1e-3)
        self.epochN = torch.zeros((2, self.r), device=self.bs.device)
      else:
        self.LipUN = ut.shift_and_zero(self.LipUN)
        self.epochN = ut.shift_and_zero(self.epochN)
    return

  def _post_reset_updates(self, ridx, reset_ids):
    """Duplicate bases from (cand_rIdx, cand_cIdx) to (rIdx, cIdx), and other
    re-initializations."""
    super()._post_reset_updates(ridx, reset_ids)
    if self.scale_grad_lip:
      # re-initialize lipschitz estimate with identity.
      self.LipUN[:, ridx, :] = 1e-3
      self.epochN[:, ridx] = 0.0
    return


class KMeansBatchAltModel(KSubspaceBatchAltBaseModel, KMeansModel):
  """K-means model as a special case of K-subspaces."""
  default_reg_params = {
      'b_frosqr_out': 0.0
  }
  assign_reg_terms = {}

  def __init__(self, k=10, dataset=None, replicates=5, reg_params={},
        reset_patience=2, reset_try_tol=0.01, reset_max_steps=50,
        reset_accept_tol=1e-3, reset_cache_size=500, **kwargs):

    kwargs['d'] = 1  # Us not used, but retained for consistency
    kwargs['affine'] = True

    super().__init__(k=k, dataset=dataset, replicates=replicates,
        reg_params=reg_params, reset_patience=reset_patience,
        reset_try_tol=reset_try_tol, reset_max_steps=reset_max_steps,
        reset_accept_tol=reset_accept_tol, reset_cache_size=reset_cache_size,
        **kwargs)
    return

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

  def __init__(self, k=10, d=10, D=100, affine=False, replicates=5,
        reg_params={}, reset_patience=100, reset_try_tol=0.01,
        reset_max_steps=50, reset_accept_tol=1e-3, reset_cache_size=500,
        scale_grad_lip=False, sparse_encode=True, sparse_decode=False,
        norm_comp_error=True, **kwargs):

    super().__init__(k=k, d=d, D=D, affine=affine, replicates=replicates,
        reg_params=reg_params, reset_patience=reset_patience,
        reset_try_tol=reset_try_tol, reset_max_steps=reset_max_steps,
        reset_accept_tol=reset_accept_tol, reset_cache_size=reset_cache_size,
        scale_grad_lip=scale_grad_lip, **kwargs)

    self.sparse_encode = sparse_encode
    self.sparse_decode = sparse_decode
    self.norm_comp_error = norm_comp_error
    self._slcUs, self._slcbs = None, None

    self.cache_z = True
    self.cache_x_ = not sparse_decode
    return

  def _pfi_fit_cluster(xcenter, x, **kwargs):
    raise NotImplementedError

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

  def encode(self, x, Us=None, bs=None):
    """Encode x by vectorized (hence parallelizable) (r*k*batch_size)
    O(pad_nnz) least-squares problems."""
    Us = self.Us if Us is None else Us
    bs = self.bs if bs is None else bs
    r, k, D, d = Us.shape
    batch_size = x.shape[0]

    if self.sparse_encode:
      # (r, k, batch_size, pad_nnz, d)
      self._slcUs = Us[:, :, x.indices, :]
      self._slcUs = self._slcUs.mul(x.omega_float.unsqueeze(2))
    else:
      # (r, k, batch_size, D, d)
      if (self._slcUs is None or self._slcUs.shape[2] != batch_size):
        self._slcUs = torch.zeros(r, k, batch_size, D, d, device=Us.device)
      self._slcUs.copy_(Us.data.unsqueeze(2))
      self._slcUs.mul_(x.omega_float_dense.unsqueeze(2))

    # (batch_size, *)
    xval = x.values if self.sparse_encode else x.values_dense
    if self.affine:
      if self.sparse_encode:
        # (r, k, batch_size, pad_nnz)
        self._slcbs = bs[:, :, x.indices]
        # (r, k, batch_size, pad_nnz)
        xval = xval.sub(self._slcbs.detach())
      else:
        # (r, k, batch_size, D)
        xval = xval.sub(bs.unsqueeze(2))
    # (r, k, batch_size, *, 1)
    xval = xval.unsqueeze(-1)

    # (r, k, batch_size, d)
    z = ut.batch_ridge(xval, self._slcUs.detach(),
        lamb=self.reg_params['z']).squeeze(4)
    return z

  def decode(self, z, Us=None, bs=None, x=None):
    """Encode x by vectorized (hence parallelizable) (r*k*batch_size)
    O(pad_nnz) mat-vec products."""
    Us = self.Us if Us is None else Us
    bs = self.bs if bs is None else bs

    # x_ = U z + b
    # Us shape (r, k, D, d)
    # bs shape (r, k, D)
    if self.sparse_decode:
      # indices shape (batch_size, pad_nnz)
      # slcUs (r, k, batch_size, pad_nnz, d)
      slcUs = Us[:, :, x.indices, :] if x is not None else self._slcUs
      # (r, k, batch_size, pad_nnz)
      x_ = torch.matmul(slcUs, z.unsqueeze(4)).squeeze(4)
      if self.affine:
        # slcbs (r, k, batch_size, pad_nnz)
        slcbs = bs[:, :, x.indices] if x is not None else self._slcbs
        x_ = x_.add(slcbs)
    else:
      # (r, k, batch_size, D)
      x_ = torch.matmul(z, Us.transpose(2, 3))
      if self.affine:
        x_ = x_.add(bs.unsqueeze(2))
    return x_

  def eval_comp_error(self, x0):
    """Evaluate completion error over observed entries in x0.

    Inputs:
      x0: either sparse format (list of sparse vectors), or dense format
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
        x0_ = self.decode(self._z, x=x0)
        # choose assigned reconstruction
        rIdx = torch.arange(self.r).view(-1, 1)
        batchIdx = torch.arange(batch_size).view(1, -1)
        # (r, batch_size, pad_nnz)
        x0_ = x0_[rIdx, self.groups.t(), batchIdx, :]
      else:
        # (r, batch_size, D)
        x0_ = self._x_

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
