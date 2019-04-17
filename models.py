from __future__ import print_function
from __future__ import division

import numpy as np
import torch
import torch.nn as nn

import utils as ut

EPS = 1e-8
EMA_DECAY = 0.9


class KSubspaceBaseModel(nn.Module):
  """Base K-subspace class."""
  default_reg_params = dict()
  assign_reg_terms = set()

  def __init__(self, k, d, D, affine=False, replicates=5, reg_params={},
        reset_metric='value', unused_thr=0.01, reset_patience=100,
        reset_obj='assign', reset_decr_tol=1e-4, reset_sigma=0.05):
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
    if reset_metric not in {'value', 'size'}:
      raise ValueError(
          "Invalid reset_metric parameter {}".format(reset_metric))
    if unused_thr < 0:
      raise ValueError("Invalid unused_thr parameter {}".format(unused_thr))
    if reset_patience < 0:
      raise ValueError("Invalid reset_patience parameter {}".format(
          reset_patience))
    if reset_obj not in {'assign', 'full'}:
      raise ValueError("Invalid reset_obj parameter {}".format(reset_obj))
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
    self.reset_metric = reset_metric
    self.unused_thr = unused_thr
    self.reset_patience = reset_patience
    self.reset_obj = reset_obj
    self.reset_decr_tol = (float('-inf') if reset_decr_tol is None
        else reset_decr_tol)
    self.reset_sigma = reset_sigma

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
    self.register_buffer('c_mean', torch.ones(self.r, k).div_(k))
    self.register_buffer('value', torch.zeros(self.r, k))
    self.register_buffer('steps', torch.zeros(self.r, k, dtype=torch.int64))
    self.register_buffer('jitter',
        torch.randint(-reset_patience//2, reset_patience//2 + 1,
            (self.r, 1), dtype=torch.int64))
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
    self._batch_assign_obj = assign_obj
    self._batch_reg_out = reg_out.data

    # reduce and compute objective, shape (r,)
    loss = loss.sum(dim=2).mean(dim=0)
    reg_in = reg_in.sum(dim=2).mean(dim=0)
    reg_out = reg_out.sum(dim=1)
    obj = loss + reg_in + reg_out
    obj_mean = obj.mean()

    # cache per replicate objective
    self._batch_obj = obj.data
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
    self.c_mean.mul_(EMA_DECAY).add_(1-EMA_DECAY, self._batch_c_mean)
    self.value.mul_(EMA_DECAY).add_(1-EMA_DECAY, self._batch_value)

    self.groups = self.groups.cpu()
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
    """Reset the least used clusters by choosing best replacement candidates
    from all other replicates.

    Returns:
      resets: np array log of resets, shape (n_resets, 9). Columns
        are (reset rep idx, reset cluster idx, reset metric, candidate rep idx,
        candidate cluster idx, reset success, obj decrease, candidate size,
        candidate value).
    """
    # reset not possible if only one replicate
    # should probably give warning
    if self.r == 1:
      return np.zeros((0, 9), dtype=object)

    # identify clusters to reset
    if self.reset_metric == 'value':
      reset_metric = self.value
      if self.reset_obj == 'full':
        # take outside reg into account
        reset_metric = reset_metric - self._batch_reg_out
    else:
      reset_metric = self.c_mean
    # clamp to avoid unlikely case where max value is not positive.
    reset_metric = reset_metric / torch.max(reset_metric,
        dim=1, keepdim=True)[0].clamp(min=EPS)
    reset_mask = reset_metric <= self.unused_thr
    reset_mask.mul_((self.steps - self.jitter) >= self.reset_patience)
    reset_rids = reset_mask.any(dim=1).nonzero().view(-1)

    resets = []
    for ridx in reset_rids:
      rep_reset_cids = reset_mask[ridx, :].nonzero().view(-1)
      # sort by metric
      rep_reset_cids = rep_reset_cids[
          torch.sort(reset_metric[ridx, rep_reset_cids])[1]]
      for cidx in rep_reset_cids:
        # find best reset candidate and substitute if decrease tol satisfied
        (cand_ridx, cand_cidx, cand_obj,
            cand_c_mean, cand_c_value) = self._reset_cluster(ridx, cidx)
        old_obj = self._batch_obj[ridx].item()
        obj_decr = 1.0 - cand_obj / old_obj
        cand_size = cand_c_mean[cidx].item()
        cand_value = cand_c_value[cidx].item()

        if self.reset_metric == 'size':
          still_unused = cand_size <= self.unused_thr * cand_c_mean.max()
        else:
          still_unused = cand_value <= self.unused_thr * cand_c_value.max()

        reset_success = (obj_decr > self.reset_decr_tol and
            (obj_decr > 0 or not still_unused))
        if reset_success:
          # reset parameters
          self.Us.data[ridx, cidx, :] = self.Us.data[cand_ridx, cand_cidx, :]
          if self.affine:
            self.bs.data[ridx, cidx, :] = self.bs.data[
                cand_ridx, cand_cidx, :]

          # reset batch metrics (matters if there are multiple resets to do)
          self._batch_assign_obj[:, ridx, cidx] = self._batch_assign_obj[
              :, cand_ridx, cand_cidx]
          self._batch_obj[ridx] = cand_obj
          self._batch_reg_out[ridx, cidx] = self._batch_reg_out[
              cand_ridx, cand_cidx]

          # reset size, value, steps
          self._batch_c_mean[ridx, :] = self.c_mean[ridx, :] = cand_c_mean
          self._batch_value[ridx, :] = self.value[ridx, :] = cand_c_value
          self.steps[ridx, cidx] = 0
        else:
          # after failing to add a new basis, don't want to reset in this
          # replicate for a while
          self.steps[ridx, :] = 0

        resets.append([ridx.item(), cidx.item(),
            reset_metric[ridx, cidx].item(), cand_ridx, cand_cidx,
            int(reset_success), obj_decr, cand_size, cand_value])

      self.jitter[ridx].random_(-self.reset_patience//2,
          self.reset_patience//2 + 1)

    if len(resets) > 0:
      resets = np.array(resets, dtype=object)
    else:
      resets = np.zeros((0, 9), dtype=object)

    # perturb all bases slightly in replicates with resets to make sure we
    # always have some diversity across replicates
    if self.reset_sigma > 0 and np.sum(resets[:, 5]) > 0:
      rIdx = np.unique(resets[resets[:, 5] == 1, 0].astype(np.int64))
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

  def _reset_cluster(self, ridx, cidx):
    """For a given replicate and empty cluster, find the best candidate to copy
    from the other replicates.

    Inputs:
      ridx, cidx: replicate and cluster index for empty cluster

    Returns:
      candridx, candcidx: replicate and cluster index for best reset candidate
      cand_obj: objective for best candidate replacement.
      cand_c_mean: fraction of points from last batch assigned to best
        candidate
    """
    assert(self.r > 1)

    # extract indices for reset candidates
    cand_mask = torch.ones(self.r, self.k, dtype=torch.uint8,
        device=self._batch_obj.device)
    cand_mask[ridx, :] = 0
    cand_Idx = cand_mask.nonzero()

    # organize alternative assignment objectives, one per reset candidate,
    # where empty cluster objective is replaced by the candidate's
    # (batch_size, 1, k)
    alt_assign_obj = self._batch_assign_obj[:, [ridx], :]
    # (batch_size, (r-1)*k, k)
    alt_assign_obj = alt_assign_obj.repeat(1, (self.r-1)*self.k, 1)
    alt_assign_obj[:, :, cidx] = self._batch_assign_obj[:, cand_Idx[:, 0],
        cand_Idx[:, 1]]

    # find assignments for each alternative
    # (batch_size, (r-1)*k, 2)
    top2obj, top2Idx = torch.topk(alt_assign_obj, 2, dim=2,
        largest=False, sorted=True)
    assign_Idx = top2Idx[:, :, 0]
    alt_obj = top2obj[:, :, 0]

    # compute candidate objectives and choose best
    # (possibly ignoring outside reg)
    # ((r-1)*k,)
    alt_obj = alt_obj.mean(dim=0)
    alt_reg_out = (self._batch_reg_out[ridx, :].sum() -
        self._batch_reg_out[ridx, cidx])
    if self.reset_obj == 'full':
      alt_reg_out = (self._batch_reg_out[cand_Idx[:, 0], cand_Idx[:, 1]] +
          alt_reg_out)
      alt_obj += alt_reg_out
    cand_obj, min_idx = torch.min(alt_obj, dim=0)
    cand_ridx, cand_cidx = cand_Idx[min_idx, 0], cand_Idx[min_idx, 1]

    # find new cluster sizes and values
    cand_c = torch.zeros((alt_assign_obj.shape[0], self.k),
        device=alt_assign_obj.device)
    cand_c.scatter_(1, assign_Idx[:, min_idx].unsqueeze(1), 1)
    cand_c_mean = cand_c.mean(dim=0)
    cand_value = torch.mean(cand_c *
        (top2obj[:, min_idx, [1]] - top2obj[:, min_idx, [0]]), dim=0)

    # Add outside reg after selection if not included
    if self.reset_obj != 'full':
      cand_obj += alt_reg_out + self._batch_reg_out[cand_ridx, cand_cidx]
    return (cand_ridx.item(), cand_cidx.item(), cand_obj.item(),
        cand_c_mean, cand_value)

  def step(self):
    """Increment steps since reset counter."""
    self.steps.add_(1)
    return

  def zero(self):
    """Zero out near zero bases.

    Numerically near zero values slows matmul performance up to 6x.
    """
    Unorms = self.Us.data.pow(2).sum(dim=(2, 3)).sqrt()
    self.Us.data[(Unorms < EPS*Unorms.max()), :, :] = 0.0
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
        reset_metric='value', unused_thr=0.01, reset_patience=100,
        reset_obj='assign', reset_decr_tol=1e-4, reset_sigma=0.05):
    super().__init__(k, d, D, affine, replicates, reg_params, reset_metric,
        unused_thr, reset_patience, reset_obj, reset_decr_tol, reset_sigma)
    self.reset_parameters()
    return

  def reset_parameters(self):
    """Initialize Us, bs with entries drawn from normal with std
    0.1/sqrt(D)."""
    std = .1 / np.sqrt(self.D)
    self.Us.data.normal_(0., std)
    if self.affine:
      self.bs.data.normal_(0., std)
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
        reset_metric='value', unused_thr=0.01, reset_patience=100,
        reset_obj='assign', reset_decr_tol=1e-4, reset_sigma=0.05):
    super().__init__(k, d, D, affine, replicates, reg_params, reset_metric,
        unused_thr, reset_patience, reset_obj, reset_decr_tol, reset_sigma)
    self.reset_parameters()
    return

  def reset_parameters(self):
    """Initialize Us, bs with entries drawn from normal with std
    0.1/sqrt(D)."""
    std = .1 / np.sqrt(self.D)
    self.Us.data.normal_(0., std)
    if self.affine:
      self.bs.data.normal_(0., std)
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
        reset_metric='value', unused_thr=0.01, reset_patience=2,
        reset_obj='assign', reset_decr_tol=1e-4, reset_sigma=0.05,
        svd_solver='randomized'):
    if svd_solver not in ('randomized', 'svds', 'svd'):
      raise ValueError("Invalid svd solver {}".format(svd_solver))

    # X assumed to be N x D.
    D = dataset.X.shape[1]
    super().__init__(k, d, D, affine, replicates, reg_params, reset_metric,
        unused_thr, reset_patience, reset_obj, reset_decr_tol, reset_sigma)

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
    return super().objective(self.X)

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
    self._updates = ((self.groups != groups_prev).sum()
        if groups_prev is not None else self.N)

    self.groups = self.groups.cpu()
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
        reset_metric='value', unused_thr=0.01, reset_patience=2,
        reset_obj='assign', reset_decr_tol=1e-4, reset_sigma=0.05,
        svd_solver='randomized'):

    super().__init__(k, d, dataset, affine, replicates, reg_params,
        reset_metric, unused_thr, reset_patience, reset_obj, reset_decr_tol,
        reset_sigma)
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
    self.steps.add_(1)
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
        reset_metric='value', unused_thr=0.01, reset_patience=2,
        reset_obj='assign', reset_decr_tol=1e-4, reset_sigma=0.05,
        svd_solver='randomized'):

    super().__init__(k, d, dataset, affine, replicates, reg_params,
        reset_metric, unused_thr, reset_patience, reset_obj, reset_decr_tol,
        reset_sigma)
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
    self.steps.add_(1)
    return
