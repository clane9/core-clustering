from __future__ import print_function
from __future__ import division

import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import lars_path

import utils as ut

EPS = 1e-8
EMA_DECAY = 0.9


class _KSubspaceBaseModel(nn.Module):
  """Base K-subspace class."""

  def __init__(self, k, d, D, affine=False, soft_assign=0.0, olpool=None,
        ol_size_scale=True):
    if soft_assign < 0:
      raise ValueError("Invalid soft-assign parameter {}".format(soft_assign))

    super(_KSubspaceBaseModel, self).__init__()
    self.k = k  # number of groups
    self.d = d  # dimension of manifold
    self.D = D  # number of data points
    self.affine = affine
    self.soft_assign = soft_assign
    self.assign_reg_terms = None  # assigned in child
    self.ol_reg_terms = None  # assigned in child
    self.ol_size_scale = ol_size_scale

    # group assignment, ultimate shape (batch_size, k)
    self.c = None
    self.groups = None
    # subspace coefficients, ultimte shape (batch_size, k, d)
    self.z = None

    self.Us = nn.Parameter(torch.Tensor(k, D, d))
    if affine:
      self.bs = nn.Parameter(torch.Tensor(k, D))
    else:
      self.register_parameter('bs', None)
    self.register_buffer('c_mean', torch.ones(k).div_(k))

    if olpool is None:
      self.olpool = OutlierPool(20*d, d)
    else:
      self.olpool = olpool
      self.olpool.reset()
    return

  def reset_parameters(self):
    """Reset model parameters."""
    raise NotImplementedError("reset_parameters not implemented")
    return

  def forward(self, x):
    """Compute representation of x wrt each subspace.

    Input:
      x: shape (batch_size, D)

    Returns:
      x_: shape (k, batch_size, D)
    """
    z = self.encode(x)
    self.z = z.data
    x_ = self.decode(z)
    return x_

  def encode(self, x):
    """Compute subspace coefficients for x.

    Input:
      x: data, shape (batch_size, D)

    Returns:
      z: latent low-dimensional codes, shape (k, batch_size, d)
    """
    raise NotImplementedError("encode not implemented")
    return

  def decode(self, z):
    """Embed low-dim code z into ambient space.

    Input:
      z: shape (k, batch_size, d)

    Returns:
      x_: shape (k, batch_size, D)
    """
    assert(z.dim() == 3 and z.size(0) == self.k and z.size(2) == self.d)

    # x_ = U z + b
    # shape (k, batch_size, D)
    x_ = torch.matmul(z, self.Us.transpose(1, 2))
    if self.affine:
      x_ = x_.add(self.bs.unsqueeze(1))
    return x_

  def objective(self, x, update_ol=True):
    """Evaluate objective function.

    Input:
      x: data, shape (batch_size, D)
      update_ol: update outlier pool

    Returns:
      obj, scale_obj, loss, reg
      x_: reconstruction, shape (k, batch_size, D)
    """
    x_ = self(x)
    losses = self.loss(x, x_)
    regs = self.reg()

    # update assignment c
    assign_obj = losses.data + sum([regs[k].data for k in regs
        if k in self.assign_reg_terms])
    self.set_assign(assign_obj)

    # update outlier pool
    if update_ol:
      ol_obj = losses.data + sum([regs[k].data for k in regs
          if k in self.ol_reg_terms])
      if self.ol_size_scale:
        ol_obj = ol_obj*self.c_mean
      ol_obj = (self.c*ol_obj).sum(dim=1)
      self.olpool.append(x, ol_obj)

    # weight loss, reg by assignment where needed, and average over batch
    losses = torch.mean(self.c*losses, dim=0)
    for k, reg in regs.items():
      if k in self.assign_reg_terms:
        reg = self.c*reg if reg.dim() == 2 else self._batch_c_mean*reg
      if reg.dim() == 2:
        reg = reg.mean(dim=0)
      regs[k] = reg
    # combine reg terms
    regs = sum(regs.values()) if len(regs) > 0 else torch.zeros_like(losses)

    loss = losses.sum()
    reg = regs.sum()
    obj = loss + reg
    return obj, loss, reg, x_

  def loss(self, x, x_):
    """Evaluate reconstruction loss

    Inputs:
      x: data, shape (batch_size, ...)
      x_: reconstruction, shape (k, batch_size, ...)

    Returns:
      loss: shape (batch_size, k)
    """
    reduce_dim = tuple(range(2, x_.dim()))
    loss = torch.sum((x.unsqueeze(0) - x_)**2, dim=reduce_dim).t()
    return loss

  def reg(self):
    """Evaluate subspace regularization."""
    raise NotImplementedError("reg not implemented")
    return

  def set_assign(self, assign_obj):
    """Compute soft-assignment.

    Inputs:
      assign_obj: shape (batch_size, k)
    """
    if self.soft_assign <= 0:
      self.c = torch.zeros_like(assign_obj.data)
      minidx = assign_obj.data.argmin(dim=1, keepdim=True)
      self.c.scatter_(1, minidx, 1)
    else:
      self.c = ut.find_soft_assign(assign_obj.data, self.soft_assign)
    self.groups = torch.argmax(self.c, dim=1)

    self._batch_c_mean = torch.mean(self.c, dim=0)
    self.c_mean.mul_(EMA_DECAY).add_(1-EMA_DECAY, self._batch_c_mean)
    return

  def eval_sprs(self):
    """measure robust sparsity of current assignment subset c"""
    cmax, _ = torch.max(self.c.data, dim=1, keepdim=True)
    sprs = torch.sum(self.c.data / cmax, dim=1)
    sprs = torch.mean(sprs)
    return sprs

  def eval_shrink(self, x, x_):
    """measure shrinkage of reconstruction wrt data"""
    # x_ is size (k, batch_size, ...)
    norm_x_ = torch.sqrt(torch.sum(x_.data.pow(2),
        dim=tuple(range(2, x_.data.dim())))).t()
    norm_x_ = torch.sum(self.c.data*norm_x_, dim=1)
    # x is size (batch_size, ...)
    norm_x = torch.sqrt(torch.sum(x.data.pow(2),
        dim=tuple(range(1, x.data.dim()))))
    norm_x_ = torch.mean(norm_x_ / (norm_x + 1e-8))
    return norm_x_

  def eval_rank(self, tol=.01, cpu=True):
    """Compute rank and singular values of subspace bases."""
    # svd is ~1000x faster on cpu for small matrices (100 x 10).
    Us = self.Us.data.cpu() if cpu else self.Us.data
    svs = torch.stack([torch.svd(Us[ii, :])[1] for ii in range(self.k)])
    ranks = (svs > tol*svs[:, 0].median()).sum(dim=1)
    return ranks.cpu().numpy(), svs.cpu().numpy()

  def reset_unused(self, reset_thr=.01):
    """Reset (nearly) unused clusters by sampling neighborhood from pool of
    outliers.

    Returns:
      reset_ids: clusters that were reset
    """
    reset_mask = self.c_mean <= reset_thr/self.k
    reset_ids = reset_mask.nonzero().view(-1)
    if reset_ids.shape[0] > 0:
      Unorm = self.Us.data.pow(2).sum(dim=(1, 2)).sqrt().max()
      for ii, idx in enumerate(reset_ids):
        newU = self.olpool.sample()
        if newU is None:
          reset_ids = reset_ids[:ii]
          break
        # re-scale to match size of current bases
        newU.mul_(Unorm/torch.norm(newU))
        self.Us.data[idx, :] = newU
        if self.affine:
          self.bs.data[idx, :] = 0.0
        self.c_mean[idx] = 1.0/self.k
    return reset_ids.cpu().numpy()


class KSubspaceModel(_KSubspaceBaseModel):
  """K-subspace model where low-dim coefficients are computed in closed
  form."""

  def __init__(self, k, d, D, affine=False, U_lamb=0.001, z_lamb=0.1,
        coh_gamma=0.0, coh_margin=0.75, soft_assign=0.0,
        assign_reg_terms=('z'), olpool=None, ol_reg_terms=('U', 'z'),
        ol_size_scale=True):
    if U_lamb < 0:
      raise ValueError("Invalid U reg parameter {}".format(U_lamb))
    if z_lamb < 0:
      raise ValueError("Invalid z reg parameter {}".format(z_lamb))
    if coh_gamma < 0:
      raise ValueError(("Invalid coherence reg "
          "parameter {}").format(coh_gamma))
    if coh_gamma > 0 and coh_margin < 0:
      raise ValueError(("Invalid coherence margin "
          "parameter {}").format(coh_margin))
    if not ('z' in assign_reg_terms or z_lamb == 0):
      raise ValueError("Assignment objective must contain z reg")

    super(KSubspaceModel, self).__init__(k, d, D, affine, soft_assign, olpool,
        ol_size_scale)
    self.U_lamb = U_lamb
    self.z_lamb = z_lamb
    self.coh_gamma = coh_gamma
    self.coh_margin = coh_margin
    self.assign_reg_terms = assign_reg_terms
    self.ol_reg_terms = ol_reg_terms
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
      z: latent low-dimensional codes (k, batch_size, d)
    """
    assert(x.dim() == 2 and x.size(1) == self.D)

    # shape (k x d x D)
    Uts = self.Us.data.transpose(1, 2)
    # (k x d x d)
    A = torch.matmul(Uts, self.Us.data)
    if self.z_lamb > 0:
      # (d x d)
      lambeye = torch.eye(self.d, dtype=A.dtype, device=A.device)
      # (1 x d x d)
      lambeye.mul_(self.z_lamb).unsqueeze_(0)
      # (k x d x d)
      A.add_(lambeye)

    # (1 x D x batch_size)
    B = x.data.t().unsqueeze(0)
    if self.affine:
      # bs shape (k, D)
      B = B.sub(self.bs.data.unsqueeze(2))
    # (k x d x batch_size)
    B = torch.matmul(Uts, B)

    # (k x d x batch_size)
    z, _ = torch.gesv(B, A)
    # (k x batch_size x d)
    z = z.transpose(1, 2)
    return z

  def reg(self):
    """Evaluate subspace regularization."""
    regs = dict()
    # (k,)
    if self.U_lamb > 0:
      regs['U'] = torch.sum(self.Us.pow(2), dim=(1, 2)).mul(self.U_lamb*0.5)

    # (batch_size, k)
    # does not affect gradients, only included to ensure objective value
    # is accurate
    if self.z_lamb > 0:
      regs['z'] = torch.sum(self.z.data.pow(2), dim=2).t().mul(self.z_lamb*0.5)

    if self.coh_gamma > 0:
      unitUs = ut.unit_normalize(self.Us, p=2, dim=1)
      # coherence (sum of squared cosine angles) between subspace bases,
      # normalized by "self-coherence".
      # (k, k)
      coh = torch.matmul(unitUs.transpose(1, 2).unsqueeze(1),
          unitUs.unsqueeze(0)).pow(2).sum(dim=(2, 3))
      coh = coh.div(coh.diag().view(-1, 1))
      # soft-threshold to incur no penalty if bases sufficiently incoherent
      if self.coh_margin > 0:
        coh = F.relu(coh - self.coh_margin)
      regs['coh'] = coh.sum(dim=1).sub(coh.diag()).mul(
          self.coh_gamma/(self.k-1))
    return regs


class KSubspaceProjModel(_KSubspaceBaseModel):
  """K-subspace model where low-dim coefficients are computed by a projection
  matrix."""

  def __init__(self, k, d, D, affine=False, symmetric=False, U_lamb=0.001,
        coh_gamma=0.0, coh_margin=0.75, soft_assign=0.0, assign_reg_terms=(),
        olpool=None, ol_reg_terms=('U',), ol_size_scale=True):
    if U_lamb < 0:
      raise ValueError("Invalid reg parameter {}".format(U_lamb))
    if coh_gamma < 0:
      raise ValueError(("Invalid coherence reg "
          "parameter {}").format(coh_gamma))
    if coh_gamma > 0 and coh_margin < 0:
      raise ValueError(("Invalid coherence margin "
          "parameter {}").format(coh_margin))

    super(KSubspaceProjModel, self).__init__(k, d, D, affine, soft_assign,
        olpool, ol_size_scale)
    self.symmetric = symmetric
    self.U_lamb = U_lamb
    self.coh_gamma = coh_gamma
    self.coh_margin = coh_margin
    self.assign_reg_terms = assign_reg_terms
    self.ol_reg_terms = ol_reg_terms

    if self.symmetric:
      self.register_parameter('Vs', None)
    else:
      self.Vs = nn.Parameter(torch.Tensor(k, D, d))
    self.reset_parameters()
    return

  def reset_parameters(self):
    """Initialize Us, Vs, bs with entries drawn from normal with std
    0.1/sqrt(D)."""
    std = .1 / np.sqrt(self.D)
    self.Us.data.normal_(0., std)
    if not self.symmetric:
      self.Vs.data.normal_(0., std)
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

    # z = U^T (x - b) or z = V (x - b)
    if self.affine:
      # shape (k, batch_size, D)
      x = x.sub(self.bs.unsqueeze(1))
    else:
      # shape (1, batch_size, D)
      x = x.unsqueeze(0)
    if self.symmetric:
      # shape (k, batch_size, d)
      z = torch.matmul(x, self.Us)
    else:
      z = torch.matmul(x, self.Vs)
    return z

  def reg(self):
    """Evaluate subspace regularization."""
    regs = dict()
    if self.U_lamb > 0:
      if self.symmetric:
        regs['U'] = torch.sum(self.Us.pow(2), dim=(1, 2)).mul(self.U_lamb)
      else:
        regs['U'] = (torch.sum(self.Us.pow(2), dim=(1, 2)) +
            torch.sum(self.Vs.pow(2), dim=(1, 2))).mul(self.U_lamb*0.5)

    if self.coh_gamma > 0:
      unitUs = ut.unit_normalize(self.Us, p=2, dim=1)
      # coherence (sum of squared cosine angles) between subspace bases,
      # normalized by "self-coherence".
      # (k, k)
      coh = torch.matmul(unitUs.transpose(1, 2).unsqueeze(1),
          unitUs.unsqueeze(0)).pow(2).sum(dim=(2, 3))
      coh = coh.div(coh.diag().view(-1, 1))
      # soft-threshold to incur no penalty if bases sufficiently incoherent
      if self.coh_margin > 0:
        coh = F.relu(coh - self.coh_margin)
      regs['coh'] = coh.sum(dim=1).sub(coh.diag()).mul(
          self.coh_gamma/(self.k-1))
    return regs


class OutlierPool(object):
  """Represent running pool of outier data points."""
  def __init__(self, maxlen, nbd_size, stale_thr=1.0, outlier_thr=0.0,
        cos_thr=None, ema_decay=0.9, sample_p=2, nbd_type='cos', svd=True):
    if stale_thr <= 0:
      raise ValueError("Invalid stale_thr {}".format(stale_thr))
    if cos_thr is not None and (cos_thr <= 0 or cos_thr > 1):
      raise ValueError("Invalid cos_thr {}".format(cos_thr))
    if ema_decay <= 0 or ema_decay >= 1:
      raise ValueError("Invalid ema_decay {}".format(ema_decay))
    if sample_p < 0:
      raise ValueError("Invalid sample_p {}".format(sample_p))
    if nbd_type not in ('cos', 'lasso'):
      raise ValueError("Invalid nbd_type {}".format(nbd_type))

    self.maxlen = maxlen
    self.nbd_size = nbd_size
    self.outlier_thr = outlier_thr
    self.stale_thr = stale_thr
    self.cos_thr = cos_thr
    self.ema_decay = ema_decay
    self.sample_p = sample_p
    self.nbd_type = nbd_type
    self.svd = svd
    self.reset()
    return

  def reset(self):
    self.error_avg = None
    self.error_var = None
    self.error_std = None
    self.errors = None
    self.error_avgs = None
    self.outliers = None
    return

  def append(self, x, errors):
    """Append outlier points in x to current pool, while clearing stale points
    and maintaining max size.

    NOTE: possibly naive implementation.
    """
    # initialize
    if self.outliers is None:
      self.outliers = torch.zeros((0, x.shape[1]),
          dtype=x.dtype, device=x.device)
      self.errors = torch.zeros((0,),
          dtype=errors.dtype, device=errors.device)
      self.error_avgs = torch.zeros((0,),
          dtype=errors.dtype, device=errors.device)

    # update error average and var
    batch_error_avg = errors.mean()
    batch_error_var = errors.var()
    if self.error_avg is None:
      self.error_avg = batch_error_avg
      self.error_var = batch_error_var
    else:
      self.error_avg = (self.ema_decay*self.error_avg +
          (1-self.ema_decay)*batch_error_avg)
      self.error_var = (self.ema_decay*self.error_var +
          (1-self.ema_decay)*batch_error_var)
    self.error_std = self.error_var.sqrt()

    # check for stale points, i.e. whose error avg at insertion time is
    # sufficiently large
    fresh_mask = (self.error_avgs - self.error_avg <
        self.stale_thr*self.error_std)
    if fresh_mask.sum() < fresh_mask.shape[0]:
      self.errors = self.errors[fresh_mask]
      self.error_avgs = self.error_avgs[fresh_mask]
      self.outliers = self.outliers[fresh_mask]

    # test for big errors
    if self.outlier_thr is not None:
      big_error_mask = (errors - self.error_avg >=
          self.outlier_thr*self.error_std)
      x = x[big_error_mask, :]
      errors = errors[big_error_mask]
    if x.shape[0] == 0:
      return 0

    outlier_count = self.outliers.shape[0]
    if outlier_count == self.maxlen:
      big_error_mask = errors > self.errors.min()
      x = x[big_error_mask, :]
      errors = errors[big_error_mask]
    if x.shape[0] == 0:
      return 0

    # test for excessive coherence (e.g. for when data points are repeated
    # across epochs)
    if outlier_count > 0 and self.cos_thr is not None and self.cos_thr < 1:
      unit_x = ut.unit_normalize(x, p=2, dim=1)
      unit_outliers = ut.unit_normalize(self.outliers, p=2, dim=1)
      cos = torch.matmul(unit_x, unit_outliers.t()).abs().max(dim=1)[0]
      cos_mask = cos < self.cos_thr
      x = x[cos_mask, :]
      errors = errors[cos_mask]
    if x.shape[0] == 0:
      return 0

    # update outliers by choosing worst from x + current pool.
    self.outliers = torch.cat((self.outliers, x))
    self.errors = torch.cat((self.errors, errors))
    self.error_avgs = torch.cat((self.error_avgs,
        torch.full_like(errors, self.error_avg)))
    Idx = torch.sort(self.errors, descending=True)[1]
    Idx = Idx[:self.maxlen]

    self.outliers = self.outliers[Idx]
    self.errors = self.errors[Idx]
    self.error_avgs = self.error_avgs[Idx]

    new_outlier_count = (Idx >= outlier_count).sum()
    return new_outlier_count

  def sample(self):
    """Sample a neighborhood of points from the current pool of outliers."""
    if self.outliers.shape[0] < self.nbd_size+1:
      return None

    # sample
    sample_prob = self.errors.pow(self.sample_p)
    idx = torch.multinomial(sample_prob, 1)

    # find nbd
    unit_outliers = ut.unit_normalize(self.outliers, p=2, dim=1)
    if self.nbd_type == 'cos':
      # (num_samples, num_outliers)
      cos = torch.matmul(unit_outliers[idx, :],
          unit_outliers.t()).abs().view(-1)
      _, nbd = torch.topk(cos, self.nbd_size, largest=True)
    else:
      # lasso nbd select
      # (D, num_outliers)
      X = unit_outliers.t()
      y = X[:, idx].view(-1)
      X[:, idx] = 0.0
      X, y = X.cpu().numpy(), y.cpu().numpy()

      # (num_outliers, num_alphas)
      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, coefs = lars_path(X, y, method='lasso')
      # choose largest alpha with sufficient support
      sprs = (np.abs(coefs) > 0).sum(axis=0)
      coef = coefs[:, (sprs == min(self.nbd_size, sprs[-1]))][:, 0]

      coef = torch.from_numpy(coef).to(device=unit_outliers.device)
      nbd = torch.arange(coef.shape[0])[coef.abs() > 0.0]

    # extract outlier neighborhood
    # (D, nbd_size)
    U = self.outliers[nbd, :].t()
    D, nbd_size = U.shape
    # add noise columns if neighborhood too small
    # this can happen if omp/lasso terminates early
    if nbd_size < self.nbd_size:
      Unorm = torch.norm(U, p=2, dim=0).min()
      nbdZ = torch.randn(D, self.nbd_size-nbd_size).mul_(
          .01*Unorm/np.sqrt(D))
      U = torch.cat((U, nbdZ), dim=1)
    if self.svd:
      U, s, _ = torch.svd(U, some=True)

    # drop nbd from current outliers
    not_nbd_mask = torch.ones_like(self.errors, dtype=torch.uint8)
    not_nbd_mask[nbd] = 0
    self.outliers = self.outliers[not_nbd_mask]
    self.errors = self.errors[not_nbd_mask]
    self.error_avgs = self.error_avgs[not_nbd_mask]
    return U
