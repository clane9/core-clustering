from __future__ import print_function
from __future__ import division

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils as ut

EPS = 1e-8
EMA_DECAY = 0.9


class _KSubspaceBaseModel(nn.Module):
  """Base K-subspace class."""

  def __init__(self, k, d, D, affine=False, soft_assign=0.0, c_sigma=0.0,
        size_scale=False):
    if soft_assign < 0:
      raise ValueError("Invalid soft-assign parameter {}".format(soft_assign))
    if c_sigma < 0:
      raise ValueError("Invalid assignment noise parameter {}".format(c_sigma))

    super(_KSubspaceBaseModel, self).__init__()
    self.k = k  # number of groups
    self.d = d  # dimension of manifold
    self.D = D  # number of data points
    self.affine = affine
    self.soft_assign = soft_assign
    self.c_sigma = c_sigma
    self.size_scale = size_scale
    self.assign_reg_terms = None  # assigned in child

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

  def objective(self, x):
    """Evaluate objective function.

    Input:
      x: data, shape (batch_size, D)

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

    # only justified as lr scaling when loss + all reg terms weighted by c_ij
    # init enforces this
    if self.size_scale:
      scale_obj = (losses.div(self.c_mean + EPS).sum() +
          regs.div(self.c_mean + EPS).sum())
    else:
      scale_obj = obj
    return obj, scale_obj, loss, reg, x_

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

    if self.c_sigma > 0:
      c_z = torch.zeros_like(self.c)
      c_z.normal_(mean=0, std=(self.c_sigma/self.k)).abs_()
      self.c.add_(c_z)
      self.c.div_(self.c.sum(dim=1, keepdim=True))

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

  def reset_unused(self, split_metric=None, sample_p=None, reset_thr=.01,
        split_thr=0.0, split_sigma=.1):
    """Reset (nearly) unused clusters by duplicating clusters likely to contain
    >1 group. By default, choose to duplicate largest clusters.

    Args:
      split_metric: shape (k,) metric used to rank groups for splitting
        (largest values chosen) (default: c_mean).
      sample_p: probability of splitting group j is proportional to
        split_metric[j]**sample_p. Default chosen so that a cluster twice as
        "bad" will be selected with at least 50% prob (default: ceil(log2(k))).
      reset_thr: size threshold for nearly empty clusters in 1/k units
        (default: .01).
      split_thr: threshold relative to median split metric a cluster must
        surpass to be considered for splitting (default: 0.0).
      split_sigma: noise added to duplicate clusters, relative to basis column
        norms (default: .1).

    Returns:
      reset_ids, split_ids
    """
    if split_metric is None:
      split_metric = self.c_mean
    if sample_p is None:
      sample_p = np.log2(self.k)
    if sample_p < 0:
      raise ValueError("Invalid sample_p {}".format(sample_p))
    sample_p = int(np.ceil(sample_p))
    if split_metric.shape != (self.k,) or split_metric.min() < 0:
      raise ValueError("Invalid splitting metric")
    if sample_p < 0:
      raise ValueError("Invalid sample power {}".format(sample_p))
    if reset_thr < 0:
      raise ValueError("Invalid reset threshold {}".format(reset_thr))
    if split_thr < 0:
      raise ValueError("Invalid split threshold {}".format(split_thr))
    if split_sigma <= 0:
      raise ValueError("Invalid split noise level {}".format(split_sigma))

    # expected value of assignment noise (folded gaussian) is sigma*sqrt(2/pi)
    reset_thr = ((self.c_sigma/self.k)*np.sqrt(2/np.pi) + reset_thr/self.k)
    reset_mask = self.c_mean <= reset_thr
    reset_ids = reset_mask.nonzero().view(-1)
    reset_count = reset_ids.size(0)

    split_thr = split_thr*split_metric[reset_mask == 0].median()
    split_cand_mask = split_metric >= split_thr
    split_cand_mask[reset_ids] = 0
    split_cands = split_cand_mask.nonzero().view(-1)
    split_cand_count = split_cands.size(0)

    reset_count = min(reset_count, split_cand_count)
    reset_ids = reset_ids[:reset_count]
    if reset_count > 0:
      split_prob = split_metric[split_cands].pow(sample_p)
      split_ids = split_cands[torch.multinomial(split_prob, reset_count)]

      # "split" clusters by duplicating bases with small perturbation
      split_Us = self.Us.data[split_ids, :]
      w = torch.randn_like(split_Us).mul_(split_sigma/np.sqrt(self.D))
      w.mul_(torch.norm(split_Us, dim=1, keepdim=True))
      self.Us.data[reset_ids, :] = split_Us + w
      self.Us.data[split_ids, :] = split_Us - w
      if self.affine:
        self.bs[reset_ids, :] = self.bs[split_ids, :]

      # important to update c_mean for split clusters when using size
      # scaled objective
      self.c_mean[split_ids] *= 0.5
      self.c_mean[reset_ids] = self.c_mean[split_ids]

      idx = torch.sort(split_metric, descending=True)[1]
      split_ranks = torch.sort(idx)[1][split_ids]
    else:
      split_ids = torch.zeros_like(reset_ids)
      split_ranks = torch.zeros_like(reset_ids)

    return (reset_ids.cpu().numpy(), split_ids.cpu().numpy(),
        split_ranks.cpu().numpy())


class KSubspaceModel(_KSubspaceBaseModel):
  """K-subspace model where low-dim coefficients are computed in closed
  form."""

  def __init__(self, k, d, D, affine=False, U_lamb=0.001, z_lamb=0.1,
        coh_gamma=0.0, coh_margin=0.75, soft_assign=0.0, c_sigma=0.0,
        assign_reg_terms=('U', 'z'), size_scale=False):
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
    if size_scale and not (('U' in assign_reg_terms or U_lamb == 0) and
          ('coh' in assign_reg_terms or coh_gamma == 0)):
      raise ValueError("Size scaled objective is only valid when "
          "assignment objective includes all reg terms (U, z, coh)")

    super(KSubspaceModel, self).__init__(k, d, D, affine, soft_assign, c_sigma,
        size_scale)
    self.U_lamb = U_lamb
    self.z_lamb = z_lamb
    self.coh_gamma = coh_gamma
    self.coh_margin = coh_margin
    self.assign_reg_terms = assign_reg_terms
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
      unitUs = self.Us.div(
          torch.norm(self.Us, p=2, dim=1, keepdim=True).add(EPS))
      # coherence (sum of squared cosine angles) between subspace bases,
      # normalized by "self-coherence".
      # (k, k)
      coh = torch.matmul(unitUs.transpose(1, 2).unsqueeze(1),
          unitUs.unsqueeze(0)).pow(2).sum(dim=(2, 3))
      coh = coh.div(coh.diag().view(-1, 1))
      # soft-threshold to incur no penalty if bases sufficiently incoherent
      coh = F.relu(coh - self.coh_margin)
      regs['coh'] = coh.sum(dim=1).sub(coh.diag()).mul(
          self.coh_gamma/(self.k-1))
    return regs


class KSubspaceProjModel(_KSubspaceBaseModel):
  """K-subspace model where low-dim coefficients are computed by a projection
  matrix."""

  def __init__(self, k, d, D, affine=False, symmetric=False, U_lamb=0.001,
        coh_gamma=0.0, coh_margin=0.75, soft_assign=0.0, c_sigma=0.0,
        assign_reg_terms=('U'), size_scale=False):
    if U_lamb < 0:
      raise ValueError("Invalid reg parameter {}".format(U_lamb))
    if coh_gamma < 0:
      raise ValueError(("Invalid coherence reg "
          "parameter {}").format(coh_gamma))
    if coh_gamma > 0 and coh_margin < 0:
      raise ValueError(("Invalid coherence margin "
          "parameter {}").format(coh_margin))
    if size_scale and not (('U' in assign_reg_terms or U_lamb == 0) and
          ('coh' in assign_reg_terms or coh_gamma == 0)):
      raise ValueError("Size scaled objective is only valid when "
          "assignment objective includes all reg terms (U, coh)")

    super(KSubspaceProjModel, self).__init__(k, d, D, affine, soft_assign,
        c_sigma, size_scale)
    self.symmetric = symmetric
    self.U_lamb = U_lamb
    self.coh_gamma = coh_gamma
    self.coh_margin = coh_margin
    self.assign_reg_terms = assign_reg_terms

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
      unitUs = self.Us.div(
          torch.norm(self.Us, p=2, dim=1, keepdim=True).add(EPS))
      # coherence (sum of squared cosine angles) between subspace bases,
      # normalized by "self-coherence".
      # (k, k)
      coh = torch.matmul(unitUs.transpose(1, 2).unsqueeze(1),
          unitUs.unsqueeze(0)).pow(2).sum(dim=(2, 3))
      coh = coh.div(coh.diag().view(-1, 1))
      # soft-threshold to incur no penalty if bases sufficiently incoherent
      coh = F.relu(coh - self.coh_margin)
      regs['coh'] = coh.sum(dim=1).sub(coh.diag()).mul(
          self.coh_gamma/(self.k-1))
    return regs
