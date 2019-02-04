from __future__ import print_function
from __future__ import division

import numpy as np
import torch
import torch.nn as nn

import utils as ut

EPS = 1e-8
EMA_DECAY = 0.9


class _KSubspaceBaseModel(nn.Module):
  """Base K-subspace class."""

  def __init__(self, k, d, D, affine=False, soft_assign=0.1, c_sigma=.01,
        size_scale=False):
    if soft_assign < 0:
      raise ValueError("Invalid soft-assign parameter {}".format(soft_assign))
    if c_sigma <= 0:
      raise ValueError("Assignment noise c_sigma > 0 is required")

    super(_KSubspaceBaseModel, self).__init__()
    self.k = k  # number of groups
    self.d = d  # dimension of manifold
    self.D = D  # number of data points
    self.affine = affine
    self.soft_assign = soft_assign
    self.c_sigma = c_sigma
    self.size_scale = size_scale

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
      z: latent code, shape (k, batch_size, d)

    Returns:
      x_: reconstruction, shape (k, batch_size, D)
    """
    raise NotImplementedError("decode not implemented")
    return

  def objective(self, x):
    """Evaluate objective function.

    Input:
      x: data, shape (batch_size, D)

    Returns:
      obj, scale_obj, loss, reg: objective, loss, regularization value
      x_: reconstruction, shape (k, batch_size, D)
    """
    x_ = self(x)
    loss = self.loss(x, x_)
    reg = self.reg()

    # update assignment c
    assign_obj = loss.detach() + reg.detach()
    self.set_assign(assign_obj)

    # shape (k,)
    loss = torch.mean(self.c*loss, dim=0)
    reg = torch.mean(self.c*reg, dim=0)

    if self.size_scale:
      scale_loss = loss.div(self.c_mean + EPS)
      scale_reg = reg.div(self.c_mean + EPS)

    loss = loss.sum()
    reg = reg.sum()
    obj = loss + reg

    if self.size_scale:
      scale_obj = (scale_loss + scale_reg).sum()
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

    batch_c_mean = torch.mean(self.c, dim=0)
    self.c_mean.mul_(EMA_DECAY).add_(1-EMA_DECAY, batch_c_mean)
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

  def eval_rank(self, tol=.001):
    """Compute rank and singular values of subspace bases."""
    ranks_svs = [ut.rank(self.Us.data[ii, :, :], tol) for ii in range(self.k)]
    ranks, svs = zip(*ranks_svs)
    return ranks, svs


class KSubspaceModel(_KSubspaceBaseModel):
  """K-subspace model where low-dim coefficients are computed in closed
  form."""

  def __init__(self, k, d, D, affine=False, U_lamb=0.001, z_lamb=0.1,
        soft_assign=0.1, c_sigma=.01, size_scale=False):
    if U_lamb < 0:
      raise ValueError("Invalid U reg parameter {}".format(U_lamb))
    if z_lamb < 0:
      raise ValueError("Invalid z reg parameter {}".format(z_lamb))

    super(KSubspaceModel, self).__init__(k, d, D, affine, soft_assign, c_sigma,
        size_scale)
    self.U_lamb = U_lamb
    self.z_lamb = z_lamb

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
    x_ = torch.matmul(self.Us, z.transpose(1, 2)).transpose(1, 2)
    if self.affine:
      x_ = x_.add(self.bs.unsqueeze(1))
    return x_

  def reg(self):
    """Evaluate subspace regularization."""
    # (k,)
    U_reg = torch.sum(self.Us.pow(2), dim=(1, 2)).mul(0.5)
    # (batch_size, k)
    z_reg = torch.sum(self.z.data.pow(2), dim=2).mul(0.5).t()
    reg = self.U_lamb*U_reg + self.z_lamb*z_reg
    return reg


class KSubspaceProjModel(_KSubspaceBaseModel):
  """K-subspace model where low-dim coefficients are computed by a projection
  matrix."""

  def __init__(self, k, d, D, affine=False, symmetric=False, lamb=0.001,
        soft_assign=0.1, c_sigma=.01, size_scale=False):
    if lamb < 0:
      raise ValueError("Invalid reg parameter {}".format(lamb))

    super(KSubspaceProjModel, self).__init__(k, d, D, affine, soft_assign,
        c_sigma, size_scale)
    self.symmetric = symmetric
    self.lamb = lamb

    if self.symmetric:
      self.register_parameter('Vs', None)
    else:
      self.Vs = nn.Parameter(torch.Tensor(k, d, D))
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
      z = torch.matmul(self.Us.transpose(1, 2),
          x.transpose(1, 2)).transpose(1, 2)
    else:
      z = torch.matmul(self.Vs, x.transpose(1, 2)).transpose(1, 2)
    return z

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
    x_ = torch.matmul(self.Us, z.transpose(1, 2)).transpose(1, 2)
    if self.affine:
      x_ = x_.add(self.bs.unsqueeze(1))
    return x_

  def reg(self):
    """Evaluate subspace regularization."""
    if self.symmetric:
      reg = torch.sum(self.Us.pow(2), dim=(1, 2))
    else:
      reg = (torch.sum(self.Us.pow(2), dim=(1, 2)) +
          torch.sum(self.Vs.pow(2), dim=(1, 2))).mul(0.5)
    reg *= self.lamb
    return reg
