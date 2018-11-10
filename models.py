from __future__ import print_function
from __future__ import division

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class KManifoldClusterModel(nn.Module):
  """Model of union of low-dimensional manifolds generalizing
  k-means/k-subspaces."""
  def __init__(self, n, d, D, N, batch_size, group_models=None):
    super(KManifoldClusterModel, self).__init__()

    self.n = n  # number of groups
    self.d = d  # dimension of manifold
    self.D = D  # ambient dimension
    self.N = N  # number of data points
    self.batch_size = batch_size

    if group_models:
      # quick check to make sure group models constructed correctly
      assert (len(group_models) == n and group_models[0].d == d and
          group_models[0].D == D)
    else:
      group_models = [SubspaceModel(d, D) for _ in range(n)]
    self.group_models = nn.ModuleList(group_models)

    # assignment and coefficient matrices, used with sparse embedding.
    self.register_buffer('C', torch.Tensor(N, n))
    self.register_buffer('V', torch.Tensor(N, d, n))
    self.c = nn.Parameter(torch.Tensor(batch_size, n))
    self.v = nn.Parameter(torch.Tensor(batch_size, d, n))
    self.reset_parameters()
    return

  def reset_parameters(self):
    """Initialize V with entries drawn from normal with std 0.1/sqrt(d), and C
    with entries from normal distribution, sigma=0.1."""
    V_std = .1 / np.sqrt(self.d)
    self.V.data.normal_(0., V_std)
    self.C.data.normal_(1., 0.01).abs_()
    self.C.data.div_(self.C.data.sum(dim=1, keepdim=True))
    return

  def forward(self, ii=None):
    """compute manifold embedding for ith data point(s)."""
    if ii is not None:
      self.set_cv(ii)
    # compute embeddings for each group and concatenate
    # NOTE: is this efficient on gpu/cpu?
    x_ = torch.stack([gm(self.v[:, :, jj])
        for jj, gm in enumerate(self.group_models)], dim=2)
    return x_

  def objective(self, x, ii=None, lamb_U=.01, lamb_V=None, wrt='all'):
    if wrt not in ['all', 'V', 'C', 'U']:
      raise ValueError(("Objective must be computed wrt 'all', "
          "'V', 'C', or 'U'"))
    lamb_V = lamb_U if lamb_V is None else lamb_V

    # evaluate loss: least-squares weighted by assignment, as in k-means.
    x_ = self(ii)
    loss = torch.sum((x.unsqueeze(2) - x_)**2, dim=1)
    if wrt in ['all', 'U']:
      loss = torch.mean(torch.sum(self.c*loss, dim=1))
    elif wrt == 'V':
      loss = torch.mean(torch.sum(loss, dim=1))
    # for wrt C, use Nb x n matrix of losses for computing closed form
    # assignment.

    # evaluate U regularizer
    if wrt in ['all', 'U']:
      Ureg = sum((gm.reg() for gm in self.group_models))
      # NOTE: division by N used to try to balance U & V regularizers
      Ureg = Ureg / self.N
    else:
      Ureg = 0.0

    # evaluate V l2 squared regularizer
    # NOTE: is there a strong need to have a U only case where Vreg is not
    # computed?
    if wrt in ['all', 'V', 'C']:
      Vreg = torch.sum(self.v**2, dim=1)
      if wrt == 'all':
        Vreg = torch.mean(torch.sum(self.c*Vreg, dim=1))
      elif wrt == 'V':
        Vreg = torch.mean(torch.sum(Vreg, dim=1))
      # for wrt C, use Nb x n matrix of V reg values for closed form
      # assignment.
    else:
      Vreg = 0.0

    reg = lamb_U*Ureg + lamb_V*Vreg
    obj = loss + reg
    return obj, loss, reg, Ureg, Vreg, x_

  def eval_sprs(self):
    """measure robust sparsity of current assignment subset c"""
    c = self.c.data
    cmax, _ = torch.max(c, dim=1, keepdim=True)
    sprs = torch.sum(c / cmax, dim=1)
    sprs = torch.mean(sprs)
    return sprs

  def eval_shrink(self, x, x_):
    """measure shrinkage of reconstruction wrt data"""
    c = self.c.data
    x = x.data
    x_ = x_.data
    norm_x_ = torch.sum(c*torch.norm(x_, 2, dim=1), dim=1)
    norm_x_ = torch.mean(norm_x_ / (torch.norm(x, 2, dim=1) + 1e-8))
    return norm_x_

  def set_cv(self, ii):
    """set c, v to reflect subset of C, V given by ii."""
    # NOTE: also tried sharing data using set_, but for some reason it didn't
    # work despite a restricted example working fine in console. Copying might
    # be safer anyway.
    self.c.data.copy_(self.C.data[ii, :])
    self.v.data.copy_(self.V.data[ii, :, :])
    return

  def set_CV(self, ii):
    """update full segmentation coefficients with values from current
    mini-batch."""
    self.C[ii, :] = self.c.data.clone()
    self.V[ii, :, :] = self.v.data.clone()
    return

  def get_groups(self):
    groups = torch.argmax(self.C.data.detach(), dim=1).numpy()
    return groups


class SubspaceModel(nn.Module):
  """Model of single low-dimensional affine or linear subspace."""

  def __init__(self, d, D, affine=False):
    super(SubspaceModel, self).__init__()

    self.d = d  # subspace dimension
    self.D = D  # ambient dimension
    self.affine = affine  # whether affine or linear subspaces.

    # construct subspace parameters and initialize
    # logic taken from pytorch Linear layer code
    self.U = nn.Parameter(torch.Tensor(D, d))
    if affine:
      self.b = nn.Parameter(torch.Tensor(D))
    else:
      self.register_parameter('b', None)
    self.reset_parameters()
    return

  def reset_parameters(self):
    """Initialize U, b with entries drawn from normal with std 1/sqrt(D)."""
    std = .1 / np.sqrt(self.D)
    self.U.data.normal_(0., std)
    if self.b is not None:
      self.b.data.normal_(0., std)
    return

  def forward(self, v):
    """Compute subspace embedding using coefficients v."""
    x = F.linear(v, self.U, self.b)
    return x

  def reg(self):
    """Compute L2 regularization on subspace basis."""
    reg = torch.sum(self.U**2)
    return reg


class ResidualManifoldModel(nn.Module):
  """Model of single low-dimensional manifold.

  Represented as affine subspace + nonlinear residual. Residual module is just
  a single hidden layer network with ReLU activation and dropout."""

  def __init__(self, d, D, H=None, drop_p=0.5, res_lamb=1.0):
    super(ResidualManifoldModel, self).__init__()

    self.d = d  # subspace dimension
    self.D = D  # ambient dimension
    self.H = H if H else D  # size of hidden layer
    self.drop_p = drop_p
    self.res_lamb = res_lamb  # residual weights regularization parameter

    self.subspace_embedding = SubspaceModel(d, D, affine=True)
    self.res_fc1 = nn.Linear(d, self.H, bias=False)
    self.res_fc2 = nn.Linear(self.H, D, bias=False)
    return

  def forward(self, v):
    """Compute residual manifold embedding using coefficients v."""
    x = self.subspace_embedding(v)
    z = F.relu(self.res_fc1(v))
    z = F.dropout(z, p=self.drop_p, training=self.training)
    z = self.res_fc2(z)
    x = x + z
    return x

  def reg(self):
    """Compute L2 squared regularization on subspace basis and residual
    module."""
    reg = torch.sum(self.subspace_embedding.U**2)
    # Intuition is for weights of residual module to be strongly regularized so
    # that residual is Lipschitz with a small constant, controlling the
    # curvature of the manifold.
    reg += self.res_lamb*(torch.sum(self.res_fc1.weight**2) +
        torch.sum(self.res_fc2.weight**2))
    return reg


class PowReLU(nn.Module):
  """Relu raised to a power p activation."""
  def __init__(self, p):
    super(PowReLU, self).__init__()
    self.p = p
    return

  def forward(self, x):
    x = F.relu(x)
    if self.p != 1:
      x = x**self.p
    return x
