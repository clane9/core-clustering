from __future__ import print_function
from __future__ import division

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class GSManifoldClusterModel(nn.Module):
  """Model of union of low-dimensional manifolds."""
  def __init__(self, n, d, D, N, group_models=None):
    super(GSManifoldClusterModel, self).__init__()

    self.n = n  # number of groups
    self.d = d  # dimension of manifold
    self.D = D  # ambient dimension
    self.N = N  # number of data points

    if group_models:
      # quick check to make sure group models constructed correctly
      assert (len(group_models) == n and group_models[0].d == d and
          group_models[0].D == D)
    else:
      group_models = [SubspaceModel(d, D) for _ in range(n)]
    self.group_models = nn.ModuleList(group_models)

    # coefficient matrices, used with sparse embedding.
    self.V = nn.Parameter(torch.Tensor(N, n, d))
    self.reset_parameters()
    return

  def reset_parameters(self):
    """Initialize V with entries drawn from normal with std 1/sqrt(d)."""
    V_std = .1 / np.sqrt(self.d)
    self.V.data.normal_(0., V_std)
    return

  def forward(self, ii):
    """compute manifold embedding for ith data point(s)."""
    # compute embeddings for each group and sum
    # NOTE: is this efficient on gpu/cpu?
    x = sum((gm(F.embedding(ii, self.V[:, jj, :])) for jj, gm in
        enumerate(self.group_models)))
    return x

  def reg(self, ii):
    """compute regularization on model parameters and representation
    coefficients."""
    # NOTE: switched reg form from product to sum with squared l2 on U
    # this will change meaning of reg parameters used in tests on 10/22/18.
    Ureg = sum((gm.reg() for gm in self.group_models))
    # NOTE: added division by N on afternoon of 10/23, to try to address
    # regularization imbalance issue.
    Ureg = Ureg / self.N
    # group sparsity norm on V: summing l2 norms of each of n coefficient
    # vectors (size d).
    Vreg = sum((torch.norm(F.embedding(ii, self.V[:, jj, :]), 2, dim=1)
        for jj in range(self.n)))
    Vreg = torch.mean(Vreg)
    # reg = Ureg + Vreg
    # Ureg = [gm.reg() for gm in self.group_models]
    # NOTE: was torch.sum during first test of batch size on 10/22/18
    # Vreg = [torch.mean(torch.norm(F.embedding(ii, self.V[:,jj,:]), 2, dim=1))
    #     for jj in range(self.n)]
    # reg = sum((ureg*vreg for ureg, vreg in zip(Ureg, Vreg)))
    return Ureg, Vreg

  def get_groups(self):
    """return group assignment."""
    C = torch.norm(self.V.detach(), 2, dim=2)
    groups = torch.argmax(C, 1).numpy()
    return groups


class ManifoldClusterModel(nn.Module):
  """Model of union of low-dimensional manifolds."""
  def __init__(self, n, d, D, N, C_sigma=0.,
        group_models=None):
    super(ManifoldClusterModel, self).__init__()

    self.n = n  # number of groups
    self.d = d  # dimension of manifold
    self.D = D  # ambient dimension
    self.N = N  # number of data points

    if group_models:
      # quick check to make sure group models constructed correctly
      assert (len(group_models) == n and group_models[0].d == d and
          group_models[0].D == D)
    else:
      group_models = [SubspaceModel(d, D) for _ in range(n)]
    self.group_models = nn.ModuleList(group_models)

    # assignment and coefficient matrices, used with sparse embedding.
    self.C = nn.Parameter(torch.Tensor(N, n))
    self.V = nn.Parameter(torch.Tensor(N, n, d))
    self.reset_parameters()
    return

  def reset_parameters(self):
    """Initialize V with entries drawn from normal with std 1/sqrt(d), and C
    with entries uniform in [0, 1]."""
    V_std = .1 / np.sqrt(self.d)
    self.V.data.normal_(0., V_std)
    C_mean = 1. / self.n
    C_std = 0.1 / self.n
    self.C.data.uniform_(C_mean - C_std, C_mean + C_std)
    return

  def forward(self, ii):
    """compute manifold embedding for ith data point(s)."""
    # find cluster assignment and compute "activation"
    c = F.embedding(ii, self.C)
    c = torch.abs(c)
    # compute embeddings for each group and weighted sum
    # NOTE: is this efficient on gpu/cpu?
    x = sum((c[:, jj].view(-1, 1)*gm(F.embedding(ii, self.V[:, jj, :]))
        for jj, gm in enumerate(self.group_models)))
    return x

  def reg(self, ii):
    """compute regularization on model parameters and representation
    coefficients."""
    Ureg = sum((gm.reg() for gm in self.group_models))
    Ureg = Ureg / self.N
    # mean l2 squared regularization on V
    Vreg = 0.5*sum((torch.mean(
        torch.sum(F.embedding(ii, self.V[:, jj, :])**2, dim=1))
        for jj in range(self.n)))
    # mean l1 regularization on C
    c = F.embedding(ii, self.C)
    c = torch.abs(c)
    Creg = torch.mean(torch.sum(c, dim=1))
    return Ureg, Vreg, Creg

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
    reg = 0.5*torch.sum(self.U**2)
    # reg = torch.norm(self.U)
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
    self.res_fc1 = nn.Linear(D, self.H, bias=True)
    self.res_fc2 = nn.Linear(self.H, D, bias=False)
    return

  def forward(self, v):
    """Compute residual manifold embedding using coefficients v."""
    x = self.subspace_embedding(v)
    # NOTE: residual computed on x, but perhaps it should be computed on v?
    # Shouldn't make a difference since both are d-dimensional.
    z = F.relu(self.res_fc1(x))
    z = F.dropout(z, p=self.drop_p, training=self.training)
    z = self.fc2(z)
    x = x + z
    return x

  def reg(self):
    """Compute L2 squared regularization on subspace basis and residual
    module."""
    reg = 0.5*torch.sum(self.subspace_embedding.U**2)
    # Intuition is for weights of residual module to be strongly regularized so
    # that residual is Lipschitz with a small constant, controlling the
    # curvature of the manifold.
    reg += self.res_lamb*0.5*(torch.sum(self.res_fc1.weight**2) +
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
