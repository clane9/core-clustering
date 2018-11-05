from __future__ import print_function
from __future__ import division

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

# NOTE: sparse gradients are necessary to avoid O(N) cost per iteration
# (absolutely necessary for good performance!!). But currently they cause
# failure during backward.
EMBED_SPARSE = False


class GSManifoldClusterModel(nn.Module):
  """Model of union of low-dimensional manifolds with segmentation coded by
  group sparsity pattern."""
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
    self.V = nn.Parameter(torch.Tensor(N, d, n))
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
    v = torch.stack([F.embedding(ii, self.V[:, :, jj], sparse=EMBED_SPARSE)
        for jj in range(self.n)], dim=2)
    x_ = sum((gm(v[:, :, jj]) for jj, gm in enumerate(self.group_models)))
    return x_, v

  def objective(self, ii, x, U_lamb, V_lamb):
    """evaluate objective, loss, regularizer, and auxiliary measures."""
    x_, v = self(ii)

    # evaluate least-squares loss
    loss = torch.mean(torch.sum((x - x_)**2, dim=1))

    # evaluate regularizer
    Ureg = sum((gm.reg() for gm in self.group_models))
    # NOTE: division by N used to try to balance U & V regularizers
    Ureg = Ureg / self.N
    Vnorms = torch.norm(v, 2, dim=1)
    Vreg = torch.sum(Vnorms, dim=1)
    Vreg = torch.mean(Vreg)

    # compute objective
    reg = U_lamb*Ureg + V_lamb*Vreg
    obj = loss + reg

    # measure robust sparsity
    Vnorms = Vnorms.detach()
    Vmaxnorm, _ = torch.max(Vnorms, dim=1, keepdim=True)
    sprs = torch.sum(Vnorms / Vmaxnorm, dim=1)
    sprs = torch.mean(sprs)

    # also whether reconstruction x_ is close in size to x.
    x_ = x_.detach()
    x = x.detach()
    norm_x_ = torch.mean(torch.norm(x_, 2, dim=1) /
        (torch.norm(x, 2, dim=1) + 1e-8))
    return obj, loss, reg, Ureg, Vreg, sprs, norm_x_

  def get_groups(self):
    """return group assignment."""
    C = torch.norm(self.V.detach(), 2, dim=2)
    groups = torch.argmax(C, 1).numpy()
    return groups


class SegManifoldClusterModel(nn.Module):
  """Model of union of low-dimensional manifolds with explicit segmentation."""
  def __init__(self, n, d, D, N, group_models=None):
    super(SegManifoldClusterModel, self).__init__()

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
    self.V = nn.Parameter(torch.Tensor(N, d, n))
    self.reset_parameters()
    return

  def reset_parameters(self):
    """Initialize V with entries drawn from normal with std 0.1/sqrt(d), and C
    with entries from normal distribution, sigma=0.1."""
    V_std = .1 / np.sqrt(self.d)
    self.V.data.normal_(0., V_std)
    self.C.data.normal_(0., 0.1)
    return

  def forward(self, ii):
    """compute manifold embedding for ith data point(s)."""
    # find cluster assignment and compute softmax "activation"
    c = F.embedding(ii, self.C, sparse=EMBED_SPARSE)
    c = self.segactiv(c)
    # compute embeddings for each group and weighted sum
    # NOTE: is this efficient on gpu/cpu?
    v = torch.stack([F.embedding(ii, self.V[:, :, jj], sparse=EMBED_SPARSE)
        for jj in range(self.n)], dim=2)
    x_ = sum((c[:, jj].view(-1, 1)*gm(v[:, :, jj]) for jj, gm in
        enumerate(self.group_models)))
    return x_, c, v

  def segactiv(self, c):
    """compute cluster assignment "activation"."""
    c = F.softmax(c, dim=1)
    return c

  def objective(self, ii, x, U_lamb, V_lamb):
    """evaluate objective, loss, regularizer, and auxiliary measures."""
    x_, c, v = self(ii)

    # evaluate least-squares loss
    loss = torch.mean(torch.sum((x - x_)**2, dim=1))

    # evaluate regularizer
    Ureg = sum((gm.reg() for gm in self.group_models))
    # NOTE: division by N used to try to balance U & V regularizers
    Ureg = Ureg / self.N
    Vreg = 0.5*torch.sum(v.view(-1, self.d*self.n)**2, dim=1)
    Vreg = torch.mean(Vreg)

    # compute objective
    reg = U_lamb*Ureg + V_lamb*Vreg
    obj = loss + reg

    # measure robust sparsity
    c = c.detach()
    cmax, _ = torch.max(c, dim=1, keepdim=True)
    sprs = torch.sum(c / cmax, dim=1)
    sprs = torch.mean(sprs)

    # measure also whether reconstruction x_ is close in size to x.
    x_ = x_.detach()
    x = x.detach()
    norm_x_ = torch.mean(torch.norm(x_, 2, dim=1) /
        (torch.norm(x, 2, dim=1) + 1e-8))
    return obj, loss, reg, Ureg, Vreg, sprs, norm_x_

  def get_groups(self):
    groups = torch.argmax(self.C.data.detach(), dim=1).numpy()
    return groups


class KManifoldClusterModel(nn.Module):
  """Model of union of low-dimensional manifolds generalizing
  k-means/k-subspaces."""
  def __init__(self, n, d, D, N, group_models=None, C_sigma=0.0):
    super(KManifoldClusterModel, self).__init__()

    self.n = n  # number of groups
    self.d = d  # dimension of manifold
    self.D = D  # ambient dimension
    self.N = N  # number of data points

    self.C_sigma = C_sigma

    if group_models:
      # quick check to make sure group models constructed correctly
      assert (len(group_models) == n and group_models[0].d == d and
          group_models[0].D == D)
    else:
      group_models = [SubspaceModel(d, D) for _ in range(n)]
    self.group_models = nn.ModuleList(group_models)

    # assignment and coefficient matrices, used with sparse embedding.
    self.C = nn.Parameter(torch.Tensor(N, n))
    self.V = nn.Parameter(torch.Tensor(N, d, n))
    self.reset_parameters()
    return

  def reset_parameters(self):
    """Initialize V with entries drawn from normal with std 0.1/sqrt(d), and C
    with entries from normal distribution, sigma=0.1."""
    V_std = .1 / np.sqrt(self.d)
    self.V.data.normal_(0., V_std)
    # self.C.data.normal_(0., 0.1)
    self.C.data.normal_(1., 0.01)
    return

  def forward(self, ii):
    """compute manifold embedding for ith data point(s)."""
    # find cluster assignment and compute "activation"
    c = F.embedding(ii, self.C, sparse=EMBED_SPARSE)
    c = self.segactiv(c)
    # compute embeddings for each group and concatenate
    # NOTE: is this efficient on gpu/cpu?
    v = torch.stack([F.embedding(ii, self.V[:, :, jj], sparse=EMBED_SPARSE)
        for jj in range(self.n)], dim=2)
    x_ = torch.stack([gm(v[:, :, jj])
        for jj, gm in enumerate(self.group_models)], dim=2)
    return x_, c, v

  def segactiv(self, c):
    """compute cluster assignment "activation"."""
    # c = F.softmax(c, dim=1)
    # NOTE: no more in place operations, might create problems for backward
    if self.training and self.C_sigma > 0:
      cmax, _ = torch.max(c.detach(), dim=1, keepdim=True)
      c = c + self.C_sigma*(torch.randn(*c.shape)*cmax)
    c = c**2
    csum = torch.sum(c, dim=1, keepdim=True)
    c = c / (csum + 1e-8)
    c = torch.clamp(c, 0.1/self.n, 1.0)
    return c

  def objective(self, ii, x, U_lamb, V_lamb):
    """evaluate objective, loss, regularizer, and auxiliary measures."""
    x_, c, v = self(ii)

    # evaluate loss: least-squares weighted by assignment, as in k-means.
    loss = torch.sum((x.view(-1, self.D, 1) - x_)**2, dim=1)
    loss = torch.mean(torch.sum(c*loss, dim=1))

    # evaluate reg: l2 squared, weighted by assignment in case of V.
    Ureg = sum((gm.reg() for gm in self.group_models))
    # NOTE: division by N used to try to balance U & V regularizers
    Ureg = Ureg / self.N
    Vreg = 0.5*torch.sum(c*torch.sum(v**2, dim=1), dim=1)
    Vreg = torch.mean(Vreg)

    # compute objective
    reg = U_lamb*Ureg + V_lamb*Vreg
    obj = loss + reg

    # measure robust sparsity
    c = c.detach()
    cmax, _ = torch.max(c, dim=1, keepdim=True)
    sprs = torch.sum(c / cmax, dim=1)
    sprs = torch.mean(sprs)

    # and also whether reconstruction x_ is close in size to x.
    x = x.detach()
    x_ = x_.detach()
    norm_x_ = torch.sum(c*torch.norm(x_, 2, dim=1), dim=1)
    norm_x_ = torch.mean(norm_x_ / (torch.norm(x, 2, dim=1) + 1e-8))
    return obj, loss, reg, Ureg, Vreg, sprs, norm_x_

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
