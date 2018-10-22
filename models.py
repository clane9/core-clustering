from __future__ import print_function
from __future__ import division

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ManifoldClusterModel(nn.Module):
  """Model of union of low-dimensional manifolds."""
  def __init__(self, n, d, D, N, C_p=1, C_sigma=0.,
        group_models=None):
    super(ManifoldClusterModel, self).__init__()

    self.n = n  # number of groups
    self.d = d  # dimension of manifold
    self.D = D  # ambient dimension
    self.N = N  # number of data points

    # assignment "activation" function
    self.C_p = C_p
    self.C_activ = PowReLU(C_p)
    # assignment noise
    self.C_sigma = C_sigma
    self.register_buffer('C_noise', Variable(torch.Tensor(n)))

    if group_models:
      # quick check to make sure group models constructed correctly
      assert (len(group_models) == n and group_models[0].d == d and
          group_models[0].D == D)
      self.group_models = group_models
    else:
      group_models = [SubspaceModel(d, D) for _ in range(n)]
    # add group models as sub-modules
    for ii in range(n):
      self.add_module('gm{}'.format(ii), group_models[ii])

    # assignment and coefficient matrices, used with sparse embedding.
    self.C = nn.Parameter(torch.Tensor(N, n))
    self.V = nn.Parameter(torch.Tensor(N, d))
    self.reset_parameters()
    return

  def reset_parameters(self):
    """Initialize V with entries drawn from normal with std 1/sqrt(d), and C
    with entries uniform in [0, 1]."""
    V_std = 1. / np.sqrt(self.d)
    self.V.data.normal_(0., V_std)
    self.C.data.uniform_(0., 1.)
    return

  def forward(self, ii):
    """compute manifold embedding for ith data point(s)."""
    # find cluster assignment and manifold coefficients for current data points
    c = F.embedding(ii, self.C)
    v = F.embedding(ii, self.V)

    # assignment noise
    if self.training and self.C_sigma > 0:
      self.C_noise.normal_(0., self.C_sigma)
      c += torch.abs(self.C_noise)
    # assignment "activation"
    c = self.C_activ(c)

    # compute embeddings for each group
    X = torch.stack([gm(v) for gm in self.group_models], dim=2)
    X = X*c
    x = torch.sum(X, 2)
    return x

  def reg(self):
    Ureg = torch.sum(torch.stack([gm.reg() for gm in self.group_models]))
    Vreg = 0.5*torch.sum(self.V**2)
    Creg = torch.sum(torch.abs(self.C))
    return Ureg, Vreg, Creg

  def get_groups(self):
    groups = torch.argmax(self.C, 1)
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
    std = 1. / np.sqrt(self.D)
    self.U.data.normal_(0., std)
    if self.b is not None:
      self.b.data.normal_(0., std)
    return

  def forward(self, v):
    """Compute subspace embedding using coefficients v."""
    x = F.linear(v, self.U, self.b)
    return x

  def reg(self):
    """Compute L2 squared regularization on subspace basis."""
    reg = 0.5*torch.sum(self.U**2)
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
