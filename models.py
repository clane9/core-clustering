from __future__ import print_function
from __future__ import division

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class KClusterModel(nn.Module):
  """Base class for k-cluster model."""

  def __init__(self, n, d, D, N, batch_size, group_models):
    super(KClusterModel, self).__init__()

    self.n = n  # number of groups
    self.d = d  # dimension of manifold
    self.D = D  # ambient dimension
    self.N = N  # number of data points
    self.batch_size = batch_size

    # quick check to make sure group models constructed correctly
    group_model_check = (len(group_models) == n and
        isinstance(group_models[0], GroupModel) and
        group_models[0].d == d and
        group_models[0].D == D)
    if not group_model_check:
      raise ValueError("Invalid sequence of group_models")
    self.group_models = nn.ModuleList(group_models)

    # NOTE: might not need to have fixed batch size in this case
    self.c = nn.Parameter(torch.Tensor(batch_size, n))
    return

  def reset_parameters(self):
    """Initialize C with entries from normal distribution centered on 1,
    sigma=0.01."""
    self.c.data.normal_(1., 0.01).abs_()
    self.c.data.div_(self.c.data.sum(dim=1, keepdim=True))
    return

  def forward(self, *args, **kwargs):
    raise NotImplementedError("forward not implemented")

  def objective(self, *args, **kwargs):
    raise NotImplementedError("objective not implemented")

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

  def get_groups(self):
    """compute group assignment."""
    groups = torch.argmax(self.c.data, dim=1).cpu().numpy()
    return groups


class KManifoldClusterModel(KClusterModel):
  """Model of union of low-dimensional manifolds generalizing
  k-means/k-subspaces."""

  def __init__(self, n, d, D, N, batch_size, group_models=None,
        use_cuda=False):
    if group_models is None:
      group_models = [SubspaceModel(d, D) for _ in range(n)]

    super(KManifoldClusterModel, self).__init__(n, d, D, N, batch_size,
        group_models)

    # assignment and coefficient matrices, used with sparse embedding.
    # NOTE: don't want these potentially huge variables ever sent to gpu
    self.C = torch.Tensor(N, n)
    self.V = torch.Tensor(N, d, n)
    if use_cuda:
      # should speed up copy to cuda memory
      self.C = self.C.pin_memory()
      self.V = self.V.pin_memory()
    self.use_cuda = use_cuda
    self.v = nn.Parameter(torch.Tensor(batch_size, d, n))
    self.reset_parameters()
    return

  def reset_parameters(self):
    """Initialize V with entries drawn from normal with std 0.1/sqrt(d), and C
    with entries from normal distribution, mean=1, sigma=0.01."""
    V_std = .1 / np.sqrt(self.d)
    self.V.data.normal_(0., V_std)
    self.C.data.normal_(1., 0.01).abs_()
    self.C.data.div_(self.C.data.sum(dim=1, keepdim=True))
    super(KManifoldClusterModel, self).reset_parameters()
    return

  def forward(self, ii=None):
    """compute manifold embedding for ith data point(s)."""
    if ii is not None:
      self.set_cv(ii)
    # compute embeddings for each group and concatenate
    # NOTE: is this efficient on gpu/cpu?
    # NOTE: could use jit trace to get some optimization.
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
    # NOTE: this call introduces memory leak of size (batch_size, D, n)
    # every iteration of AltSGD algorithm. Very strange..
    # Fixed once made sure all objective calls not requiring gradient were
    # wrapped in torch.no_grad(). Still strange though...
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
      # NOTE: disabled division by N
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

  def set_cv(self, ii):
    """set c, v to reflect subset of C, V given by ii."""
    # NOTE: in the case this is transferring cpu to gpu, does pinning memory
    # make it faster?
    self.c.data.copy_(self.C[ii, :])
    self.v.data.copy_(self.V[ii, :, :])
    return

  def set_CV(self, ii):
    """update full segmentation coefficients with values from current
    mini-batch."""
    # NOTE: is this asynchronous?
    self.C[ii, :] = self.c.data.cpu()
    self.V[ii, :, :] = self.v.data.cpu()
    return

  def get_groups(self, full=False):
    """compute group assignment."""
    if full:
      groups = torch.argmax(self.C.data, dim=1).cpu().numpy()
    else:
      groups = torch.argmax(self.c.data, dim=1).cpu().numpy()
    return groups


class KManifoldAEClusterModel(KClusterModel):
  """Model of union of low-dimensional manifolds generalizing
  k-means/k-subspaces. Coefficients computed using auto-encoder."""

  def __init__(self, n, d, D, N, batch_size, group_models=None):
    if group_models is None:
      group_models = [SubspaceAEModel(d, D) for _ in range(n)]
    super(KManifoldAEClusterModel, self).__init__(n, d, D, N, batch_size,
        group_models)
    self.reset_parameters()
    return

  def forward(self, x):
    """compute manifold ae embedding."""
    # compute embeddings for each group and concatenate
    # NOTE: is this efficient on gpu/cpu?
    # NOTE: could use jit trace to get some optimization.
    x_ = torch.stack([gm(x) for gm in self.group_models], dim=2)
    return x_

  def objective(self, x, lamb=.01, wrt='all'):
    if wrt not in ['all', 'C', 'U']:
      raise ValueError("Objective must be computed wrt 'all', 'C', or 'U'")

    # evaluate loss: least-squares weighted by assignment, as in k-means.
    x_ = self(x)
    # NOTE: this call introduces memory leak of size (batch_size, D, n)
    # every iteration of AltSGD algorithm. Very strange..
    # Fixed once made sure all objective calls not requiring gradient were
    # wrapped in torch.no_grad(). Still strange though...
    loss = torch.sum((x.unsqueeze(2) - x_)**2, dim=1)
    if wrt in ['all', 'U']:
      loss = torch.mean(torch.sum(self.c*loss, dim=1))
    # for wrt C, use (batch_size, n) matrix of losses for computing closed form
    # assignment.

    # evaluate U regularizer
    if wrt in ['all', 'U']:
      reg = sum((gm.reg() for gm in self.group_models))
    else:
      reg = 0.0

    obj = loss + lamb*reg
    return obj, loss, reg, x_


class GroupModel(nn.Module):
  """Model of single cluster."""

  def __init__(self, d, D):
    super(GroupModel, self).__init__()
    self.d = d  # subspace dimension
    self.D = D  # ambient dimension
    return

  def reset_parameters(self):
    raise NotImplementedError("reset_parameters not implemented")

  def forward(self, *args, **kwargs):
    raise NotImplementedError("forward not implemented")

  def reg(self):
    raise NotImplementedError("reg not implemented")


class SubspaceModel(GroupModel):
  """Model of single low-dimensional affine or linear subspace."""

  def __init__(self, d, D, affine=False):
    self.affine = affine  # whether affine or linear subspaces.
    super(SubspaceModel, self).__init__(d, D)

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
    """Initialize U, b with entries drawn from normal with std 0.1/sqrt(D)."""
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


class ResidualManifoldModel(GroupModel):
  """Model of single low-dimensional manifold.

  Represented as affine subspace + nonlinear residual. Residual module is just
  a single hidden layer network with ReLU activation and dropout."""

  def __init__(self, d, D, H=None, drop_p=0.5, res_lamb=1.0):
    super(ResidualManifoldModel, self).__init__(d, D)

    self.H = H if H else D  # size of hidden layer
    self.drop_p = drop_p
    self.res_lamb = res_lamb  # residual weights regularization parameter

    self.subspace_embedding = SubspaceModel(d, D, affine=True)
    self.res_fc1 = nn.Linear(d, self.H, bias=False)
    self.res_fc2 = nn.Linear(self.H, D, bias=False)
    # NOTE: not calling reset_parameters since already done.
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


class SubspaceAEModel(GroupModel):
  """Model of single low-dimensional affine or linear subspace auto-encoder
  (i.e. pca)."""

  def __init__(self, d, D, affine=False):
    super(SubspaceAEModel, self).__init__(d, D)
    self.affine = affine  # whether affine or linear subspaces.
    # NOTE: should V really be U^T, as in pca?
    self.U = nn.Parameter(torch.Tensor(D, d))
    self.V = nn.Parameter(torch.Tensor(d, D))
    if affine:
      self.b = nn.Parameter(torch.Tensor(D))
    else:
      self.register_parameter('b', None)
    self.reset_parameters()
    return

  def reset_parameters(self):
    """Initialize U, V, b with entries drawn from normal with std
    0.1/sqrt(D)."""
    std = .1 / np.sqrt(self.D)
    self.U.data.normal_(0., std)
    self.V.data.normal_(0., std)
    if self.affine:
      self.b.data.normal_(0., std)
    return

  def forward(self, x):
    """Compute subspace embedding from data x."""
    # NOTE: would it make sense to include batch norm on the coeffs v?
    v = self.encode(x)
    x_ = self.decode(v)
    return x_

  def encode(self, x):
    """Compute coefficients v from x."""
    if self.affine:
      x = x.sub(self.b)
    v = F.linear(x, self.V)
    return v

  def decode(self, v):
    """Compute reconstruction x_ from v."""
    x_ = F.linear(v, self.U, self.b)
    return x_

  def reg(self):
    """Compute L2 regularization on subspace basis."""
    reg = 0.5*(self.U.pow(2).sum() + self.V.pow(2).sum())
    return reg


class ResidualManifoldAEModel(GroupModel):
  """Model of single low-dimensional manifold.

  Represented as affine subspace + nonlinear residual. Residual module is just
  a single hidden layer network with ReLU activation and dropout."""

  def __init__(self, d, D, H=None, drop_p=0.5, res_lamb=1.0):
    super(ResidualManifoldAEModel, self).__init__(d, D)

    self.H = H if H else D  # size of hidden layer
    self.drop_p = drop_p
    self.res_lamb = res_lamb  # residual weights regularization parameter

    self.subspace_ae = SubspaceAEModel(d, D, affine=True)
    # NOTE: think about the justification for bias or not here.
    # no bias on dec_fc1 because I might want to use group sparsity
    # regularization to control dimension. but for others I have no good
    # reason.
    self.enc_fc1 = nn.Linear(D, self.H, bias=False)
    self.enc_fc2 = nn.Linear(self.H, d, bias=False)
    self.dec_fc1 = nn.Linear(d, self.H, bias=False)
    self.dec_fc2 = nn.Linear(self.H, D, bias=False)
    # NOTE: not calling reset_parameters since already done.
    return

  def forward(self, x):
    """Compute residual manifold embedding from data x."""
    # NOTE: would it make sense to include batch norm on the coeffs v?
    v = self.encode(x)
    x_ = self.decode(v)
    return x_

  def encode(self, x):
    """Compute coefficients v from x."""
    v = self.subspace_ae.encode(x)
    z = F.relu(self.enc_fc1(x))
    z = F.dropout(z, p=self.drop_p, training=self.training)
    z = self.enc_fc2(z)
    v = v + z
    return v

  def decode(self, v):
    """Compute reconstruction x_ from v."""
    x_ = self.subspace_ae.decode(v)
    z = F.relu(self.dec_fc1(v))
    z = F.dropout(z, p=self.drop_p, training=self.training)
    z = self.dec_fc2(z)
    x_ = x_ + z
    return x_

  def reg(self):
    """Compute L2 squared regularization on subspace basis and residual
    module."""
    reg = self.subspace_ae.reg()
    # Intuition is for weights of residual module to be strongly regularized so
    # that residual is Lipschitz with a small constant, controlling the
    # curvature of the manifold.
    reg += 0.5*self.res_lamb*sum([layer.weight.pow(2).sum() for layer in
        [self.enc_fc1, self.enc_fc2, self.dec_fc1, self.dec_fc2]])
    return reg
