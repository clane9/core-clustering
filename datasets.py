from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch.utils.data import Dataset

import models as mod


class SynthUoSDataset(Dataset):
  """Synthetic union of subspaces dataset."""
  def __init__(self, n, d, D, Ng, affine=False, sigma=0., seed=None):
    super(SynthUoSDataset).__init__()

    self.n = n  # number of subspaces
    self.d = d  # subspace dimension
    self.D = D  # ambient dimension
    self.Ng = Ng  # points per group
    self.N = n*Ng
    self.affine = affine
    self.sigma = sigma
    rng = np.random.RandomState(seed=seed)

    self.Us = np.zeros([D, d, n])
    self.Vs = np.zeros([d, Ng, n])
    self.bs = np.zeros([D, n]) if affine else None
    self.X = np.zeros([self.N, D])
    self.groups = np.zeros(self.N, dtype=np.int32)

    # sample data from randomnly generated (linear or affine) subspaces
    for ii in range(n):
      U, _ = np.linalg.qr(rng.randn(D, d))
      V = (1./np.sqrt(d))*rng.randn(d, Ng)
      self.Us[:, :, ii] = U
      self.Vs[:, :, ii] = V
      Xi = np.matmul(U, V)

      if affine:
        b = (1./np.sqrt(D))*rng.randn(D, 1)
        self.bs[:, ii] = b
        Xi += b

      self.X[ii*Ng:(ii+1)*Ng, :] = Xi.T
      self.groups[ii*Ng:(ii+1)*Ng] = ii

    if sigma > 0.:
      E = (sigma/np.sqrt(D))*rng.randn(self.N, D)
      self.X += E

    # permute order of data
    self.perm = np.random.permutation(self.N)
    self.X = self.X[self.perm, :]
    self.groups = self.groups[self.perm]

    self.X = torch.tensor(self.X, dtype=torch.float32)
    return

  def __len__(self):
    return self.N

  def __getitem__(self, ii):
    return torch.tensor(ii), self.X[ii, :], self.groups[ii]


class SynthUoMDataset(Dataset):
  """Synthetic union of low-dimensional manifolds dataset."""
  def __init__(self, n, d, D, Ng, H, res_weight_scale=1.0, sigma=0.,
        seed=None):
    super(SynthUoMDataset).__init__()

    self.n = n  # number of subspaces
    self.d = d  # subspace dimension
    self.D = D  # ambient dimension
    self.Ng = Ng  # points per group
    self.N = n*Ng
    self.H = H
    self.res_weight_scale = res_weight_scale
    self.sigma = sigma

    if seed is not None:
      torch.manual_seed(seed)

    self.group_models = [mod.ResidualManifoldAEModel(d, D, H,
        drop_p=0.0, res_lamb=0.0) for _ in range(n)]
    # scale weights to control smoothness
    for gm in self.group_models:
      gm.dec_fc1.weight.data.mul_(res_weight_scale)
      gm.dec_fc2.weight.data.mul_(res_weight_scale)

    # generate true groups and segmentation
    self.groups = np.arange(n, dtype=np.int64).reshape(-1, 1)
    self.groups = np.tile(self.groups, (1, Ng)).reshape(-1)

    C = torch.zeros(self.N, n)
    C.scatter_(1, torch.from_numpy(self.groups).view(-1, 1), 1)

    # generate union of manifold data
    V = torch.randn(self.N, d).div(np.sqrt(d))
    with torch.no_grad():
      self.X = torch.stack([gm.decode(V) for gm in self.group_models], dim=2)
      self.X.mul_(C.unsqueeze(1))
      self.X = self.X.sum(dim=2)

    if sigma > 0.:
      E = torch.randn(self.N, D).mul(sigma/np.sqrt(D))
      self.X += E

    # permute order of data
    self.perm = np.random.permutation(self.N)
    self.X = self.X[self.perm, :]
    self.groups = self.groups[self.perm]
    return

  def __len__(self):
    return self.N

  def __getitem__(self, ii):
    return torch.tensor(ii), self.X[ii, :], self.groups[ii]
