from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch.utils.data import Dataset


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
    self.X = np.zeros([D, self.N])
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

      self.X[:, ii*Ng:(ii+1)*Ng] = Xi
      self.groups[ii*Ng:(ii+1)*Ng] = ii

    if sigma > 0.:
      self.X += sigma*rng.randn(D, self.N)

    self.X = torch.tensor(self.X, dtype=torch.float32)
    return

  def __len__(self):
    return self.N

  def __getitem__(self, ii):
    return torch.tensor(ii), self.X[:, ii]
