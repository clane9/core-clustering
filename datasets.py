from __future__ import division
from __future__ import print_function

import os
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset

CODE_DIR = os.path.dirname(os.path.realpath(__file__))


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
    self.classes = np.arange(n)

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
        self.bs[:, ii] = b[:, 0]
        Xi += b

      self.X[ii*Ng:(ii+1)*Ng, :] = Xi.T
      self.groups[ii*Ng:(ii+1)*Ng] = ii

    if sigma > 0.:
      E = (sigma/np.sqrt(D))*rng.randn(self.N, D)
      self.X += E

    # permute order of data
    self.perm = rng.permutation(self.N)
    self.X = self.X[self.perm, :]
    self.groups = self.groups[self.perm]

    self.X = torch.tensor(self.X, dtype=torch.float32)
    self.groups = torch.tensor(self.groups, dtype=torch.int64)
    return

  def __len__(self):
    return self.N

  def __getitem__(self, ii):
    return self.X[ii, :], self.groups[ii]


class SynthUoSOnlineDataset(Dataset):
  """Synthetic union of subspaces dataset with fresh samples."""
  def __init__(self, n, d, D, N, affine=False, sigma=0., seed=None):
    super(SynthUoSOnlineDataset).__init__()

    self.n = n  # number of subspaces
    self.d = d  # subspace dimension
    self.D = D  # ambient dimension
    self.N = N  # number of points
    self.affine = affine
    self.sigma = sigma

    self.rng = torch.Generator()
    if seed is not None:
      self.rng.manual_seed(seed)
    self.seed = seed

    self.classes = np.arange(n)
    self.Us = torch.zeros((n, D, d))
    self.bs = torch.zeros((n, D)) if affine else None

    # sample data from randomnly generated (linear or affine) subspaces
    for grp in range(n):
      self.Us[grp, :, :], _ = torch.qr(torch.randn(D, d, generator=self.rng))

      if affine:
        self.bs[grp, :] = (1./np.sqrt(D))*torch.randn(D, generator=self.rng)
    return

  def __len__(self):
    return self.N

  def __getitem__(self, ii):
    # NOTE: need to use global torch generator when using worker processes to
    # generate data. Otherwise rng is duplicated and end up getting repeated
    # data. Also note that it doesn't matter if seed is changed after
    # DataLoader constructor is called.
    grp = torch.randint(high=self.n, size=(1,), dtype=torch.int64)[0]
    v = (1./np.sqrt(self.d))*torch.randn(self.d, 1)
    x = torch.matmul(self.Us[grp, :, :], v).view(-1)
    if self.affine:
      x += self.bs[grp, :]
    if self.sigma > 0:
      x += (self.sigma/np.sqrt(self.D))*torch.randn(self.D)
    return x, grp


class YouCVPR16ImageUoS(Dataset):
  """Image datasets from (You et al., CVPR 2016)."""
  def __init__(self, dataset='mnist_sc_pca'):
    if dataset == 'mnist_sc_pca':
      matfile = '{}/datasets/MNIST_SC_pca.mat'.format(CODE_DIR)
      data = loadmat(matfile)
      self.X = torch.tensor(data['MNIST_SC_DATA'].T, dtype=torch.float32)
      self.groups = torch.tensor(data['MNIST_LABEL'].reshape(-1),
          dtype=torch.int64)
    elif dataset == 'coil100':
      matfile = '{}/datasets/COIL100.mat'.format(CODE_DIR)
      data = loadmat(matfile)
      self.X = torch.tensor(data['fea'], dtype=torch.float32)
      self.groups = torch.tensor(data['gnd'].reshape(-1), dtype=torch.int64)
    else:
      raise ValueError("Invalid dataset {}".format(dataset))

    self.classes = torch.unique(self.groups, sorted=True).numpy()
    self.n = self.classes.shape[0]

    # normalize data points (rows) of X
    self.X.div_(torch.norm(self.X, p=2, dim=1).view(-1, 1).add(1e-8))
    self.N, self.D = self.X.shape
    return

  def __len__(self):
    return self.N

  def __getitem__(self, ii):
    return self.X[ii, :], self.groups[ii]
