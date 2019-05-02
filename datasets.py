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

    self.rng = np.random.RandomState(seed=seed)

    self.Us = np.zeros([D, d, n])
    self.Vs = np.zeros([d, Ng, n])
    self.bs = np.zeros([D, n]) if affine else None
    self.X = np.zeros([self.N, D])
    self.groups = np.zeros(self.N, dtype=np.int32)

    # sample data from randomnly generated (linear or affine) subspaces
    for ii in range(n):
      U, _ = np.linalg.qr(self.rng.randn(D, d))
      V = (1./np.sqrt(d))*self.rng.randn(d, Ng)
      self.Us[:, :, ii] = U
      self.Vs[:, :, ii] = V
      Xi = np.matmul(U, V)

      if affine:
        b = (1./np.sqrt(D))*self.rng.randn(D, 1)
        self.bs[:, ii] = b[:, 0]
        Xi += b

      self.X[ii*Ng:(ii+1)*Ng, :] = Xi.T
      self.groups[ii*Ng:(ii+1)*Ng] = ii

    if sigma > 0.:
      E = (sigma/np.sqrt(D))*self.rng.randn(self.N, D)
      self.X += E

    # permute order of data
    self.perm = self.rng.permutation(self.N)
    self.X = self.X[self.perm, :]
    self.groups = self.groups[self.perm]

    self.X = torch.tensor(self.X, dtype=torch.float32)
    self.groups = torch.tensor(self.groups, dtype=torch.int64)
    return

  def __len__(self):
    return self.N

  def __getitem__(self, ii):
    return self.X[ii, :], self.groups[ii]


class SynthUoSMissDataset(SynthUoSDataset):
  """Synthetic union of subspaces dataset with missing data."""
  def __init__(self, n, d, D, Ng, affine=False, sigma=0., miss_rate=0.0,
        seed=None):
    if miss_rate >= 1 or miss_rate < 0:
      raise ValueError("Invalid miss_rate {}".format(miss_rate))

    super().__init__(n, d, D, Ng, affine, sigma, seed)

    # sample observed entries uniformly
    self.miss_rate = miss_rate
    self.Omega = (self.rng.rand(self.N, self.D) >=
        miss_rate).astype(np.float32)
    self.Omega = torch.tensor(self.Omega, dtype=torch.float32)

    # NOTE: X is not masked by Omega
    self.X = torch.stack((self.X, self.Omega), dim=1)
    return


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


class SynthUoSMissOnlineDataset(SynthUoSOnlineDataset):
  """Synthetic union of subspaces dataset with fresh samples."""
  def __init__(self, n, d, D, N, affine=False, sigma=0., miss_rate=0.0,
        seed=None):
    if miss_rate >= 1 or miss_rate < 0:
      raise ValueError("Invalid miss_rate {}".format(miss_rate))

    super().__init__(n, d, D, N, affine, sigma, seed)

    self.miss_rate = miss_rate
    return

  def __getitem__(self, ii):
    x, grp = super().__getitem__(ii)
    omega = (torch.rand(self.D) >= self.miss_rate).to(torch.float32)
    # NOTE: X is not masked by Omega
    x = torch.stack((x, omega), dim=0)
    return x, grp


class ImageUoSDataset(Dataset):
  """Image datasets, mostly from (You et al., CVPR 2016)."""
  def __init__(self, dataset='mnist', center=False, sv_range=None,
        normalize=True):
    if dataset == 'mnist':
      matfile = '{}/datasets/MNIST_SC_pca.mat'.format(CODE_DIR)
      data = loadmat(matfile)
      self.X = torch.tensor(data['MNIST_SC_DATA'].T, dtype=torch.float32)
      self.groups = torch.tensor(data['MNIST_LABEL'].reshape(-1),
          dtype=torch.int64)
    elif dataset == 'coil100':
      matfile = '{}/datasets/COIL100_SC_pca.mat'.format(CODE_DIR)
      data = loadmat(matfile)
      self.X = torch.tensor(data['COIL100_SC_DATA'].T, dtype=torch.float32)
      self.groups = torch.tensor(data['COIL100_LABEL'].reshape(-1),
          dtype=torch.int64)
    elif dataset == 'coil20':
      matfile = '{}/datasets/COIL20_SC_pca.mat'.format(CODE_DIR)
      data = loadmat(matfile)
      self.X = torch.tensor(data['COIL20_SC_DATA'].T, dtype=torch.float32)
      self.groups = torch.tensor(data['COIL20_LABEL'].reshape(-1),
          dtype=torch.int64)
    elif dataset == 'yaleb':
      matfile = '{}/datasets/small_YaleB_48x42.mat'.format(CODE_DIR)
      data = loadmat(matfile)
      # tuple (scale, dim, D, N, images, labels)
      small_yale = data['small_yale'][0, 0]
      self.X = torch.tensor(small_yale[4].T, dtype=torch.float32)
      self.groups = torch.tensor(small_yale[5][0, :], dtype=torch.int64)
    else:
      raise ValueError("Invalid dataset {}".format(dataset))

    self.classes = torch.unique(self.groups, sorted=True).numpy()
    self.n = self.classes.shape[0]

    # normalize data points (rows) of X
    if center:
      self.X.sub_(self.X.mean(dim=0))
    if sv_range is not None:
      # "whitening" by removing first few svs following (Zhang, 2012)
      starti = sv_range[0] if len(sv_range) == 2 else 0
      stopi = sv_range[1] if len(sv_range) == 2 else sv_range[0]
      U, s, _ = torch.svd(self.X)
      if stopi is None or stopi <= 0:
        stopi = min(U.shape)
      self.X = U[:, starti:stopi] * s[starti:stopi]

    if normalize:
      self.X.div_(torch.norm(self.X, p=2, dim=1, keepdim=True).add(1e-8))
    self.N, self.D = self.X.shape
    return

  def __len__(self):
    return self.N

  def __getitem__(self, ii):
    return self.X[ii, :], self.groups[ii]


class SynthKMeansDataset(Dataset):
  """Synthetic k means dataset."""
  def __init__(self, k, D, Ng, separation=2.0, seed=None):
    super().__init__()

    self.k = k  # number of groups
    self.D = D  # ambient dimension
    self.Ng = Ng  # points per group
    self.N = k*Ng
    self.classes = np.arange(k)

    # c separation from Dasgupta 1999
    if separation is None or separation < 0:
      self.separation = 2.0
    else:
      self.separation = separation

    self.rng = np.random.RandomState(seed=seed)
    self.bs = (self.separation / np.sqrt(2)) * self.rng.randn(k, D)

    dists = np.sqrt(np.sum((np.expand_dims(self.bs, 1) -
        np.expand_dims(self.bs, 0))**2, axis=2))
    triuIdx = np.triu_indices(k, k=1)
    self.dists = dists[triuIdx]

    self.X = self.bs.repeat(Ng, axis=0)
    self.groups = np.arange(k, dtype=np.int32).repeat(Ng)
    self.X += self.rng.randn(Ng*k, D)

    # permute order of data
    self.perm = self.rng.permutation(self.N)
    self.X = self.X[self.perm, :]
    self.groups = self.groups[self.perm]

    self.X = torch.tensor(self.X, dtype=torch.float32)
    self.groups = torch.tensor(self.groups, dtype=torch.int64)
    return

  def __len__(self):
    return self.N

  def __getitem__(self, ii):
    return self.X[ii, :], self.groups[ii]
