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
  def __init__(self, n, d, D, Ng, affine=False, sigma=0., theta=None,
        normalize=False, seed=None):
    super(SynthUoSDataset).__init__()

    if theta is not None:
      if theta <= 0:
        raise ValueError("Invalid principal angle {}".format(theta))
      if n*d > D:
        raise ValueError("Can only specify principal angle for independent "
            "subspaces")

    self.n = n  # number of subspaces
    self.d = d  # subspace dimension
    self.D = D  # ambient dimension
    self.Ng = Ng  # points per group
    self.N = n*Ng
    self.affine = affine
    self.sigma = sigma
    self.theta = theta
    self.normalize = normalize
    self.classes = np.arange(n)

    self.rng = np.random.RandomState(seed=seed)

    self.Us = np.zeros([n, D, d])
    self.Vs = np.zeros([n, d, Ng])
    self.bs = np.zeros([n, D]) if affine else None
    self.X = np.zeros([self.N, D])
    self.groups = np.zeros(self.N, dtype=np.int32)

    # generate bases
    if theta is None:
      # bases sampled uniformly at random
      for ii in range(n):
        self.Us[ii, :] = np.linalg.qr(self.rng.randn(D, d))[0]
    else:
      # bases sampled with fixed principal angles.
      alpha = np.sqrt(np.cos(theta))
      beta = np.sqrt(1.0 - alpha**2)

      # all bases will be a perturbation of U0
      U0 = np.linalg.qr(self.rng.randn(D, d), mode='complete')[0]
      U0_comp = U0[:, d:]
      U0 = U0[:, :d]
      # apply random rotation to U0 complement
      Q = np.linalg.qr(self.rng.randn(D-d, D-d))[0]
      U0_comp = np.matmul(U0_comp, Q)
      for ii in range(n):
        P = np.linalg.qr(self.rng.randn(d, d))[0]
        self.Us[ii, :] = (alpha*np.matmul(U0, P) +
            beta*U0_comp[:, ii*d:(ii+1)*d])

    # sample coefficients
    for ii in range(n):
      V = (1./np.sqrt(d))*self.rng.randn(d, Ng)
      self.Vs[ii, :] = V
      Xi = np.matmul(self.Us[ii, :], V)

      if affine:
        b = (1./np.sqrt(D))*self.rng.randn(D, 1)
        self.bs[ii, :] = b[:, 0]
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

    if normalize:
      self.X.div_(torch.norm(self.X, dim=1, keepdim=True))
    return

  def __len__(self):
    return self.N

  def __getitem__(self, ii):
    return self.X[ii, :], self.groups[ii]


class SynthUoSMissDataset(SynthUoSDataset):
  """Synthetic union of subspaces dataset with missing data."""
  def __init__(self, n, d, D, Ng, affine=False, sigma=0., theta=None,
        miss_rate=0.0, normalize=False, sparse_format=False, test_frac=0.05,
        seed=None):
    if miss_rate >= 1 or miss_rate < 0:
      raise ValueError("Invalid miss_rate {}".format(miss_rate))

    super().__init__(n, d, D, Ng, affine, sigma, theta, normalize, seed)

    self.miss_rate = miss_rate
    self.test_frac = test_frac
    self.sparse_format = sparse_format

    # sample observed entries uniformly
    self.Omega = (torch.tensor(self.rng.rand(self.N, self.D),
        dtype=torch.float32) >= miss_rate)
    OmegaIdx = self.Omega.nonzero()
    test_mask = (torch.tensor(self.rng.rand(OmegaIdx.shape[0]),
        dtype=torch.float32) <= test_frac)
    self.Omega[OmegaIdx[test_mask, 0], OmegaIdx[test_mask, 1]] = 0
    self.Omega_test = torch.zeros_like(self.Omega)
    self.Omega_test[OmegaIdx[test_mask, 0], OmegaIdx[test_mask, 1]] = 1
    return

  def __getitem__(self, ii):
    x, omega, omega_test, grp = (self.X[ii, :], self.Omega[ii, :],
        self.Omega_test[ii, :], self.groups[ii])
    if self.sparse_format:
      omegaIdx = omega.nonzero().view(-1)
      omega_testIdx = omega_test.nonzero().view(-1)
      if omega_testIdx.shape[0] > 0:
        omega_testIdx = omega_testIdx.view(1, -1)
      if omegaIdx.shape[0] > 0:
        omegaIdx = omegaIdx.view(1, -1)
      x0 = torch.sparse.FloatTensor(omega_testIdx, x[omega_testIdx].view(-1),
          (self.D,))
      x_miss = torch.sparse.FloatTensor(omegaIdx, x[omegaIdx].view(-1),
          (self.D,))
    else:
      x_miss = torch.zeros_like(x).mul_(np.nan)
      x0 = torch.zeros_like(x).mul_(np.nan)
      x_miss[omega] = x[omega]
      x0[omega_test] = x[omega_test]
    return x_miss, grp, x0


class SynthUoSOnlineDataset(SynthUoSDataset):
  """Synthetic union of subspaces dataset with fresh samples."""
  def __init__(self, n, d, D, N, affine=False, sigma=0., theta=None,
        normalize=False, seed=None):
    super().__init__(n, d, D, 10, affine, sigma, theta, normalize, seed)
    self.Us = torch.tensor(self.Us, dtype=torch.float32)
    self.N = N

    # parent constructor called to generate bases, but coefficients and X don't
    # matter
    self.Vs = None
    self.X = None
    self.groups = None
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
    if self.normalize:
      x.div_(torch.norm(x))
    return x, grp


class SynthUoSMissOnlineDataset(SynthUoSOnlineDataset):
  """Synthetic union of subspaces dataset with fresh samples."""
  def __init__(self, n, d, D, N, affine=False, sigma=0., theta=None,
        miss_rate=0.0, normalize=False, sparse_format=False, test_frac=0.05,
        seed=None):
    if miss_rate >= 1 or miss_rate < 0:
      raise ValueError("Invalid miss_rate {}".format(miss_rate))

    super().__init__(n, d, D, N, affine, sigma, theta, normalize, seed)

    self.miss_rate = miss_rate
    self.test_frac = test_frac
    self.sparse_format = sparse_format
    return

  def __getitem__(self, ii):
    x, grp = super().__getitem__(ii)
    omega = (torch.rand(self.D) >= self.miss_rate)
    omegaIdx = omega.nonzero().view(-1)
    test_mask = (torch.rand(omegaIdx.shape[0]) <= self.test_frac)
    omega_testIdx = omegaIdx[test_mask]
    omegaIdx = omegaIdx[test_mask == 0]

    if self.sparse_format:
      if omega_testIdx.shape[0] > 0:
        omega_testIdx = omega_testIdx.view(1, -1)
      if omegaIdx.shape[0] > 0:
        omegaIdx = omegaIdx.view(1, -1)
      x0 = torch.sparse.FloatTensor(omega_testIdx, x[omega_testIdx].view(-1),
          (self.D,))
      x_miss = torch.sparse.FloatTensor(omegaIdx, x[omegaIdx].view(-1),
          (self.D,))
    else:
      x_miss = torch.zeros_like(x).mul_(np.nan)
      x0 = torch.zeros_like(x).mul_(np.nan)
      x_miss[omegaIdx] = x[omegaIdx]
      x0[omega_testIdx] = x[omega_testIdx]
    return x_miss, grp, x0


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
        stopi = min(U.shape) + 1
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


def sparse_miss_collate(batch):
  x, grp, x0 = zip(*batch)
  grp = torch.stack(grp)
  return x, grp, x0
