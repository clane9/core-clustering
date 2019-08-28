from __future__ import division
from __future__ import print_function

import os
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset

DATA_DIR = os.path.realpath(os.path.dirname(__file__)) + '/../datasets'


class SynthUoSDataset(Dataset):
  """Synthetic union of subspaces dataset."""
  def __init__(self, k, d, D, Ng, affine=False, sigma=0., theta=None,
        normalize=False, seed=None):
    super(SynthUoSDataset).__init__()

    self.theta_mode = theta is not None and theta > 0 and theta < np.pi/2
    if self.theta_mode:
      if (k+1)*d > D:
        raise ValueError("Can only specify principal angle for independent "
            "subspaces")

    self.k = k  # number of subspaces
    self.d = d  # subspace dimension
    self.D = D  # ambient dimension
    self.Ng = Ng  # points per group
    self.N = k*Ng
    self.affine = affine
    self.sigma = sigma
    self.theta = theta
    self.normalize = normalize
    self.classes = np.arange(k)

    self.rng = np.random.RandomState(seed=seed)

    self.Us = np.zeros([k, D, d])
    self.Vs = np.zeros([k, d, Ng])
    self.bs = np.zeros([k, D]) if affine else None
    self.X = np.zeros([self.N, D])
    self.groups = np.zeros(self.N, dtype=np.int32)

    # generate bases
    if not self.theta_mode:
      # bases sampled uniformly at random
      for ii in range(k):
        self.Us[ii, :] = np.linalg.qr(self.rng.randn(D, d))[0]
    else:
      # bases sampled with fixed principal angles.
      # theta in [0, pi/2]
      alpha = np.sqrt(np.cos(theta))
      beta = np.sqrt(1.0 - alpha**2)

      # all bases will be a perturbation of U0
      U0 = np.linalg.qr(self.rng.randn(D, d), mode='complete')[0]
      U0_comp = U0[:, d:]
      U0 = U0[:, :d]
      # apply random rotation to U0 complement
      Q = np.linalg.qr(self.rng.randn(D-d, D-d))[0]
      U0_comp = np.matmul(U0_comp, Q)
      for ii in range(k):
        P = np.linalg.qr(self.rng.randn(d, d))[0]
        # with this construction, have
        #
        #   U_i^T U_i = alpha^2 (P_i^T U_0^T U_0 P_i) + beta^2 Z_i^T Z_i
        #             = alpha^2 I + beta^2 I = I
        #
        #   U_i^T U_j = alpha^2 (P_i^T U_0^T U_0 P_j)
        #             = alpha^2 P_i^T P_j
        #
        # So, || U_i^T U_j ||_2 = alpha^2 = cos(theta)
        self.Us[ii, :] = (alpha*np.matmul(U0, P) +
            beta*U0_comp[:, ii*d:(ii+1)*d])

    # sample coefficients
    for ii in range(k):
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
  def __init__(self, k, d, D, Ng, affine=False, sigma=0., theta=None,
        miss_rate=0.0, normalize=False, store_sparse=True, store_dense=False,
        test_frac=0.05, seed=None):
    if miss_rate >= 1 or miss_rate < 0:
      raise ValueError("Invalid miss_rate {}".format(miss_rate))

    super().__init__(k, d, D, Ng, affine, sigma, theta, normalize, seed)

    self.miss_rate = miss_rate
    self.test_frac = test_frac
    self.store_sparse = store_sparse
    self.store_dense = store_dense

    # sample observed entries uniformly
    self.Omega = (torch.tensor(self.rng.rand(self.N, self.D),
        dtype=torch.float32) <= (1 + test_frac)*(1 - miss_rate))
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

    x_miss = MissingDataSample(x, omega=omega, store_sparse=self.store_sparse,
        store_dense=self.store_dense)
    x0 = MissingDataSample(x, omega=omega_test, store_sparse=self.store_sparse,
        store_dense=self.store_dense)
    return x_miss, grp, x0


class SynthUoSOnlineDataset(SynthUoSDataset):
  """Synthetic union of subspaces dataset with fresh samples."""
  def __init__(self, k, d, D, N, affine=False, sigma=0., theta=None,
        normalize=False, seed=None):
    super().__init__(k, d, D, 10, affine, sigma, theta, normalize, seed)
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
    grp = torch.randint(high=self.k, size=(1,), dtype=torch.int64)[0]
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
  def __init__(self, k, d, D, N, affine=False, sigma=0., theta=None,
        miss_rate=0.0, normalize=False, store_sparse=False, store_dense=False,
        test_frac=0.05, seed=None):
    if miss_rate >= 1 or miss_rate < 0:
      raise ValueError("Invalid miss_rate {}".format(miss_rate))

    super().__init__(k, d, D, N, affine, sigma, theta, normalize, seed)

    self.miss_rate = miss_rate
    self.test_frac = test_frac
    self.store_sparse = store_sparse
    self.store_dense = store_dense
    return

  def __getitem__(self, ii):
    x, grp = super().__getitem__(ii)
    omega = (torch.rand(self.D) <= (1 + self.test_frac)*(1 - self.miss_rate))
    omegaIdx = omega.nonzero().view(-1)
    test_mask = (torch.rand(omegaIdx.shape[0]) <= self.test_frac)
    omega_testIdx = omegaIdx[test_mask]
    omegaIdx = omegaIdx[test_mask == 0]

    x_miss = MissingDataSample(x[omegaIdx], indices=omegaIdx, D=self.D,
        store_sparse=self.store_sparse, store_dense=self.store_dense)
    x0 = MissingDataSample(x[omega_testIdx], indices=omega_testIdx, D=self.D,
        store_sparse=self.store_sparse, store_dense=self.store_dense)
    return x_miss, grp, x0


class ImageUoSDataset(Dataset):
  """Image datasets, mostly from (You et al., CVPR 2016)."""
  def __init__(self, dataset='mnist', center=False, sv_range=None,
        normalize=True):
    if dataset in {'mnist', 'cifar10'}:
      fname = '{}/{}/{}_scat_pca.npz'.format(DATA_DIR, dataset, dataset)
      with open(fname, 'rb') as f:
        f = np.load(f)
        self.X = torch.tensor(f['data'], dtype=torch.float32)
        self.groups = torch.tensor(f['labels'], dtype=torch.int64)
    elif dataset == 'coil100':
      matfile = '{}/COIL100_SC_pca.mat'.format(DATA_DIR)
      data = loadmat(matfile)
      self.X = torch.tensor(data['COIL100_SC_DATA'].T, dtype=torch.float32)
      self.groups = torch.tensor(data['COIL100_LABEL'].reshape(-1),
          dtype=torch.int64)
    elif dataset == 'coil20':
      matfile = '{}/COIL20_SC_pca.mat'.format(DATA_DIR)
      data = loadmat(matfile)
      self.X = torch.tensor(data['COIL20_SC_DATA'].T, dtype=torch.float32)
      self.groups = torch.tensor(data['COIL20_LABEL'].reshape(-1),
          dtype=torch.int64)
    elif dataset == 'yaleb':
      matfile = '{}/small_YaleB_48x42.mat'.format(DATA_DIR)
      data = loadmat(matfile)
      # tuple (scale, dim, D, N, images, labels)
      small_yale = data['small_yale'][0, 0]
      self.X = torch.tensor(small_yale[4].T, dtype=torch.float32)
      self.groups = torch.tensor(small_yale[5][0, :], dtype=torch.int64)
    else:
      raise ValueError("Invalid dataset {}".format(dataset))

    self.classes = torch.unique(self.groups, sorted=True).numpy()
    self.k = self.classes.shape[0]

    # normalize data points (rows) of X
    if center:
      self.X.sub_(self.X.mean(dim=0))
    if sv_range is not None:
      # "whitening" by removing first few svs following (Zhang, 2012)
      starti = sv_range[0] if len(sv_range) == 2 else 0
      stopi = sv_range[1] if len(sv_range) == 2 else sv_range[0]
      if (starti, stopi) not in {(0, -1), (0, self.X.shape[1])}:
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
    # will have EE ||b_i - b_j||_2 ~= separation
    self.bs = ((self.separation / (np.sqrt(2)*np.sqrt(D))) *
        self.rng.randn(k, D))

    dists = np.sqrt(np.sum((np.expand_dims(self.bs, 1) -
        np.expand_dims(self.bs, 0))**2, axis=2))
    triuIdx = np.triu_indices(k, k=1)
    self.dists = dists[triuIdx]

    self.X = self.bs.repeat(Ng, axis=0)
    self.groups = np.arange(k, dtype=np.int32).repeat(Ng)
    # will have EE || x_i - b ||_2 = 1
    self.X += (1/np.sqrt(D)) * self.rng.randn(Ng*k, D)

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


class SynthKMeansOnlineDataset(SynthKMeansDataset):
  """Synthetic k means dataset with fresh samples."""
  def __init__(self, k, D, N, separation=2.0, seed=None):
    super().__init__(k, D, 10, separation, seed)
    self.bs = torch.tensor(self.bs, dtype=torch.float32)
    self.N = N

    # parent constructor called to generate centers, but X, groups don't matter
    self.X = None
    self.perm = None
    self.groups = None
    return

  def __len__(self):
    return self.N

  def __getitem__(self, ii):
    grp = torch.randint(high=self.k, size=(1,), dtype=torch.int64)[0]
    x = self.bs[grp, :] + torch.randn(self.D).div(np.sqrt(self.D))
    return x, grp


class NetflixDataset(Dataset):
  def __init__(self, dataset='nf_17k', center=True, normalize=False,
        store_sparse=True, store_dense=False):
    if dataset not in {'nf_17k', 'nf_1k'}:
      raise ValueError("Dataset {} not supported".format(dataset))

    fname = {'nf_17k': 'nf_prize_446460x16885.npz',
        'nf_1k': 'nf_prize_422889x889.npz'}[dataset]
    fpath = '{}/nf_prize_preprocessed/{}.npz'.format(DATA_DIR, fname)

    with open(fpath, 'rb') as f:
      f = np.load(f, allow_pickle=True)
      # tolist required since np.savez doesn't know how to properly handle
      # sparse matrices. there is probably a better way.
      self.X = f['nf_train_mat'].tolist().astype(np.float32)
      self.X_test = f['nf_test_mat'].tolist().astype(np.float32)
    if center:
      train_mean = self.X.data.mean()
      self.X.data -= train_mean
      self.X_test.data -= train_mean

    self.dataset = dataset
    self.fname = fname
    self.center = center
    self.normalize = normalize
    self.store_sparse = store_sparse
    self.store_dense = store_dense
    self.N, self.D = self.X.shape
    self.groups, self.classes = None, None
    return

  def __len__(self):
    return self.N

  def __getitem__(self, ii):
    x_miss = self.X[ii, :]
    x0 = self.X_test[ii, :]

    if self.normalize:
      xnorm = np.sqrt(self.D * (x_miss.data ** 2).mean())
      x_miss.data = x_miss.data / xnorm
      x0.data = x0.data / xnorm

    x_miss = MissingDataSample(torch.from_numpy(x_miss.data),
        indices=torch.from_numpy(x_miss.indices), D=self.D,
        store_sparse=self.store_sparse, store_dense=self.store_dense)
    x0 = MissingDataSample(torch.from_numpy(x0.data),
        indices=torch.from_numpy(x0.indices), D=self.D,
        store_sparse=self.store_sparse, store_dense=self.store_dense)
    grp = torch.tensor(0)
    return x_miss, grp, x0


class MissingDataBatch(object):
  def __init__(self, missing_samples):
    """Represent a batch of missing data samples. Effectively a container for
    (values, indices, omega)."""
    self.nnz = torch.tensor([v.nnz for v in missing_samples],
        dtype=torch.int64)
    self.D = missing_samples[0].D
    self.store_sparse = missing_samples[0].store_sparse
    self.store_dense = missing_samples[0].store_dense
    self.max_nnz = self.nnz.max().item()
    self.pad_nnz = max(self.max_nnz, 2)

    if self.store_sparse:
      indices, values, omega = [], [], []
    if self.store_dense:
      values_dense, omega_dense = [], []
    for v in missing_samples:
      if self.store_sparse:
        v.pad(self.pad_nnz)
        indices.append(v.indices)
        values.append(v.values)
        omega.append(v.omega)
      if self.store_dense:
        values_dense.append(v.values_dense)
        omega_dense.append(v.omega_dense)

    if self.store_sparse:
      self.indices = torch.stack(indices)
      self.values = torch.stack(values)
      self.omega = torch.stack(omega)
      self.omega_float = self.omega.float()
    else:
      self.indices, self.values, self.omega = None, None, None
      self.omega_float = None

    if self.store_dense:
      self.values_dense = torch.stack(values_dense)
      self.omega_dense = torch.stack(omega_dense)
      self.omega_float_dense = self.omega_dense.float()
    else:
      self.values_dense, self.omega_dense = None, None
      self.omega_float_dense = None

    batch_size = (self.values.shape[0] if self.store_sparse else
        self.values_dense.shape[0])
    self.shape = (batch_size, self.D)
    self.device = (self.values.device if self.store_sparse else
        self.values_dense.device)
    return

  def to(self, device):
    if not isinstance(device, torch.device):
      raise ValueError("to only supported for devices.")
    if self.store_sparse:
      self.indices = self.indices.to(device)
      self.values = self.values.to(device)
      self.omega = self.omega.to(device)
      self.omega_float = self.omega_float.to(device)
    if self.store_dense:
      self.values_dense = self.values_dense.to(device)
      self.omega_dense = self.omega_dense.to(device)
      self.omega_float_dense = self.omega_float_dense.to(device)
    self.device = device
    return self


class MissingDataSample(object):
  def __init__(self, values, omega=None, indices=None, D=None,
        store_sparse=True, store_dense=False):
    """Represent a single sample from a dataset with missing entries. Can be
    constructed either from a mask or indices. Represented in a padded sparse
    format, or dense format.

    Args:
      values: data values, either (nnz,) (when indices given) or (D,) (when
        omega given).
      omega: observed entry mask (D,).
      indices: data indices (nnz,).
      D: data ambient dimension, required if omega not provided.
      store_dense, store_dense: whether to store sparse, dense representations
        of data.
    """
    if not (store_sparse or store_dense):
      raise ValueError("Either store_sparse or store_dense required.")
    if indices is None and omega is None:
      raise ValueError("Either indices or omega required.")
    elif indices is None:
      if D is None:
        D = omega.numel()
      elif D != omega.numel():
        raise ValueError("D doesn't match size of omega.")
      if values.numel() != omega.numel():
        raise ValueError("Sizes of values and omega don't match.")

      values = values.view(-1).float()
      omega = omega.view(-1).byte()
      indices = omega.nonzero().view(-1)
      nnz = indices.numel()
      if store_dense:
        # mask by missing entry mask to make sure there's no leakage.
        values_dense = values * omega.float()
        omega_dense = omega
      else:
        values_dense, omega_dense = None, None
      if store_sparse:
        values = values[indices]
        omega = omega[indices]
      else:
        values, omega, indices = None, None, None
    else:
      if D is None:
        raise ValueError("D required for indices inputs.")
      if values.numel() != indices.numel():
        raise ValueError("Sizes of values and indices don't match.")

      indices, sortIdx = indices.view(-1).long().sort()
      values = values.view(-1).float()[sortIdx]
      nnz = indices.numel()
      omega = torch.ones(nnz, dtype=torch.uint8)
      if store_dense:
        values_dense = torch.zeros(D, dtype=torch.float32)
        omega_dense = torch.zeros(D, dtype=torch.uint8)
        values_dense[indices] = values
        omega_dense[indices] = 1
      else:
        values_dense, omega_dense = None, None
      if not store_sparse:
        values, omega, indices = None, None, None

    # avoid issues of zero size dimensions in 0.4.1 by padding
    # use pad_nnz = 2 to avoid possible unwanted squeezing
    if nnz == 0 and store_sparse:
      indices = torch.tensor([0, 1], dtype=torch.int64)
      values = torch.tensor([0.0, 0.0], dtype=torch.float32)
      omega = torch.tensor([0, 0], dtype=torch.uint8)
      pad_nnz = 2
    else:
      pad_nnz = nnz

    self.values = values
    self.omega = omega
    self.indices = indices
    self.values_dense = values_dense
    self.omega_dense = omega_dense
    self.D = D
    self.store_sparse = store_sparse
    self.store_dense = store_dense
    self.nnz = nnz
    self.pad_nnz = pad_nnz
    self.shape = (D,)
    return

  def pad(self, pad_nnz):
    """Pad with pad_nnz zeros selected from initial indices. Only applies to
    sparse format.
    """
    if pad_nnz > self.D:
      raise ValueError("pad_nnz > D")
    if self.nnz > pad_nnz:
      raise ValueError("sparse vector nnz > pad_nnz.")

    # do nothing if not storing sparse format
    if not self.store_sparse:
      return

    if self.pad_nnz > pad_nnz:
      # strip leading padded values
      nstrip = self.pad_nnz - pad_nnz
      self.indices = self.indices[nstrip:]
      self.values = self.values[nstrip:]
      self.omega = self.omega[nstrip:]
    elif self.pad_nnz < pad_nnz:
      # pad with leading zeros
      npad = pad_nnz - self.pad_nnz
      self.indices = torch.cat([torch.zeros(npad, dtype=torch.int64),
          self.indices])
      self.values = torch.cat([torch.zeros(npad, dtype=torch.float32),
          self.values])
      self.omega = torch.cat([torch.zeros(npad, dtype=torch.uint8),
          self.omega])
    return


def generate_synth_uos_dataset(k, d, D, Ng, N=None, affine=False, sigma=0.0,
      theta=None, miss_rate=0.0, normalize=False, online=False,
      comp_test_frac=0.05, miss_store_sparse=True, miss_store_dense=False,
      seed=None):
  """Generate synthetic UoS dataset.

  Args:
    k: number of subspaces.
    d: subspace dimension.
    D: ambient dimension.
    Ng: points per group.
    N: total num data points, takes precedent over Ng (default: None).
    affine: affine setting (default: False).
    sigma: gaussian noise sigma (default: 0.0).
    theta: principal angle between subspaces in radians (default: None).
    miss_rate: missing data rate (default: 0.0).
    online: online data setting (default: False).
    normalize: project data onto unit sphere (default: False).
    comp_test_frac: fraction of observed entries to hold out as test in missing
      data setting (default: 0.05).
    miss_store_sparse: store sparse representation of missing data (default:
      True).
    miss_store_dense: store dense representation of missing data (default:
      False).
    seed: random seed (default: None).

  Returns:
    dataset: Synthetic UoS dataset instance.
  """
  # N takes precedent if provided.
  if N is not None and N > 0:
    Ng = N // k
  else:
    N = k * Ng

  if miss_rate <= 0:
    if not online:
      dataset = SynthUoSDataset(k, d, D, Ng, affine=affine, sigma=sigma,
          theta=theta, normalize=normalize, seed=seed)
    else:
      dataset = SynthUoSOnlineDataset(k, d, D, N, affine=affine, sigma=sigma,
          theta=theta, normalize=normalize, seed=seed)
  else:
    if not online:
      dataset = SynthUoSMissDataset(k, d, D, Ng, affine=affine, sigma=sigma,
          theta=theta, miss_rate=miss_rate, normalize=normalize,
          store_sparse=miss_store_sparse, store_dense=miss_store_dense,
          test_frac=comp_test_frac, seed=seed)
    else:
      dataset = SynthUoSMissOnlineDataset(k, d, D, N, affine=affine,
          sigma=sigma, theta=theta, miss_rate=miss_rate, normalize=normalize,
          store_sparse=miss_store_sparse, store_dense=miss_store_dense,
          test_frac=comp_test_frac, seed=seed)
  return dataset


def missing_data_collate(batch):
  """Collate missing data samples into batch."""
  # NOTE: maybe it would be more elegant to combine the two classes.
  x, grp, x0 = zip(*batch)
  grp = torch.stack(grp)
  x = MissingDataBatch(x)
  x0 = MissingDataBatch(x0)
  return x, grp, x0
