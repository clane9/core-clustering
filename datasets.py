from __future__ import division
from __future__ import print_function

import math
import numpy as np
from scipy.io import loadmat
import torch
from torch.utils.data import Dataset, Sampler
from torchvision.datasets import MNIST
import torch.distributed as dist

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
        self.bs[:, ii] = b
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
    self.Idx = torch.arange(self.N)
    return

  def __len__(self):
    return self.N

  def __getitem__(self, ii):
    return self.Idx[ii], self.X[ii, :], self.groups[ii]


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
    return torch.tensor(ii), x, grp


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
    self.classes = np.arange(n)

    if seed is not None:
      torch.manual_seed(seed)
    rng = np.random.RandomState(seed=seed)

    self.group_models = [mod.ResidualManifoldModel(d, D, H,
        drop_p=0.0, res_lamb=0.0) for _ in range(n)]
    # scale weights to control smoothness
    for gm in self.group_models:
      gm.res_fc1.weight.data.mul_(res_weight_scale)
      gm.res_fc2.weight.data.mul_(res_weight_scale)

    self.planted_model = mod.KManifoldClusterModel(n, d, D, self.N, self.N,
        self.group_models)
    self.planted_model.eval()

    # generate true groups and segmentation
    self.groups = np.arange(n, dtype=np.int64).reshape(-1, 1)
    self.groups = np.tile(self.groups, (1, Ng)).reshape(-1)

    self.planted_model.C.zero_()
    self.planted_model.C.scatter_(1,
        torch.from_numpy(self.groups).view(-1, 1), 1)

    # generate union of manifold data
    ii = torch.arange(self.N, dtype=torch.int64)
    with torch.no_grad():
      self.X = self.planted_model(ii)
      self.X.mul_(self.planted_model.C.unsqueeze(1))
      self.X = self.X.sum(dim=2)

    if sigma > 0.:
      E = torch.randn(self.N, D).mul(sigma/np.sqrt(D))
      self.X += E

    # permute order of data
    self.perm = rng.permutation(self.N)
    self.X = self.X[self.perm, :]
    self.groups = self.groups[self.perm]

    self.groups = torch.tensor(self.groups, dtype=torch.int64)
    self.Idx = torch.arange(self.N)
    return

  def __len__(self):
    return self.N

  def __getitem__(self, ii):
    return self.Idx[ii], self.X[ii, :], self.groups[ii]


class MNISTUoM(MNIST):
  """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
      root (string): Root directory of dataset where ``processed/training.pt``
        and  ``processed/test.pt`` exist.
      train (bool, optional): If True, creates dataset from ``training.pt``,
        otherwise from ``test.pt``.
      download (bool, optional): If true, downloads the dataset from the
        internet and puts it in root directory. If dataset is already
        downloaded, it is not downloaded again.
      transform (callable, optional): A function/transform that  takes in an
        PIL image and returns a transformed version. E.g,
        ``transforms.RandomCrop``
      target_transform (callable, optional): A function/transform that takes in
        the target and transforms it.
      classes (sequence, optional): Subset of digits to include. Note that
        original digit labels are kept.
    """
  def __init__(self, root, train=True, transform=None, target_transform=None,
        download=False, classes=None, batch_size=100):
    super(MNISTUoM, self).__init__(root, train, transform, target_transform,
        download)

    self.classes = classes
    if self.classes is not None:
      self.classes = np.array(self.classes)
      if np.setdiff1d(self.classes, np.arange(10)).size > 0:
        raise ValueError("Invalid classes.")

      if self.train:
        self.train_data, self.train_labels = self._subset_classes(
            self.train_data, self.train_labels, self.classes)
      else:
        self.test_data, self.test_labels = self._subset_classes(
            self.test_data, self.test_labels, self.classes)
    else:
      self.classes = np.arange(10)

    # make sure batch_size divides N
    N = len(self)
    if batch_size <= 0 or batch_size > N:
      batch_size = N
    self.batch_size = batch_size
    N = (N // batch_size)*batch_size
    # NOTE: separately named train_* and test_* is annoying, and fixed in more
    # recent torchvision
    if self.train:
      self.train_data = self.train_data[:N, :]
      self.train_labels = self.train_labels[:N]
    else:
      self.test_data = self.test_data[:N, :]
      self.test_labels = self.test_labels[:N]

    self.Idx = torch.arange(len(self))
    return

  def _subset_classes(self, data, labels, classes):
    """Restrict data to subset of classes."""
    mask = (labels.view(-1, 1) == torch.tensor(classes).view(1, -1)).any(dim=1)
    data = data[mask, :]
    labels = labels[mask]
    return data, labels

  def __getitem__(self, index):
    """
    Args:
      index (int): Index
    Returns:
      tuple: (index, image, target) where target is index of the target class.
    """
    img, target = super(MNISTUoM, self).__getitem__(index)
    return self.Idx[index], img, target


class MNISTScatPCAUoS(Dataset):
  """MNIST after scattering transform feature extraction and PCA to D=500.

  Following (You et al., CVPR 2016)."""
  def __init__(self, matfile='MNIST_SC_pca.mat'):
    mnist_sc_pca = loadmat(matfile)

    self.X = torch.tensor(mnist_sc_pca['X'].T, dtype=torch.float32)
    self.groups = torch.tensor(mnist_sc_pca['MNIST_LABEL'], dtype=torch.int64).view(-1)
    self.classes = np.arange(10)
    self.n = 10

    # normalize data points (rows) of X
    self.X.div_(torch.norm(self.X, p=2, dim=1).view(-1, 1).add(1e-8))
    self.N, self.D = self.X.shape
    self.Idx = torch.arange(self.N, dtype=torch.int64)
    return

  def __len__(self):
    return self.N

  def __getitem__(self, ii):
    return self.Idx[ii], self.X[ii, :], self.groups[ii]


# taken from pytorch 1.0.0 so that data sampling order consistent regardless
# world size.
class DistributedSampler(Sampler):
  """Sampler that restricts data loading to a subset of the dataset.

  It is especially useful in conjunction with
  :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
  process can pass a DistributedSampler instance as a DataLoader sampler,
  and load a subset of the original dataset that is exclusive to it.

  .. note::
    Dataset is assumed to be of constant size.

  Arguments:
    dataset: Dataset used for sampling.
    num_replicas (optional): Number of processes participating in
      distributed training.
    rank (optional): Rank of the current process within num_replicas.
  """

  def __init__(self, dataset, num_replicas=None, rank=None):
    if num_replicas is None:
      if not dist.is_available():
        raise RuntimeError("Requires distributed package to be available")
      num_replicas = dist.get_world_size()
    if rank is None:
      if not dist.is_available():
        raise RuntimeError("Requires distributed package to be available")
      rank = dist.get_rank()
    self.dataset = dataset
    self.num_replicas = num_replicas
    self.rank = rank
    self.epoch = 0
    self.num_samples = int(math.ceil(len(self.dataset) *
        (1.0 / self.num_replicas)))
    self.total_size = self.num_samples * self.num_replicas

  def __iter__(self):
    # deterministically shuffle based on epoch
    g = torch.Generator()
    g.manual_seed(self.epoch)
    indices = torch.randperm(len(self.dataset), generator=g).tolist()

    # add extra samples to make it evenly divisible
    indices += indices[:(self.total_size - len(indices))]
    assert len(indices) == self.total_size

    # subsample
    indices = indices[self.rank:self.total_size:self.num_replicas]
    assert len(indices) == self.num_samples

    return iter(indices)

  def __len__(self):
    return self.num_samples

  def set_epoch(self, epoch):
    self.epoch = epoch
