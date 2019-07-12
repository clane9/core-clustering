"""Extract scattering network features from image datasets."""

import time

import numpy as np
from kymatio import Scattering2D
import torch
from torchvision import datasets, transforms
from sklearn.utils.extmath import randomized_svd


def extract_scattering(dataset, scattering, transform=None, pca_d=None,
      num_workers=4, use_cuda=False):
  dataset = dataset.lower()
  if dataset not in {'mnist', 'cifar10'}:
    raise ValueError("Only mnist and cifar10 supported.")
  use_cuda = use_cuda and torch.cuda.is_available()
  device = torch.device('cuda' if use_cuda else 'cpu')
  if use_cuda:
    scattering = scattering.cuda()

  # load data
  if dataset == 'mnist':
    data_train = datasets.MNIST('datasets/mnist', train=True,
        transform=transform)
    data_test = datasets.MNIST('datasets/mnist', train=False,
        transform=transform)
  else:
    data_train = datasets.CIFAR10('datasets/cifar10/CIFAR10', train=True,
        transform=transform)
    data_test = datasets.CIFAR10('datasets/cifar10/CIFAR10', train=False,
        transform=transform)
  train_loader = torch.utils.data.DataLoader(data_train, batch_size=100,
      shuffle=False, num_workers=num_workers, pin_memory=use_cuda)
  test_loader = torch.utils.data.DataLoader(data_test, batch_size=100,
      shuffle=False, num_workers=num_workers, pin_memory=use_cuda)

  # transform and append data
  data, labels = [], []
  nbatches = 4
  itr = 0
  stop = False
  tic = time.time()
  for loader in [train_loader, test_loader]:
    for x, y in loader:
      itr += 1
      x = normalize_and_flatten(scattering(x.to(device)))
      data.append(x)
      labels.append(y)
      rtime = time.time() - tic
      print('done {} batch {} ({:.4f}s)'.format(dataset, itr, rtime))
      tic = time.time()
      if itr == nbatches:
        stop = True
        break
    if stop:
      break

  data = torch.cat(data).cpu().numpy()
  labels = torch.cat(labels).numpy()

  # optional pca
  if pca_d is not None:
    data, _ = pca(data, pca_d, center=False)
  return data, labels


def normalize_and_flatten(x):
  # x output from scattering transform has shape (*, M, p, p)
  # will normalize along last two dimensions
  shape = x.shape
  newshape = shape[:-2] + (-1,)
  x = x.view(newshape)

  xnorm = x.abs().max(dim=-1, keepdim=True)[0]
  x = x.div(xnorm)

  # flatten again, this time all but batch dim
  x = x.view(x.shape[0], -1)
  return x


def pca(X, d, center=True):
  """PCA on X (n_samples, n_features)."""
  if center:
    X = X - X.mean(axis=0)

  V, s, Ut = randomized_svd(X, d)
  Y = V * s
  U = Ut.T
  return Y, U


if __name__ == '__main__':
  pca_d = 500
  use_cuda = True
  num_workers = 4
  do_mnist = True
  do_cifar10 = True

  if do_mnist:
    mnist_scattering = Scattering2D(J=3, shape=(32, 32))
    mnist_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
    mnist_data, mnist_labels = extract_scattering('mnist', mnist_scattering,
        transform=mnist_transform, pca_d=pca_d, num_workers=num_workers,
        use_cuda=use_cuda)
    with open('datasets/mnist/mnist_scat_pca.npz', 'wb') as f:
      np.savez(f, data=mnist_data, labels=mnist_labels)

  if do_cifar10:
    cifar10_scattering = Scattering2D(J=3, shape=(32, 32))
    cifar10_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    cifar10_data, cifar10_labels = extract_scattering('cifar10',
        cifar10_scattering, transform=cifar10_transform, pca_d=pca_d,
        num_workers=num_workers, use_cuda=use_cuda)
    with open('datasets/cifar10/cifar10_scat_pca.npz', 'wb') as f:
      np.savez(f, data=cifar10_data, labels=cifar10_labels)
