from __future__ import print_function
from __future__ import division

import numpy as np
import torch


class StackedPadSparseVector(object):
  def __init__(self, sparse_vectors):
    """Store a set of N sparse vectors as an (N x max nnz ) matrix with each
    row zero-padded as necessary.

    Args:
      sparse_vectors: list of PadSparseVectors, each with the same shape.
    """
    nnz, shapes = zip(*[(v.nnz, v.shape) for v in sparse_vectors])
    shapes = set(shapes)
    if len(shapes) > 1:
      raise ValueError("inputs must all have the same shape.")
    self.nnz = torch.tensor(nnz)
    self.max_nnz = self.nnz.max().item()

    indices, values, omega = [], [], []
    for v in sparse_vectors:
      v.pad(self.max_nnz)
      indices.append(v.indices)
      values.append(v.values)
      omega.append(v.omega)

    self.indices = torch.stack(indices)
    self.values = torch.stack(values)
    self.omega = torch.stack(omega)

    self.shape = (self.indices.shape[0], shapes.pop()[0])
    self.device = self.indices.device
    return

  def to(self, device):
    if not isinstance(device, torch.device):
      raise ValueError("to only supported for devices.")
    self.indices = self.indices.to(device)
    self.values = self.values.to(device)
    self.omega = self.omega.to(device)
    self.device = device
    return self


class PadSparseVector(object):
  def __init__(self, indices, values, shape):
    if max(indices.dim(), values.dim(), len(shape)) > 1:
      raise ValueError("inputs must be vectors.")
    if values.shape[0] != indices.shape[0]:
      raise ValueError("indices, values must be the same size.")
    if indices.shape[0] > 0 and indices.max() >= shape[0]:
      raise ValueError("shape is too small for given indices.")

    self.nnz = indices.shape[0]
    self.shape = tuple(shape)
    if self.nnz == 0:
      # to avoid errors with concatenating empty tensors
      self.indices = torch.tensor([0, 1], dtype=torch.int64)
      self.values = torch.tensor([0.0, 0.0], dtype=torch.float32)
      self.omega = torch.tensor([0, 0], dtype=torch.uint8)
      self.pad_nnz = 2
    else:
      sortIdx = torch.sort(indices)[1]
      self.indices = indices[sortIdx]
      self.values = values[sortIdx].float()
      self.omega = torch.ones(self.nnz, dtype=torch.uint8)
      self.pad_nnz = self.nnz
    return

  def pad(self, pad_nnz):
    """Pad with pad_nnz zeros selected from initial indices."""
    if self.nnz > pad_nnz:
      raise RuntimeError("sparse vector nnz > pad_nnz.")
    elif self.pad_nnz == pad_nnz:
      return

    indices = self.indices.cpu().numpy()
    values = self.values.cpu().numpy()
    omega = self.omega.cpu().numpy() > 0
    if self.pad_nnz > self.nnz:
      indices = indices[omega]
      values = values[omega]

    pad_indices = np.setdiff1d(np.arange(pad_nnz), indices, assume_unique=True)
    pad_indices = pad_indices[:pad_nnz - self.nnz]
    pad_indices = np.union1d(pad_indices, indices)
    pad_omega = np.isin(pad_indices, indices, assume_unique=True)
    pad_values = np.zeros(pad_indices.shape, dtype=values.dtype)
    pad_values[pad_omega] = values

    self.pad_nnz = pad_nnz
    self.indices = torch.from_numpy(pad_indices)
    self.values = torch.from_numpy(pad_values)
    self.omega = torch.from_numpy(pad_omega.astype(np.uint8))
    return


def pad_sparse_collate(batch):
  x, grp, x0 = zip(*batch)
  grp = torch.stack(grp)
  x = StackedPadSparseVector(x)
  x0 = StackedPadSparseVector(x0)
  return x, grp, x0
