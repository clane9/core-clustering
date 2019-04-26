from __future__ import print_function
from __future__ import division

import numpy as np
import shutil
from scipy.optimize import linear_sum_assignment, bisect
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd
import torch
import torch.nn.functional as F

EPS = 1e-8


class AverageMeter(object):
  """Computes and stores the average and current value.

  From: https://github.com/pytorch/examples/blob/master/imagenet/main.py
  """
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count


def get_learning_rate(optimizer):
  return np.median([param_group['lr']
    for param_group in optimizer.param_groups])


def adjust_learning_rate(optimizer, new_lr):
  for param_group in optimizer.param_groups:
    param_group['lr'] = new_lr


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar',
      best_filename='model_best.pth.tar'):
  """Save restart checkpoint.

  From: https://github.com/pytorch/examples/blob/master/imagenet/main.py
  """
  torch.save(state, filename)
  if is_best:
    shutil.copyfile(filename, best_filename)


def eval_cluster_error(*args, **kwargs):
  """Evaluate clustering error.

  Examples:
    cluster_error, conf_mat = eval_cluster_error(conf_mat)
    cluster_error, conf_mat = eval_cluster_error(groups, true_groups, k)

  Args:
    conf_mat: (n, n) group confusion matrix
    groups: (N,) group assignment
    true_groups: (N,) true group assignment
  """
  if len(args) not in (1, 2):
    raise ValueError("Invalid number of arguments")

  if len(args) == 1:
    conf_mat = args[0]
  else:
    groups = args[0]
    true_groups = args[1]
    # number of groups and labels will be inferred from true_groups.
    conf_mat = eval_confusion(groups, true_groups, k=kwargs.get('k'))

  if torch.is_tensor(conf_mat):
    conf_mat = conf_mat.numpy()
  if conf_mat.ndim != 2:
    raise ValueError("Invalid format for confusion matrix")
  conf_mat = np.round(conf_mat).astype(np.int64)

  row_ind, col_ind = linear_sum_assignment(-conf_mat)
  correct = conf_mat[row_ind, col_ind].sum()
  N = conf_mat.sum()
  cluster_error = 1.0 - correct/N

  if row_ind.size < conf_mat.shape[0]:
    row_ind = np.concatenate([row_ind,
      np.setdiff1d(np.arange(conf_mat.shape[0]), row_ind)])
  if col_ind.size < conf_mat.shape[1]:
    col_ind = np.concatenate([col_ind,
      np.setdiff1d(np.arange(conf_mat.shape[1]), col_ind)])
  map_Idx = (row_ind, col_ind)
  return cluster_error, map_Idx


def eval_confusion(groups, true_groups, k, true_classes=None):
  """compute confusion matrix between assigned and true groups

  Note: group labels assumed to be integers (0, ..., k-1).
  """
  if torch.is_tensor(groups):
    groups = groups.cpu().numpy()
  if torch.is_tensor(true_groups):
    true_groups = true_groups.cpu().numpy()
  if groups.size != true_groups.size:
    raise ValueError("groups true_groups must have the same size")

  if true_classes is not None:
    true_groups = np.argmax(true_groups.reshape(-1, 1) ==
        true_classes.reshape(1, -1), axis=1)
    true_k = true_classes.shape[0]
  else:
    true_labels, true_groups = np.unique(true_groups, return_inverse=True)
    true_k = true_labels.shape[0]
  conf_mat = np.zeros((k, true_k))

  groups = groups.reshape(-1)
  true_groups = true_groups.reshape(-1)
  groups_stack = np.stack((groups, true_groups), axis=1)
  Idx, counts = np.unique(groups_stack, axis=0, return_counts=True)
  conf_mat[Idx[:, 0], Idx[:, 1]] = counts
  return conf_mat


def rank(X, tol=.01):
  """Evaluate approximate rank of X."""
  _, svs, _ = torch.svd(X)
  return (svs > tol*svs.max()).sum(), svs


def find_soft_assign(losses, T=1.):
  """soft assignment found by shifting up negative losses by T, thresholding
  and normalizing.

  Args:
    losses (Tensor): (Nb, n) matrix of loss values.
    T (float): soft assignment parameter in (0, \infty).

  .. note::
    For any T there is a tau such that the returned c is a solution to

      min_c c^T l  s.t.  c >= 0, c^T 1 = 1, ||c||_2^2 <= tau

    Since the optimality conditions of this problem yield

      c^* = 1/gamma( lambda \1 - l)_+

    for gamma, lambda Lagrange multipliers.
  """
  if T <= 0:
    raise ValueError("soft assignment T must be > 0.")
  # normalize loss to that T has consistent interpretation.
  # NOTE: minimum loss can't be exactly zero
  loss_min, _ = losses.min(dim=1, keepdim=True)
  losses = losses.div(loss_min)
  c = torch.clamp(T + 1. - losses, min=0.)
  c = c.div(c.sum(dim=1, keepdim=True))
  return c


def coherence(U1, U2, normalize=False):
  """Compute coherence between U1, U2.

  Args:
    U1, U2: (D x d) matrices
    normalize: normalize columns of U1, U2 (default: False)

  Returns:
    coh: scalar coherence loss
  """
  d = U1.size(1)
  if normalize:
    U1 = unit_normalize(U1, p=2, dim=0)
    U2 = unit_normalize(U2, p=2, dim=0)
  coh = torch.matmul(U1.t(), U2).pow(2).sum().div(d)
  return coh


def unit_normalize(X, p=2, dim=None):
  """Normalize X

  Args:
    X: tensor
    p: norm order
    dim: dimension to normalize

  returns:
    unitX: normalized tensor
  """
  unitX = X.div(torch.norm(X, p=p, dim=dim, keepdim=True).add(EPS))
  return unitX


def batch_svd(X, out=None):
  shape = X.shape
  if len(shape) < 3:
    raise ValueError("Invalid X value, should have >= 1 batch dim.")

  X = X.contiguous()  # to ensure viewing works as expected
  batch_dims = shape[:-2]
  batch_D = np.prod(batch_dims)
  m, n = shape[-2:]
  d = min(m, n)
  X = X.view(batch_D, m, n)

  if out is not None:
    U, s, V = out
    # check strides
    if U.stride()[-2:] != (1, m):
      raise ValueError("U has invalid stride, must be stored as transpose")
    if s.stride()[-1] != 1:
      raise ValueError("s has invalid stride")
    if V.stride()[-2:] != (n, 1):
      raise ValueError("V has invalid stride")

    U = U.view(batch_D, m, d)
    s = s.view(batch_D, d)
    V = V.view(batch_D, n, d)
  else:
    U = torch.zeros((batch_D, d, m), dtype=X.dtype, device=X.device)
    # must be stored as transpose
    U = U.transpose(1, 2)
    s = torch.zeros((batch_D, d), dtype=X.dtype, device=X.device)
    V = torch.zeros((batch_D, n, d), dtype=X.dtype, device=X.device)

  for idx in range(batch_D):
    torch.svd(X[idx, :], some=True, out=(U[idx, :], s[idx, :], V[idx, :]))

  U = U.view(batch_dims + (m, d))
  s = s.view(batch_dims + (d,))
  V = V.view(batch_dims + (n, d))
  return U, s, V


def reg_pca(X, d, form='proj', lamb=0.0, gamma=0.0, affine=False,
      solver='randomized'):
  """Solve one of the regularized PCA problems

  (proj):

  min_U  1/2 || (X - 1 b^T) -  (X - 1 b^T) U U^T ||_F^2
      + lambda || U ||_F^2 + gamma || U U^T ||_F

  (mf):

  min_{U,V}  1/2 || (X - 1 b^T) -  U V^T ||_F^2
      + lambda/2 (|| U ||_F^2 + || V ||_F^2)

  Args:
    X: dataset (N, D).
    form: either 'proj' or 'mf' (default: 'proj')
    lamb, gamma: regularization parameters (default: 0.0).
    affine: Use affine PCA model with bias b.
    solver: SVD solver ('randomized', 'svds', 'svd').

  Returns:
    U: subspace basis with containing top singular vectors, possibly with
      column norm shrinkage if lamb, gamma > 0 (D, d).
    b: subspace bias (d,).
  """
  if form not in {'proj', 'mf'}:
    raise ValueError("Invalid form {}".format(form))

  if not torch.is_tensor(X):
    X = torch.from_numpy(X)
  device = X.device

  if affine:
    b = X.mean(dim=0)
    X = X.sub(b)
  else:
    b = None

  if X.shape[0] < d:
    # special case where N < d
    _, s, U = torch.svd(X, some=False)
    s = torch.cat((s,
        torch.zeros(d - s.shape[0], dtype=s.dtype, device=s.device)))
    U = U[:, :d]
  elif solver == 'svd':
    _, s, U = torch.svd(X)
    s, U = s[:d], U[:, :d]
  else:
    X = X.cpu().numpy()
    if solver == 'svds':
      _, s, Ut = svds(X, d)
    else:
      _, s, Ut = randomized_svd(X, d)
    s, Ut = [torch.from_numpy(z).to(device) for z in (s, Ut)]
    U = Ut.t()

  # shrink column norms
  if max(lamb, gamma) > 0:
    nz_mask = s >= max(s[0], 1)*EPS
    s_nz = s[nz_mask]

    if form == 'mf':
      z = F.relu(s_nz - lamb).sqrt()
    else:
      s_sqr = s_nz.pow(2)
      s_sqr_shrk = F.relu(s_sqr - lamb)
      if gamma == 0:
        z = s_sqr_shrk.div(s_sqr)
      else:
        if torch.norm(s_sqr_shrk) <= gamma + EPS:
          z = torch.zeros_like(s_nz)
        else:
          # find ||z||_2 by bisection
          def norm_test(beta):
            z = s_sqr_shrk.div(s_sqr + gamma/beta)
            return torch.norm(z) - beta

          a, b = 1e-4, np.sqrt(s.shape[0])
          while norm_test(a) < 0:
            a /= 10
            if a <= 1e-16:
              raise RuntimeError("Bisection zero-finding failed.")

          beta = bisect(norm_test, a, b, xtol=EPS, rtol=EPS)
          z = s_sqr_shrk.div(s_sqr + gamma/beta)
      z = z.sqrt()
    U[:, nz_mask] *= z
    U[:, nz_mask == 0] = 0.0
  return U, b
