from __future__ import print_function
from __future__ import division

from collections import OrderedDict
import shutil
import gc

import numpy as np
from scipy.optimize import linear_sum_assignment, bisect
from scipy.sparse.linalg import svds
from sklearn.utils.extmath import randomized_svd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd

EPS = 1e-8

# use torch.solve in >=1.1.0 and torch.gesv in 0.4.1
solve = torch.solve if hasattr(torch, 'solve') else torch.gesv


class _BSScheduler(object):
  def __init__(self, dataset, dl_kwargs, last_epoch=0):
    self.dataset = dataset
    self.dl_kwargs = dl_kwargs
    self.last_epoch = last_epoch
    self.batch_size = dl_kwargs['batch_size']

  def update_bs(self):
    raise NotImplementedError

  def step(self, epoch=None):
    # returns a new DataLoader instance when batch size updated. this is not
    # too elegant, but not overly expensive either.
    if epoch is None:
      epoch = self.last_epoch + 1
    self.last_epoch = epoch
    return self.update_bs()

  def new_data_loader(self):
    return DataLoader(self.dataset, **self.dl_kwargs)


class LambdaBS(_BSScheduler):
  def __init__(self, dataset, dl_kwargs, bs_lambda, last_epoch=0):
    self.bs_lambda = bs_lambda
    super(LambdaBS, self).__init__(dataset, dl_kwargs, last_epoch)

  def update_bs(self):
    batch_size = self.bs_lambda(self.last_epoch)
    updated = False
    if batch_size != self.batch_size:
      self.batch_size = self.dl_kwargs['batch_size'] = batch_size
      updated = True
    return updated


class ClampDecay(object):
  def __init__(self, initval, step_size, gamma, min=None, max=None):
    self.initval = initval
    self.step_size = step_size
    self.gamma = gamma
    self.minval = min
    self.maxval = max

  def __call__(self, ii):
    val = clamp(self.initval * (self.gamma ** (ii // self.step_size)),
        minval=self.minval, maxval=self.maxval)
    return val


def clamp(x, minval=None, maxval=None):
  if minval is not None:
    x = max(x, minval)
  if maxval is not None:
    x = min(x, maxval)
  return x


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


def boolarg(arg):
  return bool(int(arg))


def reset_optimizer_state(model, optimizer, rIdx):
  """Reset optimizer states to zero for re-initialized replicates &
  clusters. Or, if copy=True, copy states from duplicated clusters."""
  for p in [model.Us, model.bs]:
    if p is None or not p.requires_grad:
      continue
    state = optimizer.state[p]
    for key, val in state.items():
      if isinstance(val, torch.Tensor) and val.shape == p.shape:
        # assuming all parameters have (r, k) as first dims.
        val[rIdx, :] = 0.0
  return


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


def assign_and_value(assign_obj, compute_c=True):
  """Compute assignments and cluster size & values."""
  batch_size, r, k = assign_obj.shape
  device = assign_obj.device

  if k > 1:
    top2obj, top2idx = torch.topk(assign_obj, 2, dim=2, largest=False,
        sorted=True)
    groups = top2idx[:, :, 0]
    min_assign_obj = top2obj[:, :, 0]
  else:
    groups = torch.zeros(batch_size, r, device=device, dtype=torch.int64)
    min_assign_obj = assign_obj.squeeze(2)

  if compute_c:
    c = torch.zeros_like(assign_obj)
    c.scatter_(2, groups.unsqueeze(2), 1)
    c_mean = c.mean(dim=0)
  else:
    c = c_mean = None

  if k > 1:
    value = torch.zeros_like(assign_obj)
    value = value.scatter_(2, groups.unsqueeze(2),
        (top2obj[:, :, 1] - top2obj[:, :, 0]).unsqueeze(2)).mean(dim=0)
  else:
    value = torch.ones(r, k, device=device)
  return groups, min_assign_obj, c, c_mean, value


def update_ema_metric(val, ema, ema_decay=0.99, inplace=True):
  """Update an exponential moving average tensor.

  Args:
    val: current value.
    ema: moving average (possibly with nan entries).
    ema_decay: (default: 0.99)
    inplace: (default: True)

  Returns:
    ema
  """
  if not inplace:
    ema = torch.clone(ema)
  nan_mask = torch.isnan(ema)
  ema[nan_mask] = val[nan_mask]
  ema.mul_(ema_decay).add_(1-ema_decay, val)
  return ema


def cos_knn(y, X, k, normalize=False):
  """Find knn in absolute cosine distance.

  Args:
    y: targets, shape (n, D).
    X: dataset, shape (N, D).

  Returns:
    knnIdx: indices of nearest neighbors, shape (n, k).
  """
  with torch.no_grad():
    if y.dim() == 1:
      y = y.view(1, -1)
    if normalize:
      y = unit_normalize(y, dim=1)
      X = unit_normalize(X, dim=1)

    # (N, n)
    cos_abs = torch.matmul(X, y.t()).abs()
    # (n, k)
    knnIdx = (torch.topk(cos_abs, k, dim=0)[1]).t()
  return knnIdx


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


def shift_and_zero(buf):
  """Shift 1 -> 0, and zero out 1."""
  buf[0, :] = buf[1, :]
  buf[1, :] = 0.0
  return buf


def min_med_max(x):
  """Compute min, median, max of tensor x."""
  return [x.min().item(), x.median().item(), x.max().item()]


def batch_svd(X, out=None):
  """Compute batch svd of X

  Args:
    X: shape (*, m, n)

  Returns:
    U, s, V: batch svds, shape (*, m, d), (*, d), (*, n, d) (d = min(m, n))
  """
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


def batch_ridge(B, A, lamb=0.0):
  """Solve regularized least-squares problem

      min_X 1/2 || A X - B ||_F^2 + \lambda/2 || X ||_F^2

  by explicitly solving normal equations

      (A^T A + \lambda I) X = A^T B

  Args:
    B: shape (*, m, n)
    A: shape (*, m, p)

  Returns:
    X: shape (*, p, n)
  """
  dim = A.dim()
  p = A.shape[-1]

  At = A.transpose(dim-2, dim-1)
  AtA = torch.matmul(At, A)
  AtB = torch.matmul(At, B)

  if lamb > 0:
    lambeye = torch.eye(p, dtype=A.dtype, device=A.device).mul_(lamb)
    AtA = AtA.add_(lambeye)

  X, _ = solve(AtB, AtA)
  return X


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


def print_cuda_tensors():
  """Print active cuda tensors.

  Adapted from:
  https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741
  """
  objs = gc.get_objects()
  obj_counts = {}
  obj_dev_shapes = []
  for obj in objs:
    try:
      if (torch.is_tensor(obj) or (hasattr(obj, 'data') and
            torch.is_tensor(obj.data))):
        device = str(obj.device)
        shape = tuple(obj.shape)
        key = (device, shape)
        if key in obj_counts:
          obj_counts[key] += 1
        else:
          obj_dev_shapes.append(key)
          obj_counts[key] = 1
    except Exception:
      pass

  for key in sorted(obj_dev_shapes):
    if 'cuda' in key[0]:
      print('{}: {}'.format(key, obj_counts[key]))
  return


def unique_resets(reset_cids):
  """Choose last occurrence of each cidx."""
  revIdx = np.arange(reset_cids.shape[0] - 1, -1, -1)
  _, uniqIdx = np.unique(reset_cids[revIdx], axis=0, return_index=True)
  uniqIdx = revIdx[uniqIdx]
  return uniqIdx


def aggregate_resets(resets):
  """Aggregate resets for each epoch and replicate."""
  columns = ['epoch', 'step', 'ridx', 'core.step', 'cidx', 'cand.ridx',
      'cand.cidx', 'obj.decr', 'cumu.obj.decr']
  dtypes = 7*[int] + 2*[float]
  resets_dict = {columns[ii]: resets[:, ii].astype(dtypes[ii])
      for ii in range(len(columns))}
  resets = pd.DataFrame(data=resets_dict)
  grouped = resets.groupby(['epoch', 'step', 'ridx'], as_index=False)
  agg_resets = grouped.agg(OrderedDict([
      ('core.step', ['count']),
      ('obj.decr', ['min', 'median', 'max']),
      ('cumu.obj.decr', ['min', 'median', 'max'])]))
  agg_resets.columns = (
      ['epoch', 'step', 'ridx', 'core.steps'] +
      ['{}.{}'.format(met, meas) for met in ['obj.decr', 'cumu.obj.decr']
          for meas in ['min', 'med', 'max']])
  return agg_resets


def set_auto_reg_params(k, d, D, Ng, sigma_hat, min_size=0.0):
  """Compute "optimal" regularization parameters based on expected singular
  value distribution.

  Key resource is (Gavish & Donoho, 2014). In general, we know that the noise
  singular values will have a right bulk edge at:
      (sqrt{N_j} + sqrt{D - d}) (\sigma / sqrt{D})
  whereas the data singular values will have a right bulk edge at:
      (sqrt{N_j} + sqrt{d}) sqrt{1/ d + \sigma^2 / D}
  The regularization parameters are set so that the "inside reg" will threshold
  all noise svs, whereas the "outside reg" will threshold all noise + data svs
  as soon as the cluster becomes too small.
  """
  if sigma_hat < 0:
    raise ValueError("sigma hat {} should be >= 0.".format(sigma_hat))
  if min_size >= 1 or min_size < 0:
    raise ValueError("min size {} should be in [0, 1).".format(min_size))
  U_frosqr_in_lamb = ((1.0 + np.sqrt((D - d)/Ng))**2 * (sigma_hat**2 / D))
  U_frosqr_out_lamb = ((min_size / k) * (1.0 / d + sigma_hat**2 / D))
  z_lamb = 0.01 if max(U_frosqr_in_lamb, U_frosqr_out_lamb) > 0 else 0.0
  return U_frosqr_in_lamb, U_frosqr_out_lamb, z_lamb
