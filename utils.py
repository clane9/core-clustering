from __future__ import print_function
from __future__ import division

import numpy as np
import shutil
import torch
from scipy.optimize import linear_sum_assignment


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
    cluster_error, conf_mat = eval_cluster_error(groups, true_groups)

  Args:
    conf_mat: (n, n) group confusion matrix
    groups: (N,) group assignment
    true_groups: (N,) true group assignment
    sort_conf_mat: (bool) return sorted confusion matrix sort confusion matrix.
  """
  if len(args) not in (1, 2):
    raise ValueError("Invalid number of arguments")

  if len(args) == 1:
    conf_mat = args[0]
  else:
    groups = args[0]
    true_groups = args[1]
    # number of groups and labels will be inferred from groups and true_groups.
    conf_mat = eval_confusion(groups, true_groups)

  if torch.is_tensor(conf_mat):
    conf_mat = conf_mat.numpy()
  if conf_mat.ndim != 2:
    raise ValueError("Invalid format for confusion matrix")
  conf_mat = np.round(conf_mat).astype(np.int64)

  row_ind, col_ind = linear_sum_assignment(-conf_mat)
  correct = conf_mat[row_ind, col_ind].sum()
  N = conf_mat.sum()
  cluster_error = 1.0 - correct/N

  # re-order conf_mat
  if 'sort_conf_mat' in kwargs and kwargs['sort_conf_mat']:
    if row_ind.size < conf_mat.shape[0]:
      row_ind = np.concatenate([row_ind,
        np.setdiff1d(np.arange(conf_mat.shape[0]), row_ind)])
    if col_ind.size < conf_mat.shape[1]:
      col_ind = np.concatenate([col_ind,
        np.setdiff1d(np.arange(conf_mat.shape[1]), col_ind)])
    conf_mat = conf_mat[row_ind, :][:, col_ind]
  return cluster_error, conf_mat


def eval_confusion(groups, true_groups, k=None, true_k=None,
      true_classes=None):
  """compute confusion matrix between assigned and true groups"""
  if isinstance(groups, np.ndarray):
    groups = torch.from_numpy(groups)
  if isinstance(true_groups, np.ndarray):
    true_groups = torch.from_numpy(true_groups)
  if groups.numel() != true_groups.numel():
    raise ValueError("groups true_groups must have the same size")

  with torch.no_grad():
    if k is not None:
      classes = torch.arange(k).view(1, -1)
      if true_classes is not None:
        true_classes = torch.tensor(true_classes).view(1, -1)
      elif true_k is not None:
        true_classes = torch.arange(true_k).view(1, -1)
      else:
        true_classes = classes
    else:
      classes = torch.unique(groups).view(1, -1)
      true_classes = torch.unique(true_groups).view(1, -1)

    groups = groups.view(-1, 1)
    true_groups = true_groups.view(-1, 1)

    groups_onehot = (groups == classes).type(torch.int64)
    true_groups_onehot = (true_groups == true_classes).type(torch.int64)
    # float32 used here so that single all reduce can be used in distributed
    # setting. Is there a case where float32 might be inaccurate?
    conf_mat = torch.matmul(groups_onehot.t(),
        true_groups_onehot).type(torch.float32)
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
