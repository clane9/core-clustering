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
    cluster_error = eval_cluster_error(conf_mat)
    cluster_error = eval_cluster_error(groups, true_groups, n=None)

  Args:
    conf_mat: (n, n) group confusion matrix
    groups: (N,) group assignment
    true_groups: (N,) true group assignment
    n (optional): number of groups (default: infer from true_groups)
  """
  if len(args) < 1 or len(args) > 3:
    raise ValueError("Invalid number of arguments")
  elif len(args) == 1:
    conf_mat = args[0]
    if ((not isinstance(conf_mat, np.ndarray)) or
        (len(conf_mat.shape) != 2) or
            (conf_mat.shape[0] != conf_mat.shape[1])):
      raise ValueError("Invalid format for confusion matrix")
  else:
    groups = args[0]
    true_groups = args[1]
    if len(args) == 3:
      n = args[2]
    elif 'n' in kwargs:
      n = kwargs['n']
    else:
      n = None
    conf_mat = eval_confusion(groups, true_groups, n)

  row_ind, col_ind = linear_sum_assignment(-conf_mat)
  correct = conf_mat[row_ind, col_ind].sum()
  N = conf_mat.sum()
  cluster_error = 1.0 - correct/N
  return cluster_error


def eval_confusion(groups, true_groups, n=None):
  """compute confusion matrix between assigned and true groups"""
  if torch.is_tensor(groups):
    groups = groups.cpu().numpy()
  if torch.is_tensor(true_groups):
    true_groups = true_groups.cpu().numpy()
  if np.size(groups) != np.size(true_groups):
    raise ValueError("groups true_groups must have the same size")

  if n is not None:
    labels = np.arange(n).reshape((1, n))
    labels_true = labels
  else:
    labels = np.unique(groups)
    labels_true = np.unique(true_groups)

  groups = groups.reshape((-1, 1))
  true_groups = true_groups.reshape((-1, 1))

  groups_onehot = (groups == labels).astype(np.int64)
  true_groups_onehot = (true_groups == labels_true).astype(np.int64)
  conf_mat = np.matmul(groups_onehot.T, true_groups_onehot)
  return conf_mat
