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
    cluster_error = eval_cluster_error(groups, true_groups)

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
