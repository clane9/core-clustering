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


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar',
      best_filename='model_best.pth.tar'):
  """Save restart checkpoint.

  From: https://github.com/pytorch/examples/blob/master/imagenet/main.py
  """
  torch.save(state, filename)
  if is_best:
    shutil.copyfile(filename, best_filename)


def eval_cluster_error(groups, true_groups):
  """Evaluate clustering error between groups and true_groups.

  Returns cluster error and relabeled groups to match true_groups."""
  groups = best_map(groups, true_groups)
  cluster_error = 1.0 - np.mean(groups == true_groups)
  return cluster_error, groups


def best_map(groups1, groups2):
  """Find relabeling of groups1 that best matches groups2 using Hungarian
  algorithm.

  Ports matlab function bestMap writting by Deng Cai (dengcai AT gmail.com).
  (Although that function relabels groups2 to match groups1.)
  """
  labels1 = np.unique(groups1)
  labels2 = np.unique(groups2)
  nclass1 = np.size(labels1)
  nclass2 = np.size(labels2)
  nclass = np.max([nclass1, nclass2])

  C = np.zeros([nclass, nclass])
  for ii in range(nclass1):
    for jj in range(nclass2):
      C[ii, jj] = np.sum((groups1 == labels1[ii])*(groups2 == labels2[jj]))

  row_ind, col_ind = linear_sum_assignment(-C)
  new_groups1 = np.zeros(groups1.shape)
  for ii in row_ind:
    new_groups1[groups1 == labels1[ii]] = labels2[col_ind[ii]]
  return new_groups1
