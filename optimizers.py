from __future__ import print_function
from __future__ import division

import torch
import torch.optim as optim

C_EPS = .005


class KManifoldSparseSGD(optim.Optimizer):
  """Implements stochastic gradient descent (optionally with momentum) for
  special K-manifold case.

  Args:
    params (iterable): iterable of parameter group dicts. assumes each group
      has a 'name' key, and takes special care of 'C' and 'V' groups where
      gradients are sparse.
    n (int): number of groups
    N (int): number of examples in dataset
    lr (float): learning rate
    momentum (float, optional): momentum factor (default: 0)
    nesterov (bool, optional): enables Nesterov momentum (default: False)
    soft_assign (float, optional): update segmentation using soft assignment.
      0=exact, 1=uniform assignment (default: 0)
  """

  def __init__(self, params, n, N, lr, momentum=0., nesterov=False,
        soft_assign=0.0):
    if lr < 0.0:
      raise ValueError("Invalid learning rate: {}".format(lr))
    if momentum < 0.0:
      raise ValueError("Invalid momentum value: {}".format(momentum))
    if nesterov and momentum <= 0:
      raise ValueError("Nesterov momentum requires a momentum")

    defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
    super(KManifoldSparseSGD, self).__init__(params, defaults)

    self.n = n
    self.N = N
    self.set_soft_assign(soft_assign)
    return

  def __setstate__(self, state):
    super(KManifoldSparseSGD, self).__setstate__(state)
    for group in self.param_groups:
      group.setdefault('nesterov', False)
    return

  def set_soft_assign(self, soft_assign=0.0):
    """set soft assignment parameter. larger values mean softer, closer to
    uniform distribution."""
    if soft_assign <= 0.0:
      raise ValueError("soft assignment must be > 0")
    self.soft_assign = soft_assign
    return

  def step(self, ii, losses, prevc):
    """Performs a single optimization step.

    Args:
      ii (LongTensor): indices corresponding to current mini-batch. Used to
        look up correct momentum buffers for C, V.
      losses (Tensor): (batch_size, n) losses for exact C update.
      prevc (Tensor): (batch_size, n) previous segmentation coefficients for
        scaling V gradients.
    """
    for group in self.param_groups:
      group_name = group['name']
      momentum = group['momentum']
      nesterov = group['nesterov']

      if group_name == 'C':
        # update C in closed form
        p = group['params'][0]
        if self.soft_assign <= 0:
          minidx = losses.argmin(dim=1, keepdim=True)
          p.data.zero_()
          p.data.scatter_(1, minidx, 1)
        else:
          p.data.copy_(find_soft_assign(losses, self.soft_assign))
        # NOTE: add small eps to avoid divide by zero in case using V_scale
        # this will also affect U updates very slightly.
        p.data.add_(C_EPS / self.n)
        p.data.div_(p.data.sum(dim=1, keepdim=True))
        continue

      for p in group['params']:
        if p.grad is None:
          continue
        d_p = p.grad.data

        if group_name == 'V':
          # rescale gradients to compensate for c scaling on objective wrt V.
          d_p.div_(prevc.unsqueeze(1))

        if momentum != 0:
          param_state = self.state[p]
          if 'momentum_buffer' not in param_state:
            # for C and V need to keep track of full momentum buffers
            if group_name in ('C', 'V'):
              buf_shape = (self.N,) + p.shape[1:]
              buf = param_state['momentum_buffer'] = torch.zeros(buf_shape,
                  dtype=p.data.dtype, layout=p.data.layout,
                  device=p.data.device)
            else:
              buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
          else:
            buf = param_state['momentum_buffer']

          if group_name in ('C', 'V'):
            buf = buf[ii, :]
          buf.mul_(momentum).add_(d_p)

          if nesterov:
            d_p = d_p.add(momentum, buf)
          else:
            d_p = buf

          # update full momentum buffers
          if group_name in ('C', 'V'):
            param_state['momentum_buffer'][ii, :] = buf

        p.data.add_(-group['lr'], d_p)
    return


def find_soft_assign(losses, T=1.):
  """soft assignment found by shifting up negative losses by T, thresholding
  and normalizing.

  Args:
    losses (Tensor): (batch_size, n) matrix of loss values.
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
