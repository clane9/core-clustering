from __future__ import print_function
from __future__ import division

import torch
import torch.optim as optim

import ipdb

C_EPS = .05


class KManifoldSparseSGD(optim.Optimizer):
  """Implements stochastic gradient descent (optionally with momentum) for
  special K-manifold case.

  Args:
    params (iterable): iterable of parameter group dicts. assumes each group
      has a 'name' key, and takes special care of 'C' and 'V' groups where
      gradients are sparse.
    N (int): number of examples in dataset.
    lr (float): learning rate
    momentum (float, optional): momentum factor (default: 0)
    nesterov (bool, optional): enables Nesterov momentum (default: False)
    C_exact (bool, optional): update segmentation C in closed form.
    V_scale (bool, optional): scale V step by 1/C to accelerate.
  """

  def __init__(self, params, N, lr, momentum=0., nesterov=False,
        C_exact=False, V_scale=False):
    if lr < 0.0:
      raise ValueError("Invalid learning rate: {}".format(lr))
    if momentum < 0.0:
      raise ValueError("Invalid momentum value: {}".format(momentum))
    if nesterov and momentum <= 0:
      raise ValueError("Nesterov momentum requires a momentum")

    defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
    super(KManifoldSparseSGD, self).__init__(params, defaults)

    self.N = N
    self.C_exact = C_exact
    self.V_scale = V_scale
    return

  def __setstate__(self, state):
    super(KManifoldSparseSGD, self).__setstate__(state)
    for group in self.param_groups:
      group.setdefault('nesterov', False)
    return

  def step(self, ii, losses=None, cactiv=None):
    """Performs a single optimization step.

    Args:
      ii (LongTensor): indices corresponding to current mini-batch. Used to
        look up correct momentum buffers for C, V.
    """
    for group in self.param_groups:
      group_name = group['name']
      momentum = group['momentum']
      nesterov = group['nesterov']

      if group_name == 'C' and self.C_exact and (not losses is None):
        # update C in closed form
        p = group['params'][0]
        assignidx = torch.argmin(losses, dim=1, keepdim=True)
        # NOTE: add small eps to avoid divide by zero in case using V_scale
        # this will also affect U updates very slightly.
        n = p.data.shape[1]
        p.data.zero_().add_(C_EPS / n)
        p.data.scatter_(1, assignidx, 1-C_EPS*(n-1)/n)
        continue

      for p in group['params']:
        if p.grad is None:
          continue
        d_p = p.grad.data

        if group_name == 'V' and self.V_scale and (not cactiv is None):
          # rescale gradients to compensate for c scaling on objective wrt V.
          d_p.div_(cactiv.unsqueeze(1))

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
