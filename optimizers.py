from __future__ import print_function
from __future__ import division

import torch
import torch.optim as optim

import utils as ut
from models import KManifoldAEClusterModel

# import ipdb

C_EPS = .01


class KManifoldAESGD(optim.Optimizer):
  """Implements variant of SGD for special k-manifold case. Alternates between
  (1) closed form update to (soft) assignment C, (2) stochastic gradient update
  on U,V (manifold ae model variables).

  Args:
    KMmodel (KManifoldAEClusterModel instance): model to optimize.
    lr (float): learning rate
    lamb (float): regularization parameter
    momentum (float, optional): momentum factor (default: 0)
    nesterov (bool, optional): enables Nesterov momentum (default: False)
    soft_assign (float, optional): update segmentation using soft assignment.
      0=exact, 1=uniform assignment (default: 0)
  """
  def __init__(self, KMmodel, lr, lamb, momentum=0.0, nesterov=False,
        soft_assign=0.0):

    if not isinstance(KMmodel, KManifoldAEClusterModel):
      raise ValueError("Must provide k-manifold ae model.")
    if lr < 0.0:
      raise ValueError("Invalid learning rate: {}".format(lr))
    if lamb < 0.0:
      raise ValueError("Invalid regularization lambda: {}".format(lamb))
    if momentum < 0.0:
      raise ValueError("Invalid momentum value: {}".format(momentum))
    if nesterov and momentum <= 0:
      raise ValueError("Nesterov momentum requires a momentum")

    # NOTE: lr, momentum, nesterov not used for C
    defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
    params = [{'name': 'C', 'params': [KMmodel.c]},
        {'name': 'UV', 'params': KMmodel.group_models.parameters()}]

    # disable gradients wrt C
    KMmodel.c.requires_grad = False

    super(KManifoldAESGD, self).__init__(params, defaults)
    self.params_dict = {group['name']: group for group in self.param_groups}

    self.KMmodel = KMmodel
    self.n = KMmodel.n
    self.N = KMmodel.N
    self.lamb = lamb
    self.set_soft_assign(soft_assign)
    return

  def __setstate__(self, state):
    super(KManifoldAESGD, self).__setstate__(state)
    for group in self.param_groups:
      group.setdefault('nesterov', False)
    return

  def set_soft_assign(self, soft_assign=0.0):
    """set soft assignment parameter. larger values mean softer, closer to
    uniform distribution."""
    if soft_assign < 0.0:
      raise ValueError("soft assignment must be >= 0")
    self.soft_assign = soft_assign
    return

  def step(self, x, groups):
    """Performs a single optimization step with alternating V, C, U updates.

    Args:
      x (FloatTensor): current minibatch data
      groups (LongTensor): current minibatch true group assignment
    """
    self._step_C(x)
    obj, loss, reg, x_ = self._step_U_V(x)

    sprs = self.KMmodel.eval_sprs()
    norm_x_ = self.KMmodel.eval_shrink(x, x_)
    conf_mat = ut.eval_confusion(self.KMmodel.get_groups(), groups, n=self.n)
    return obj, loss, reg, sprs, norm_x_, conf_mat

  def _step_U_V(self, x):
    """stochastic gradient step on U, V variables (manifold models and
    coeffcients)"""
    obj, loss, reg, x_ = self.KMmodel.objective(x, lamb=self.lamb, wrt='all')
    self.zero_grad()
    obj.backward()

    group = self.params_dict['UV']
    momentum = group['momentum']
    nesterov = group['nesterov']
    lr = group['lr']

    for p in group['params']:
      if p.grad is None:
        continue
      d_p = p.grad.data

      if momentum > 0:
        param_state = self.state[p]
        if 'momentum_buffer' not in param_state:
          param_state['momentum_buffer'] = torch.zeros_like(p.data)
        buf = param_state['momentum_buffer']

        buf.mul_(momentum).add_(d_p)

        if nesterov:
          d_p = d_p.add(momentum, buf)
        else:
          d_p = buf

      p.data.add_(-lr, d_p)
    return obj.data, loss.data, reg.data, x_.data

  def _step_C(self, x):
    """update assignment c in closed form."""
    c = self.params_dict['C']['params'][0]
    # Nb x n matrix of loss + V reg values.
    with torch.no_grad():
      obj, _, _, _ = self.KMmodel.objective(x, lamb=self.lamb, wrt='C')
    obj = obj.data
    if self.soft_assign <= 0:
      minidx = obj.argmin(dim=1, keepdim=True)
      c.data.zero_()
      c.data.scatter_(1, minidx, 1)
    else:
      c.data.copy_(find_soft_assign(obj, self.soft_assign))
    # NOTE: add small eps, seems to help U updates slightly. But probably
    # should prune.
    if C_EPS > 0:
      c.data.add_(C_EPS / self.n)
      c.data.div_(c.data.sum(dim=1, keepdim=True))
    return


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
