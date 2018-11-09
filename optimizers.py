from __future__ import print_function
from __future__ import division

import numpy as np
import torch
import torch.optim as optim
from models import KManifoldClusterModel

# import ipdb

C_EPS = .01


class KManifoldAltSGD(optim.Optimizer):
  """Implements variant of SGD for special k-manifold case. Alternates between
  (1) one or a few exact gradient updates on V (coefficients), (2) closed form
  update to (soft) assignment C, (3) stochastic gradient update on U (manifold
  model variables).

  Args:
    KMmodel (KManifoldClusterModel instance): model to optimize.
    lr (float): learning rate
    lamb_U (float): U regularization parameter
    lamb_V (float): V regularization parameter.
    momentum (float, optional): momentum factor (default: 0)
    nesterov (bool, optional): enables Nesterov momentum (default: False)
    soft_assign (float, optional): update segmentation using soft assignment.
      0=exact, 1=uniform assignment (default: 0)
    maxit_V (int, optional): number of gradient iterations for V update.
  """
  def __init__(self, KMmodel, lr, lamb_U, lamb_V=None, momentum=0.0,
        nesterov=False, soft_assign=0.0, maxit_V=1):

    lamb_V = lamb_U if lamb_V is None else lamb_V

    if not isinstance(KMmodel, KManifoldClusterModel):
      raise ValueError("Must proved k-manifold model.")
    if lr < 0.0:
      raise ValueError("Invalid learning rate: {}".format(lr))
    if np.min([lamb_U, lamb_V]) < 0.0:
      raise ValueError("Invalid regularization lambda: {}".format(
          np.min([lamb_U, lamb_V])))
    if momentum < 0.0:
      raise ValueError("Invalid momentum value: {}".format(momentum))
    if nesterov and momentum <= 0:
      raise ValueError("Nesterov momentum requires a momentum")
    if maxit_V < 0:
      raise ValueError("Invalid V maxit value: {}".format(maxit_V))

    defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
    params = [{'name': 'C', 'params': [KMmodel.c]},
        {'name': 'V', 'params': [KMmodel.v]},
        {'name': 'U', 'params': KMmodel.group_models.parameters()}]

    super(KManifoldAltSGD, self).__init__(params, defaults)
    self.params_dict = {group['name']: group for group in self.param_groups}

    self.KMmodel = KMmodel
    self.n = KMmodel.n
    self.N = KMmodel.N
    self.lamb_U = lamb_U
    self.lamb_V = lamb_V
    self.set_soft_assign(soft_assign)
    self.maxit_V = maxit_V
    return

  def __setstate__(self, state):
    super(KManifoldAltSGD, self).__setstate__(state)
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

  def step(self, ii, x):
    """Performs a single optimization step with alternating V, C, U updates.

    Args:
      ii (LongTensor): indices for current minibatch
      x (FloatTensor): current minibatch data
    """
    self.KMmodel.set_cv(ii)
    obj_decr_V = self._step_V(ii, x)
    self._step_C(x)
    self._step_U(x)
    self.KMmodel.set_CV(ii)

    obj, loss, reg, Ureg, Vreg, x_ = self.KMmodel.objective(x,
        lamb_U=self.lamb_U, lamb_V=self.lamb_V, wrt='all')
    sprs = self.KMmodel.eval_sprs()
    norm_x_ = self.KMmodel.eval_shrink(x, x_)
    return obj, loss, reg, Ureg, Vreg, obj_decr_V, sprs, norm_x_

  def _step_V(self, ii, x):
    """one or a few exact gradient steps on V variable (manifold
    coefficients)."""
    group = self.params_dict['V']
    momentum = group['momentum']
    nesterov = group['nesterov']
    lr = group['lr']

    v = group['params'][0]
    v_state = self.state[v]

    # NOTE: how useful is it to maintain a global momentum buffer? Maybe these
    # get stale quickly with all the U updates?
    if momentum > 0:
      # initialize momentum buffer for full dataset
      if 'momentum_buffer' not in v_state:
        v_state['momentum_buffer'] = torch.zeros_like(self.KMmodel.V)
      buf = v_state['momentum_buffer'][ii, :, :]

    # freeze C, U
    v.requires_grad = True
    for name in ['C', 'U']:
      for p in self.params_dict[name]['params']:
        p.requires_grad = False

    # gradient descent iteration with O(1/k) decaying steps
    obj = self.KMmodel.objective(x, lamb_V=self.lamb_V, wrt='V')[0]
    init_obj = obj.data
    for kk in range(1, self.maxit_V+1):
      # bw pass
      self.zero_grad()
      obj.backward()

      # momentum step
      d_v = v.grad.data
      if momentum > 0:
        buf.mul_(momentum).add_(d_v)
        if nesterov:
          d_v = d_v.add(momentum, buf)
        else:
          d_v = buf
      # NOTE: What's the best way to choose step sizes here?
      v.data.add_(-lr/kk, d_v)

      obj = self.KMmodel.objective(x, lamb_V=self.lamb_V, wrt='V')[0]

    # update subset of momentum buffer for next iteration.
    v_state['momentum_buffer'][ii, :, :] = buf

    self.zero_grad()

    obj_decr = (init_obj - obj.data)/init_obj
    return obj_decr

  def _step_C(self, x):
    """update assignment c in closed form."""
    c = self.params_dict['C']['params'][0]
    # Nb x n matrix of loss + V reg values.
    obj = self.KMmodel.objective(x, lamb_V=self.lamb_V, wrt='C')[0]
    if self.soft_assign <= 0:
      minidx = obj.argmin(dim=1, keepdim=True)
      c.data.zero_()
      c.data.scatter_(1, minidx, 1)
    else:
      c.data.copy_(find_soft_assign(obj, self.soft_assign))
    # NOTE: add small eps, seems to help U updates slightly. But probably
    # should prune this once find the right soft assignment setting.
    # this will also affect U updates very slightly.
    if C_EPS > 0:
      c.data.add_(C_EPS / self.n)
      c.data.div_(c.data.sum(dim=1, keepdim=True))
    return obj.data

  def _step_U(self, x):
    """stochastic gradient step on U variables (manifold models)"""
    group = self.params_dict['U']
    momentum = group['momentum']
    nesterov = group['nesterov']
    lr = group['lr']

    # freeze V, C
    for p in group['params']:
      p.requires_grad = True
    for name in ['C', 'V']:
      for p in self.params_dict[name]['params']:
        p.requires_grad = False

    obj = self.KMmodel.objective(x, lamb_U=self.lamb_U, wrt='U')[0]
    self.zero_grad()
    obj.backward()

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
    self.zero_grad()
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
