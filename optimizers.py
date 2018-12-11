from __future__ import print_function
from __future__ import division

import numpy as np
import torch
import torch.optim as optim
import torch.distributed as dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from models import KClusterModel

C_EPS = .01


class KClusterOptimizer(optim.Optimizer):
  """Base class for special purpose k-cluster optimizers."""
  def __init__(self, model, params, lr, lamb, momentum=0.0, nesterov=False,
        soft_assign=0.0):

    if not isinstance(model, KClusterModel):
      raise ValueError("Must provide k-cluster model instance.")
    if lr < 0.0:
      raise ValueError("Invalid learning rate: {}".format(lr))
    if lamb < 0.0:
      raise ValueError("Invalid regularization lambda: {}".format(lamb))
    if momentum < 0.0:
      raise ValueError("Invalid momentum value: {}".format(momentum))
    if nesterov and momentum <= 0:
      raise ValueError("Nesterov momentum requires a momentum")

    defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
    super(KClusterOptimizer, self).__init__(params, defaults)
    self.params_dict = {group['name']: group for group in self.param_groups}

    self.model = model
    self.k = model.k
    self.N = model.N
    self.lamb = lamb
    self.set_soft_assign(soft_assign)

    # disable gradients wrt C
    model.c.requires_grad = False
    return

  def __setstate__(self, state):
    super(KClusterOptimizer, self).__setstate__(state)
    # this from pytorch SGD, not sure why it's important
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

  def step(self, *args, **kwargs):
    raise NotImplementedError("step not implemented")

  def _step_C(self, x):
    """update assignment c in closed form."""
    c = self.model.c
    with torch.no_grad():
      # batch_size x n matrix of obj values.
      obj = self.objective(x, wrt='C')[0]
    obj = obj.data
    if self.soft_assign <= 0:
      minidx = obj.argmin(dim=1, keepdim=True)
      c.data.zero_()
      c.data.scatter_(1, minidx, 1)
    else:
      c.data.copy_(find_soft_assign(obj, self.soft_assign))
    # NOTE: add small eps, seems to help U updates slightly. But probably
    # should prune this.
    if C_EPS > 0:
      c.data.add_(C_EPS / self.k)
      c.data.div_(c.data.sum(dim=1, keepdim=True))
    return

  def objective(self, x, wrt='all'):
    """encapsulate model objective function. A little inelegant..."""
    return self.model.objective(x, lamb=self.lamb, wrt=wrt)


class KManifoldSGD(KClusterOptimizer):
  """Implements variant of SGD for special k-manifold case. Alternates between
  (1) closed form update to (soft) assignment C, (3) stochastic gradient update
  on U, V jointly (manifold model variables and coefficients).

  Args:
    model (KManifoldClusterModel instance): model to optimize.
    lr (float): learning rate
    lamb_U (float): U regularization parameter
    lamb_V (float): V regularization parameter.
    momentum (float, optional): momentum factor (default: 0)
    nesterov (bool, optional): enables Nesterov momentum (default: False)
    soft_assign (float, optional): update segmentation using soft assignment.
      0=exact, 1=uniform assignment (default: 0)
  """
  def __init__(self, model, lr, lamb_U, lamb_V=None, momentum=0.0,
        nesterov=False, soft_assign=0.0):

    lamb_V = lamb_U if lamb_V is None else lamb_V
    min_lamb = np.min([lamb_U, lamb_V])
    if min_lamb < 0.0:
      raise ValueError("Invalid regularization lambda: {}".format(min_lamb))

    params = [{'name': 'C', 'params': [model.c]},
        {'name': 'V', 'params': [model.v]},
        {'name': 'U', 'params': model.group_models.parameters()}]

    super(KManifoldSGD, self).__init__(model, params, lr, lamb_U, momentum,
        nesterov, soft_assign)

    self.lamb_U = lamb_U
    self.lamb_V = lamb_V
    return

  def step(self, ii, x):
    """Performs a single optimization step with alternating V, C, U updates.

    Args:
      ii (LongTensor): indices for current minibatch
      x (FloatTensor): current minibatch data
    """
    self.model.set_cv(ii)
    self._step_C(x)
    obj, loss, reg, Ureg, Vreg, x_ = self._step_U_V(ii, x)
    self.model.set_CV(ii)

    sprs = self.model.eval_sprs()
    norm_x_ = self.model.eval_shrink(x, x_)

    return obj, loss, reg, Ureg, Vreg, sprs, norm_x_

  def _step_U_V(self, ii, x):
    """stochastic gradient step on U, V variables (manifold models and
    coeffcients)"""
    obj, loss, reg, Ureg, Vreg, x_ = self.objective(x, wrt='all')
    self.zero_grad()
    obj.backward()

    # prefetch subset of momentum buffer for V
    if self.params_dict['V']['momentum'] > 0:
      v = self.params_dict['V']['params'][0]
      v_state = self.state[v]
      if 'momentum_buffer' not in v_state:
        v_state['momentum_buffer'] = torch.zeros_like(
            self.model.V.data)
        if self.model.use_cuda:
          v_state['momentum_buffer'] = v_state['momentum_buffer'].pin_memory()
      v_buf = v_state['momentum_buffer'][ii, :, :].to(v.data.device,
          non_blocking=True)

    for group_name in ['U', 'V']:
      group = self.params_dict[group_name]
      momentum = group['momentum']
      nesterov = group['nesterov']
      lr = group['lr']
      V_step = group_name == 'V'

      for p in group['params']:
        if p.grad is None:
          continue
        d_p = p.grad.data

        if V_step:
          # rescale gradients to compensate for c scaling on objective wrt V.
          d_p.div_(self.model.c.data.unsqueeze(1))

        if momentum > 0:
          param_state = self.state[p]
          if V_step:
            buf = v_buf
          else:
            if 'momentum_buffer' not in param_state:
              param_state['momentum_buffer'] = torch.zeros_like(p.data)
            buf = param_state['momentum_buffer']

          buf.mul_(momentum).add_(d_p)

          if V_step:
            # update momentum variable subset
            param_state['momentum_buffer'][ii, :, :] = buf.to(
                param_state['momentum_buffer'].device, non_blocking=True)

          if nesterov:
            d_p = d_p.add(momentum, buf)
          else:
            d_p = buf

        p.data.add_(-lr, d_p)
    return obj.data, loss.data, reg.data, Ureg.data, Vreg.data, x_.data

  def objective(self, x, wrt='all'):
    """encapsulate model objective function. A little inelegant..."""
    return self.model.objective(x, lamb_U=self.lamb_U,
        lamb_V=self.lamb_V, wrt=wrt)


class KSubspaceAltSGD(KClusterOptimizer):
  """Implements variant of SGD for special k-subspace case. Alternates between
  (1) closed form v solution by solving least-squares, (2) closed form update
  to (soft) assignment C, (3) stochastic gradient update on U (manifold model
  variables).

  Args:
    model (KManifoldClusterModel instance): model to optimize.
    lr (float): learning rate
    lamb_U (float): U regularization parameter
    lamb_V (float): V regularization parameter.
    momentum (float, optional): momentum factor (default: 0)
    nesterov (bool, optional): enables Nesterov momentum (default: False)
    soft_assign (float, optional): update segmentation using soft assignment.
      0=exact, 1=uniform assignment (default: 0)
    dist_mode (bool, optional): whether in mpi distributed mode
      (default: False).
  """
  def __init__(self, model, lr, lamb_U, lamb_V=None, momentum=0.0,
        nesterov=False, soft_assign=0.0, dist_mode=False):

    lamb_V = lamb_U if lamb_V is None else lamb_V
    min_lamb = np.min([lamb_U, lamb_V])
    if min_lamb < 0.0:
      raise ValueError("Invalid regularization lambda: {}".format(min_lamb))

    if dist_mode and not (dist.is_initialized()):
      raise RuntimeError("Distributed package not initialized")

    params = [{'name': 'C', 'params': [model.c]},
        {'name': 'V', 'params': [model.v]},
        {'name': 'U', 'params': model.group_models.parameters()}]

    super(KSubspaceAltSGD, self).__init__(model, params, lr, lamb_U, momentum,
        nesterov, soft_assign)

    # disable grads for V
    for p in self.params_dict['V']['params']:
      p.requires_grad = False

    self.lamb_U = lamb_U
    self.lamb_V = lamb_V
    self.lamb = None
    self.dist_mode = dist_mode

    if dist_mode:
      if dist.get_rank() == 0:
        print("Distributed mode with world={}, syncing parameters.".format(
            dist.get_world_size()))
      _sync_params(self.params_dict['U'])
    return

  def step(self, _, x):
    """Performs a single optimization step with alternating V, C, U updates.

    Args:
      ii (LongTensor): indices for current minibatch (not used, included for
        consistency)
      x (FloatTensor): current minibatch data
    """
    self._step_V(x)
    self._step_C(x)
    obj, loss, reg, Ureg, Vreg, x_ = self._step_U(x)

    sprs = self.model.eval_sprs()
    norm_x_ = self.model.eval_shrink(x, x_)
    return obj, loss, reg, Ureg, Vreg, sprs, norm_x_

  def _step_V(self, x):
    """closed form least-squares update to V variable (coefficients)."""
    v = self.model.v
    d = v.data.shape[1]

    # solve least squares by computing batched solution to normal equations.
    # (k x D x d)
    U = torch.stack([self.model.group_models[jj].U.data
        for jj in range(self.k)], dim=0)
    Ut = U.transpose(1, 2)
    # (k x d x d)
    UtU = torch.matmul(Ut, U)
    # (d x d)
    lambeye = torch.eye(d, dtype=UtU.dtype, device=UtU.device).mul(self.lamb_V)
    # (k x d x d)
    A = UtU.add(lambeye.unsqueeze(0))

    # (k x D x batch_size)
    B = x.data.t().unsqueeze(0).expand(self.k, -1, -1)
    if self.model.group_models[0].affine:
      B = B.sub(torch.stack([self.model.group_models[jj].b.data
          for jj in range(self.k)], dim=0).unsqueeze(2))
    # (k x d x batch_size)
    B = torch.matmul(Ut, B)

    # (k x d x batch_size)
    vt, _ = torch.gesv(B, A)
    v.data.copy_(vt.permute(2, 1, 0))
    return

  def _step_U(self, x):
    """stochastic gradient step on U variables (manifold models)"""
    group = self.params_dict['U']
    momentum = group['momentum']
    nesterov = group['nesterov']
    lr = group['lr']

    obj, loss, reg, Ureg, Vreg, x_ = self.objective(x, wrt='all')
    self.zero_grad()
    obj.backward()
    if self.dist_mode:
      _average_grads(group)

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
    return obj.data, loss.data, reg.data, Ureg.data, Vreg.data, x_.data

  def objective(self, x, wrt='all'):
    """encapsulate model objective function. A little inelegant..."""
    return self.model.objective(x, lamb_U=self.lamb_U,
        lamb_V=self.lamb_V, wrt=wrt)


class KManifoldAESGD(KClusterOptimizer):
  """Implements variant of SGD for special k-manifold case. Alternates between
  (1) closed form update to (soft) assignment C, (2) stochastic gradient update
  on U,V (manifold ae model variables).

  Args:
    model (KManifoldAEClusterModel instance): model to optimize.
    lr (float): learning rate
    lamb (float): regularization parameter
    momentum (float, optional): momentum factor (default: 0)
    nesterov (bool, optional): enables Nesterov momentum (default: False)
    soft_assign (float, optional): update segmentation using soft assignment.
      0=exact, 1=uniform assignment (default: 0)
    dist_mode (bool, optional): whether in mpi distributed mode
      (default: False).
  """
  def __init__(self, model, lr, lamb, momentum=0.0, nesterov=False,
        soft_assign=0.0, dist_mode=False):

    if dist_mode and not dist.is_initialized():
      raise RuntimeError("Distributed package not initialized")

    params = [{'name': 'C', 'params': [model.c]},
        {'name': 'U_V', 'params': model.group_models.parameters()}]
    super(KManifoldAESGD, self).__init__(model, params, lr, lamb, momentum,
        nesterov, soft_assign)

    self.dist_mode = dist_mode
    if dist_mode:
      if dist.get_rank() == 0:
        print("Distributed mode with world={}, syncing parameters.".format(
            dist.get_world_size()))
      _sync_params(self.params_dict['U_V'])
    return

  def step(self, _, x):
    """Performs a single optimization step with alternating C, U_V updates.

    Args:
      ii (LongTensor): indices for current minibatch (not used since c updated
        first in closed form)
      x (FloatTensor): current minibatch data
    """
    self._step_C(x)
    obj, loss, reg, x_ = self._step_U_V(x)
    # included for consistency
    Ureg = reg
    Vreg = 0.0

    sprs = self.model.eval_sprs()
    norm_x_ = self.model.eval_shrink(x, x_)
    return obj, loss, reg, Ureg, Vreg, sprs, norm_x_

  def _step_U_V(self, x):
    """stochastic gradient step on U, V variables (manifold models and
    coeffcients)"""
    group = self.params_dict['U_V']
    momentum = group['momentum']
    nesterov = group['nesterov']
    lr = group['lr']

    obj, loss, reg, x_ = self.objective(x, wrt='all')
    self.zero_grad()
    obj.backward()
    if self.dist_mode:
      _average_grads(group)

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


def _average_grads(group):
  """All reduce U gradients across processes.

  Follows: pytorch DistributedDataParallelCPU.
  """
  params = [p for p in group['params'] if p.requires_grad and
      p.grad is not None]
  grads = [p.grad.data for p in params]
  coalesced = _flatten_dense_tensors(grads)
  dist.all_reduce(coalesced, op=dist.reduce_op.SUM)
  coalesced /= dist.get_world_size()
  sync_grads = _unflatten_dense_tensors(coalesced, grads)
  for buf, synced in zip(grads, sync_grads):
    buf.copy_(synced)
  return


def _sync_params(group):
  """Broadcast current parameters from process 0.

  Follows: pytorch DistributedDataParallelCPU.
  """
  # only one time broadcast so multiple calls should be fine.
  for p in group['params']:
    dist.broadcast(p.data, 0)
  return
