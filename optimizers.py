from __future__ import print_function
from __future__ import division

import numpy as np
import torch
import torch.optim as optim

from models import KClusterModel
import utils as ut

# import ipdb

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
    self.n = model.n
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
      c.data.add_(C_EPS / self.n)
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

    params = [{'name': 'C', 'params': [model.c]},
        {'name': 'V', 'params': [model.v]},
        {'name': 'U', 'params': model.group_models.parameters()}]

    super(KManifoldSGD, self).__init__(model, params, lr, lamb_U, momentum,
        nesterov, soft_assign)

    self.lamb_U = lamb_U
    self.lamb_V = lamb_U if lamb_V is None else lamb_V
    return

  def step(self, ii, x, groups):
    """Performs a single optimization step with alternating V, C, U updates.

    Args:
      ii (LongTensor): indices for current minibatch
      x (FloatTensor): current minibatch data
      groups (numpy ndarray): minibatch true groups
    """
    self.model.set_cv(ii)
    self._step_C(x)
    obj, loss, reg, Ureg, Vreg, x_ = self._step_U_V(ii, x)
    self.model.set_CV(ii)

    sprs = self.model.eval_sprs()
    norm_x_ = self.model.eval_shrink(x, x_)
    conf_mat = ut.eval_confusion(self.model.get_groups(), groups, n=self.n)
    return obj, loss, reg, Ureg, Vreg, sprs, norm_x_, conf_mat

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


class KManifoldAltSGD(KManifoldSGD):
  """Implements variant of SGD for special k-manifold case. Alternates between
  (1) one or a few exact gradient updates on V (coefficients), (2) closed form
  update to (soft) assignment C, (3) stochastic gradient update on U (manifold
  model variables).

  Args:
    model (KManifoldClusterModel instance): model to optimize.
    lr (float): learning rate
    lamb_U (float): U regularization parameter
    lamb_V (float): V regularization parameter.
    momentum (float, optional): momentum factor (default: 0)
    nesterov (bool, optional): enables Nesterov momentum (default: False)
    maxit_V (int, optional): number of gradient iterations for V update
      (default: 20).
    soft_assign (float, optional): update segmentation using soft assignment.
      0=exact, 1=uniform assignment (default: 0)
  """
  def __init__(self, model, lr, lamb_U, lamb_V=None, momentum=0.0,
        nesterov=False, maxit_V=20, soft_assign=0.0):
    if maxit_V <= 0:
      raise ValueError("Invalid max V iters: {}".format(maxit_V))

    super(KManifoldAltSGD, self).__init__(model, lr, lamb_U, lamb_V,
        momentum, nesterov, soft_assign)
    self.maxit_V = maxit_V
    return

  def step(self, ii, x, groups):
    """Performs a single optimization step with alternating V, C, U updates.

    Args:
      ii (LongTensor): indices for current minibatch
      x (FloatTensor): current minibatch data
      groups (numpy ndarray): minibatch true groups
    """
    self.model.set_cv(ii)
    self._step_V(ii, x)
    self._step_C(x)
    obj, loss, reg, Ureg, Vreg, x_ = self._step_U(x)
    self.model.set_CV(ii)

    sprs = self.model.eval_sprs()
    norm_x_ = self.model.eval_shrink(x, x_)
    conf_mat = ut.eval_confusion(self.model.get_groups(), groups, n=self.n)
    return obj, loss, reg, Ureg, Vreg, sprs, norm_x_, conf_mat

  def _step_V(self, ii, x):
    """one or a few exact gradient steps on V variable (manifold
    coefficients). no acceleration."""
    group = self.params_dict['V']
    lr = group['lr']
    v = group['params'][0]

    # adaptive step size constants following trust region framework
    lr_decr, lr_min, lr_max = 0.5, 1e-4, 1e4
    success_thr, fail_thr = 0.9, .01
    # NOTE: should these scalars be transferred off the gpu if using cuda?
    tol = torch.norm(v.data).mul(1e-5)

    # freeze C, U
    v.requires_grad = True
    for name in ['C', 'U']:
      for p in self.params_dict[name]['params']:
        p.requires_grad = False

    # gradient descent iteration with steps chosen by trust region success
    # condition
    # NOTE: will this strategy work well if manifold embedding is not smooth?
    obj = self.objective(x, wrt='V')[0]
    prev_obj = obj.data
    g_norm = torch.ones_like(tol).mul(np.inf)
    backtracking = False
    itr = 0
    while itr < self.maxit_V and g_norm >= tol:
      # bw pass
      if not backtracking:
        self.zero_grad()
        obj.backward()
        g_v = v.grad.data
        g_normsqr = g_v.pow(2).sum()
        g_norm = g_normsqr.sqrt()

      # grad step
      v.data.add_(-lr, g_v)
      obj = self.objective(x, wrt='V')[0]

      # adaptive step size using trust region success condition
      pred_decr = g_normsqr.mul(lr)
      obs_decr = prev_obj - obj.data
      model_fit = obs_decr/pred_decr

      if model_fit < fail_thr:
        # failure case
        # reset variable and decrease step
        v.data.add_(lr, g_v)
        if lr <= lr_min:
          # increment iteration counter if already hit lr_min
          itr += 1
        else:
          lr = np.clip(lr_decr*lr, lr_min, lr_max)
        backtracking = True
      else:
        if model_fit > success_thr:
          # very successful case, increase step
          lr = np.clip(lr/lr_decr, lr_min, lr_max)
        prev_obj = obj.data
        itr += 1
        backtracking = False

      # print(('k={:d}, obj={:.5e}, |g|={:.3e}, pd={:.3e}, od={:.3e}, '
      #     'fit={:.3f}, lr={:.3e}').format(itr, obj.data, g_norm, pred_decr,
      #         obs_decr, model_fit, lr))
    return

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

    obj, loss, reg, Ureg, Vreg, x_ = self.objective(x, wrt='all')
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
    return obj.data, loss.data, reg.data, Ureg.data, Vreg.data, x_.data


class KSubspaceAltSGD(KManifoldAltSGD):
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
  """
  def _step_V(self, _, x):
    """closed form least-squares update to V variable (coefficients)."""
    v = self.model.v
    d = v.data.shape[1]

    # solve least squares by computing batched solution to normal equations.
    # (n x D x d)
    U = torch.stack([self.model.group_models[jj].U.data
        for jj in range(self.n)], dim=0)
    # (n x d x d)
    Ut = U.transpose(1, 2)
    UtU = torch.matmul(Ut, U)
    # (d x d)
    lambeye = torch.eye(d, dtype=UtU.dtype, device=UtU.device).mul(self.lamb_V)
    # (n x d x d)
    A = UtU.add(lambeye.unsqueeze(0))

    # (n x D x batch_size)
    B = x.data.t().unsqueeze(0).expand(self.n, -1, -1)
    if self.model.group_models[0].affine:
      B = B.sub(torch.stack([self.model.group_models[jj].b.data
          for jj in range(self.n)], dim=0).unsqueeze(2))
    # (n x d x batch_size)
    B = torch.matmul(Ut, B)

    # (n x d x batch_size)
    vt, _ = torch.gesv(B, A)
    v.data.copy_(vt.permute(2, 1, 0))
    return


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
  """
  def __init__(self, model, lr, lamb, momentum=0.0, nesterov=False,
        soft_assign=0.0):

    params = [{'name': 'C', 'params': [model.c]},
        {'name': 'U_V', 'params': model.group_models.parameters()}]
    super(KManifoldAESGD, self).__init__(model, params, lr, lamb, momentum,
        nesterov, soft_assign)
    return

  def step(self, _, x, groups):
    """Performs a single optimization step with alternating V, C, U updates.

    Args:
      ii (LongTensor): indices for current minibatch (not used)
      x (FloatTensor): current minibatch data
      groups (numpy ndarray): minibatch true groups
    """
    self._step_C(x)
    obj, loss, reg, x_ = self._step_U_V(x)
    # included for consistency
    Ureg = reg
    Vreg = 0.0

    sprs = self.model.eval_sprs()
    norm_x_ = self.model.eval_shrink(x, x_)
    conf_mat = ut.eval_confusion(self.model.get_groups(), groups, n=self.n)
    return obj, loss, reg, Ureg, Vreg, sprs, norm_x_, conf_mat

  def _step_U_V(self, x):
    """stochastic gradient step on U, V variables (manifold models and
    coeffcients)"""
    obj, loss, reg, x_ = self.objective(x, wrt='all')
    self.zero_grad()
    obj.backward()

    group = self.params_dict['U_V']
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
