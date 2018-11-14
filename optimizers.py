from __future__ import print_function
from __future__ import division

import numpy as np
import torch
import torch.optim as optim
from models import KManifoldClusterModel

# import ipdb

C_EPS = .01


class KManifoldSGD(optim.Optimizer):
  """Implements variant of SGD for special k-manifold case. Alternates between
  (1) closed form update to (soft) assignment C, (3) stochastic gradient update
  on U, V jointly (manifold model variables and coefficients).

  Args:
    KMmodel (KManifoldClusterModel instance): model to optimize.
    lr (float): learning rate
    lamb_U (float): U regularization parameter
    lamb_V (float): V regularization parameter.
    momentum (float, optional): momentum factor (default: 0)
    nesterov (bool, optional): enables Nesterov momentum (default: False)
    soft_assign (float, optional): update segmentation using soft assignment.
      0=exact, 1=uniform assignment (default: 0)
  """
  def __init__(self, KMmodel, lr, lamb_U, lamb_V=None, momentum=0.0,
        nesterov=False, soft_assign=0.0):

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

    # NOTE: lr, momentum, nesterov not used for C
    defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov)
    params = [{'name': 'C', 'params': [KMmodel.c]},
        {'name': 'V', 'params': [KMmodel.v]},
        {'name': 'U', 'params': KMmodel.group_models.parameters()}]

    # disable gradients wrt C
    KMmodel.c.requires_grad = False

    super(KManifoldSGD, self).__init__(params, defaults)
    self.params_dict = {group['name']: group for group in self.param_groups}

    self.KMmodel = KMmodel
    self.n = KMmodel.n
    self.N = KMmodel.N
    self.lamb_U = lamb_U
    self.lamb_V = lamb_V
    self.set_soft_assign(soft_assign)
    return

  def __setstate__(self, state):
    super(KManifoldSGD, self).__setstate__(state)
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
    obj, loss, reg, Ureg, Vreg, x_ = self._step_U_V(ii, x)
    self._step_C(x)
    self.KMmodel.set_CV(ii)

    sprs = self.KMmodel.eval_sprs()
    norm_x_ = self.KMmodel.eval_shrink(x, x_)
    return obj, loss, reg, Ureg, Vreg, sprs, norm_x_

  def _step_U_V(self, ii, x):
    """stochastic gradient step on U, V variables (manifold models and
    coeffcients)"""
    obj, loss, reg, Ureg, Vreg, x_ = self.KMmodel.objective(
        x, lamb_U=self.lamb_U, lamb_V=self.lamb_V, wrt='all')
    self.zero_grad()
    obj.backward()

    # prefetch subset of momentum buffer for V
    if self.params_dict['V']['momentum'] > 0:
      v = self.params_dict['V']['params'][0]
      v_state = self.state[v]
      if 'momentum_buffer' not in v_state:
        v_state['momentum_buffer'] = torch.zeros_like(
            self.KMmodel.V.data)
        if self.KMmodel.use_cuda:
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
          d_p.div_(self.KMmodel.c.data.unsqueeze(1))

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

  def _step_C(self, x):
    """update assignment c in closed form."""
    c = self.params_dict['C']['params'][0]
    # Nb x n matrix of loss + V reg values.
    with torch.no_grad():
      obj, _, _, _, _, _ = self.KMmodel.objective(x, lamb_V=self.lamb_V,
          wrt='C')
    obj = obj.data
    if self.soft_assign <= 0:
      minidx = obj.argmin(dim=1, keepdim=True)
      c.data.zero_()
      c.data.scatter_(1, minidx, 1)
    else:
      c.data.copy_(find_soft_assign(obj, self.soft_assign))
    # NOTE: add small eps, seems to help U updates slightly. Also needed to
    # avoid divide by zero in V update, due to scaling gradient by 1/c.
    c.data.add_(C_EPS / self.n)
    c.data.div_(c.data.sum(dim=1, keepdim=True))
    return


class KManifoldAltSGD(KManifoldSGD):
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
    maxit_V (int, optional): number of gradient iterations for V update
      (default: 20).
    soft_assign (float, optional): update segmentation using soft assignment.
      0=exact, 1=uniform assignment (default: 0)
  """
  def __init__(self, KMmodel, lr, lamb_U, lamb_V=None, momentum=0.0,
        nesterov=False, maxit_V=20, soft_assign=0.0):
    if maxit_V <= 0:
      raise ValueError("Invalid max V iters: {}".format(maxit_V))

    super(KManifoldAltSGD, self).__init__(KMmodel, lr, lamb_U, lamb_V,
        momentum, nesterov, soft_assign)
    self.maxit_V = maxit_V
    return

  def step(self, ii, x):
    """Performs a single optimization step with alternating V, C, U updates.

    Args:
      ii (LongTensor): indices for current minibatch
      x (FloatTensor): current minibatch data
    """
    self.KMmodel.set_cv(ii)
    self._step_V(ii, x)
    self._step_C(x)
    self._step_U(x)
    self.KMmodel.set_CV(ii)

    with torch.no_grad():
      obj, loss, reg, Ureg, Vreg, x_ = self.KMmodel.objective(x,
          lamb_U=self.lamb_U, lamb_V=self.lamb_V, wrt='all')
    sprs = self.KMmodel.eval_sprs()
    norm_x_ = self.KMmodel.eval_shrink(x, x_)

    # torch.cuda.empty_cache()
    return obj, loss, reg, Ureg, Vreg, sprs, norm_x_

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
    obj, _, _, _, _, _ = self.KMmodel.objective(x, lamb_V=self.lamb_V, wrt='V')
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
      obj, _, _, _, _, _ = self.KMmodel.objective(x, lamb_V=self.lamb_V,
          wrt='V')

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

    obj, _, _, _, _, _ = self.KMmodel.objective(x, lamb_U=self.lamb_U, wrt='U')
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
    return


class KSubspaceAltSGD(KManifoldAltSGD):
  """Implements variant of SGD for special k-subspace case. Alternates between
  (1) closed form v solution by solving least-squares, (2) closed form update
  to (soft) assignment C, (3) stochastic gradient update on U (manifold model
  variables).

  Args:
    KMmodel (KManifoldClusterModel instance): model to optimize.
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
    v = self.KMmodel.v
    d = v.data.shape[1]

    # solve least squares by computing batched solution to normal equations.
    # (n x D x d)
    U = torch.stack([self.KMmodel.group_models[jj].U.data
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
    if self.KMmodel.group_models[0].affine:
      B = B.sub(torch.stack([self.KMmodel.group_models[jj].b.data
          for jj in range(self.n)], dim=0).unsqueeze(2))
    # (n x d x batch_size)
    B = torch.matmul(Ut, B)

    # (n x d x batch_size)
    vt, _ = torch.gesv(B, A)
    v.data.copy_(vt.permute(2, 1, 0))
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
