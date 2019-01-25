from __future__ import print_function
from __future__ import division

from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

import utils as ut


class KManifoldAEClusterModel(nn.Module):
  """Model of union of low-dimensional manifolds generalizing
  k-means/k-subspaces. Coefficients computed using auto-encoder."""

  def __init__(self, k, d, D, affine=False, symmetric=False, loss='l2_sqr',
        trunks=None, soft_assign=0.1, c_sigma=.01, ema_decay=0.9):
    if loss not in ('l2_sqr', 'l1'):
      raise ValueError("Invalid loss mode {}.".format(loss))
    if len(trunks) != 2 or not (isinstance(trunks[0], nn.Module) and
          isinstance(trunks[1], nn.Module)):
      raise ValueError("Invalid trunk encoder/decoder pair.")
    if soft_assign < 0:
      raise ValueError("Invalid soft-assign parameter {}".format(soft_assign))
    if c_sigma < 0:
      raise ValueError("Invalid assign noise parameter {}".format(c_sigma))

    super(KManifoldAEClusterModel, self).__init__()
    self.k = k  # number of groups
    self.d = d  # dimension of manifold
    self.D = D  # number of data points
    self.affine = affine
    self.symmetric = symmetric
    self.loss_mode = loss
    self.soft_assign = soft_assign
    self.c_sigma = c_sigma
    self.ema_decay = ema_decay

    # group assignment, ultimate shape (batch_size, k)
    self.c = None
    self.groups = None
    # low-dimensional coefficients, ultimate shape (batch_size, k, d)
    self.z = None
    # mean and cov for latent low-dim coeffs.
    self.register_buffer('running_z_mean', torch.zeros(k, d))
    self.register_buffer('running_z_cov', torch.eye(d, d).repeat(k, 1, 1))
    # chol factorization of covariance matrices
    self.register_buffer('z_cov_sqrt', torch.eye(d, d).repeat(k, 1, 1))

    # (trunk_encoder, trunk_decoder)
    self.trunks = nn.ModuleList(trunks) if trunks is not None else None
    self.Us = nn.Parameter(torch.Tensor(k, D, d))
    if self.symmetric:
      self.register_parameter('Vs', None)
    else:
      self.Vs = nn.Parameter(torch.Tensor(k, d, D))
    if affine:
      self.bs = nn.Parameter(torch.Tensor(k, D))
    else:
      self.register_parameter('bs', None)
    self.reset_parameters()
    return

  def reset_parameters(self):
    """Initialize Us, Vs, bs with entries drawn from normal with std
    0.1/sqrt(D)."""
    std = .1 / np.sqrt(self.D)
    self.Us.data.normal_(0., std)
    if not self.symmetric:
      self.Vs.data.normal_(0., std)
    if self.affine:
      self.bs.data.normal_(0., std)
    return

  def forward(self, x):
    """Compute AE embedding.

    Input:
      x: shape (batch_size, D) if trunks is None, or (batch_size, ...)

    Returns:
      x_: shape (k, batch_size, D) if trunks is None, or (k, batch_size, ...)
    """
    z = self.encode(x)
    self.z = z.data
    x_ = self.decode(z)
    return x_

  def encode(self, x):
    """Project x into each of k low-dimensional spaces.

    Input:
      x: shape (batch_size, D) if trunks is None, or (batch_size, ...)

    Returns:
      z: latent low-dimensional codes (k, batch_size, d, 1)
    """
    if self.trunks is not None:
      x = self.trunks[0](x)
    assert(x.dim() == 2 and x.size(1) == self.D)

    # z = U^T (x - b) or z = V (x - b)
    if self.affine:
      # shape (k, batch_size, D)
      x = x.sub(self.bs.unsqueeze(1))
    else:
      # shape (1, batch_size, D)
      x = x.unsqueeze(0)
    if self.symmetric:
      # shape (k, batch_size, d, 1)
      z = torch.matmul(self.Us.transpose(1, 2).unsqueeze(1), x.unsqueeze(3))
    else:
      z = torch.matmul(self.Vs.unsqueeze(1), x.unsqueeze(3))
    # shape (k, batch_size, d)
    z = z.squeeze()
    return z

  def decode(self, z):
    """Embed low-dim code z into ambient space.

    Input:
      z: shape (k, batch_size, d)

    Returns:
      x_: shape (k, batch_size, D) if trunks is None, or (k, batch_size, ...)
    """
    assert(z.dim() == 3 and z.size(0) == self.k and z.size(2) == self.d)
    # x_ = U z + b
    # shape (k, batch_size, D)
    x_ = torch.matmul(self.Us.unsqueeze(1), z.unsqueeze(3)).squeeze()
    if self.affine:
      x_ = x_.add(self.bs.unsqueeze(1))
    if self.trunks is not None:
      x_ = self.trunks[1](x_.view(-1, self.D))
      # shape (k, batch_size, ...)
      x_ = x_.view((self.k, -1) + x_.shape[1:])
    return x_

  def objective(self, x, lamb=.01):
    """Evaluate objective function."""
    x_ = self(x)
    loss = self.loss(x, x_)
    reg = self.reg() if lamb > 0 else 0.0
    assign_obj = loss + lamb*reg
    self.set_assign(assign_obj)

    loss = torch.mean(torch.sum(self.c*loss, dim=1))
    if lamb > 0:
      reg = torch.mean(torch.sum(self.c*reg, dim=1))
    else:
      reg = torch.zeros_like(loss)

    reg = lamb*reg
    obj = loss + reg
    return obj, loss, reg, x_

  def loss(self, x, x_):
    """Evaluate reconstruction loss

    Inputs:
      x: data, shape (batch_size, ...)
      x_, reconstruction, shape (k, batch_size, ...)

    Returns:
      loss: shape (batch_size, k)
    """
    reduce_dim = tuple(range(2, x_.dim()))
    if self.loss_mode == 'l1':
      loss = torch.sum((x.unsqueeze(0) - x_).abs(), dim=reduce_dim).t()
    else:
      loss = torch.sum((x.unsqueeze(0) - x_)**2, dim=reduce_dim).t()
    return loss

  def reg(self):
    """Evaluate subspace and trunk regularization."""
    if self.symmetric:
      reg = torch.sum(self.Us**2, dim=(1, 2))
    else:
      reg = 0.5*(torch.sum(self.Us**2, dim=(1, 2)) +
          torch.sum(self.Vs**2, dim=(1, 2)))

    if self.trunks is not None:
      for ii in (0, 1):
        if hasattr(self.trunks[ii], 'reg'):
          reg += self.trunks[ii].reg()
    return reg

  def set_assign(self, assign_obj):
    """Compute soft-assignment.

    Inputs:
      assign_obj: shape (batch_size, k)
    """
    if self.soft_assign <= 0:
      self.c = torch.zeros_like(assign_obj.data)
      minidx = assign_obj.data.argmin(dim=1, keepdim=True)
      self.c.scatter_(1, minidx, 1)
    else:
      self.c = ut.find_soft_assign(assign_obj.data, self.soft_assign)
    self.groups = torch.argmax(self.c, dim=1)

    if self.c_sigma > 0:
      c_z = torch.zeros_like(self.c)
      c_z.normal_(mean=0, std=(self.c_sigma/self.k)).abs_()
      self.c.add_(c_z)
      self.c.div_(self.c.sum(dim=1, keepdim=True))
    return

  def update_z_mean_cov(self):
    """Update running estimates of latent space mean and covariance.

    Used when sampling from latent space.
    """
    for ii in range(self.k):
      group_mask = self.groups == ii
      if group_mask.sum() < 2:
        continue
      group_z = self.z[ii, group_mask, :]

      group_z_mean = group_z.mean(dim=0)
      self.running_z_mean[ii, :].mul_(self.ema_decay)
      self.running_z_mean[ii, :].add_(1-self.ema_decay, group_z_mean)

      group_z_center = group_z.sub(self.running_z_mean[ii, :])
      group_z_cov = torch.matmul(group_z_center.t(), group_z_center)
      # use unbiased cov estimate
      group_z_cov.div_(group_z.size(0)-1)
      self.running_z_cov[ii, :, :].mul_(self.ema_decay)
      self.running_z_cov[ii, :, :].add_(1-self.ema_decay, group_z_cov)

      # compute covariance sqrt by eigendecomp
      # NOTE: this might be too costly, for moderate d.
      z_cov_e, z_cov_V = torch.symeig(self.running_z_cov[ii, :, :],
          eigenvectors=True)
      self.z_cov_sqrt[ii, :, :] = z_cov_V.mul(z_cov_e.sqrt())
    return

  def eval_sprs(self):
    """measure robust sparsity of current assignment subset c"""
    c = self.c.data
    cmax, _ = torch.max(c, dim=1, keepdim=True)
    sprs = torch.sum(c / cmax, dim=1)
    sprs = torch.mean(sprs)
    return sprs

  def eval_shrink(self, x, x_):
    """measure shrinkage of reconstruction wrt data"""
    c = self.c.data
    x = x.data
    x_ = x_.data
    # x_ is size (n, batch_size, ...)
    norm_x_ = torch.sqrt(torch.sum(x_**2, dim=tuple(range(2, x_.dim())))).t()
    norm_x_ = torch.sum(c*norm_x_, dim=1)
    # x is size (batch_size, ...)
    norm_x = torch.sqrt(torch.sum(x**2, dim=tuple(range(1, x.dim()))))
    norm_x_ = torch.mean(norm_x_ / (norm_x + 1e-8))
    return norm_x_

  def eval_rank(self, tol=.001):
    """Compute rank and singular values of subspace bases."""
    ranks_svs = [ut.rank(self.Us.data[ii, :, :], tol) for ii in range(self.k)]
    ranks, svs = zip(*ranks_svs)
    return ranks, svs


class MNISTDCTrunkEncoder(nn.Module):
  def __init__(self, conv_filters=10, H_dim=100):
    super(MNISTDCTrunkEncoder, self).__init__()

    self.conv_filters = conv_filters
    self.H_dim = H_dim
    self.feat_dim = 4*conv_filters*4*4

    self.conv_feat = nn.Sequential(OrderedDict([
        # input (1, 32, 32)
        ('conv1', nn.Conv2d(1, conv_filters, 4, 2, 1, bias=True)),
        ('lrelu1', nn.LeakyReLU(0.2, inplace=True)),
        # (conv_filters, 16, 16)
        ('conv2', nn.Conv2d(conv_filters, conv_filters*2, 4, 2, 1,
            bias=False)),
        ('bn2', nn.BatchNorm2d(conv_filters*2)),
        ('lrelu2', nn.LeakyReLU(0.2, inplace=True)),
        # (2*conv_filters, 8, 8)
        ('conv3', nn.Conv2d(conv_filters*2, conv_filters*4, 4, 2, 1,
            bias=False)),
        ('bn3', nn.BatchNorm2d(conv_filters*4)),
        ('lrelu3', nn.LeakyReLU(0.2, inplace=True))
        # (4*conv_filters, 4, 4)
    ]))

    self.uos_proj = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(self.feat_dim, H_dim, bias=True)),
        ('relu1', nn.ReLU(True))]))
    return

  def forward(self, x):
    x = self.conv_feat(x)
    x = x.view(x.size(0), self.feat_dim)
    x = self.uos_proj(x)
    return x

  def reg(self):
    # for compatibility
    return 0.0


class MNISTDCDiscriminator(nn.Module):
  def __init__(self, conv_filters=10):
    super(MNISTDCDiscriminator, self).__init__()

    self.conv_filters = conv_filters

    self.main = nn.Sequential(OrderedDict([
        # input (1, 32, 32)
        ('conv1', nn.Conv2d(1, conv_filters, 4, 2, 1, bias=True)),
        ('lrelu1', nn.LeakyReLU(0.2, inplace=True)),
        # (conv_filters, 16, 16)
        ('conv2', nn.Conv2d(conv_filters, conv_filters*2, 4, 2, 1,
            bias=False)),
        ('bn2', nn.BatchNorm2d(conv_filters*2)),
        ('lrelu2', nn.LeakyReLU(0.2, inplace=True)),
        # (2*conv_filters, 8, 8)
        ('conv3', nn.Conv2d(conv_filters*2, conv_filters*4, 4, 2, 1,
            bias=False)),
        ('bn3', nn.BatchNorm2d(conv_filters*4)),
        ('lrelu3', nn.LeakyReLU(0.2, inplace=True)),
        # (4*conv_filters, 4, 4)
        ('fc4', nn.Conv2d(conv_filters*4, 1, 4, 1, 0, bias=True)),
        ('sig', nn.Sigmoid())
    ]))
    return

  def forward(self, x):
    return self.main(x).squeeze()


class MNISTDCTrunkDecoder(nn.Module):
  def __init__(self, conv_filters=10, H_dim=100):
    super(MNISTDCTrunkDecoder, self).__init__()

    self.H_dim = H_dim
    self.conv_filters = conv_filters

    self.uos_embed = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(H_dim, 4*conv_filters*4*4, bias=True)),
        ('relu1', nn.ReLU(True))]))

    self.deconv_generator = nn.Sequential(OrderedDict([
        # input from uos_embed (4*conv_filters, 4, 4)
        ('convt1', nn.ConvTranspose2d(4*conv_filters, 2*conv_filters, 4, 2, 1,
            bias=False)),
        ('bn1', nn.BatchNorm2d(2*conv_filters)),
        ('relu1', nn.ReLU(True)),
        # (2*conv_filters, 8, 8)
        ('convt2', nn.ConvTranspose2d(2*conv_filters, conv_filters, 4, 2, 1,
            bias=False)),
        ('bn2', nn.BatchNorm2d(conv_filters)),
        ('relu2', nn.ReLU(True)),
        # (conv_filters, 16, 16)
        ('convt3', nn.ConvTranspose2d(conv_filters, 1, 4, 2, 1, bias=True)),
        ('tanh', nn.Tanh())
        # (1, 32, 32)
    ]))
    return

  def forward(self, x):
    x = self.uos_embed(x)
    x = x.view(x.size(0), 4*self.conv_filters, 4, 4)
    x = self.deconv_generator(x)
    return x

  def reg(self):
    # for compatibility
    return 0.0


class NoiseInject(nn.Module):
  def __init__(self, sigma=0.05):
    super(NoiseInject, self).__init__()
    if sigma < 0:
      raise ValueError("Invalid noise sigma {}".format(sigma))
    self.sigma = sigma
    self.z = None
    self.sqrtD = None
    self.ndim = None
    return

  def forward(self, x):
    if not self.training or self.sigma == 0:
      return x
    if self.z is None or self.z.shape != x.data.shape:
      self.z = torch.zeros_like(x.data)
      self.sqrtD = np.sqrt(np.prod(x.data.shape[1:]))
      self.ndim = x.data.dim()
    self.z.normal_()
    xnorm = x.data.pow(2).sum(dim=tuple(range(1, self.ndim)),
        keepdim=True).sqrt()
    sigma = xnorm.mul(self.sigma/self.sqrtD)
    self.z.mul_(sigma)
    x = x.add(self.z)
    return x
