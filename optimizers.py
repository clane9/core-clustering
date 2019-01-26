from __future__ import print_function
from __future__ import division

import torch
from torch import optim, nn

from models import KManifoldAEClusterModel


class KManifoldAEMetaOptimizer(optim.Optimizer):
  """Meta optimizer for k-manifold AE formulations. Step alternates between
  (1) closed form update to (soft) assignment C, (2) SGD type step on manifold
  auto-encoders.

  Args:
    model (KManifoldAEClusterModel instance): model to optimize.
    Optimizer (Optimizer class): optimizer to use for SGD updates.
    lr (float): learning rate
    lamb (float): regularization parameter
    adv_lamb (float, optional): adversarial embedding regularization parameter
      (default: 0.0)
    discriminator (Module instance, optional): discriminator for adversarial
      embedding regularizer (default: None).
    opt_kwargs (dict or list of dict, optional): Optimizer kwargs. Optionally
      can be list of separate kwargs for manifold model and discriminator
      optimizers (default: {}).
  """
  def __init__(self, model, Optimizer, lr, lamb, adv_lamb=0.0,
        discriminator=None, opt_kwargs={}):

    if not isinstance(model, KManifoldAEClusterModel):
      raise ValueError("Must provide k-manifold model instance.")
    if not issubclass(Optimizer, optim.Optimizer):
      raise ValueError("Optimizer must be sub-class of torch.optim.Optimizer.")
    if lr < 0.0:
      raise ValueError("Invalid learning rate: {}".format(lr))
    if lamb < 0.0:
      raise ValueError("Invalid regularization lambda: {}".format(lamb))
    if adv_lamb < 0.0:
      raise ValueError("Invalid adversarial lambda {}".format(adv_lamb))
    if adv_lamb > 0 and not isinstance(discriminator, nn.Module):
      raise ValueError("Invalid discriminator, must be Module if adv_lamb > 0")
    if not (isinstance(opt_kwargs, dict) or len(opt_kwargs) != 2):
      raise ValueError(("Invalid opt_kwargs, should be dict "
          "or len 2 list of dicts."))

    params = [model.Us]
    if not model.symmetric:
      params.append(model.Vs)
    if model.affine:
      params.append(model.bs)
    if model.trunks is not None:
      params.extend(model.trunks.parameters())
    params = [{'name': 'AE', 'params': params}]

    discriminator = discriminator if adv_lamb > 0 else None
    if discriminator is not None:
      params.append({'name': 'D', 'params': discriminator.parameters()})

    super(KManifoldAEMetaOptimizer, self).__init__(params,
        defaults=dict(lr=lr))
    self.params_dict = {group['name']: group for group in self.param_groups}

    self.model = model
    self.k = model.k
    self.d = model.d
    self.lr = lr
    self.lamb = lamb
    self.adv_lamb = adv_lamb
    self.discriminator = discriminator
    if self.adv_lamb > 0:
      self.discrim_loss = nn.BCELoss()

    if isinstance(opt_kwargs, dict):
      discrim_opt_kwargs = opt_kwargs
    else:
      discrim_opt_kwargs = opt_kwargs[1]
      opt_kwargs = opt_kwargs[0]

    self.AE_optimizer = Optimizer([self.params_dict['AE']], **opt_kwargs)
    if discriminator is not None:
      self.D_optimizer = Optimizer([self.params_dict['D']],
          **discrim_opt_kwargs)
    return

  def step(self, x):
    """Performs a single optimization step.

    Args:
      x (FloatTensor): current minibatch data
    """
    batch_size = x.size(0)
    device = x.device
    # update discriminator if necessary
    if self.adv_lamb > 0:
      # real data
      self.discriminator.zero_grad()
      pred_real = self.discriminator(x)
      label = torch.ones(batch_size, device=device)
      lossD_real = self.discrim_loss(pred_real, label)
      lossD_real.backward()
      D_x = pred_real.mean().item()
      # model samples
      # noise sampled from gaussian fit to latent low-dim codes
      noise = torch.randn(self.k, self.d, batch_size // self.k, device=device)
      noise = torch.matmul(self.model.z_cov_sqrt, noise).transpose(1, 2)
      noise.add_(self.model.running_z_mean.unsqueeze(1))
      samples = self.model.decode(noise)
      # shape is (k, batch_size, ...), flatten
      samples = samples.view((-1,) + samples.shape[2:])
      label.fill_(0)
      pred_samples = self.discriminator(samples.detach())
      lossD_samples = self.discrim_loss(pred_samples, label[:samples.size(0)])
      lossD_samples.backward()
      D_G_z = pred_samples.mean().item()
      self.D_optimizer.step()
    else:
      D_x, D_G_z = 0.0, 0.0

    # update AE models (U_V)
    self.AE_optimizer.zero_grad()
    if self.adv_lamb > 0:
      label.fill_(1)
      pred_samples = self.discriminator(samples)
      lossG = self.discrim_loss(pred_samples, label[:samples.size(0)])
      adv_reg = self.adv_lamb*lossG
      adv_reg.backward()
      # NOTE: Do not want to accumulate gradients on any features shared
      # between encoder and discriminator during adv reg step
      self.discriminator.zero_grad()
    else:
      adv_reg = 0.0
      lossG = 0.0
    obj, loss, reg, x_ = self.model.objective(x, lamb=self.lamb)
    obj.backward()
    obj += adv_reg
    self.AE_optimizer.step()

    sprs = self.model.eval_sprs()
    norm_x_ = self.model.eval_shrink(x, x_)
    self.model.update_z_mean_cov()
    return obj, loss, reg, adv_reg, D_x, D_G_z, sprs, norm_x_
