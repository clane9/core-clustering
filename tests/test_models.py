"""Test k-subspace model functionality."""

from __future__ import division
from __future__ import print_function

from collections import OrderedDict

import numpy as np
import torch
from torch import nn

from corecluster import models as mod

RTOL = 1e-5
ATOL = 1e-8


def test_mf_model_objective():
  k, d, D = 10, 4, 100
  torch.manual_seed(2019)
  model = mod.KSubspaceMFModel(k, d, D, affine=False, replicates=6,
      reg_params={'U_frosqr_in': 1e-3, 'U_frosqr_out': 1e-6, 'z': 0.01})
  x = torch.randn(64, D)
  x = x.div_(torch.norm(x, dim=1, keepdim=True))

  with torch.no_grad():
    x_ = model.forward(x)
    obj = model.objective()[0]

  expected_x_prefix = np.array(
      [0.00939088, 0.01115249, -0.02560702, -0.00749532, -0.02842807,
       0.00571623, 0.00904365, 0.02388666, -0.00559709, 0.00187919,
       -0.02105073, -0.01211026, -0.00083121, 0.00170494, -0.01204916,
       -0.00806474, 0.00125662, 0.00221543, -0.00617624, -0.00086401],
      dtype=np.float32)
  x_prefix = x_[:2, :2, :5].contiguous().view(-1).numpy()
  assert np.allclose(x_prefix, expected_x_prefix, rtol=RTOL, atol=ATOL)

  expected_obj = 0.477435
  obj = obj.item()
  assert np.isclose(obj, expected_obj)


def test_proj_model_objective():
  k, d, D = 10, 4, 100
  torch.manual_seed(2019)
  model = mod.KSubspaceProjModel(k, d, D, affine=True, replicates=6,
      reg_params={'U_frosqr_in': 1e-3, 'U_frosqr_out': 1e-6})
  x = torch.randn(64, D)
  x = x.div_(torch.norm(x, dim=1, keepdim=True))

  with torch.no_grad():
    x_ = model.forward(x)
    obj = model.objective()[0]

  expected_x_prefix = np.array(
      [1.8649189e-04, 2.5565317e-04, -5.2430463e-04, -1.4049212e-04,
       -5.7174423e-04, 1.3389569e-04, 2.0334467e-04, 5.4405129e-04,
       -1.2284910e-04, 5.1941577e-05, -3.9342023e-04, -2.6168377e-04,
       2.0518971e-06, 5.1536215e-05, -2.4044792e-04, -1.9192464e-04,
       7.4743753e-06, 2.1633583e-05, -1.3819062e-04, 2.3944507e-05],
      dtype=np.float32)

  x_prefix = x_[:2, :2, :5].contiguous().view(-1).numpy()
  assert np.allclose(x_prefix, expected_x_prefix, rtol=RTOL, atol=ATOL)

  expected_obj = 0.499118
  obj = obj.item()
  assert np.isclose(obj, expected_obj)


def test_deep_proj_model_objective():
  torch.manual_seed(2019)
  encoder = nn.Sequential(OrderedDict([
      # shape (16, 28, 28)
      ('conv1', nn.Conv2d(1, 16, 3, padding=1)),
      ('relu1', nn.ReLU()),
      # shape (16, 14, 14)
      ('pool1', nn.MaxPool2d(2, 2)),
      # shape (4, 14, 14)
      ('conv2', nn.Conv2d(16, 4, 3, padding=1)),
      ('relu2', nn.ReLU()),
      # shape (4, 7, 7) = 196
      ('pool2', nn.MaxPool2d(2, 2))]))

  decoder = nn.Sequential(OrderedDict([
      # shape (16, 14, 14)
      ('tconv1', nn.ConvTranspose2d(4, 16, 2, stride=2)),
      ('relu1', nn.ReLU()),
      # shape (1, 28, 28)
      ('tconv2', nn.ConvTranspose2d(16, 1, 2, stride=2)),
      ('sigmoid', nn.Sigmoid())]))

  k, d, r = 10, 8, 8
  D = 196
  affine = True
  reg_params = {'U_frosqr_in': 1e-3, 'U_frosqr_out': 1e-4}
  model = mod.DeepKSubspaceProjModel(k, d, D, encoder, decoder, affine=affine,
      replicates=r, reg_params=reg_params, criterion=None, ksub_obj_weight=1.0)

  x = torch.rand(20, 1, 28, 28)

  with torch.no_grad():
    x_ = model.forward(x)
    obj = model.objective()[0]

  expected_x_prefix = np.array(
      [0.54113835, 0.51413935, 0.5411339, 0.5141498, 0.48489517, 0.51316774,
       0.48490095, 0.51317364, 0.5411327, 0.5141528, 0.54113686, 0.514142,
       0.48489472, 0.5131643, 0.48489502, 0.5131741], dtype=np.float32)
  x_prefix = x_[0, 0, 0, :4, :4].contiguous().view(-1).numpy()
  assert np.allclose(x_prefix, expected_x_prefix, rtol=RTOL, atol=ATOL)

  expected_obj = 0.09362324
  obj = obj.item()
  assert np.isclose(obj, expected_obj)
