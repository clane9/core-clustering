"""Test k-subspace model functionality."""

from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from corecluster import models as mod

RTOL = 1e-5
ATOL = 1e-8


def test_mf_model_objective():
  k, d, D = 10, 4, 100
  torch.manual_seed(2019)
  mf_model = mod.KSubspaceMFModel(k, d, D, affine=False, replicates=6,
      reg_params={'U_frosqr_in': 1e-3, 'U_frosqr_out': 1e-6, 'z': 0.01})
  x = torch.randn(64, D)
  x = x.div_(torch.norm(x, dim=1, keepdim=True))

  with torch.no_grad():
    x_ = mf_model.forward(x)
    obj = mf_model.objective()[0]

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
