"""Test CoRe re-initialization."""

from __future__ import division
from __future__ import print_function

import pytest
import numpy as np
import torch

from corecluster.core_reset import reset_replicate
from corecluster import models as mod

RTOL = 1e-4
ATOL = 1e-6

torch.manual_seed(2019)


@pytest.fixture(scope='module')
def assign_obj_and_reg_out():
  """Construct assignment objective with a few planted columns with small
  objective values."""
  cache_size, r, k = 200, 8, 4
  assign_obj = torch.randn(cache_size, r, k).abs()
  assign_obj[:100, 1, 0] = torch.randn(100).abs().mul(1e-6)
  assign_obj[100:150, 3, 2] = torch.randn(50).abs().mul(1e-6)
  assign_obj[150:190, 5, 1] = torch.randn(40).abs().mul(1e-6)
  assign_obj[190:, 7, 3] = torch.randn(10).abs().mul(1e-6)
  reg_out = torch.randn(8, 4).abs().mul(1e-6)
  return assign_obj, reg_out


def test_reset_replicate(assign_obj_and_reg_out):
  assign_obj, reg_out = assign_obj_and_reg_out

  temps = [0.1, 0.0]
  expected_obj_decrs = [0.9999, 0.9999]
  expected_first_swaps = [(0, 3, 3), (2, 1, 0)]
  for temp, eod, efs in zip(temps, expected_obj_decrs, expected_first_swaps):
    success, resets = reset_replicate(0, assign_obj, reg_out, temp=temp,
        max_steps=10, accept_tol=1e-3)
    obj_decr = resets[:, 6].max()
    assert np.isclose(obj_decr, eod, rtol=RTOL, atol=ATOL)

    first_swap = tuple(resets[0, 1:4])
    assert first_swap == efs


def test_core_reset(assign_obj_and_reg_out):
  k, d, D, r, cache_size = 4, 2, 20, 8, 200
  assign_obj, reg_out = assign_obj_and_reg_out

  mf_model = mod.KSubspaceMFModel(k, d, D, affine=False, replicates=r,
      reset_patience=20, reset_try_tol=-1, reset_max_steps=10,
      reset_accept_tol=1e-3, reset_cache_size=cache_size,
      temp_scheduler=mod.ConstantTempScheduler(init_temp=0.1))

  # plant assign obj and set num bad steps to trigger re-initialization
  mf_model._cache_assign_obj.copy_(assign_obj)
  mf_model._reg_out_per_cluster = reg_out
  mf_model.num_bad_steps[:] = 40
  resets = mf_model.core_reset()

  obj_decr = resets[:, 6].max()
  expected_obj_decr = 0.9999
  assert np.allclose(obj_decr, expected_obj_decr, rtol=RTOL, atol=ATOL)

  first_swap = tuple(resets[0, 1:4])
  expected_first_swap = (0, 1, 3)
  assert first_swap == expected_first_swap
