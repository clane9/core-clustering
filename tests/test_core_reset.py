"""Test CoRe re-initialization."""

from __future__ import division
from __future__ import print_function

import pytest
import numpy as np
import torch

from corecluster.core_reset import reset_replicate
from corecluster import models as mod
from corecluster.utils import aggregate_resets

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

  expected_obj_decr = 0.9999
  expected_first_swap = (3, 1, 0)

  # resets cols: ridx,  step, cidx, cand_ridx, cand_cidx, obj_decr,
  # cumu_obj_decr
  resets = reset_replicate(0, assign_obj, reg_out, max_steps=10,
      accept_tol=1e-3)

  assert resets.shape[0] > 0

  obj_decr = resets[:, 6].max()
  assert np.isclose(obj_decr, expected_obj_decr, rtol=RTOL, atol=ATOL)

  first_swap = tuple(resets[0, 2:5])
  assert first_swap == expected_first_swap


def test_core_reset(assign_obj_and_reg_out):
  k, d, D, r, cache_size = 4, 2, 20, 8, 200
  assign_obj, reg_out = assign_obj_and_reg_out

  mf_model = mod.KSubspaceMFModel(k=k, d=d, D=D, affine=False, replicates=r,
      reset_patience=20, reset_try_tol=-1, reset_max_steps=10,
      reset_accept_tol=1e-3, reset_cache_size=cache_size)

  # plant assign obj and set num bad steps to trigger re-initialization
  mf_model._cache_assign_obj.copy_(assign_obj)
  mf_model._reg_out_per_cluster = reg_out
  mf_model.num_bad_steps[:] = 40
  resets = mf_model.core_reset()

  assert resets.shape[0] > 0

  pad_resets = np.insert(resets, [0, 0], [0, 0], axis=1)
  agg_resets = aggregate_resets(pad_resets)
  obj_decr = agg_resets['cumu.obj.decr.max'].min()
  expected_obj_decr = 0.9999
  assert np.isclose(obj_decr, expected_obj_decr, rtol=RTOL, atol=ATOL)

  first_swap = tuple(resets[0, 2:5])
  expected_first_swap = (3, 1, 0)
  assert first_swap == expected_first_swap
