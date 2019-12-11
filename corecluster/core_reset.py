from __future__ import print_function
from __future__ import division

import numpy as np
import torch

from .utils import assign_and_value

EPS = 1e-8
# ridx,  step, cidx, cand_ridx, cand_cidx, obj_decr, cumu_obj_decr
RESET_NCOL = 7


def reset_replicate(ridx, assign_obj, reg_out, max_steps=10, accept_tol=1e-3):
  """Re-initialize a given replicate by cooperative re-initialization (CoRe).
  On each iteration, we first find the best candidate from the set of all rk
  clusters to add to the current replicate. We then find the best of the
  replicate's clusters to drop. Both decisions made based on objective value.
  The relative objective decrease offered by the swap must pass a tolerance to
  be accepted.

  Args:
    ridx: replicate index
    assign_obj: cached assignment objectives, (cache_size, r, k).
    reg_out: outside regularization values, (r, k).
    max_steps: maximum iterations (default: 10).
    accept_tol: relative objective decrease accept tolerance (default: 1e-3).

  Returns
    resets: log of swap updates, shape (n_resets, 7). Columns are (reset rep
      idx, iteration, reset cluster idx, candidate rep idx, candidate cluster
      idx, obj decrease, cumulative obj decrease).
  """
  if accept_tol <= 0 or max_steps <= 0:
    raise ValueError("accept tol and max steps must be > 0")

  # make copies since these will be modified.
  assign_obj, reg_out = assign_obj.clone(), reg_out.clone()
  cache_size, r, k = assign_obj.shape
  device = assign_obj.device

  # initialize large tensors used in computing alternate objectives.
  alt_obj_buffers = (
      # alt_assign_obj
      torch.zeros((cache_size, r*k, 2), device=device),
      # alt_min_assign_obj
      torch.zeros((cache_size, r*k), device=device),
      # tmp_Idx
      torch.zeros((cache_size, r*k), dtype=torch.int64, device=device))

  rep_obj = old_obj = (assign_obj[:, ridx, :].min(dim=1)[0].mean() +
      reg_out[ridx, :].sum()).item()

  resets = []
  for kk in range(max_steps):
    # evaluate assignment objectives for every candidate new cluster.
    alt_obj = _eval_alt_obj(ridx, assign_obj, alt_obj_buffers)

    # choose candidate offering smallest objective.
    idx = alt_obj.view(-1).argmin().item()
    cand_ridx, cand_cidx = np.unravel_index(idx, (r, k))
    # if we pick a candidate from current replicate, time to stop.
    if cand_ridx == ridx:
      break

    # choose existing cluster to swap out, based on value (objective increase).
    # temporary assign obj and out reg for replicate with one added cluster.
    tmp_rep_assign_obj = torch.cat((assign_obj[:, cand_ridx, [cand_cidx]],
        assign_obj[:, ridx, :]), dim=1)
    tmp_rep_reg_out = torch.cat((reg_out[cand_ridx, [cand_cidx]],
        reg_out[ridx, :]), dim=0)
    tmp_rep_value, tmp_rep_min_assign_obj = _eval_value(tmp_rep_assign_obj,
        tmp_rep_reg_out)
    cidx = tmp_rep_value.argmin().item() - 1

    # measure objective and objective decrease after completing swap.
    # note that value of a cluster is the amount the objective would increase
    # if cluster were removed.
    cand_obj = (tmp_rep_min_assign_obj.mean() + tmp_rep_reg_out.sum() +
        tmp_rep_value[cidx+1]).item()
    obj_decr = (rep_obj - cand_obj) / rep_obj

    # should always have cidx >= 0 when obj_decr > 0, but make sure just in
    # case of numerical error.
    if obj_decr > accept_tol and cidx >= 0:
      # update replicate assign obj
      assign_obj[:, ridx, cidx] = assign_obj[:, cand_ridx, cand_cidx]
      reg_out[ridx, cidx] = reg_out[cand_ridx, cand_cidx]

      rep_obj = cand_obj
      cumu_obj_decr = (old_obj - rep_obj) / old_obj
      resets.append([ridx, kk+1, cidx, cand_ridx, cand_cidx, obj_decr,
          cumu_obj_decr])
    else:
      break

  if len(resets) > 0:
    resets = np.array(resets, dtype=object)
  else:
    resets = np.zeros((0, RESET_NCOL), dtype=object)
  return resets


def _eval_value(assign_obj, reg_out):
  """Evaluate value of every replicate, cluster based on cache objective."""
  assert(reg_out.dim() == assign_obj.dim() - 1)

  # in this case assign obj/reg_out correspond to a fixed replicate.
  # unsqueeze rep dimension.
  if assign_obj.dim() == 2:
    assign_obj = assign_obj.unsqueeze(1)
    reg_out = reg_out.unsqueeze(0)

  _, min_assign_obj, _, _, value = assign_and_value(assign_obj,
      compute_c=False)
  value = value.sub_(reg_out)
  return value.squeeze(0), min_assign_obj.squeeze(1)


def _eval_alt_obj(ridx, assign_obj, buffers):
  """Find the best candidate cluster to add to given replicate in terms of
  assignment objective."""
  cache_size, r, k = assign_obj.shape
  alt_assign_obj, alt_min_assign_obj, tmp_Idx = buffers

  # min assign objs, shape (cache_size,)
  rep_min_assign_obj = assign_obj[:, ridx, :].min(dim=1)[0]

  # alternative assignment objectives for each candidate, (cache_size, rk, 2)
  alt_assign_obj[:, :, 0] = rep_min_assign_obj.unsqueeze(1)
  alt_assign_obj[:, :, 1] = assign_obj.view(-1, r*k)

  # (cache_size, rk)
  alt_min_assign_obj, _ = torch.min(alt_assign_obj, dim=2,
      out=(alt_min_assign_obj, tmp_Idx))

  # average over cache (rk,)
  alt_obj = alt_min_assign_obj.mean(dim=0)
  return alt_obj.view(r, k)
