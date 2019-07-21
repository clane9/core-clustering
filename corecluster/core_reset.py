from __future__ import print_function
from __future__ import division

import numpy as np
import torch

from .utils import assign_and_value

EPS = 1e-8


def reset_replicate(ridx, assign_obj, reg_out, temp, max_steps=10,
      accept_tol=1e-3, cand_metric='obj_decr'):
  """Re-initialize a given replicate by cooperative re-initialization. This
  consists of several iterations of simulated annealing over the graph of all
  (rk choose k) subsets of clusters. On each iteration we choose a cluster to
  replace along with a candidate replacement from the replicate's siblings. The
  swapped cluster is sampled with prob inversely proportional to its value,
  while the replacement can be chosen based on either value or objective
  decrease (greedier, more sensitive). The replacement is then accepted with
  probability depending on the objective increase. At termination we mark the
  best iterate surpassing the accept tolerance (if one exists).

  Args:
    ridx: replicate index
    assign_obj: cached assignment objectives, (cache_size, r, k).
    reg_out: outside regularization values, (r, k).
    temp: simulated annealing temperature.
    accept_tol: relative objective decrease accept tolerance (default: 1e-3).
    cand_metric: candidate replacement selection metric, either 'value' or
      obj_decr (default: 'obj_decr').

  Returns
    success: whether we surpassed accept tolerance.
    resets: log of swap updates, shape (n_resets, 8). Columns are (reset rep
      idx, reset cluster idx, candidate rep idx, candidate cluster idx, reset
      success, obj decrease, cumulative obj decrease, temperature)
    rep_assign_obj, rep_reg_out: new assign objective, outside regularization
      values for ridx.
  """
  assign_obj, reg_out = assign_obj.clone(), reg_out.clone()
  cache_size, r, k = assign_obj.shape
  device = assign_obj.device
  resets = []

  # initialize large tensors used in computing alternate objectives
  if cand_metric == 'obj_decr':
    alt_obj_buffers = (
        # alt_assign_obj
        torch.zeros((cache_size, r*k, 2), device=device),
        # alt_obj
        torch.zeros((cache_size, r*k), device=device),
        # tmp_Idx
        torch.zeros((cache_size, r*k), dtype=torch.int64, device=device))

  value, min_assign_obj = _eval_value(assign_obj, reg_out)
  rep_obj = (min_assign_obj[:, ridx].mean() + reg_out[ridx, :].sum()).item()
  best_obj = old_obj = rep_obj

  # first dim: (tmp, best)
  rep_assign_obj = torch.zeros(2, cache_size, k, device=device).copy_(
      assign_obj[:, ridx, :].unsqueeze(0))
  rep_reg_out = torch.zeros(2, k, device=device).copy_(reg_out[ridx, :])

  for _ in range(max_steps):
    swap_sample_prob = 1.0 / value[ridx, :].clamp(min=EPS)
    cidx = torch.multinomial(swap_sample_prob, 1)[0].item()
    cand_ridx, cand_cidx = _sample_swap_cand(ridx, cidx, value, rep_obj,
        assign_obj, reg_out, cand_metric, alt_obj_buffers)

    # update temp assign obj
    rep_assign_obj[0, :, cidx] = assign_obj[:, cand_ridx, cand_cidx]
    rep_reg_out[0, cidx] = reg_out[cand_ridx, cand_cidx]
    cand_obj = (rep_assign_obj[0, :].min(dim=1)[0].mean() +
        rep_reg_out[0, :].sum()).item()
    obj_decr = (rep_obj - cand_obj) / rep_obj

    # accept with probability decaying exponentially in obj increase, as in
    # simulated annealing.
    accept_prob = np.exp(min(obj_decr - accept_tol, 0) / temp)
    success = torch.rand(1)[0].item() <= accept_prob

    if success:
      # copy assign obj from temp to current
      assign_obj[:, ridx, cidx] = rep_assign_obj[0, :, cidx]
      reg_out[ridx, cidx] = rep_reg_out[0, cidx]
      rep_obj = cand_obj

      # update replicate value
      value[ridx, :], _ = _eval_value(rep_assign_obj[0, :], rep_reg_out[0, :])

      # update best assign obj and copy assign obj from current to best
      if rep_obj < best_obj:
        best_obj = rep_obj
        rep_assign_obj[1, :] = rep_assign_obj[0, :]
        rep_reg_out[1, :] = rep_reg_out[0, :]
    else:
      # revert temporary assign obj to previous
      rep_assign_obj[0, :, cidx] = assign_obj[:, ridx, cidx]
      rep_reg_out[0, cidx] = reg_out[ridx, cidx]

    cumu_obj_decr = (old_obj - rep_obj) / old_obj
    resets.append([ridx, cidx, cand_ridx, cand_cidx, int(success), obj_decr,
        cumu_obj_decr, temp])

  resets = np.array(resets, dtype=object)

  # adjust reset success after the fact to terminate after best iteration
  success = (old_obj - best_obj) / old_obj >= accept_tol
  if success:
    bestitr = np.argmax(resets[:, 6])
    # code reset successes after best as -1
    resets[bestitr+1:, 4] *= -1
  else:
    resets[:, 4] *= -1
  return success, resets, rep_assign_obj[1, :], rep_reg_out[1, :]


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


def _sample_swap_cand(ridx, cidx, value, rep_obj, assign_obj, reg_out,
      cand_metric, alt_obj_buffers):
  """Sample a candidate replicate, cluster to swap for given ridx, cidx."""
  r, k = assign_obj.shape[1:]
  if cand_metric == 'value':
    cand_sample_prob = value.clamp(min=EPS)
  else:
    # obj_decr
    alt_obj = _eval_alt_obj(ridx, cidx, assign_obj, reg_out, alt_obj_buffers)
    alt_obj_decr = (rep_obj - alt_obj) / rep_obj
    # clamp so objective increase steps have some small non-zero probability
    cand_sample_prob = alt_obj_decr.clamp(min=EPS)
  cand_sample_prob[ridx, :] = 0.0
  idx = torch.multinomial(cand_sample_prob.view(-1), 1)[0].item()
  cand_ridx, cand_cidx = np.unravel_index(idx, (r, k))
  return cand_ridx, cand_cidx


def _eval_alt_obj(ridx, cidx, assign_obj, reg_out, buffers):
  """Find the best candidate swap for given ridx, cidx."""
  cache_size, r, k = assign_obj.shape
  alt_assign_obj, alt_obj, tmp_Idx = buffers

  # (cache_size,) of min assign objs with cidx dropped.
  drop_assign_obj = assign_obj[:, ridx, :].clone()
  drop_assign_obj[:, cidx] = np.inf
  drop_assign_obj = drop_assign_obj.min(dim=1)[0]

  # scalar reg value for single dropped cluster cidx
  drop_reg_out = reg_out[ridx, :].sum() - reg_out[ridx, cidx]

  # (cache_size, rk, 2)
  alt_assign_obj[:, :, 0] = drop_assign_obj.unsqueeze(1)
  alt_assign_obj[:, :, 1] = assign_obj.view(-1, r*k)

  # (cache_size, rk)
  alt_obj, _ = torch.min(alt_assign_obj, dim=2, out=(alt_obj, tmp_Idx))

  # average over cache and add regularization, (rk,)
  alt_obj = alt_obj.mean(dim=0)
  alt_obj.add_(drop_reg_out)
  alt_obj.add_(reg_out.view(-1))
  alt_obj = alt_obj.view(r, k)
  return alt_obj
