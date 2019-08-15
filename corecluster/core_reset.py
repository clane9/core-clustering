from __future__ import print_function
from __future__ import division

import numpy as np
import torch

from .utils import assign_and_value

EPS = 1e-8


def reset_replicate(ridx, assign_obj, reg_out, temp=0, max_steps=10,
      accept_tol=1e-3):
  """Re-initialize a given replicate by cooperative re-initialization. This
  consists of several iterations of simulated annealing over the graph of all
  (rk choose k) subsets of clusters. On each iteration we choose a cluster to
  replace along with a candidate replacement from the replicate's siblings. The
  swapped cluster is sampled with prob inversely proportional to its value,
  while the replacement is chosen either deterministically (when temp <= 0) or
  probabilistically based on objective decrease. The replacement is then
  accepted if the objective decrease surpasses the accept tolerance (when temp
  <= 0) or with probability exponential in the objective decrease. At
  termination we mark the best iterate surpassing the accept tolerance (if one
  exists).

  Args:
    ridx: replicate index
    assign_obj: cached assignment objectives, (cache_size, r, k).
    reg_out: outside regularization values, (r, k).
    temp: simulated annealing temperature (default: 0).
    accept_tol: relative objective decrease accept tolerance (default: 1e-3).

  Returns
    success: whether we surpassed accept tolerance.
    resets: log of swap updates, shape (n_resets, 8). Columns are (reset rep
      idx, reset cluster idx, candidate rep idx, candidate cluster idx, reset
      success, obj decrease, cumulative obj decrease, temperature)
  """
  if accept_tol <= 0 or max_steps <= 0:
    raise ValueError("accept tol and max steps must be > 0")
  step_accept_tol = accept_tol / max_steps

  # make copies since these will be modified.
  assign_obj, reg_out = assign_obj.clone(), reg_out.clone()
  cache_size, r, k = assign_obj.shape
  device = assign_obj.device
  resets = []

  # initialize large tensors used in computing alternate objectives
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

  for _ in range(max_steps):
    # sample cluster to swap with prob inversely proportional to value.
    # clamp at EPS so that negative value have a very high chance of being
    # chosen.
    cidx = _multinom_single(1.0 / value[ridx, :].clamp(min=EPS))

    # alternate objectives for every possible swap, (rk,)
    alt_obj = _eval_alt_obj(ridx, cidx, assign_obj, reg_out, alt_obj_buffers)
    alt_obj_decr = (rep_obj - alt_obj) / rep_obj
    if temp <= 0:
      # choose best replacement deterministically
      idx = alt_obj_decr.max(dim=0)[1].item()
    else:
      # sample with prob proportional to decrease when doing SA.
      # clamped at step accept tol so that ascent steps still possible when no
      # good descent direction exists.
      sample_prob = alt_obj_decr.clamp(min=step_accept_tol).view(r, k)
      sample_prob[ridx, :] = 0.0
      idx = _multinom_single(sample_prob.view(-1))
    cand_ridx, cand_cidx = np.unravel_index(idx, (r, k))
    cand_obj, obj_decr = alt_obj[idx].item(), alt_obj_decr[idx].item()

    # accept with probability decaying exponentially in obj increase, as in
    # simulated annealing.
    if temp <= 0:
      success = obj_decr > step_accept_tol
    else:
      accept_prob = np.exp(min(obj_decr - step_accept_tol, 0) / temp)
      success = torch.rand(1)[0].item() <= accept_prob

    if success:
      # update assign obj and replicate value
      assign_obj[:, ridx, cidx] = assign_obj[:, cand_ridx, cand_cidx]
      reg_out[ridx, cidx] = reg_out[cand_ridx, cand_cidx]
      value[ridx, :], _ = _eval_value(assign_obj[:, ridx, :], reg_out[ridx, :])
      rep_obj = cand_obj
      best_obj = min(best_obj, cand_obj)

    cumu_obj_decr = (old_obj - rep_obj) / old_obj
    resets.append([ridx, cidx, cand_ridx, cand_cidx, int(success), obj_decr,
        cumu_obj_decr, temp])

  # identify best iterate and adjust reset success after the fact
  resets = np.array(resets, dtype=object)
  success = (old_obj - best_obj) / old_obj >= accept_tol
  if success:
    bestitr = np.argmax(resets[:, 6])
    # code reset successes after best as -1
    resets[bestitr+1:, 4] *= -1
  else:
    resets[:, 4] *= -1
  return success, resets


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
  return alt_obj


def _multinom_single(sample_prob):
  """Draw a single sample from a discrete distribution."""
  return torch.multinomial(sample_prob, 1)[0].item()
