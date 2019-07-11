"""Test a variety of evaluation settings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import Namespace
# from datetime import datetime

from eval_core_clustering import eval_core_clustering


# test_dir = 'test_results/tests_{}'.format(
#     datetime.now().strftime('%Y%m%d%H%M'))
test_dir = 'test_results/tests_20190710165104'

synth_uos_base_args = {
    'setting': 'synth-uos',

    # data
    'k': 10,
    'd': 20,
    'D': 100,
    'Ng': 1000,
    'N': None,
    'affine': False,
    'sigma': 0.4,
    'theta': None,
    'miss_rate': 0.0,
    'online': False,
    'normalize': False,

    # model
    'form': 'mf',
    'init': 'random',
    'model_k': None,
    'model_d': None,
    'auto_reg': True,
    'sigma_hat': None,
    'min_size': 0.0,
    'U_frosqr_in_lamb': 1e-4,
    'U_frosqr_out_lamb': 0.0,
    'z_lamb': 0.01,

    # training
    'epochs': 10,
    'epoch_size': None,
    'batch_size': 100,
    'optim': 'sgd',
    'init_lr': 0.1,
    'scale_grad_mode': None,
    'scale_grad_update_freq': 20,
    'sparse_encode': True,
    'sparse_decode': True,
    'serial_eval': None,

    # re-initialization
    'reps': 6,
    'core_reset': True,
    'reset_metric': 'obj_decr',
    'reset_patience': 50,
    'reset_try_tol': 0.01,
    'reset_steps': 20,
    'reset_accept_tol': 0.001,
    'reset_cache_size': 500,

    # general configuration
    'cuda': False,
    'num_threads': 1,
    'num_workers': 0,
    'data_seed': 2001,
    'seed': 3001,
    'eval_rank': True,
    'chkp_freq': None,
    'stop_freq': None,
    'save_large_data': False
}


def test_base_synth_uos():
  args = synth_uos_base_args.copy()
  args['out_dir'] = '{}/base_synth_uos'.format(test_dir)
  expected_obj = 2.720434666e-01
  evaluate_and_compare('base synth uos', args, expected_obj)


def test_online_synth_uos():
  args = synth_uos_base_args.copy()
  args['out_dir'] = '{}/online_synth_uos'.format(test_dir)
  args['online'] = True
  expected_obj = 2.722297311e-01
  evaluate_and_compare('online synth uos', args, expected_obj)


def test_missing_data_synth_uos():
  args = synth_uos_base_args.copy()
  args['out_dir'] = '{}/missing_data_synth_uos'.format(test_dir)
  args['miss_rate'] = 0.2
  args['num_workers'] = 4
  args['epochs'] = 2
  expected_obj = 3.934202790e-01
  evaluate_and_compare('missing data synth uos', args, expected_obj)


def test_dense_missing_data_synth_uos():
  args = synth_uos_base_args.copy()
  args['out_dir'] = '{}/dense_missing_data_synth_uos'.format(test_dir)
  args['miss_rate'] = 0.2
  args['num_workers'] = 4
  args['sparse_encode'] = False
  args['sparse_decode'] = False
  args['epochs'] = 2
  expected_obj = 3.934202790e-01
  evaluate_and_compare('dense missing data synth uos', args, expected_obj)


def is_close(a, b, tol=1e-6):
  """Test if a is approximately equal to b."""
  return abs(a - b) / b <= tol


def evaluate_and_compare(test_name, args, expected_obj):
  args = Namespace(**args)
  metrics, _, _, _, _ = eval_core_clustering(args)
  final_obj = metrics[-1, 1, :].min()
  print(('({}) final objective: {:.9e}, expected: {:.9e}').format(test_name,
      final_obj, expected_obj))
  assert is_close(final_obj, expected_obj)


if __name__ == '__main__':
  test_base_synth_uos()
  test_online_synth_uos()
  test_missing_data_synth_uos()
  test_dense_missing_data_synth_uos()
