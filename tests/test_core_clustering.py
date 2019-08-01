"""Test a variety of evaluation settings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from argparse import Namespace
from datetime import datetime

import numpy as np

from corecluster.eval_core_clustering import eval_core_clustering


test_dir = 'test_results/tests_{}'.format(
    datetime.now().strftime('%Y%m%d'))

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

    # re-initialization
    'reps': 6,
    'core_reset': True,
    'reset_patience': 50,
    'reset_try_tol': 0.01,
    'reset_steps': 20,
    'reset_accept_tol': 0.001,
    'reset_temp': 0.1,
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
  """Expected output:

    (epoch 1/10) lr=1.0e-01 err=0.610,0.674,0.728 obj=4.87e-01,4.92e-01,4.94e-01 loss=4.0e-01,4.1e-01,4.1e-01 reg.in=8.4e-02,8.5e-02,8.7e-02 reg.out=0.0e+00,0.0e+00,0.0e+00 rank=20,20,20 resets=0 samp/s=7760 rtime=1.29 data.rt=0.11 reset.rt=0.03
    (epoch 2/10) lr=1.0e-01 err=0.137,0.229,0.333 obj=3.75e-01,3.89e-01,4.05e-01 loss=2.4e-01,2.6e-01,2.8e-01 reg.in=1.3e-01,1.3e-01,1.4e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 rank=20,20,20 resets=0 samp/s=7612 rtime=1.31 data.rt=0.12 reset.rt=0.04
    (epoch 3/10) lr=1.0e-01 err=0.001,0.083,0.126 obj=2.91e-01,3.16e-01,3.32e-01 loss=1.2e-01,1.5e-01,1.8e-01 reg.in=1.6e-01,1.6e-01,1.7e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 rank=20,20,20 resets=0 samp/s=7208 rtime=1.39 data.rt=0.12 reset.rt=0.04
    (epoch 4/10) lr=1.0e-01 err=0.000,0.005,0.101 obj=2.73e-01,2.87e-01,2.99e-01 loss=9.4e-02,1.1e-01,1.3e-01 reg.in=1.7e-01,1.7e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 rank=20,20,20 resets=0 samp/s=7696 rtime=1.30 data.rt=0.11 reset.rt=0.03
    (epoch 5/10) lr=1.0e-01 err=0.000,0.000,0.099 obj=2.72e-01,2.73e-01,2.95e-01 loss=9.1e-02,9.3e-02,1.2e-01 reg.in=1.7e-01,1.8e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 rank=20,20,20 resets=0 samp/s=7670 rtime=1.30 data.rt=0.11 reset.rt=0.03
    (epoch 6/10) lr=1.0e-01 err=0.000,0.000,0.120 obj=2.72e-01,2.72e-01,2.92e-01 loss=9.1e-02,9.1e-02,1.2e-01 reg.in=1.7e-01,1.8e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 rank=20,20,20 resets=32 samp/s=7205 rtime=1.39 data.rt=0.12 reset.rt=0.07
    (epoch 7/10) lr=1.0e-01 err=0.000,0.000,0.000 obj=2.71e-01,2.72e-01,2.72e-01 loss=9.1e-02,9.1e-02,9.1e-02 reg.in=1.8e-01,1.8e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 rank=20,20,20 resets=0 samp/s=7094 rtime=1.41 data.rt=0.12 reset.rt=0.14
    (epoch 8/10) lr=1.0e-01 err=0.000,0.000,0.000 obj=2.71e-01,2.71e-01,2.71e-01 loss=9.1e-02,9.1e-02,9.1e-02 reg.in=1.8e-01,1.8e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 rank=20,20,20 resets=0 samp/s=6703 rtime=1.49 data.rt=0.11 reset.rt=0.23
    (epoch 9/10) lr=1.0e-01 err=0.000,0.000,0.000 obj=2.71e-01,2.71e-01,2.71e-01 loss=9.1e-02,9.1e-02,9.1e-02 reg.in=1.8e-01,1.8e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 rank=20,20,20 resets=0 samp/s=6835 rtime=1.46 data.rt=0.11 reset.rt=0.21
    (epoch 10/10) lr=1.0e-01 err=0.000,0.000,0.000 obj=2.72e-01,2.72e-01,2.72e-01 loss=9.1e-02,9.1e-02,9.1e-02 reg.in=1.8e-01,1.8e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 rank=20,20,20 resets=0 samp/s=6524 rtime=1.53 data.rt=0.11 reset.rt=0.25
  """
  args = synth_uos_base_args.copy()
  args['out_dir'] = '{}/base_synth_uos'.format(test_dir)
  expected_obj = 2.715032101e-01
  evaluate_and_compare('base synth uos', args, expected_obj)


def test_online_synth_uos():
  """Expected output:

    (epoch 1/10) lr=1.0e-01 err=0.632,0.684,0.731 obj=4.87e-01,4.91e-01,4.93e-01 loss=4.0e-01,4.1e-01,4.1e-01 reg.in=8.4e-02,8.5e-02,8.6e-02 reg.out=0.0e+00,0.0e+00,0.0e+00 rank=20,20,20 resets=0 samp/s=4550 rtime=2.20 data.rt=0.96 reset.rt=0.04
    (epoch 2/10) lr=1.0e-01 err=0.167,0.286,0.317 obj=3.76e-01,4.02e-01,4.10e-01 loss=2.4e-01,2.7e-01,2.9e-01 reg.in=1.3e-01,1.3e-01,1.4e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 rank=20,20,20 resets=0 samp/s=4598 rtime=2.17 data.rt=0.95 reset.rt=0.04
    (epoch 3/10) lr=1.0e-01 err=0.039,0.095,0.196 obj=3.06e-01,3.16e-01,3.34e-01 loss=1.4e-01,1.5e-01,1.8e-01 reg.in=1.6e-01,1.6e-01,1.7e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 rank=20,20,20 resets=0 samp/s=4622 rtime=2.16 data.rt=0.94 reset.rt=0.04
    (epoch 4/10) lr=1.0e-01 err=0.000,0.028,0.193 obj=2.76e-01,2.96e-01,3.19e-01 loss=9.9e-02,1.2e-01,1.6e-01 reg.in=1.6e-01,1.7e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 rank=20,20,20 resets=0 samp/s=4629 rtime=2.16 data.rt=0.95 reset.rt=0.04
    (epoch 5/10) lr=1.0e-01 err=0.000,0.000,0.193 obj=2.72e-01,2.75e-01,3.19e-01 loss=9.2e-02,9.6e-02,1.5e-01 reg.in=1.7e-01,1.7e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 rank=20,20,20 resets=0 samp/s=4622 rtime=2.16 data.rt=0.94 reset.rt=0.04
    (epoch 6/10) lr=1.0e-01 err=0.000,0.000,0.263 obj=2.72e-01,2.72e-01,3.02e-01 loss=9.1e-02,9.2e-02,1.3e-01 reg.in=1.7e-01,1.8e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 rank=20,20,20 resets=35 samp/s=4552 rtime=2.20 data.rt=0.93 reset.rt=0.09
    (epoch 7/10) lr=1.0e-01 err=0.000,0.000,0.000 obj=2.72e-01,2.72e-01,2.72e-01 loss=9.1e-02,9.2e-02,9.2e-02 reg.in=1.8e-01,1.8e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 rank=20,20,20 resets=0 samp/s=4325 rtime=2.31 data.rt=0.94 reset.rt=0.20
    (epoch 8/10) lr=1.0e-01 err=0.000,0.000,0.000 obj=2.73e-01,2.73e-01,2.73e-01 loss=9.2e-02,9.2e-02,9.2e-02 reg.in=1.8e-01,1.8e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 rank=20,20,20 resets=0 samp/s=4265 rtime=2.34 data.rt=0.95 reset.rt=0.21
    (epoch 9/10) lr=1.0e-01 err=0.000,0.000,0.000 obj=2.72e-01,2.72e-01,2.72e-01 loss=9.1e-02,9.1e-02,9.1e-02 reg.in=1.8e-01,1.8e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 rank=20,20,20 resets=0 samp/s=4152 rtime=2.41 data.rt=0.96 reset.rt=0.25
    (epoch 10/10) lr=1.0e-01 err=0.000,0.000,0.000 obj=2.72e-01,2.72e-01,2.72e-01 loss=9.1e-02,9.1e-02,9.1e-02 reg.in=1.8e-01,1.8e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 rank=20,20,20 resets=0 samp/s=4224 rtime=2.37 data.rt=0.95 reset.rt=0.25
  """
  args = synth_uos_base_args.copy()
  args['out_dir'] = '{}/online_synth_uos'.format(test_dir)
  args['online'] = True
  expected_obj = 2.721022069e-01
  evaluate_and_compare('online synth uos', args, expected_obj)


def test_missing_data_synth_uos():
  """Expected output:

    (epoch 1/2) lr=1.0e-01 err=0.744,0.793,0.812 obj=4.04e-01,4.04e-01,4.05e-01 loss=3.3e-01,3.3e-01,3.3e-01 reg.in=7.5e-02,7.6e-02,7.6e-02 reg.out=0.0e+00,0.0e+00,0.0e+00 comp.err=1.00,1.01,1.01 rank=20,20,20 resets=0 samp/s=342 rtime=29.23 data.rt=0.24 reset.rt=0.04
    (epoch 2/2) lr=1.0e-01 err=0.395,0.559,0.621 obj=3.57e-01,3.68e-01,3.71e-01 loss=2.5e-01,2.7e-01,2.7e-01 reg.in=9.9e-02,1.0e-01,1.1e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 comp.err=0.90,0.94,0.94 rank=20,20,20 resets=0 samp/s=346 rtime=28.93 data.rt=0.20 reset.rt=0.04
  """
  args = synth_uos_base_args.copy()
  args['out_dir'] = '{}/missing_data_synth_uos'.format(test_dir)
  args['miss_rate'] = 0.2
  args['num_workers'] = 4
  args['epochs'] = 2
  expected_obj = 3.568494022e-01
  evaluate_and_compare('missing data synth uos', args, expected_obj)


def test_dense_missing_data_synth_uos():
  """Expected output:

    (epoch 1/2) lr=1.0e-01 err=0.744,0.793,0.812 obj=4.04e-01,4.04e-01,4.05e-01 loss=3.3e-01,3.3e-01,3.3e-01 reg.in=7.5e-02,7.6e-02,7.6e-02 reg.out=0.0e+00,0.0e+00,0.0e+00 comp.err=1.00,1.01,1.01 rank=20,20,20 resets=0 samp/s=686 rtime=14.58 data.rt=0.17 reset.rt=0.04
    (epoch 2/2) lr=1.0e-01 err=0.395,0.559,0.621 obj=3.57e-01,3.68e-01,3.71e-01 loss=2.5e-01,2.7e-01,2.7e-01 reg.in=9.9e-02,1.0e-01,1.1e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 comp.err=0.90,0.94,0.94 rank=20,20,20 resets=0 samp/s=699 rtime=14.32 data.rt=0.18 reset.rt=0.03
  """
  args = synth_uos_base_args.copy()
  args['out_dir'] = '{}/dense_missing_data_synth_uos'.format(test_dir)
  args['miss_rate'] = 0.2
  args['num_workers'] = 4
  args['sparse_encode'] = False
  args['sparse_decode'] = False
  args['epochs'] = 2
  expected_obj = 3.568494022e-01
  evaluate_and_compare('dense missing data synth uos', args, expected_obj)


def evaluate_and_compare(test_name, args, expected_obj):
  args = Namespace(**args)
  metrics, _, _, _, _ = eval_core_clustering(args)
  final_obj = metrics[-1, 1, :].min()
  print(('({}) final objective: {:.9e}, expected: {:.9e}').format(test_name,
      final_obj, expected_obj))
  assert np.isclose(final_obj, expected_obj, rtol=1e-5, atol=1e-8)
