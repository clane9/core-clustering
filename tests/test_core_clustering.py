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
    'model_k': None,
    'model_d': None,
    'auto_reg': True,
    'sigma_hat': None,
    'min_size': 0.0,
    'U_frosqr_in_lamb': 1e-4,
    'U_frosqr_out_lamb': 0.0,
    'z_lamb': 0.01,

    # training
    'init': 'random',
    'pfi_init_size': 500,
    'pfi_n_cands': None,
    'epochs': 10,
    'epoch_size': None,
    'optim': 'sgd',
    'init_lr': 0.1,
    'lr_step_size': None,
    'lr_gamma': 0.1,
    'lr_min': 1e-8,
    'init_bs': 100,
    'bs_step_size': None,
    'bs_gamma': 4,
    'bs_max': 1000,
    'scale_grad_lip': False,
    'sparse_encode': True,
    'sparse_decode': True,

    # re-initialization
    'reps': 6,
    'core_reset': True,
    'reset_patience': 50,
    'reset_try_tol': 0.01,
    'reset_steps': 5,
    'reset_accept_tol': 1e-3,
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
    'save_large_data': False,
    'config': None,
}


def test_base_synth_uos():
  """Expected output:
    (epoch 1/10) lr=1.0e-01 bs=100 err=0.610,0.674,0.728 obj=4.87e-01,4.92e-01,4.94e-01 loss=4.0e-01,4.1e-01,4.1e-01 reg.in=8.4e-02,8.5e-02,8.7e-02 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=20,20,20 samp/s=6692 rtime=1.50 data.rt=0.15 reset.rt=0.04
    (epoch 2/10) lr=1.0e-01 bs=100 err=0.137,0.229,0.333 obj=3.75e-01,3.89e-01,4.05e-01 loss=2.4e-01,2.6e-01,2.8e-01 reg.in=1.3e-01,1.3e-01,1.4e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=20,20,20 samp/s=6717 rtime=1.49 data.rt=0.15 reset.rt=0.04
    (epoch 3/10) lr=1.0e-01 bs=100 err=0.001,0.083,0.126 obj=2.91e-01,3.16e-01,3.32e-01 loss=1.2e-01,1.5e-01,1.8e-01 reg.in=1.6e-01,1.6e-01,1.7e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=20,20,20 samp/s=6731 rtime=1.49 data.rt=0.15 reset.rt=0.04
    (epoch 4/10) lr=1.0e-01 bs=100 err=0.000,0.005,0.101 obj=2.73e-01,2.87e-01,2.99e-01 loss=9.4e-02,1.1e-01,1.3e-01 reg.in=1.7e-01,1.7e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=20,20,20 samp/s=6749 rtime=1.48 data.rt=0.15 reset.rt=0.04
    (epoch 5/10) lr=1.0e-01 bs=100 err=0.000,0.000,0.099 obj=2.72e-01,2.73e-01,2.95e-01 loss=9.1e-02,9.3e-02,1.2e-01 reg.in=1.7e-01,1.8e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=20,20,20 samp/s=6755 rtime=1.48 data.rt=0.15 reset.rt=0.04
    (epoch 6/10) lr=1.0e-01 bs=100 err=0.000,0.000,0.087 obj=2.72e-01,2.72e-01,2.92e-01 loss=9.1e-02,9.1e-02,1.2e-01 reg.in=1.7e-01,1.8e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=4 rank=20,20,20 samp/s=6658 rtime=1.50 data.rt=0.15 reset.rt=0.05
    (epoch 7/10) lr=1.0e-01 bs=100 err=0.000,0.000,0.000 obj=2.71e-01,2.71e-01,2.71e-01 loss=9.1e-02,9.1e-02,9.1e-02 reg.in=1.8e-01,1.8e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=20,20,20 samp/s=6700 rtime=1.49 data.rt=0.15 reset.rt=0.05
    (epoch 8/10) lr=1.0e-01 bs=100 err=0.000,0.000,0.000 obj=2.72e-01,2.72e-01,2.72e-01 loss=9.1e-02,9.1e-02,9.1e-02 reg.in=1.8e-01,1.8e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=20,20,20 samp/s=6675 rtime=1.50 data.rt=0.15 reset.rt=0.05
    (epoch 9/10) lr=1.0e-01 bs=100 err=0.000,0.000,0.000 obj=2.72e-01,2.72e-01,2.72e-01 loss=9.1e-02,9.1e-02,9.1e-02 reg.in=1.8e-01,1.8e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=20,20,20 samp/s=6676 rtime=1.50 data.rt=0.15 reset.rt=0.05
    (epoch 10/10) lr=1.0e-01 bs=100 err=0.000,0.000,0.000 obj=2.72e-01,2.72e-01,2.72e-01 loss=9.1e-02,9.1e-02,9.1e-02 reg.in=1.8e-01,1.8e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=20,20,20 samp/s=6643 rtime=1.51 data.rt=0.15 reset.rt=0.05
  """
  args = synth_uos_base_args.copy()
  args['out_dir'] = '{}/base_synth_uos'.format(test_dir)
  expected_obj = 2.715106308e-01
  evaluate_and_compare('base synth uos', args, expected_obj)

def test_scale_grad_sgd():
  """Expected output:

    (epoch 1/10) lr=1.0e-01 bs=100 err=0.178,0.271,0.380 obj=3.62e-01,3.80e-01,3.99e-01 loss=2.1e-01,2.3e-01,2.6e-01 reg.in=1.4e-01,1.5e-01,1.6e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=20,20,20 lip=1.09e-01,1.38e-01,1.60e-01 lip.err=0.85,0.89,0.92 samp/s=7331 rtime=1.37 data.rt=0.11 reset.rt=0.03
    (epoch 2/10) lr=1.0e-01 bs=100 err=0.000,0.051,0.128 obj=2.83e-01,2.98e-01,3.06e-01 loss=1.0e-01,1.2e-01,1.3e-01 reg.in=1.7e-01,1.7e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=20,20,20 lip=1.12e-01,1.43e-01,1.75e-01 lip.err=0.90,0.90,0.93 samp/s=7369 rtime=1.36 data.rt=0.11 reset.rt=0.03
    (epoch 3/10) lr=1.0e-01 bs=100 err=0.000,0.000,0.137 obj=2.80e-01,2.82e-01,3.03e-01 loss=9.7e-02,9.9e-02,1.3e-01 reg.in=1.7e-01,1.7e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=20,20,20 lip=1.21e-01,1.35e-01,1.93e-01 lip.err=0.90,0.92,0.93 samp/s=7374 rtime=1.36 data.rt=0.11 reset.rt=0.03
    (epoch 4/10) lr=1.0e-01 bs=100 err=0.000,0.000,0.147 obj=2.82e-01,2.82e-01,3.03e-01 loss=9.8e-02,9.9e-02,1.3e-01 reg.in=1.7e-01,1.7e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=20,20,20 lip=1.20e-01,1.21e-01,1.95e-01 lip.err=0.89,0.92,0.92 samp/s=7410 rtime=1.35 data.rt=0.11 reset.rt=0.03
    (epoch 5/10) lr=1.0e-01 bs=100 err=0.000,0.000,0.117 obj=2.81e-01,2.82e-01,2.96e-01 loss=9.8e-02,9.8e-02,1.2e-01 reg.in=1.8e-01,1.8e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=26 rank=20,20,20 lip=1.29e-01,1.30e-01,1.74e-01 lip.err=0.68,0.85,0.89 samp/s=6860 rtime=1.46 data.rt=0.11 reset.rt=0.06
    (epoch 6/10) lr=1.0e-01 bs=100 err=0.000,0.000,0.000 obj=2.80e-01,2.81e-01,2.82e-01 loss=9.7e-02,9.8e-02,9.8e-02 reg.in=1.8e-01,1.8e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=20,20,20 lip=1.22e-01,1.22e-01,1.23e-01 lip.err=0.91,0.92,0.93 samp/s=6829 rtime=1.46 data.rt=0.11 reset.rt=0.05
    (epoch 7/10) lr=1.0e-01 bs=100 err=0.000,0.000,0.000 obj=2.82e-01,2.82e-01,2.82e-01 loss=9.9e-02,9.9e-02,9.9e-02 reg.in=1.8e-01,1.8e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=20,20,20 lip=1.20e-01,1.21e-01,1.22e-01 lip.err=0.92,0.92,0.93 samp/s=5935 rtime=1.69 data.rt=0.13 reset.rt=0.06
    (epoch 8/10) lr=1.0e-01 bs=100 err=0.000,0.000,0.000 obj=2.82e-01,2.82e-01,2.82e-01 loss=9.9e-02,9.9e-02,9.9e-02 reg.in=1.8e-01,1.8e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=20,20,20 lip=1.19e-01,1.20e-01,1.22e-01 lip.err=0.91,0.92,0.93 samp/s=5960 rtime=1.68 data.rt=0.13 reset.rt=0.06
    (epoch 9/10) lr=1.0e-01 bs=100 err=0.000,0.000,0.000 obj=2.82e-01,2.82e-01,2.82e-01 loss=9.9e-02,9.9e-02,9.9e-02 reg.in=1.8e-01,1.8e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=20,20,20 lip=1.20e-01,1.20e-01,1.21e-01 lip.err=0.92,0.93,0.94 samp/s=6423 rtime=1.56 data.rt=0.12 reset.rt=0.05
    (epoch 10/10) lr=1.0e-01 bs=100 err=0.000,0.000,0.000 obj=2.82e-01,2.82e-01,2.82e-01 loss=9.9e-02,9.9e-02,9.9e-02 reg.in=1.8e-01,1.8e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=20,20,20 lip=1.20e-01,1.20e-01,1.21e-01 lip.err=0.92,0.92,0.93 samp/s=6374 rtime=1.57 data.rt=0.13 reset.rt=0.05
  """
  args = synth_uos_base_args.copy()
  args['scale_grad_lip'] = True
  args['out_dir'] = '{}/scale_grad_sgd_lip'.format(test_dir)
  expected_obj = 2.820871472e-01
  evaluate_and_compare('scale grad sgd lip', args, expected_obj)


def test_online_synth_uos():
  """Expected output:

    (epoch 1/10) lr=1.0e-01 bs=100 err=0.632,0.684,0.731 obj=4.87e-01,4.91e-01,4.93e-01 loss=4.0e-01,4.1e-01,4.1e-01 reg.in=8.4e-02,8.5e-02,8.6e-02 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=20,20,20 samp/s=4533 rtime=2.21 data.rt=0.96 reset.rt=0.03
    (epoch 2/10) lr=1.0e-01 bs=100 err=0.166,0.286,0.317 obj=3.76e-01,4.02e-01,4.10e-01 loss=2.4e-01,2.7e-01,2.9e-01 reg.in=1.3e-01,1.3e-01,1.4e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=20,20,20 samp/s=4635 rtime=2.16 data.rt=0.97 reset.rt=0.03
    (epoch 3/10) lr=1.0e-01 bs=100 err=0.039,0.095,0.196 obj=3.06e-01,3.16e-01,3.34e-01 loss=1.4e-01,1.5e-01,1.8e-01 reg.in=1.6e-01,1.6e-01,1.7e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=20,20,20 samp/s=4533 rtime=2.21 data.rt=0.98 reset.rt=0.03
    (epoch 4/10) lr=1.0e-01 bs=100 err=0.000,0.028,0.194 obj=2.76e-01,2.96e-01,3.19e-01 loss=9.9e-02,1.2e-01,1.6e-01 reg.in=1.6e-01,1.7e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=20,20,20 samp/s=4588 rtime=2.18 data.rt=0.96 reset.rt=0.03
    (epoch 5/10) lr=1.0e-01 bs=100 err=0.000,0.000,0.193 obj=2.72e-01,2.75e-01,3.19e-01 loss=9.2e-02,9.6e-02,1.5e-01 reg.in=1.7e-01,1.7e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=20,20,20 samp/s=4602 rtime=2.17 data.rt=0.95 reset.rt=0.03
    (epoch 6/10) lr=1.0e-01 bs=100 err=0.000,0.000,0.189 obj=2.72e-01,2.72e-01,3.18e-01 loss=9.1e-02,9.2e-02,1.5e-01 reg.in=1.7e-01,1.8e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=4 rank=20,20,20 samp/s=4357 rtime=2.30 data.rt=1.04 reset.rt=0.05
    (epoch 7/10) lr=1.0e-01 bs=100 err=0.000,0.000,0.033 obj=2.72e-01,2.72e-01,2.80e-01 loss=9.1e-02,9.1e-02,1.0e-01 reg.in=1.8e-01,1.8e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=4 rank=20,20,20 samp/s=4490 rtime=2.23 data.rt=1.01 reset.rt=0.05
    (epoch 8/10) lr=1.0e-01 bs=100 err=0.000,0.000,0.000 obj=2.72e-01,2.72e-01,2.72e-01 loss=9.1e-02,9.1e-02,9.1e-02 reg.in=1.8e-01,1.8e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=20,20,20 samp/s=4529 rtime=2.21 data.rt=0.98 reset.rt=0.05
    (epoch 9/10) lr=1.0e-01 bs=100 err=0.000,0.000,0.000 obj=2.73e-01,2.73e-01,2.73e-01 loss=9.2e-02,9.2e-02,9.2e-02 reg.in=1.8e-01,1.8e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=20,20,20 samp/s=4562 rtime=2.19 data.rt=0.96 reset.rt=0.05
    (epoch 10/10) lr=1.0e-01 bs=100 err=0.000,0.000,0.000 obj=2.73e-01,2.73e-01,2.73e-01 loss=9.1e-02,9.1e-02,9.1e-02 reg.in=1.8e-01,1.8e-01,1.8e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=20,20,20 samp/s=4577 rtime=2.18 data.rt=0.95 reset.rt=0.05
  """
  args = synth_uos_base_args.copy()
  args['out_dir'] = '{}/online_synth_uos'.format(test_dir)
  args['online'] = True
  expected_obj = 2.725076675e-01
  evaluate_and_compare('online synth uos', args, expected_obj)


def test_missing_data_synth_uos():
  """Expected output:

    (epoch 1/2) lr=1.0e-01 bs=100 err=0.744,0.792,0.812 obj=4.04e-01,4.04e-01,4.05e-01 loss=3.3e-01,3.3e-01,3.3e-01 reg.in=7.5e-02,7.6e-02,7.6e-02 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=20,20,20 comp.err=1.00,1.01,1.01 samp/s=347 rtime=28.80 data.rt=0.24 reset.rt=0.03
    (epoch 2/2) lr=1.0e-01 bs=100 err=0.395,0.559,0.621 obj=3.57e-01,3.68e-01,3.71e-01 loss=2.5e-01,2.7e-01,2.7e-01 reg.in=9.9e-02,1.0e-01,1.1e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=20,20,20 comp.err=0.90,0.94,0.94 samp/s=341 rtime=29.35 data.rt=0.20 reset.rt=0.03
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

    (epoch 1/2) lr=1.0e-01 bs=100 err=0.744,0.792,0.812 obj=4.04e-01,4.04e-01,4.05e-01 loss=3.3e-01,3.3e-01,3.3e-01 reg.in=7.5e-02,7.6e-02,7.6e-02 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=20,20,20 comp.err=1.00,1.01,1.01 samp/s=676 rtime=14.80 data.rt=0.19 reset.rt=0.03
    (epoch 2/2) lr=1.0e-01 bs=100 err=0.395,0.559,0.621 obj=3.57e-01,3.68e-01,3.71e-01 loss=2.5e-01,2.7e-01,2.7e-01 reg.in=9.9e-02,1.0e-01,1.1e-01 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=20,20,20 comp.err=0.90,0.94,0.94 samp/s=674 rtime=14.85 data.rt=0.16 reset.rt=0.03
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


def test_synth_kmeans_sgd():
  """Expected output:

    (epoch 1/10) lr=1.0e-01 bs=100 err=0.213,0.295,0.500 obj=9.68e-01,1.03e+00,1.18e+00 loss=9.7e-01,1.0e+00,1.2e+00 reg.in=0.0e+00,0.0e+00,0.0e+00 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=1,1,1 samp/s=21502 rtime=0.47 data.rt=0.11 reset.rt=0.03
    (epoch 2/10) lr=1.0e-01 bs=100 err=0.200,0.205,0.499 obj=6.86e-01,7.31e-01,1.06e+00 loss=6.9e-01,7.3e-01,1.1e+00 reg.in=0.0e+00,0.0e+00,0.0e+00 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=1,1,1 samp/s=21364 rtime=0.47 data.rt=0.11 reset.rt=0.03
    (epoch 3/10) lr=1.0e-01 bs=100 err=0.200,0.200,0.499 obj=6.81e-01,7.24e-01,1.06e+00 loss=6.8e-01,7.2e-01,1.1e+00 reg.in=0.0e+00,0.0e+00,0.0e+00 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=1,1,1 samp/s=21430 rtime=0.47 data.rt=0.11 reset.rt=0.03
    (epoch 4/10) lr=1.0e-01 bs=100 err=0.200,0.200,0.417 obj=6.81e-01,7.23e-01,9.73e-01 loss=6.8e-01,7.2e-01,9.7e-01 reg.in=0.0e+00,0.0e+00,0.0e+00 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=5 rank=1,1,1 samp/s=21139 rtime=0.47 data.rt=0.11 reset.rt=0.04
    (epoch 5/10) lr=1.0e-01 bs=100 err=0.100,0.198,0.294 obj=6.21e-01,6.91e-01,8.06e-01 loss=6.2e-01,6.9e-01,8.1e-01 reg.in=0.0e+00,0.0e+00,0.0e+00 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=17 rank=1,1,1 samp/s=20668 rtime=0.48 data.rt=0.11 reset.rt=0.05
    (epoch 6/10) lr=1.0e-01 bs=100 err=0.000,0.000,0.100 obj=5.00e-01,5.00e-01,6.11e-01 loss=5.0e-01,5.0e-01,6.1e-01 reg.in=0.0e+00,0.0e+00,0.0e+00 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=4 rank=1,1,1 samp/s=20826 rtime=0.48 data.rt=0.11 reset.rt=0.04
    (epoch 7/10) lr=1.0e-01 bs=100 err=0.000,0.000,0.022 obj=5.00e-01,5.00e-01,5.23e-01 loss=5.0e-01,5.0e-01,5.2e-01 reg.in=0.0e+00,0.0e+00,0.0e+00 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=2 rank=1,1,1 samp/s=21004 rtime=0.48 data.rt=0.11 reset.rt=0.04
    (epoch 8/10) lr=1.0e-01 bs=100 err=0.000,0.000,0.000 obj=5.00e-01,5.00e-01,5.00e-01 loss=5.0e-01,5.0e-01,5.0e-01 reg.in=0.0e+00,0.0e+00,0.0e+00 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=1,1,1 samp/s=20971 rtime=0.48 data.rt=0.11 reset.rt=0.04
    (epoch 9/10) lr=1.0e-01 bs=100 err=0.000,0.000,0.000 obj=5.00e-01,5.00e-01,5.00e-01 loss=5.0e-01,5.0e-01,5.0e-01 reg.in=0.0e+00,0.0e+00,0.0e+00 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=1,1,1 samp/s=21012 rtime=0.48 data.rt=0.11 reset.rt=0.04
    (epoch 10/10) lr=1.0e-01 bs=100 err=0.000,0.000,0.000 obj=5.00e-01,5.00e-01,5.00e-01 loss=5.0e-01,5.0e-01,5.0e-01 reg.in=0.0e+00,0.0e+00,0.0e+00 reg.out=0.0e+00,0.0e+00,0.0e+00 resets=0 rank=1,1,1 samp/s=20961 rtime=0.48 data.rt=0.11 reset.rt=0.04
  """
  args = synth_uos_base_args.copy()
  args['setting'] = 'synth-kmeans'
  args['out_dir'] = '{}/synth_kmeans_sgd'.format(test_dir)
  args['sep'] = 2.0
  args['b_frosqr_out_lamb'] = 0.0
  expected_obj = 4.998686910e-01
  evaluate_and_compare('base synth uos', args, expected_obj)


def evaluate_and_compare(test_name, args, expected_obj):
  args = Namespace(**args)
  metrics, _, _, _, _ = eval_core_clustering(args)
  final_obj = metrics[-1, 1, :].min()
  print(('({}) final objective: {:.9e}, expected: {:.9e}').format(test_name,
      final_obj, expected_obj))
  assert np.isclose(final_obj, expected_obj, rtol=1e-5, atol=1e-8)
