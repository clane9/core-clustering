"""Train and evaluate factorized manifold clustering model on MNIST data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil
import pickle

import numpy as np
import torch
from torch import optim
from torchvision import transforms

import datasets as dat
import models as mod
import optimizers as opt
import training as tr

CHKP_FREQ = 50
STOP_FREQ = -1


def main():
  use_cuda = args.cuda and torch.cuda.is_available()
  device = torch.device('cuda' if use_cuda else 'cpu')
  torch.set_num_threads(args.num_threads)

  torch.manual_seed(args.seed)
  np.random.seed(args.seed)

  # construct dataset
  mnist_dataset = dat.MNISTUoM(args.data_dir, train=True,
      transform=transforms.Compose([
          transforms.Resize((32, 32)),
          transforms.ToTensor(),
          transforms.Normalize((0.5,), (0.5,))]),
      download=True, classes=args.classes)
  kwargs = {'num_workers': args.num_workers}
  if use_cuda:
    kwargs['pin_memory'] = True
  mnist_data_loader = torch.utils.data.DataLoader(mnist_dataset,
      batch_size=args.batch_size, shuffle=True, drop_last=True, **kwargs)

  # construct model
  model_n = mnist_dataset.n if args.n is None else args.n
  trunk_encoder = mod.MNISTDCTrunkEncoder(args.conv_filters, args.H_dim)
  discriminator = mod.MNISTDCDiscriminator(args.conv_filters)
  trunk_decoder = mod.MNISTDCTrunkDecoder(args.conv_filters, args.H_dim)
  model = mod.KManifoldAEClusterModel(model_n, args.d, args.H_dim, affine=True,
      symmetric=True, loss='l2_sqr', trunks=(trunk_encoder, trunk_decoder),
      soft_assign=args.soft_assign, c_sigma=.01)
  model.to(device)

  # optimizer
  optimizer = opt.KManifoldAEMetaOptimizer(model, optim.Adam, lr=args.init_lr,
      lamb=args.lamb, adv_lamb=args.adv_lamb, discriminator=discriminator,
      opt_kwargs={'betas': (0.9, 0.9)})

  metric_printformstr = tr.metric_printformstrs['adv_reg']
  metric_logheader = tr.metric_logheaders['adv_reg']
  tr.train_loop(model, mnist_data_loader, device, optimizer,
      metric_printformstr, args.out_dir, metric_logheader, args.epochs,
      CHKP_FREQ, STOP_FREQ, eval_rank=False)
  return


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Cluster MNIST using Adv AE')
  parser.add_argument('--out-dir', type=str, required=True,
                      help='Output directory.')
  parser.add_argument('--data-dir', type=str,
                      default='~/Documents/Datasets/MNIST',
                      help=('Data directory '
                          '[default: ~/Documents/Datasets/MNIST].'))
  parser.add_argument('--classes', type=int, default=None, nargs='+',
                      help='Subset of digit classes [default: all]')
  # model settings
  parser.add_argument('--n', type=int, default=None,
                      help='Number of manifold models [default: # classes]')
  parser.add_argument('--d', type=int, default=2,
                      help='Manifold dimension [default: 2]')
  parser.add_argument('--H-dim', type=int, default=100,
                      help=('Latent space dimensionality [default: 100]'))
  parser.add_argument('--conv-filters', type=int, default=10,
                      help=('Encoder/decoder/discriminator conv filters '
                          '[default: 10]'))
  parser.add_argument('--lamb', type=float, default=1e-4,
                      help='Reg parameter [default: 1e-4]')
  parser.add_argument('--adv-lamb', type=float, default=0.1,
                      help='Adversarial reg parameter [default: 0.1]')
  parser.add_argument('--soft-assign', type=float, default=0.1,
                      help='soft assignment parameter [default: 0.1]')
  # training settings
  parser.add_argument('--batch-size', type=int, default=100,
                      help='Input batch size for training [default: 100]')
  parser.add_argument('--epochs', type=int, default=50,
                      help='Number of epochs to train [default: 50]')
  parser.add_argument('--init-lr', type=float, default=0.5,
                      help='Initial learning rate [default: 0.5]')
  parser.add_argument('--cuda', action='store_true', default=False,
                      help='Enables CUDA training')
  parser.add_argument('--num-threads', type=int, default=1,
                      help='Number of parallel threads to use [default: 1]')
  parser.add_argument('--num-workers', type=int, default=1,
                      help='Number of workers for data loading [default: 1]')
  parser.add_argument('--seed', type=int, default=2018,
                      help='Training random seed [default: 2018]')
  args = parser.parse_args()

  # create output directory, deleting any existing results.
  if os.path.exists(args.out_dir):
    shutil.rmtree(args.out_dir)
  os.mkdir(args.out_dir)
  # save args
  with open('{}/args.pkl'.format(args.out_dir), 'wb') as f:
    pickle.dump(args, f)
  main()
