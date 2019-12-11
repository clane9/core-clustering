"""Test k-subspace model functionality."""

from __future__ import division
from __future__ import print_function

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

from corecluster import models as mod
from corecluster import datasets as dat
from corecluster import utils as ut

RTOL = 1e-5
ATOL = 1e-8

k, d, D, Ng, sigma = 10, 4, 100, 50, 0.1
affine = True
replicates = 6
batch_size = 64


@pytest.fixture(scope='module')
def synth_uos_dataset():
  torch.manual_seed(2001)
  dataset = dat.generate_synth_uos_dataset(k, d, D, Ng, affine=affine,
      sigma=sigma, miss_rate=0.0, normalize=True, seed=2001)
  kwargs = {'num_workers': 0, 'batch_size': batch_size, 'shuffle': True}
  data_loader = DataLoader(dataset, **kwargs)
  x, _ = next(iter(data_loader))
  return dataset, data_loader, x


@pytest.fixture(scope='module')
def synth_kmeans_dataset():
  torch.manual_seed(2001)
  dataset = dat.SynthKMeansDataset(k, D, Ng, separation=2.0, seed=2001)
  kwargs = {'num_workers': 0, 'batch_size': batch_size, 'shuffle': True}
  data_loader = DataLoader(dataset, **kwargs)
  x, _ = next(iter(data_loader))
  return dataset, data_loader, x


@pytest.fixture(scope='module')
def synth_mc_dataset():
  torch.manual_seed(2001)
  dataset = dat.generate_synth_uos_dataset(k, d, D, Ng, affine=affine,
      sigma=sigma, miss_rate=0.2, normalize=True, miss_store_sparse=True,
      miss_store_dense=True, seed=2001)
  kwargs = {'num_workers': 0, 'batch_size': batch_size, 'shuffle': True,
      'collate_fn': dat.missing_data_collate}
  data_loader = DataLoader(dataset, **kwargs)
  x, _, x0 = next(iter(data_loader))
  return dataset, data_loader, x, x0


def test_mf_model_objective(synth_uos_dataset):
  dataset, data_loader, x = synth_uos_dataset

  torch.manual_seed(2019)
  model = mod.KSubspaceMFModel(k=k, d=d, D=D, affine=affine,
      replicates=replicates,
      reg_params={'U_frosqr_in': 1e-3, 'U_frosqr_out': 1e-6, 'z': 0.01})

  with torch.no_grad():
    x_ = model.forward(x)
    obj = model.objective()[0]

  expected_x_prefix = np.array(
      [-0.02013446, -0.00486054, -0.00243031, -0.01523567, 0.00813505,
      0.00165438, -0.01336181, 0.02123059, 0.00971936, -0.01408227,
      0.01172347, 0.00429059, -0.00268211, -0.04007174, -0.00300111,
      -0.01072975, 0.00339037, 0.0144486, -0.00545396, 0.00340225],
      dtype=np.float32)
  x_prefix = x_[:2, :2, :5].contiguous().view(-1).numpy()
  assert np.allclose(x_prefix, expected_x_prefix, rtol=RTOL, atol=ATOL)

  expected_obj = 0.47707966
  obj = obj.item()
  assert np.isclose(obj, expected_obj)


def test_proj_model_objective(synth_uos_dataset):
  dataset, data_loader, x = synth_uos_dataset

  torch.manual_seed(2019)
  model = mod.KSubspaceProjModel(k=k, d=d, D=D, affine=affine,
      replicates=replicates,
      reg_params={'U_frosqr_in': 1e-3, 'U_frosqr_out': 1e-6})

  with torch.no_grad():
    x_ = model.forward(x)
    obj = model.objective()[0]

  expected_x_prefix = np.array(
      [-4.4353388e-04, -1.4909614e-04, -6.1778810e-05, -2.7829624e-04,
      1.7707214e-04, 4.1063624e-05, -2.2891119e-04, 3.9358105e-04,
      1.5634160e-04, -2.3627513e-04, 2.3868230e-04, 7.7640405e-05,
      -1.1694831e-04, -8.1144070e-04, -8.0584919e-05, -1.9985480e-04,
      1.0086927e-04, 2.7064842e-04, -1.2854450e-04, 5.1062831e-05],
      dtype=np.float32)
  x_prefix = x_[:2, :2, :5].contiguous().view(-1).numpy()
  assert np.allclose(x_prefix, expected_x_prefix, rtol=RTOL, atol=ATOL)

  expected_obj = 0.49909949
  obj = obj.item()
  assert np.isclose(obj, expected_obj)


def test_km_model_objective(synth_kmeans_dataset):
  dataset, data_loader, x = synth_kmeans_dataset

  torch.manual_seed(2019)
  model = mod.KMeansModel(k=k, D=D, replicates=replicates,
      reg_params={'b_frosqr_out': 0.0})

  with torch.no_grad():
    x_ = model.forward(x)
    obj = model.objective()[0]

  expected_x_prefix = np.array(
      [0.00271328, -0.00508405, -0.0072806, 0.00122242, -0.00482553,
      -0.00589834, -0.00838365, 0.01280026, 0.01253189, 0.00720055, 0.01377965,
      -0.00826924, 0.00310045, 0.0120761, 0.01169455, 0.01264404, 0.00317396,
      -0.00987673, -0.00915308, 0.02370972],
      dtype=np.float32)
  x_prefix = x_[:2, :2, :5].contiguous().view(-1).numpy()
  assert np.allclose(x_prefix, expected_x_prefix, rtol=RTOL, atol=ATOL)

  expected_obj = 1.60742127895
  obj = obj.item()
  assert np.isclose(obj, expected_obj)


def test_batch_mf_model_objective(synth_uos_dataset):
  dataset, data_loader, _ = synth_uos_dataset

  torch.manual_seed(2019)
  model = mod.KSubspaceBatchAltMFModel(k=k, d=d, dataset=dataset,
      affine=affine, replicates=replicates,
      reg_params={'U_frosqr_in': 1e-3, 'U_frosqr_out': 1e-6, 'z': 0.01})

  with torch.no_grad():
    obj = model.objective()[0]

  expected_obj = 0.4774763882
  obj = obj.item()
  assert np.isclose(obj, expected_obj)

  expected_err = 0.606
  err = ut.eval_cluster_error(model.groups[:, 0], model.true_groups,
      k=model.k)[0]
  assert np.isclose(err, expected_err)


def test_batch_proj_model_objective(synth_uos_dataset):
  dataset, data_loader, _ = synth_uos_dataset

  torch.manual_seed(2019)
  model = mod.KSubspaceBatchAltProjModel(k=k, d=d, dataset=dataset,
      affine=affine, replicates=replicates,
      reg_params={'U_frosqr_in': 1e-3, 'U_frosqr_out': 1e-6})

  with torch.no_grad():
    obj = model.objective()[0]

  expected_obj = 0.4991159737
  obj = obj.item()
  assert np.isclose(obj, expected_obj)

  expected_err = 0.602
  err = ut.eval_cluster_error(model.groups[:, 0], model.true_groups,
      k=model.k)[0]
  assert np.isclose(err, expected_err)


def test_batch_km_model_objective(synth_kmeans_dataset):
  dataset, data_loader, _ = synth_kmeans_dataset

  torch.manual_seed(2019)
  model = mod.KMeansBatchAltModel(k=k, dataset=dataset, replicates=replicates,
      reg_params={'b_frosqr_out': 0.0})

  with torch.no_grad():
    obj = model.objective()[0]

  expected_obj = 1.587741494
  obj = obj.item()
  assert np.isclose(obj, expected_obj)

  expected_err = 0.538
  err = ut.eval_cluster_error(model.groups[:, 0], model.true_groups,
      k=model.k)[0]
  assert np.isclose(err, expected_err)


def test_mc_model_objective(synth_mc_dataset):
  dataset, data_loader, x, x0 = synth_mc_dataset

  torch.manual_seed(2019)
  model = mod.KSubspaceMCModel(k=k, d=d, D=D, affine=affine,
      replicates=replicates,
      reg_params={'U_frosqr_in': 1e-3, 'U_frosqr_out': 1e-6, 'z': 0.01},
      sparse_encode=True, sparse_decode=True)

  with torch.no_grad():
    obj = model.objective(x)[0].item()
  expected_obj = 0.37360477447509766
  assert np.isclose(obj, expected_obj)

  comp_err = model.eval_comp_error(x0)[0].item()
  expected_comp_err = 1.0230462551116943
  assert np.isclose(comp_err, expected_comp_err)

  model.sparse_encode = False
  model.sparse_decode = False
  model._slcUs, model._slcbs = None, None
  model.cache_x_ = True
  with torch.no_grad():
    obj = model.objective(x)[0].item()
  assert np.isclose(obj, expected_obj)

  comp_err = model.eval_comp_error(x0)[0].item()
  expected_comp_err = 1.0230462551116943
  assert np.isclose(comp_err, expected_comp_err)


def test_batch_mf_pfi_init(synth_uos_dataset):
  dataset, data_loader, _ = synth_uos_dataset

  torch.manual_seed(2019)
  model = mod.KSubspaceBatchAltMFModel(k=k, d=d, dataset=dataset,
      affine=affine, replicates=replicates,
      reg_params={'U_frosqr_in': 1e-3, 'U_frosqr_out': 1e-6, 'z': 0.01})

  model.pfi_init(dataset.X)

  with torch.no_grad():
    obj = model.objective()[0]

  expected_obj = 0.11262587
  obj = obj.item()
  assert np.isclose(obj, expected_obj)

  expected_err = 0.004
  err = ut.eval_cluster_error(model.groups[:, 0], model.true_groups,
      k=model.k)[0]
  assert np.isclose(err, expected_err)


def test_batch_proj_pfi_init(synth_uos_dataset):
  dataset, data_loader, _ = synth_uos_dataset

  torch.manual_seed(2019)
  model = mod.KSubspaceBatchAltProjModel(k=k, d=d, dataset=dataset,
      affine=affine, replicates=replicates,
      reg_params={'U_frosqr_in': 1e-3, 'U_frosqr_out': 1e-6})

  model.pfi_init(dataset.X)

  with torch.no_grad():
    obj = model.objective()[0]

  expected_obj = 0.1163978577
  obj = obj.item()
  assert np.isclose(obj, expected_obj)

  expected_err = 0.002
  err = ut.eval_cluster_error(model.groups[:, 0], model.true_groups,
      k=model.k)[0]
  assert np.isclose(err, expected_err)


def test_batch_km_pfi_init(synth_kmeans_dataset):
  dataset, data_loader, _ = synth_kmeans_dataset

  torch.manual_seed(2019)
  model = mod.KMeansBatchAltModel(k=k, dataset=dataset, replicates=replicates,
      reg_params={'b_frosqr_out': 0.0})

  model.pfi_init(dataset.X)

  with torch.no_grad():
    obj = model.objective()[0]

  expected_obj = 0.961146593
  obj = obj.item()
  assert np.isclose(obj, expected_obj)

  expected_err = 0.0
  err = ut.eval_cluster_error(model.groups[:, 0], model.true_groups,
      k=model.k)[0]
  assert np.isclose(err, expected_err)


def test_mf_scale_grad(synth_uos_dataset):
  dataset, data_loader, x = synth_uos_dataset

  torch.manual_seed(2019)
  model = mod.KSubspaceMFModel(k=k, d=d, D=D, affine=False,
      replicates=replicates,
      reg_params={'U_frosqr_in': 1e-3, 'U_frosqr_out': 1e-6, 'z': 0.01},
      scale_grad_lip=True)

  model.epoch_init()
  obj = model.objective(x)[0]
  obj.backward()

  expected_Lip = np.array(
      [0.25192788, 0.27584976, 0.23968105, 0.24420282, 0.36183926, 0.1280791],
      dtype=np.float32)
  Lip = model.Lip.numpy()
  assert np.allclose(Lip, expected_Lip, rtol=RTOL, atol=ATOL)


def test_km_scale_grad(synth_kmeans_dataset):
  dataset, data_loader, x = synth_kmeans_dataset

  torch.manual_seed(2019)
  model = mod.KMeansModel(k=k, D=D, replicates=replicates,
      reg_params={'b_frosqr_out': 0.0}, scale_grad_lip=True)

  model.epoch_init()
  obj = model.objective(x)[0]
  obj.backward()

  expected_Lip = np.array(
      [0.2978906, 0.2822656, 0.21976563, 0.3135156, 0.2822656, 0.2978906],
      dtype=np.float32)
  Lip = model.Lip.numpy()
  assert np.allclose(Lip, expected_Lip, rtol=RTOL, atol=ATOL)
