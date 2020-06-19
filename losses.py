import torch
from torch.nn.functional import cosine_similarity


def prepare_dimensions_and_compute_N(prediction, gt, mask):
  if len(prediction.shape) == 3:
    prediction = prediction[:, None, :, :]
  if len(gt.shape) == 3:
    gt = gt[:, None, :, :]
  assert prediction.shape == gt.shape
  if not mask is None:
    assert prediction.shape[0] == mask.shape[0]
    assert prediction.shape[-2:] == mask.shape[-2:]
    if len(mask.shape) == 3:
      mask = mask[:, None, :, :]
    assert mask.shape[1] == 1
    N = prediction.shape[1] * mask.sum(-1).sum(-1).sum(-1)
    N[N == 0] = 1
  else:
    N = prediction.shape[1] * prediction.shape[2] * prediction.shape[3]
  return prediction, gt, mask, N


def NormalsLoss(predicted_normals, normals_gt, mask):
  if len(mask.shape) == 4:
    assert mask.shape[1] == 1
    mask = mask[:, 0, :, :]
  m_normals_loss = (1 / 2 - 1 / 2 * cosine_similarity(predicted_normals, normals_gt, dim=1)) * mask
  return m_normals_loss.sum() / (mask.sum() + 1)


def MyRMSELoss(prediction, gt, mask=None):
  prediction, gt, mask, N = prepare_dimensions_and_compute_N(prediction, gt, mask)
  d_diff = gt - prediction
  if not mask is None:
    d_diff = d_diff * mask
  mse = (d_diff ** 2).sum(-1).sum(-1).sum(-1) / N
  rmse = torch.sqrt(mse + 1e-16)
  return rmse.mean()


import numpy as np
#From: https://github.com/mrharicot/monodepth/blob/master/utils/evaluation_utils.py
def compute_depth_metrics(all_predicted, all_gt, gt_depth_mask):
  if len(all_predicted.shape) != 2:
    raise Exception('Evaluation should be performed per sample!')
  assert all_predicted.shape == all_gt.shape == gt_depth_mask.shape
  assert type(all_predicted) == np.ndarray

  gt = all_gt[gt_depth_mask]
  pred = all_predicted[gt_depth_mask]
  thresh = np.maximum((gt / pred), (pred / gt))
  a1 = (thresh < 1.25).mean()
  a2 = (thresh < 1.25 ** 2).mean()
  a3 = (thresh < 1.25 ** 3).mean()

  rmse = (gt - pred) ** 2
  rmse = np.sqrt(rmse.mean())

  rmse_log = (np.log(gt) - np.log(pred)) ** 2
  rmse_log = np.sqrt(rmse_log.mean())

  rmse_log10 = (np.log10(gt) - np.log10(pred)) ** 2
  rmse_log10 = np.sqrt(rmse_log10.mean())

  log10 = float(np.abs(np.log10(gt) - np.log10(pred)).mean())
  logmae = float(np.abs(np.log(gt) - np.log(pred)).mean())

  abs_rel = np.mean(np.abs(gt - pred) / gt)

  sq_rel = np.mean(((gt - pred) ** 2) / gt)

  results = dict()
  results['abs_rel'] = abs_rel
  results['sq_rel'] = sq_rel
  results['rmse'] = rmse
  results['rmse_log'] = rmse_log
  results['rmse_log10'] = rmse_log10
  results['log10'] = log10
  results['logmae'] = logmae
  results['a1'] = a1
  results['a2'] = a2
  results['a3'] = a3

  meand_depth = pred*0 + gt.mean()
  rmse_mean_depth = (gt - meand_depth) ** 2
  rmse_mean_depth = np.sqrt(rmse_mean_depth.mean())
  results['rmse_mean_gt_depth'] = rmse_mean_depth

  return results