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
