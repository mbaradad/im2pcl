# we need to import open3d this before torch:
# https://github.com/pytorch/pytorch/issues/19739

import argparse
import datetime
import os
import pickle
import shutil
import socket
from pathlib import Path
import random
import numpy as np
import torch
from plyfile import PlyData, PlyElement


def get_hostname():
  return socket.gethostname()


def select_gpus(gpus_arg):
  # so that default gpu is one of the selected, instead of 0
  if len(gpus_arg) > 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_arg
    gpus = list(range(len(gpus_arg.split(','))))
  else:
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    gpus = []
  print('CUDA_VISIBLE_DEVICES={}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
  return gpus


def count_trainable_parameters(network, return_as_string):
  n_parameters = sum(p.numel() for p in network.parameters() if p.requires_grad)
  if return_as_string:
    return f"{n_parameters:,}"
  else:
    return n_parameters


def gettimedatestring():
  return datetime.datetime.now().strftime("%m-%d-%H:%M")


def str2bool(v):
  assert type(v) is str
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean (yes, true, t, y or 1, lower or upper case) string expected.')


def str2intlist(v):
  if len(v) == 0:
    return []
  return [int(k) for k in v.split(',')]


def list_of_lists_into_single_list(list_of_lists):
  flat_list = [item for sublist in list_of_lists for item in sublist]
  return flat_list


def tonumpy(tensor):
  if type(tensor) is list:
    return np.array(tensor)
  if type(tensor) is np.ndarray:
    return tensor
  if tensor.requires_grad:
    tensor = tensor.detach()
  if type(tensor) is torch.autograd.Variable:
    tensor = tensor.data
  if tensor.is_cuda:
    tensor = tensor.cpu()
  return tensor.detach().numpy()


def totorch(array):
  if type(array) is torch.Tensor:
    return array
  if not type(array) is np.ndarray:
    array = np.array(array)
  return torch.FloatTensor(array)


def dump_pointcloud(coords, colors, file_name, valid_mask=None, max_points=10000000, subsample_by_distance=False):
  if not valid_mask is None:
    coords = coords[valid_mask]
    colors = colors[valid_mask]
  if max_points != -1 and coords.shape[0] > max_points:
    if subsample_by_distance:
      # just get the ones whose neighbors are further away
      distances = -1 * (np.sqrt(((coords[:-1] - coords[1:]) ** 2).sum(-1)))
      distances_and_indices = list(zip(distances, range(len(distances))))
      distances_and_indices.sort()
      selected_positions = [k[1] for k in distances_and_indices[:max_points]]
    else:
      selected_positions = random.sample(range(coords.shape[0]), max_points)
    coords = coords[selected_positions]
    colors = colors[selected_positions]
  coords = tonumpy(coords)
  colors = tonumpy(colors)
  if coords.shape[0] == 3:
    coords = coords.transpose()
    colors = colors.transpose()
  data_np = np.concatenate((coords, colors), axis=1)
  tupled_data = [tuple(k) for k in data_np.tolist()]
  data = np.array(tupled_data, dtype=[('x', 'f4'),
                                      ('y', 'f4'),
                                      ('z', 'f4'),
                                      ('red', 'u1'),
                                      ('green', 'u1'),
                                      ('blue', 'u1')])

  vertex = PlyElement.describe(data, 'vertex')
  vertex.data = data
  plydata = PlyData([vertex])
  if not os.path.exists(os.path.dirname(file_name)):
    os.makedirs(os.path.dirname(file_name))
  plydata.write(file_name + '.ply')


def dump_to_pickle(filename, obj):
  try:
    with open(filename, 'wb') as handle:
      pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
  except:
    with open(filename, 'wb') as handle:
      pickle.dump(obj, handle)


def save_checkpoint_pickles(save_path, others_to_pickle, is_best):
  for (prefix, object) in others_to_pickle.items():
    dump_to_pickle(save_path / ('{}_latest.pckl').format(prefix), object)
  if is_best:
    for prefix in others_to_pickle.keys():
      shutil.copyfile(save_path / ('{}_latest.pckl').format(prefix),
                      save_path / ('{}_best.pckl').format(prefix))


def save_checkpoint(save_path, nets_to_save, is_best, other_objects_to_pickle=None):
  print('Saving checkpoint in: ' + str(save_path))
  save_path = Path(save_path)
  for (prefix, state) in nets_to_save.items():
    torch.save(state, save_path / ('{}_latest.pth.tar').format(prefix))
  if is_best:
    for prefix in nets_to_save.keys():
      shutil.copyfile(save_path / ('{}_latest.pth.tar').format(prefix),
                      save_path / ('{}_best.pth.tar').format(prefix))
  if not other_objects_to_pickle is None:
    save_checkpoint_pickles(save_path, other_objects_to_pickle, is_best)
  print('Saved in: ' + str(save_path))


def xrotation_deg_torch(deg, four_dims=False):
  return xrotation_rad_torch(torch_deg2rad(deg), four_dims)


def yrotation_deg_torch(deg, four_dims=False):
  return yrotation_rad_torch(torch_deg2rad(deg), four_dims)


def zrotation_deg_torch(deg, four_dims=False):
  return zrotation_rad_torch(torch_deg2rad(deg), four_dims)


def xrotation_rad_torch(rad, four_dims=False):
  c = torch.cos(rad)
  s = torch.sin(rad)
  zeros = torch.zeros_like(rad)
  ones = torch.ones_like(rad)
  if not four_dims:
    return torch.cat([
      torch.cat([ones[:, None], zeros[:, None], zeros[:, None]], dim=-1)[:, :, None],
      torch.cat([zeros[:, None], c[:, None], s[:, None]], dim=-1)[:, :, None],
      torch.cat([zeros[:, None], -s[:, None], c[:, None]], dim=-1)[:, :, None]
    ], dim=-1)
  else:
    raise Exception('Not implemented!')
    return torch.cat([[ones, zeros, zeros, zeros],
                      [zeros, c, s, zeros],
                      [zeros, -s, c, zeros],
                      [zeros, zeros, zeros, ones]])


def yrotation_rad_torch(rad, four_dims=False):
  c = torch.cos(rad)
  s = torch.sin(rad)
  zeros = torch.zeros_like(rad)
  ones = torch.ones_like(rad)
  if not four_dims:
    return torch.cat([
      torch.cat([c[:, None], zeros[:, None], s[:, None]], dim=-1)[:, :, None],
      torch.cat([zeros[:, None], ones[:, None], zeros[:, None]], dim=-1)[:, :, None],
      torch.cat([-s[:, None], zeros[:, None], c[:, None]], dim=-1)[:, :, None]
    ], dim=-1)
  else:
    raise Exception('Not implemented!')
    return torch.cat([[c, zeros, s, zeros],
                      [zeros, ones, zeros, zeros],
                      [-s, zeros, c, zeros],
                      [zeros, zeros, zeros, ones]])


def zrotation_rad_torch(rad, four_dims=False):
  c = torch.cos(rad)
  s = torch.sin(rad)
  zeros = torch.zeros_like(rad)
  ones = torch.ones_like(rad)
  if not four_dims:
    return torch.cat([
      torch.cat([c[:, None], -s[:, None], zeros[:, None]], dim=-1)[:, :, None],
      torch.cat([s[:, None], c[:, None], zeros[:, None]], dim=-1)[:, :, None],
      torch.cat([zeros[:, None], zeros[:, None], ones[:, None]], dim=-1)[:, :, None]
    ], dim=-1)
  else:
    raise Exception('Not implemented!')
    return torch.cat([[c, zeros, s, zeros],
                      [zeros, ones, zeros, zeros],
                      [-s, zeros, c, zeros],
                      [zeros, zeros, zeros, ones]])


def xrotation_deg(deg, four_dims=False):
  return xrotation_rad(np.deg2rad(deg), four_dims)


def yrotation_deg(deg, four_dims=False):
  return yrotation_rad(np.deg2rad(deg), four_dims)


def zrotation_deg(deg, four_dims=False):
  return zrotation_rad(np.deg2rad(deg), four_dims)


def xrotation_rad(th, four_dims=False):
  c = np.cos(th)
  s = np.sin(th)
  if not four_dims:
    return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
  else:
    return np.array([[1, 0, 0, 0], [0, c, s, 0], [0, -s, c, 0], [0, 0, 0, 1]])


def yrotation_rad(th, four_dims=False):
  c = np.cos(th)
  s = np.sin(th)
  if not four_dims:
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
  else:
    return np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]])


def zrotation_rad(th, four_dims=False):
  c = np.cos(th)
  s = np.sin(th)
  if not four_dims:
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
  else:
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


class ImageFolderCenterCroppLoader():
  def __init__(self, folder_or_img_list, height, width, extension='jpg'):
    if type(folder_or_img_list) == list:
      self.img_list = folder_or_img_list
    elif os.path.isdir(folder_or_img_list):
      self.img_list = [folder_or_img_list + '/' + k for k in os.listdir(folder_or_img_list) if k.endswith(extension)]
    self.img_list.sort()
    self.height = height
    self.width = width

  def __len__(self):
    return len(self.img_list)

  def __getitem__(self, item):
    image = cv2_imread(self.img_list[item])
    img = best_centercrop_image(image, self.height, self.width)
    to_return = {'image': np.array(img / 255.0, dtype='float32'),
                 'path': self.img_list[item].split('/')[-1],
                 'full_path': self.img_list[item]}
    return to_return



