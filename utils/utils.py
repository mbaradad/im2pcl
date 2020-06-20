# we need to import open3d this before torch:
# https://github.com/pytorch/pytorch/issues/19739

import argparse
import datetime
import json
import os
import pickle
import random
import shutil
import socket
from pathlib import Path

import cv2
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


def load_from_pickle(filename):
  try:
    with open(filename, 'rb') as handle:
      return pickle.load(handle)
  except:
    pass
  try:
    with open(filename, 'rb') as handle:
      return pickle.load(handle, encoding='latin1')
  except:
    pass
  try:
    with open(filename, 'rb') as handle:
      return pickle.load(open(filename, "rb"))
  except:
    raise Exception('Failed to unpickle!')


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


def torch_deg2rad(angle_deg):
  return (angle_deg * np.pi) / 180


def xrotation_deg_torch(deg, four_dims=False):
  return xrotation_rad_torch(torch_deg2rad(deg), four_dims)


def yrotation_deg_torch(deg, four_dims=False):
  return yrotation_rad_torch(torch_deg2rad(deg), four_dims)


def zrotation_deg_torch(deg, four_dims=False):
  return zrotation_rad_torch(torch_deg2rad(deg), four_dims)


def listdir(folder, prepend_folder=False, extension=None, type=None):
  assert type in [None, 'file', 'folder'], "Type must be None, 'file' or 'folder'"
  files = [k for k in os.listdir(folder) if (True if extension is None else k.endswith(extension))]
  if type == 'folder':
    files = [k for k in files if os.path.isdir(folder + '/' + k)]
  elif type == 'file':
    files = [k for k in files if not os.path.isdir(folder + '/' + k)]
  if prepend_folder:
    files = [folder + '/' + f for f in files]
  return files


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


def cv2_resize(image, target_height_width, interpolation=cv2.INTER_NEAREST):
  if len(image.shape) == 2:
    return cv2.resize(image, target_height_width[::-1], interpolation=interpolation)
  else:
    return cv2.resize(image.transpose((1, 2, 0)), target_height_width[::-1], interpolation=interpolation).transpose((2, 0, 1))


def crop_center(img, crop):
  cropy, cropx = crop
  if len(img.shape) == 3:
    _, y, x = img.shape
  else:
    y, x = img.shape
  startx = x // 2 - (cropx // 2)
  starty = y // 2 - (cropy // 2)
  if len(img.shape) == 3:
    return img[:, starty:starty + cropy, startx:startx + cropx]
  else:
    return img[starty:starty + cropy, startx:startx + cropx]


def best_centercrop_image(image, height, width, return_rescaled_size=False, interpolation=cv2.INTER_NEAREST):
  if height == -1 and width == -1:
    if return_rescaled_size:
      return image, image.shape
    return image
  image_height, image_width = image.shape[-2:]
  im_crop_height_shape = (int(height), int(image_width * height / image_height))
  im_crop_width_shape = (int(image_height * width / image_width), int(width))
  # if we crop on the height dimension, there must be enough pixels on the width
  if im_crop_height_shape[1] >= width:
    rescaled_size = im_crop_height_shape
  else:
    # crop over width
    rescaled_size = im_crop_width_shape
  resized_image = cv2_resize(image, rescaled_size, interpolation=interpolation)
  center_cropped = crop_center(resized_image, (height, width))
  if return_rescaled_size:
    return center_cropped, rescaled_size
  else:
    return center_cropped


def to_cv2(image):
  return image.transpose((1, 2, 0))


def from_cv2(image):
  return image.transpose((2, 0, 1))


def cv2_imwrite(im, file, normalize=False, jpg_quality=None):
  if len(im.shape) == 3 and im.shape[0] == 3 or im.shape[0] == 4:
    im = im.transpose(1, 2, 0)
  if normalize:
    im = (im - im.min()) / (im.max() - im.min())
    im = np.array(255.0 * im, dtype='uint8')
  if jpg_quality is None:
    # The default jpg quality seems to be 95
    if im.shape[-1] == 3:
      cv2.imwrite(file, im[:, :, ::-1])
    else:
      raise Exception('Alpha not working correctly')
      im_reversed = np.concatenate((im[:, :, 3:0:-1], im[:, :, -2:-1]), axis=2)
      cv2.imwrite(file, im_reversed)
  else:
    cv2.imwrite(file, im[:, :, ::-1], [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])


def cv2_imread(file, return_BGR=False):
  im = cv2.imread(file)
  if im is None:
    raise Exception('Image {} could not be read!'.format(file))
  im = im.transpose(2, 0, 1)
  if return_BGR:
    return im
  return im[::-1, :, :]


def png_16_bits_imread(file):
  return cv2.imread(file, -cv2.IMREAD_ANYDEPTH)


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


def read_text_file_lines(filename, stop_at=-1):
  lines = list()
  with open(filename, 'r') as f:
    for line in f:
      if stop_at > 0 and len(lines) >= stop_at:
        return lines
      lines.append(line.replace('\n', ''))
  return lines


def write_text_file_lines(lines, file):
  with open(file, 'w') as file_handler:
    for item in lines:
      file_handler.write("%s\n" % item)


def load_json(file_name, replace_nans=False):
  with open(file_name) as handle:
    json_string = handle.read()
    if replace_nans:
      json_string = json_string.replace('-nan', 'NaN').replace('nan', 'NaN')
    parsed_json = json.loads(json_string)
    return parsed_json


def dump_json(json_dict, filename):
  with open(filename, 'w') as fp:
    json.dump(json_dict, fp, indent=4)


def find_all_files_recursively(folder, prepend_path=False, extension=None, progress=False):
  if extension is None:
    glob_expresion = '*'
  else:
    glob_expresion = '*' + extension
  all_files = []
  for f in Path(folder).rglob(glob_expresion):
    all_files.append((str(f) if prepend_path else f.name))
  return all_files
