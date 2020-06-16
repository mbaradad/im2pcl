from __future__ import division

import math

import numpy as np
import torch
from torch.autograd import Variable
from kornia import pixel2cam as k_pixel2cam

# Borrowed from: https://github.com/ClementPinard/SfmLearner-Pytorch
pixel_coords = None
pixel_coords_cpu = None
pixel_coords_cuda = None


def get_id_grid(depth_image, new_grid=False):
  global pixel_coords, pixel_coords_cpu, pixel_coords_cuda
  b, h, w = depth_image.size()
  if (pixel_coords is None) or pixel_coords.size(2) < h or new_grid:
    set_id_grid(depth_image)
  if depth_image.is_cuda:
    return pixel_coords_cuda
  else:
    return pixel_coords_cpu


def get_flow(new_coordinates):
  id_grid = get_scaled_id_grid(new_coordinates[:, :, :, 0])
  flow = new_coordinates - id_grid
  # mask things that fall outside, which are marked with a coordinate == 2
  flow = flow * (new_coordinates[0, :, :, :] != 2).float()
  return flow


def get_scaled_id_grid(depth_image):
  b, h, w = depth_image.size()

  pcoords = get_id_grid(depth_image)
  X = pcoords[:, 0]
  Y = pcoords[:, 1]
  Z = pcoords[:, 2].clamp(min=1e-3)

  X_norm = 2 * (X / Z) / (w - 1) - 1  # Normalized, -1 if on extreme left, 1 if on extreme right (x = w-1) [B, H*W]
  Y_norm = 2 * (Y / Z) / (h - 1) - 1  # Idem [B, H*W]
  coords = torch.cat([X_norm, Y_norm])[None, :, :, :]
  return coords.transpose(1, 3).transpose(1, 2)


def set_id_grid(depth):
  # the pixel coords is a HxW grid, where each element contains the position as (x,y,1)
  global pixel_coords_cpu, pixel_coords_cuda
  b, h, w = depth.size()
  i_range = Variable(torch.arange(0, h).view(1, h, 1).expand(1, h, w)).type_as(depth)  # [1, H, W]
  j_range = Variable(torch.arange(0, w).view(1, 1, w).expand(1, h, w)).type_as(depth)  # [1, H, W]
  ones = Variable(torch.ones(1, h, w)).type_as(depth)

  pixel_coords = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]
  if depth.is_cuda:
    pixel_coords_cuda = pixel_coords.cuda()
  else:
    pixel_coords_cpu = pixel_coords.cpu()


def check_sizes(input, input_name, expected):
  condition = [input.ndimension() == len(expected)]
  for i, size in enumerate(expected):
    if size.isdigit():
      condition.append(input.size(i) == int(size))
  assert (all(condition)), "wrong size for {}, expected {}, got  {}".format(input_name, 'x'.join(expected), list(input.size()))


def get_id_grid(depth_image, new_grid=False):
  global pixel_coords, pixel_coords_cpu, pixel_coords_cuda
  b, h, w = depth_image.size()
  if (pixel_coords is None) or pixel_coords.size(2) < h or new_grid:
    set_id_grid(depth_image)
  if depth_image.is_cuda:
    return pixel_coords_cuda
  else:
    return pixel_coords_cpu


def pixel2cam(depth, K):
  global pixel_coords_cpu, pixel_coords_cuda
  if len(depth.shape) == 4:
    assert depth.shape[1] == 1
    depth = depth[1]
  assert len(depth.shape) == 3
  assert K.shape[1] == K.shape[2]
  assert depth.shape[0] == K.shape[0]

  K = make_4x4_K(K)
  intrinsics_inv = torch.inverse(K)

  height, width = depth.shape[-2:]
  if depth.is_cuda:
    # to avoid recomputing the id_grid if it is not necessary
    if (pixel_coords_cuda is None) or pixel_coords_cuda.size(2) != height or pixel_coords_cuda.size(3) != width:
      set_id_grid(height, width, to_cuda=True)
    pixel_coords = pixel_coords_cuda
  else:
    if (pixel_coords_cpu is None) or pixel_coords_cpu.size(2) != height or pixel_coords_cpu.size(3) != width:
      set_id_grid(height, width, to_cuda=False)
    pixel_coords = pixel_coords_cpu

  batch_size = depth.shape[0]
  pcl = k_pixel2cam(depth[:,None,:,:], intrinsics_inv, pixel_coords.expand(batch_size, -1, -1, -1))
  return pcl.permute(0,3,1,2)



def mat2euler_rotation(R):
  '''
  Convert rotation matrix to euler angles.

  Reference: https://www.learnopencv.com/rotation-matrix-to-euler-angles/
  '''
  sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

  singular = sy < 1e-6

  if not singular:
    x = math.atan2(R[2, 1], R[2, 2])
    y = math.atan2(-R[2, 0], sy)
    z = math.atan2(R[1, 0], R[0, 0])
  else:
    x = math.atan2(-R[1, 2], R[1, 1])
    y = math.atan2(-R[2, 0], sy)
    z = 0

  return np.array([x, y, z])


def euler2mat_opencv(euler_theta):
  R_x = np.array([[1, 0, 0],
                  [0, math.cos(euler_theta[0]), -math.sin(euler_theta[0])],
                  [0, math.sin(euler_theta[0]), math.cos(euler_theta[0])]
                  ])

  R_y = np.array([[math.cos(euler_theta[1]), 0, math.sin(euler_theta[1])],
                  [0, 1, 0],
                  [-math.sin(euler_theta[1]), 0, math.cos(euler_theta[1])]
                  ])

  R_z = np.array([[math.cos(euler_theta[2]), -math.sin(euler_theta[2]), 0],
                  [math.sin(euler_theta[2]), math.cos(euler_theta[2]), 0],
                  [0, 0, 1]
                  ])

  R = np.dot(R_z, np.dot(R_y, R_x))

  return R


def euler2mat(angle):
  """Convert euler angles to rotation matrix.

   Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174

  Args:
    angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
  Returns:
    Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
  """
  B = angle.size(0)
  x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

  cosz = torch.cos(z)
  sinz = torch.sin(z)

  zeros = z.detach() * 0
  ones = zeros.detach() + 1
  zmat = torch.stack([cosz, -sinz, zeros,
                      sinz, cosz, zeros,
                      zeros, zeros, ones], dim=1).view(B, 3, 3)

  cosy = torch.cos(y)
  siny = torch.sin(y)

  ymat = torch.stack([cosy, zeros, siny,
                      zeros, ones, zeros,
                      -siny, zeros, cosy], dim=1).view(B, 3, 3)

  cosx = torch.cos(x)
  sinx = torch.sin(x)

  xmat = torch.stack([ones, zeros, zeros,
                      zeros, cosx, -sinx,
                      zeros, sinx, cosx], dim=1).view(B, 3, 3)

  # rotMat = xmat.bmm(ymat).bmm(zmat)
  # changed to match opencv and conversion euler->mat/mat->euler
  rotMat = torch.bmm(zmat, torch.bmm(ymat, xmat))

  return rotMat


def quat2mat(quat):
  """Convert quaternion coefficients to rotation matrix.

  Args:
    quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
  Returns:
    Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
  """
  norm_quat = torch.cat([quat[:, :1].detach() * 0 + 1, quat], dim=1)
  norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
  w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

  B = quat.size(0)

  w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
  wx, wy, wz = w * x, w * y, w * z
  xy, xz, yz = x * y, x * z, y * z

  rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                        2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                        2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
  return rotMat


def pose_vec2mat(vec, rotation_mode='euler'):
  """
  Convert 6DoF parameters to transformation matrix.

  Args:s
    vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
  Returns:
    A transformation matrix -- [B, 3, 4]
  """
  translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
  rot = vec[:, 3:]
  if rotation_mode == 'euler':
    rot_mat = euler2mat(rot)  # [B, 3, 3]
  elif rotation_mode == 'quat':
    rot_mat = quat2mat(rot)  # [B, 3, 3]
  transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
  return transform_mat


def mat2pose_vec(mat):
  rot_mat = mat[:3, :3]
  euler_translation = mat[:3, 3]
  euler_rot = mat2euler_rotation(rot_mat)
  pose_vec = np.concatenate((euler_translation, euler_rot))
  return pose_vec


def displace_to_pose(img, depth, pose, intrinsics, intrinsics_inv, rotation_mode='euler', padding_mode='zeros', return_coordinates=False):
  """
    Inverse warp a source image to the target image plane.

    Args:
      img: the source image (where to sample pixels) -- [B, 3, H, W]
      depth: depth map of the source image -- [B, H, W]
      pose: 6DoF pose parameters from source to target-- [B, 6]
      intrinsics: camera intrinsic matrix -- [B, 3, 3]
      intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
      Source image warped to the target image plane
    """
  check_sizes(img, 'img', 'B3HW')

  src_pixel_coords = get_displacement_pixel_transformation(img, depth, pose, intrinsics, intrinsics_inv, rotation_mode=rotation_mode, padding_mode=padding_mode)
  projected_img = torch.nn.functional.grid_sample(img, src_pixel_coords, padding_mode=padding_mode)
  if return_coordinates:
    return projected_img, src_pixel_coords
  else:
    return projected_img


def inverse_warp(img, depth, pose, intrinsics, intrinsics_inv, rotation_mode='euler', padding_mode='zeros', return_coordinates=False):
  """
    Inverse warp a source image to the target image plane.

    Args:
      img: the source image (where to sample pixels) -- [B, 3, H, W]
      depth: depth map of the target image -- [B, H, W]
      pose: 6DoF pose parameters from target to source -- [B, 6]
      intrinsics: camera intrinsic matrix -- [B, 3, 3]
      intrinsics_inv: inverse of the intrinsic matrix -- [B, 3, 3]
    Returns:
      Source image warped to the target image plane
    """
  check_sizes(img, 'img', 'B3HW')

  src_pixel_coords = get_warp_pixel_transformation(img, depth, pose, intrinsics, intrinsics_inv, rotation_mode=rotation_mode, padding_mode=padding_mode)
  projected_img = torch.nn.functional.grid_sample(img, src_pixel_coords, padding_mode=padding_mode, align_corners=False)
  if return_coordinates:
    return projected_img, src_pixel_coords
  else:
    return projected_img


def inverse_warp_camera_coords(img, pcoords, pose_mat, intrinsics, padding_mode='zeros', rotation_mode='euler'):
  b, _, h, w = pcoords.size()

  # Get projection matrix for tgt camera frame to source pixel frame
  proj_cam_to_src_pixel = intrinsics.bmm(pose_mat)  # [B, 3, 4]

  src_pixel_coords = cam2pixel(pcoords, proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:], padding_mode=padding_mode)  # [B,H,W,2]
  new_coordinates = src_pixel_coords
  projected_img = torch.nn.functional.grid_sample(img, src_pixel_coords, padding_mode=padding_mode)
  return projected_img, new_coordinates


def transform_camera_coords(cam_coords, actual_pose_mat):
  original_shape = cam_coords.shape

  rotation = actual_pose_mat[:, :, :3]
  translation = actual_pose_mat[:, :, -1:]
  b, _, h, w = cam_coords.size()
  cam_coords_flat = cam_coords.view(b, 3, -1)  # [B, 3, H*W]
  pcoords = rotation.bmm(cam_coords_flat)

  pcoords = pcoords + translation  # [B, 3, H*W]

  return pcoords.view(original_shape)


def common_pixel_transform_input(img, depth, pose, intrinsics, intrinsics_inv, rotation_mode):
  check_sizes(depth, 'depth', 'BHW')
  if rotation_mode == 'mat':
    check_sizes(pose, 'pose', 'B34')
  else:
    check_sizes(pose, 'pose', 'B6')
  check_sizes(intrinsics, 'intrinsics', 'B33')
  check_sizes(intrinsics_inv, 'intrinsics', 'B33')

  assert (intrinsics_inv.size() == intrinsics.size())

  cam_coords = pixel2cam(depth, intrinsics_inv)  # [B,3,H,W]

  if rotation_mode != 'mat':
    pose_mat = pose_vec2mat(pose, rotation_mode)  # [B,3,4]
  else:
    # already mat
    pose_mat = pose  # [B,3,4]

  return cam_coords, pose_mat


def get_displacement_pixel_transformation(img, depth, pose, intrinsics, intrinsics_inv, rotation_mode='euler', padding_mode='zeros'):
  cam_coords, pose_mat = common_pixel_transform_input(img, depth, pose, intrinsics, intrinsics_inv, rotation_mode)

  # Get projection matrix for tgt camera frame to source pixel frame
  proj_cam_to_src_pixel = intrinsics.bmm(pose_mat)  # [B, 3, 4]

  src_pixel_coords = cam2pixel(cam_coords, proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:], padding_mode=padding_mode)  # [B,H,W,2]
  return src_pixel_coords


def get_warp_pixel_transformation(img, depth, pose, intrinsics, intrinsics_inv, rotation_mode='euler', padding_mode='zeros'):
  cam_coords, pose_mat = common_pixel_transform_input(img, depth, pose, intrinsics, intrinsics_inv, rotation_mode)

  # Get projection matrix for tgt camera frame to source pixel frame
  proj_cam_to_src_pixel = intrinsics.bmm(pose_mat)  # [B, 3, 4]

  src_pixel_coords = cam2pixel(cam_coords, proj_cam_to_src_pixel[:, :, :3], proj_cam_to_src_pixel[:, :, -1:], padding_mode=padding_mode)  # [B,H,W,2]
  return src_pixel_coords


def compute_normals_from_closest_image_coords(coords, mask=None):
  # TODO: maybe compute normals with bigger region?
  x_coords = coords[:, 0, :, :]
  y_coords = coords[:, 1, :, :]
  z_coords = coords[:, 2, :, :]

  if type(coords) is torch.Tensor or type(coords) is torch.nn.parameter.Parameter:
    ts = torch.cat((x_coords[:, None, :-1, 1:], y_coords[:, None, :-1, 1:], z_coords[:, None, :-1, 1:]), dim=1)
    ls = torch.cat((x_coords[:, None, 1:, :-1], y_coords[:, None, 1:, :-1], z_coords[:, None, 1:, :-1]), dim=1)
    cs = torch.cat((x_coords[:, None, 1:, 1:], y_coords[:, None, 1:, 1:], z_coords[:, None, 1:, 1:]), dim=1)

    n = torch.cross((ls - cs), (ts - cs), dim=1)
    n_norm = n / (torch.sqrt(torch.abs((n * n).sum(1) + 1e-20))[:, None, :, :])
  else:
    ts = np.concatenate((x_coords[:, None, :-1, 1:], y_coords[:, None, :-1, 1:], z_coords[:, None, :-1, 1:]), axis=1)
    ls = np.concatenate((x_coords[:, None, 1:, :-1], y_coords[:, None, 1:, :-1], z_coords[:, None, 1:, :-1]), axis=1)
    cs = np.concatenate((x_coords[:, None, 1:, 1:], y_coords[:, None, 1:, 1:], z_coords[:, None, 1:, 1:]), axis=1)

    n = np.cross((ls - cs), (ts - cs), axis=1)
    n_norm = n / (np.sqrt(np.abs((n * n).sum(1) + 1e-20))[:, None, :, :])

  if not mask is None:
    assert len(mask.shape) == 4
    valid_ts = mask[:, :, :-1, 1:]
    valid_ls = mask[:, :, 1:, :-1]
    valid_cs = mask[:, :, 1:, 1:]
    if type(mask) is torch.tensor:
      final_mask = valid_ts * valid_ls * valid_cs
    else:
      final_mask = np.logical_and(np.logical_and(valid_ts, valid_ls), valid_cs)
    return n_norm, final_mask
  else:
    return n_norm


def compute_normals_from_pcl(coords, viewpoint, radius=0.2, max_nn=30):
  assert len(viewpoint.shape) == 1
  assert len(coords.shape) == 2
  assert type(coords) == np.ndarray
  assert coords.shape[0] == 3
  pcd = o3d.geometry.PointCloud()
  coords_to_pcd = coords.transpose()
  pcd.points = o3d.utility.Vector3dVector(coords_to_pcd)
  pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
  estimated_normals = np.array(pcd.normals)
  # orient towards viewpoint
  normal_viewpoint_signs = np.sign((estimated_normals * (coords_to_pcd - viewpoint[None, :])).sum(-1))
  corrected_normals = (estimated_normals * normal_viewpoint_signs[:, None]).transpose().reshape(coords.shape)
  return corrected_normals


def torch_deg2rad(angle_deg):
  return (angle_deg * np.pi) / 180


def intrinsics_to_fov_x_deg(intrinsics):
  # assumes zero centered
  width = intrinsics[0, 2] * 2
  fov_x_rad = 2 * np.arctan(width / 2.0 / intrinsics[0, 0])
  fov_x_deg = np.rad2deg(fov_x_rad)
  return fov_x_deg


def intrinsics_to_fov_y_deg(intrinsics):
  # assumes zero centered
  height = intrinsics[1, 2] * 2
  fov_y_rad = 2 * np.arctan(height / 2.0 / intrinsics[1, 1])
  fov_y_deg = np.rad2deg(fov_y_rad)
  return fov_y_deg


def fov_x_to_intrinsic_deg(fov_x_deg, width, height, return_inverse=True):
  fov_y_rad = fov_x_deg / 180.0 * np.pi
  return fov_x_to_intrinsic_rad(fov_y_rad, width, height, return_inverse=return_inverse)


def fov_x_to_intrinsic_rad(fov_x_rad, width, height, return_inverse=True):
  if type(fov_x_rad) is torch.Tensor:
    f = width / (2 * torch.tan(fov_x_rad / 2))
    zero = torch.FloatTensor(np.zeros(fov_x_rad.shape[0]))
    one = torch.FloatTensor(np.ones(fov_x_rad.shape[0]))
    if fov_x_rad.is_cuda:
      zero = zero.cuda()
      one = one.cuda()
    intrinsics_0 = torch.stack((f, zero, zero)).transpose(0, 1)
    intrinsics_1 = torch.stack((zero, f, zero)).transpose(0, 1)
    intrinsics_2 = torch.stack((width / 2, height / 2, one)).transpose(0, 1)
    intrinsics = torch.cat((intrinsics_0[:, :, None], intrinsics_1[:, :, None], intrinsics_2[:, :, None]), dim=-1)
  else:
    f = width / (2 * np.tan(fov_x_rad / 2))
    intrinsics = np.array(((f, 0, width / 2),
                           (0, f, height / 2),
                           (0, 0, 1)), dtype='float32')
  if return_inverse:
    if type(fov_x_rad) is torch.Tensor:
      return intrinsics, torch.inverse(intrinsics)
    else:
      return intrinsics, np.linalg.inv(intrinsics)
  else:
    return intrinsics
