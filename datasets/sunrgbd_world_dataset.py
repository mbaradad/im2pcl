import scipy.io as sio
from tqdm import tqdm
from transforms3d.euler import mat2euler

from paths import *

META_3D_PATH = SUNRGBD_PATH + '/meta3d.pckl'
POSE_STATS_PATH = SUNRGBD_PATH + '/pose_stats.pckl'

from utils.geom_utils import *
from utils.visdom_utils import *

def getSUNRGBDDirectoryList(path):
  directoryList = []
  # return nothing if path is a file
  if os.path.isfile(path):
    return []

  # add dir to directorylist if it contains .txt files
  if len([f for f in os.listdir(path) if 'extrinsics' in f or 'intrinsics.txt' in f or 'image' in f]) == 3:
    directoryList.append(path)

  dirs = os.listdir(path)
  dirs.sort()
  for d in dirs:
    new_path = os.path.join(path, d)
    if os.path.isdir(new_path):
      directoryList += getSUNRGBDDirectoryList(new_path)
  if len(directoryList) > 0: print(directoryList[-1])
  return directoryList


class SUNRGBDWorld():
  def __init__(self, height, width, split, use_bfx_depth=False, nyu_only=False, base_path=SUNRGBD_PATH, return_semantics=False, knn_normals=False,
               catch_exception=True, try_random_on_exception=False):
    if height == -1 or width == -1:
      assert height == width
    self.height = height
    self.width = width
    self.base_path = base_path
    self.return_semantics = return_semantics
    self.knn_normals = knn_normals
    self.catch_exception = catch_exception

    self.use_bfx_depth = use_bfx_depth

    split_mat = sio.loadmat(base_path + '/SUNRGBDtoolbox/traintestSUNRGBD/allsplit.mat')

    train_samples = [str(k[0]).replace('/n/fs/sun3d/data/SUNRGBD/', '') for k in split_mat['alltrain'][0]]
    val_samples = [str(k[0]).replace('/n/fs/sun3d/data/SUNRGBD/', '') for k in split_mat['alltest'][0]]
    self.try_random_on_exception = try_random_on_exception
    self.split = split
    if split == 'train':
      self.samples = train_samples
    elif split == 'val':
      self.samples = val_samples
    elif split == 'all':
      self.samples = [*train_samples, *val_samples]
    else:
      raise Exception('Split {} invalid!'.format(split))
    if nyu_only:
      self.samples = [k for k in self.samples if k.split('/')[1] == 'NYUdata']

    # strip last '/'
    self.samples = [k[:-1] if k[-1] == '/' else k for k in self.samples]
    self.meta_3d = self.load_3d_meta()
    self.samples.sort()
    self.last_return = None
    while self.last_return is None:
      rand_item = random.randint(0, self.__len__() - 1)
      self.last_return = self.__getitem__(rand_item)

  def get_cache_name(self, index):
    return self.split + '/' + '/'.join(self.samples[index].split('/')[-3:])

  def load_3d_meta(self):
    if os.path.exists(META_3D_PATH):
      return load_from_pickle(META_3D_PATH)
    meta_3dbb = sio.loadmat(self.base_path + '/SUNRGBD/SUNRGBDMeta3DBB_v2.mat')['SUNRGBDMeta'][0]
    columns = [k[0] for k in meta_3dbb[0].dtype.descr]
    meta_3d_per_sample = dict()
    for elem in meta_3dbb:
      splitted = elem[columns.index('depthpath')][0].replace('/n/fs/sun3d/data/SUNRGBD/', '').split('/')[:-2]
      splitted = [k for k in splitted if len(k) > 0]
      path = '/'.join(splitted)
      extrinsics = elem[columns.index('Rtilt')]
      anno_extrinsics = elem[columns.index('anno_extrinsics')]
      intrinsics = elem[columns.index('K')]
      valid = elem[columns.index('valid')]
      bboxes = []
      mat_bboxes = elem[columns.index('groundtruth3DBB')]
      if len(mat_bboxes) > 0:
        mat_bboxes = mat_bboxes[0]
        bboxes_columns = [k[0] for k in mat_bboxes.dtype.descr]
        for bbox in mat_bboxes:
          basis = bbox[bboxes_columns.index('basis')]
          coeffs = bbox[bboxes_columns.index('coeffs')]
          parsed_bbox = dict(class_name=bbox[bboxes_columns.index('classname')][0],
                             basis=basis,
                             coeffs=coeffs,
                             centroid=bbox[bboxes_columns.index('centroid')],
                             orientation=bbox[bboxes_columns.index('orientation')],
                             label=bbox[bboxes_columns.index('label')])
          bboxes.append(parsed_bbox)
      meta_3d_per_sample[path] = dict(extrinsics=extrinsics,
                                      anno_extrinsics=anno_extrinsics,
                                      intrinsics=intrinsics,
                                      valid=valid,
                                      bboxes=bboxes)
    # dump_to_pickle(META_3D_PATH, meta_3d_per_sample)
    return meta_3d_per_sample

  def __len__(self):
    return len(self.samples)

  @staticmethod
  def get_camera_stats_per_file(parallel=False, plot=False):
    if os.path.exists(POSE_STATS_PATH):
      params = load_from_pickle(POSE_STATS_PATH)
    else:
      dataset = SUNRGBDWorld(192, 240, 'all', return_semantics=False)

      def get_single_element(i):
        elem = dataset.__getitem__(i)
        simplified_path = elem['path'].replace(dataset.base_path + '/SUNRGBD/', '')
        return [simplified_path, elem['params']]

      if parallel:
        from p_tqdm import p_map
        params = p_map(get_single_element, list(range(len(dataset))), num_cpus=32)
      else:
        params = []
        for i in tqdm(range(len(dataset))):
          params.append(get_single_element(i))
      params = dict(params)
      dump_to_pickle(POSE_STATS_PATH, params)

    all_params = np.concatenate([k[None, :] for k in params.values()])
    if plot:
      visdom_histogram(all_params[:, 0], title='SUNRGBD_fov_x_deg')
      visdom_histogram(all_params[:, 1], title='SUNRGBD_height')
      visdom_histogram(all_params[:, 2], title='SUNRGBD_pitch_deg')
      visdom_histogram(all_params[:, 3], title='SUNRGBD_roll_deg')
    return params

  def get_all_poses(self):
    all_file_poses = self.get_camera_stats_per_file()
    sample_poses = []
    for file in self.samples:
      try:
        sample_poses.append(all_file_poses[file].tolist())
      except:
        sample_poses.append([-1, -1, -1, -1])
    all_file_poses = np.array(sample_poses)
    all_poses = {'FOVs_deg': all_file_poses[:, 0],
                 'heights': all_file_poses[:, 1],
                 'pitches': all_file_poses[:, 2],
                 'rolls': all_file_poses[:, 3]}
    return all_poses

  def get_ranges(self):
    all_poses = self.get_all_poses()
    fov_degs = all_poses['FOVs_deg'][all_poses['FOVs_deg'] > 0]
    FOV_deg_min, FOV_deg_max = fov_degs.min(), fov_degs.max()

    param_ranges = dict()
    pitch_min, pitch_max = np.quantile(all_poses['pitches'], 0.01), np.quantile(all_poses['pitches'], 0.99)
    roll_min, roll_max = np.quantile(all_poses['rolls'], 0.01), np.quantile(all_poses['rolls'], 0.99)
    height_min, height_max = np.quantile(all_poses['heights'], 0.01), np.quantile(all_poses['heights'], 0.99)

    param_ranges['FOV_deg'] = (FOV_deg_min, FOV_deg_max)
    param_ranges['pitch_deg'] = (pitch_min, pitch_max)
    param_ranges['roll_deg'] = (roll_min, roll_max)
    param_ranges['height'] = (height_min, height_max)

    return param_ranges

  def bbox_to_usorted_corners(self, bbox):
    # replicates code from mBB/get_corners_of_bb3d.m
    basis, coeffs, centroid = bbox['basis'], bbox['coeffs'][0], bbox['centroid'][0]
    corners = np.zeros((8, 3))
    coeffs = np.abs(coeffs)
    for i in range(8):
      actual_signs = np.array(((-1) ** (i // 4), (-1) ** (i // 2), (-1) ** (i % 2)))
      corners[i, :] = (actual_signs[0] * coeffs[0] * basis[0, :] + \
                       actual_signs[1] * coeffs[1] * basis[1, :] + \
                       actual_signs[2] * coeffs[2] * basis[2, :])
    corners = corners + centroid[None, :]
    # and change the coordinate system to ours (i.e. swap z and y)
    corners = np.array((corners[:, 0], -1 * corners[:, 2], corners[:, 1]))
    return corners.transpose()

  def __get_basic_item__(self, item, return_semantics=False):
    path = self.samples[item]
    meta_3d = self.meta_3d[path]
    full_path = self.base_path + '/SUNRGBD/' + path

    intrinsics = np.array(meta_3d['intrinsics'])

    image_dir = full_path + '/image'
    image = cv2_imread(image_dir + '/' + os.listdir(image_dir)[0])

    if self.use_bfx_depth:
      depth_dir = full_path + '/depth_bfx'
    else:
      depth_dir = full_path + '/depth'

    # replicate  SUNRGBDToolBox/read3dPoints.m
    depthVis = png_16_bits_imread(depth_dir + '/' + os.listdir(depth_dir)[0])
    # depthInpaint = np.bitwise_or(np.left_shift(depthVis, 3), np.right_shift(depthVis, 16-3))
    depth = np.bitwise_or(np.right_shift(depthVis, 3), np.left_shift(depthVis, 16 - 3))
    depth = depth / 1000.0

    original_height, original_width = image.shape[-2:]

    image = best_centercrop_image(image, self.height, self.width)
    depth, rescaled_size = best_centercrop_image(depth, self.height, self.width, return_rescaled_size=True)

    final_height, final_width = image.shape[1:]
    intrinsics[0, 0] = intrinsics[0, 0] / original_height * rescaled_size[0]
    intrinsics[1, 1] = intrinsics[1, 1] / original_width * rescaled_size[1]

    intrinsics[0, 2] = intrinsics[0, 2] / original_width * final_width
    intrinsics[1, 2] = intrinsics[1, 2] / original_height * final_height

    fov_x_rad = 2 * np.arctan(final_width / 2.0 / intrinsics[0, 0])
    fov_y_rad = 2 * np.arctan(final_height / 2.0 / intrinsics[1, 1])
    fov_x_deg = np.rad2deg(fov_x_rad)
    fov_y_deg = np.rad2deg(fov_y_rad)

    valid_depth_mask = depth != 0
    # also remove the < 1 and > 99 percentile, as this removes outliers
    valid_depth_mask[depth <= np.percentile(depth, 1)] = 0
    valid_depth_mask[depth >= np.percentile(depth, 99)] = 0

    intrinsics_inv = np.linalg.inv(intrinsics)

    if return_semantics:
      seg = sio.loadmat(full_path + '/seg.mat')
      semantics = seg['seglabel']

      if 'kv2' in full_path or 'xtion' in full_path:
        class_names = [str(k[0]) for k in seg['names'][0]]
      else:
        class_names = [str(k[0][0]) for k in seg['names']]

      semantics = best_centercrop_image(semantics, self.height, self.width)

      return full_path, meta_3d, depth, image, intrinsics, fov_x_deg, valid_depth_mask, intrinsics_inv, semantics, class_names
    else:
      return full_path, meta_3d, depth, image, intrinsics, fov_x_deg, valid_depth_mask, intrinsics_inv

  def __getitem__(self, item):
    try:
      full_path, meta_3d, depth, image, intrinsics, fov_x_deg, valid_depth_mask, intrinsics_inv, semantics, class_names = self.__get_basic_item__(item, return_semantics=True)

      if self.knn_normals:
        knn_normals = np.array(np.load(full_path + '/computed_normals_radius_0.2_max_nn_300/normals.npz')['computed_normals'], dtype='float32')
        knn_normals = best_centercrop_image(knn_normals, self.height, self.width)

        # drop the last row column for easier compatibility with normals computed from closest pixels
        knn_normals = knn_normals

      extrinsics = meta_3d['anno_extrinsics']

      pcl = pixel2cam(torch.FloatTensor(depth[None, :, :]), torch.FloatTensor(intrinsics[None, :, :]))[0]
      world_pcl = np.matmul(extrinsics[:3, :3], pcl.reshape(3, -1)).reshape(pcl.shape)

      # Heuristics to get the height, as not all samples are annotated
      camera_height_from_object_bboxes = -1
      camera_height_from_room_layout = -1
      camera_height_from_semantics = -1

      # First try to get it from annotations, but there are several samples that are not annotated,
      # and sometimes the room annotations are not good enough
      try:
        annotation3DLayout = load_json(full_path + '/annotation3Dlayout/index.json')
        room = [k for k in annotation3DLayout['objects'] if not k is None and len(k) > 0 and 'roo' in k['name']][0]
        room_dims = room['polygon'][0]
        if 'NYU' in full_path or 'xtion/xtion_align_data' in full_path:
          # TODO: no idea why, but matches demo.m in SUNCGtoolbox
          camera_height_from_room_layout = room_dims['Ymin']
        else:
          camera_height_from_room_layout = room_dims['Ymax']
      except:
        pass

      # Then try to get it from semantics (the depth of the floor pixels):
      floor_indices = [k + 1 for k in range(len(class_names)) if class_names[k] == 'floor']
      floor_mask = np.zeros(semantics.shape)
      for index in floor_indices:
        floor_mask = floor_mask + semantics == index
      floor_mask = floor_mask > 0
      floor_mask = floor_mask * valid_depth_mask
      # be safe with borders with other classes
      kernel = np.ones((5, 5), np.uint8)
      floor_mask = cv2.erode(np.array(floor_mask, np.uint8), kernel, iterations=1)
      if floor_mask.sum() > 50:
        camera_height_from_semantics = np.abs(world_pcl[1, np.array(floor_mask, dtype='bool')].mean())

      # Also try to get it from object bboxes
      # Assuming some object is laying on the floor
      corners_pcl = []
      try:
        annotation3Dobjects = meta_3d['bboxes']
        if len(annotation3Dobjects) > 0:
          Y_max_obj = float('-inf')
          Y_min_obj = float('inf')
          for obj in annotation3Dobjects:
            bbox_corners = self.bbox_to_usorted_corners(obj)
            corners_pcl.extend(bbox_corners.tolist())
            Y_max_obj = max(Y_max_obj, np.max(bbox_corners[:, 1]))
            Y_min_obj = min(Y_min_obj, np.min(bbox_corners[:, 1]))
          camera_height_from_object_bboxes = np.abs(Y_max_obj)
      except:
        pass

      # Then we use the layout floor when available, if not the lowest bbox and if not the distance to the pixels labeled as floor.
      camera_height = camera_height_from_object_bboxes if camera_height_from_object_bboxes != -1 else \
        camera_height_from_room_layout if camera_height_from_room_layout != -1 else \
          camera_height_from_semantics
      if camera_height == -1:
        raise Exception('Detected bad layout and other heuristics did not work!')

      world_pcl[1, :] -= camera_height

      yaw_rad, pitch_rad, roll_rad = mat2euler(extrinsics, axes='syxz')

      pitch_deg = np.rad2deg(pitch_rad)
      roll_deg = np.rad2deg(roll_rad)

      params = np.array((fov_x_deg, camera_height, pitch_deg, roll_deg))

      # world_normals_from_pcl = np.zeros_like(world_pcl)
      # world_normals_from_pcl[:,mask] = compute_normals_from_pcl(tonumpy(world_pcl)[:,mask].reshape(3,-1), np.array((0, -height, 0)))
      if self.knn_normals:
        world_normals = knn_normals[:, :-1, :-1]
        normals_mask = valid_depth_mask[:-1, :-1]
      else:
        world_normals, normals_mask = compute_normals_from_closest_image_coords(world_pcl[None, :, :, :], mask=valid_depth_mask[None, None, :, :])
        world_normals = world_normals[0]
        normals_mask = normals_mask[0]

      to_return = dict(image=np.array(image / 255.0, dtype='float32'),
                       depth=np.array(depth, dtype='float32'),
                       mask=np.array(valid_depth_mask, dtype='float32'),
                       intrinsics=intrinsics,
                       intrinsics_inv=intrinsics_inv,
                       world_normals=np.array(world_normals),
                       normals_mask=np.array(normals_mask[0] * 1.0, dtype='float32'),
                       pcl=np.array(pcl, dtype='float32'),
                       world_pcl=np.array(world_pcl, dtype='float32'),
                       params=np.array(params, dtype='float32'),
                       path=full_path,
                       proper_scale=np.array((1), dtype=bool))

      if self.return_semantics:
        semantics_dict = dict(semantics=torch.IntTensor(np.array(semantics, dtype='uint8')),
                              class_names=class_names)
        to_return.update(semantics_dict)

      self.last_return = to_return
      return self.last_return
    except Exception as e:
      print("Exception on dataset!")
      print(e)
      if self.try_random_on_exception:
        elem = np.random.randint(0, self.__len__())
        return self.__getitem__(elem)
      if self.catch_exception:
        return self.last_return
      else:
        raise e


def get_image_ressolution_stats(dataset):
  ressolutions = dict()
  from tqdm import tqdm
  for k in tqdm(dataset.samples):
    try:
      image_dir = dataset.base_path + '/SUNRGBD/' + k + 'image'
      image_file = image_dir + '/' + os.listdir(image_dir)[0]
      ressolution = get_image_ressolution_fast_jpg(image_file)
      try:
        ressolutions[ressolution] += 1
      except:
        ressolutions[ressolution] = 1
    except Exception as e:
      print(e)
      continue
  print(ressolutions)


def get_y_max_y_min_stats(dataset):
  Y_mins_by_type = dict()
  Y_maxs_by_type = dict()
  from tqdm import tqdm
  for k in tqdm(dataset.samples):
    try:
      annotation3DLayout = load_json(dataset.base_path + '/SUNRGBD/' + k + 'annotation3Dlayout/index.json')
      room = [k for k in annotation3DLayout['objects'] if not k is None and len(k) > 0 and 'roo' in k['name']][0]
      room_dims = room['polygon'][0]
      type = '/'.join(k.split('/')[:2])
      try:
        Y_mins_by_type[type].append(room_dims['Ymin'])
        Y_maxs_by_type[type].append(room_dims['Ymax'])
      except:
        Y_mins_by_type[type] = [room_dims['Ymin']]
        Y_maxs_by_type[type] = [room_dims['Ymax']]
    except Exception as e:
      # print(e)
      continue
  for k in Y_mins_by_type.keys():
    Y_mins_by_type[k] = np.array(Y_mins_by_type[k])
    Y_maxs_by_type[k] = np.array(Y_maxs_by_type[k])

  for k in Y_mins_by_type.keys():
    visdom_histogram(Y_maxs_by_type[k], title=k + '_Y_max')
    visdom_histogram(Y_mins_by_type[k], title=k + '_Y_min')
  print('=====================')


def compute_normals_knn():
  dataset = SUNRGBDWorld(1, 1, 'all', return_semantics=True)

  from tqdm import tqdm
  for image in tqdm(dataset.samples):
    try:
      path = dataset.base_path + '/SUNRGBD/' + image
      depth_dir = path + '/depth_bfx'

      # replicate  SUNRGBDToolBox/read3dPoints.m
      depthVis = png_16_bits_imread(depth_dir + '/' + os.listdir(depth_dir)[0])
      depth = np.bitwise_or(np.right_shift(depthVis, 3), np.left_shift(depthVis, 16 - 3))
      depth = depth / 1000.0

      intrinsics = np.loadtxt(path + '/intrinsics.txt', dtype='float32').reshape(3, 3)

      pcl = pixel2cam(torch.FloatTensor(depth[None, :, :]), torch.FloatTensor(intrinsics[None, :, :]))[0]

      viewpoint = np.array((0, 0, 0))
      radius = 0.2
      max_nn = 300
      computed_normals = compute_normals_from_pcl(tonumpy(pcl).reshape(3, -1), viewpoint, radius=radius, max_nn=max_nn).reshape((3, *pcl.shape[-2:]))
      normals_dir = path + '/computed_normals_radius_{:.5}_max_nn_{}'.format(radius, max_nn)
      if not os.path.exists(normals_dir):
        os.makedirs(normals_dir)
      np.savez_compressed(normals_dir + '/normals', computed_normals=np.array(computed_normals, dtype='float32'), radius=radius, max_nn=max_nn)

    except Exception as e:
      print(e)
      print("Failed file: {}".format(image))


def sunrgbd_vs_nyu():
  random.seed(34)
  dataset = SUNRGBDWorld(288, 384, 'all', knn_normals=True, return_semantics=True, nyu_only=True)
  then = time.time()

  for i in range(1, len(dataset), 1):
    elem = dataset.__getitem__(i)

    original_nyu_color = cv2_imread('/data/vision/torralba/datasets/NYUv2_depth/annotated/images/' + str(i).zfill(4) + '.png')
    original_nyu_depth = np.load('/data/vision/torralba/datasets/NYUv2_depth/annotated/depth/' + str(i).zfill(4) + '.npy')

    sunrgbd_image = elem['image']

    print('Average time per batch: {}'.format((time.time() - then)))
    imshow(elem['mask'], title='mask')
    sunrgbd_depth = elem['depth']
    imshow(sunrgbd_depth, title='depth')
    imshow(elem['normals_mask'], title='normals_mask')
    imshow(elem['world_normals'], title='normals')
    imshow(sunrgbd_image, title='image')
    imshow(elem['semantics'], title='semantics')
    show_pointcloud(elem['world_pcl'], elem['image'], title='pcl_sunrgbd')

    fov_x_deg = intrinsics_to_fov_x_deg(elem['intrinsics'])
    fov_x_deg += 20
    height, width = original_nyu_depth.shape
    intrinsics = fov_x_to_intrinsic_deg(fov_x_deg, width, height, return_inverse=False)

    pcl = tonumpy(pixel2cam(totorch(original_nyu_depth[None, :, :]), totorch(intrinsics[None, :, :]))[0])
    imshow(original_nyu_depth, title='depth_original')
    imshow(original_nyu_color, title='color_original')
    pcl[1, :, :] = pcl[1, :, :] - elem['params'][1]
    show_pointcloud(pcl, original_nyu_color, title='pcl_original')

    # they seem to match. check sunrgbd intrinsics!
    bound = 4.1
    imshow(original_nyu_color * (1 - np.logical_and(original_nyu_depth > bound, original_nyu_depth < bound + 0.2)), title='original_masked')
    imshow(sunrgbd_image * (1 - np.logical_and(sunrgbd_depth > bound, sunrgbd_depth < bound + 0.2)), title='sunrgbd_masked')

    nyu_depth = best_centercrop_image(original_nyu_depth, sunrgbd_depth.shape[0], sunrgbd_depth.shape[1])

    # dump_pointcloud(nyu_depth, )
    then = time.time()
    continue

  camera_stats_per_file = dataset.get_camera_stats_per_file(plot=True)
  bad_files = []
  for k, v in camera_stats_per_file.items():
    if v[1] > float(3):
      if k in dataset.samples:
        bad_files.append(dataset.samples.index(k))

  for k in bad_files:
    dataset.__getitem__(k)


def test_crop():
  dataset = SUNRGBDWorld(288, 384, 'val', knn_normals=True, return_semantics=True, nyu_only=True)
  then = time.time()

  dataset.height = 288
  dataset.width = 384
  for i in range(len(dataset)):
    elem_cropped = dataset.__getitem__(i)
    imshow(elem_cropped['image'])
    print(elem_cropped['path'])
    continue
    show_pointcloud(elem_cropped['world_pcl'], elem_cropped['image'], title='pcl_cropped')
    dataset.height = -1
    dataset.width = -1

    elem = dataset.__getitem__(i)
    show_pointcloud(elem['world_pcl'], elem['image'], title='pcl_not_cropped')
    dataset.height = 288
    dataset.width = 384

  return


if __name__ == '__main__':
  test_crop()
