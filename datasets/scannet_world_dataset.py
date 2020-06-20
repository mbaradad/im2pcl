import os

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6

from torch.utils.data import Dataset
from tqdm import tqdm
from paths import *

from utils.visdom_utils import *
from utils.geom_utils import *

from transforms3d.euler import mat2euler, euler2mat

TRAIN_PATH = '{}/scans'.format(SCANNET_PATH)
TEST_PATH = '{}/scans_test'.format(SCANNET_PATH)
SCANNET_CODE_PATH = '{}/code'.format(SCANNET_PATH)
POSE_STATS_PATH = os.path.dirname(__file__) + '/../assets/pose_stats.pckl'


def color_to_depth_path(color_path):
  return color_path.replace('/color/', '/depth/').replace('.jpg', '.png')


def color_to_pose_path(color_path):
  return color_path.replace('/color/', '/pose/').replace('.jpg', '.txt')


def color_to_extrinsics_and_intrinsics_path(colors_path):
  intrinsics_folder = colors_path.replace('/color', '/intrinsic')
  return [intrinsics_folder + '/' + k for k in ['extrinsic_color.txt', 'extrinsic_depth.txt',
                                                'intrinsic_color.txt', 'intrinsic_depth.txt']]


def color_to_labels_path(color_path):
  return color_path.replace('/color/', '/label-filt/').replace('.jpg', '.png')


def quantize_normals(non_quantized_normals):
  return np.array((1 + np.round(non_quantized_normals, decimals=2)) * 100, dtype='uint8')


def dequantize_normals(quantized_normals):
  return quantized_normals / 100.0 - 1


# Default normal computations
DEFAULT_NN_RADIUS = 1.0
DEFAULT_MAX_NN = 300


def color_to_knn_normals_path(color_path, radius=DEFAULT_NN_RADIUS, max_nn=DEFAULT_MAX_NN):
  base_dir = '/'.join(color_path.split('/')[:-2])
  filename = color_path.split('/')[-1].replace('.jpg', '')
  normals_dir = base_dir + '/computed_normals_radius_{:.5}_max_nn_{}'.format(radius, max_nn)
  normals_filename = '{}/{}.npz'.format(normals_dir, filename)

  return normals_filename


def get_scannet_id_to_class():
  import pandas as pd
  df = pd.read_csv(SCANNET_PATH + '/scannetv2-labels.combined.tsv', delimiter='\t', encoding='utf-8')
  scannet_label_to_class = dict()
  for row in df.iterrows():
    scannet_label_to_class[row[1].id] = row[1].category
  return scannet_label_to_class


class ScannetWorld(Dataset):
  def __init__(self, height, width, split, subsample=8, knn_normals=False, read_n_samples=-1,
               catch_exception=True, try_random_on_exception=False):
    self.height = height
    self.width = width
    self.knn_normals = knn_normals
    self.catch_exception = catch_exception
    self.try_random_on_exception = try_random_on_exception

    all_color_examples = self.get_all_examples(knn_normals, read_n_samples)
    if subsample > 0:
      all_color_examples = all_color_examples[::subsample]
    all_color_examples.sort()

    splits_path = SCANNET_CODE_PATH + '/Tasks/Benchmark'

    val_scenes = read_text_file_lines(splits_path + '/scannetv2_val.txt')
    test_scenes = read_text_file_lines(splits_path + '/scannetv2_test.txt')
    train_scenes = read_text_file_lines(splits_path + '/scannetv2_train.txt')
    self.split = split
    if split == 'val':
      self.scenes = val_scenes
    elif split == 'train':
      self.scenes = train_scenes
    elif split == 'test':
      self.scenes = test_scenes
    elif split == 'all':
      self.scenes = [*train_scenes, *val_scenes, *test_scenes]
    else:
      raise Exception('Split not implemented')
    self.last_return = None
    self.scenes = set(self.scenes)
    self.samples = [k for k in all_color_examples if k.split('/')[-3] in self.scenes]
    self.scannet_label_to_class = get_scannet_id_to_class()
    self.scannet_class_to_label = dict([(v, k) for k, v in self.scannet_label_to_class.items()])
    self.scannet_floor_id = self.scannet_class_to_label['floor']

  def get_cache_name(self, index):
    return self.split + '/' + '/'.join(self.samples[index].split('/')[-3:])

  @staticmethod
  def get_all_examples(knn_normals, read_n_samples=-1):
    all_examples_file = TRAIN_PATH + '/all_examples' + ('_knn_normals' if knn_normals else '')
    if os.path.exists(all_examples_file):
      return read_text_file_lines(all_examples_file, read_n_samples)
    else:
      #Creating file with all available samples
      print("Creating a text file with all available samples. This can take a while.")
      all_scenes = os.listdir(TRAIN_PATH)

      def process_single(scene):
        examples = []
        color_path = TRAIN_PATH + '/' + scene + '/color'
        if not os.path.exists(color_path):
          return []
        ext_intrinsics = color_to_extrinsics_and_intrinsics_path(color_path)
        if any([not os.path.exists(k) for k in ext_intrinsics]):
          return []
        color_paths = [color_path + '/' + k for k in os.listdir(color_path)]
        for color_path_i in color_paths:
          required_things = [color_to_depth_path(color_path_i),
                             color_to_pose_path(color_path_i),
                             color_to_labels_path(color_path_i)]
          if knn_normals:
            required_things.append(color_to_knn_normals_path(color_path_i))

          if not any([not os.path.exists(thing) for thing in required_things]):
            examples.append(color_path_i)
        return examples

      from p_tqdm import p_map
      all_examples = p_map(process_single, all_scenes, num_cpus=50)
      all_examples = list_of_lists_into_single_list(all_examples)
      all_examples_to_sort = [(k.split('/')[-3] + '_' + str(int(k.split('/')[-1].replace('.jpg', ''))).zfill(5), k) for k in all_examples]
      all_examples_to_sort.sort()
      all_examples_sorted = [k[1] for k in all_examples_to_sort]
      write_text_file_lines(all_examples_sorted, all_examples_file)
      return all_examples

  @staticmethod
  def delete_uncessesary_files(parallel=True):
    all_scenes = listdir(TRAIN_PATH, True)
    all_scenes.extend(listdir(TEST_PATH, True))

    def del_single_scene(scene):
      if not os.path.isdir(scene):
        return
      all_files = listdir(scene, True)
      files_to_rm = [f for f in all_files if f.endswith('.zip') or f.endswith('.ply') or f.endswith('.sens')]
      for f in files_to_rm:
        os.remove(f)

    if parallel:
      from p_tqdm import p_map
      p_map(del_single_scene, all_scenes)
    else:
      for scene in tqdm(all_scenes):
        del_single_scene(scene)

  @staticmethod
  def get_camera_stats_per_file():
    if os.path.exists(POSE_STATS_PATH):
      return load_from_pickle(POSE_STATS_PATH)
    load_from_pickle('/data/vision/torralba/globalstructure/datasets/scannet/pose_stats.pckl')
    print("Computeing pose stats from Scannet. This can take a while!")
    dataset = ScannetWorld(240, 320, split='all', subsample=100, knn_normals=True)
    params = []
    for sample in tqdm(dataset.samples):
      try:
        _, camera_height, pitch_deg, roll_deg, intrinsics = dataset.get_camera_params(sample)
        width = intrinsics[0, 2] * 2
        height = intrinsics[1, 2] * 2
        fov_x_rad = 2 * np.arctan(width / 2.0 / intrinsics[0, 0])
        fov_x_deg = np.rad2deg(fov_x_rad)

        params.append((sample, (fov_x_deg, camera_height, pitch_deg, roll_deg)))
      except:
        continue
    params = dict(params)
    dump_to_pickle(POSE_STATS_PATH, params)
    return params

  def get_all_poses(self):
    all_file_poses = self.get_camera_stats_per_file()
    sample_poses = []
    all_samples_as_set = set(self.samples)
    for k, v in all_file_poses.items():
      if k in all_samples_as_set:
        sample_poses.append(v)
    all_file_poses = np.array(sample_poses)
    all_poses = {'FOVs_deg': all_file_poses[:, 0],
                 'heights': all_file_poses[:, 1],
                 'pitches': all_file_poses[:, 2],
                 'rolls': all_file_poses[:, 3]}
    return all_poses

  def get_ranges(self, plot=False):
    pose_stats = self.get_all_poses()
    if plot:
      visdom_histogram(pose_stats['rolls'], title='rolls')
      visdom_histogram(pose_stats['pitches'], title='pitches')
      visdom_histogram(pose_stats['heights'], title='heights')

    FOV_deg_min, FOV_deg_max = pose_stats['FOVs_deg'].min(), pose_stats['FOVs_deg'].max()

    param_ranges = dict()
    pitch_min, pitch_max = np.quantile(pose_stats['pitches'], 0.01), np.quantile(pose_stats['pitches'], 0.99)
    roll_min, roll_max = np.quantile(pose_stats['rolls'], 0.01), np.quantile(pose_stats['rolls'], 0.99)
    height_min, height_max = np.quantile(pose_stats['heights'], 0.01), np.quantile(pose_stats['heights'], 0.99)

    param_ranges['FOV_deg'] = (FOV_deg_min, FOV_deg_max)
    param_ranges['pitch_deg'] = (pitch_min, pitch_max)
    param_ranges['roll_deg'] = (roll_min, roll_max)
    param_ranges['height'] = (height_min, height_max)

    return param_ranges

  def __len__(self):
    return len(self.samples)

  def get_camera_params(self, color_path):
    pose = np.loadtxt(color_to_pose_path(color_path))
    rotation = pose[:3, :3]
    translation = pose[:3, 3]

    theirs_deg_euler_x, theirs_deg_euler_y, theirs_deg_euler_z = np.rad2deg(np.array(mat2euler(rotation, axes='sxyz')))

    pitch_deg = theirs_deg_euler_x + 90
    roll_deg = theirs_deg_euler_y
    rotation = euler2mat(np.deg2rad(pitch_deg), 0, np.deg2rad(roll_deg))

    camera_height = translation[-1]
    if camera_height < 0:
      raise Exception("Camera height smaller than 0: {}".format(camera_height))

    ext_intrinsics_paths = color_to_extrinsics_and_intrinsics_path('/'.join(color_path.split('/')[:-1]))
    intrinsics_depth = np.loadtxt(ext_intrinsics_paths[-1])[:3, :3]
    intrinsics = np.array(intrinsics_depth)

    return rotation, camera_height, pitch_deg, roll_deg, intrinsics

  def get_floor_wall_masks(self, semantics, class_names, valid_depth_mask):
    floor_indices = [k + 1 for k in range(len(class_names)) if class_names[k] == 'floor']
    wall_indices = [k + 1 for k in range(len(class_names)) if class_names[k] == 'wall']
    floor_mask = np.zeros(semantics.shape)
    wall_mask = np.zeros(semantics.shape)
    for index in floor_indices:
      floor_mask = floor_mask + semantics == index
    for index in wall_indices:
      wall_mask = wall_mask + semantics == index

    floor_mask = (floor_mask > 0) * valid_depth_mask
    wall_mask = (wall_mask > 0) * valid_depth_mask
    # be safe with borders with other classes
    kernel = np.ones((5, 5), np.uint8)
    floor_mask = cv2.erode(np.array(floor_mask, np.uint8), kernel, iterations=1)
    wall_mask = cv2.erode(np.array(wall_mask, np.uint8), kernel, iterations=1)
    floor_mask = np.array(floor_mask, dtype='bool')
    wall_mask = np.array(wall_mask, dtype='bool')

    return floor_mask, wall_mask

  def __getitem__(self, item):
    try:
      color_path = self.samples[item]
      base_path = '/'.join(color_path.split('/')[:-2])
      scene_name = color_path.split('/')[-3]
      image = cv2_imread(color_path)
      depth = png_16_bits_imread(color_to_depth_path(color_path)) / 1000
      labels = png_16_bits_imread(color_to_labels_path(color_path))

      rotation_annotated, camera_height_annotated, _, roll_deg_annotated, intrinsics = self.get_camera_params(color_path)

      # resize to same size as depth.
      image = cv2_resize(image, depth.shape, interpolation=cv2.INTER_LINEAR)
      labels = cv2_resize(labels, depth.shape)
      if self.knn_normals:
        local_normals = dequantize_normals(np.load(color_to_knn_normals_path(color_path))['computed_normals'])
        local_normals = cv2_resize(local_normals, depth.shape)

      original_height, original_width = depth.shape[-2:]

      image = best_centercrop_image(image, self.height, self.width)
      depth = best_centercrop_image(depth, self.height, self.width)
      if self.knn_normals:
        local_normals = best_centercrop_image(local_normals, self.height, self.width)

      labels, rescaled_size = best_centercrop_image(labels, self.height, self.width, return_rescaled_size=True)
      final_height, final_width = image.shape[1:]

      intrinsics[0, 0] = intrinsics[0, 0] / original_height * rescaled_size[0]
      intrinsics[1, 1] = intrinsics[1, 1] / original_width * rescaled_size[1]

      intrinsics[0, 2] = intrinsics[0, 2] / original_width * final_width
      intrinsics[1, 2] = intrinsics[1, 2] / original_height * final_height

      fov_x_rad = 2 * np.arctan(final_width / 2.0 / intrinsics[0, 0])
      fov_y_rad = 2 * np.arctan(final_height / 2.0 / intrinsics[1, 1])
      fov_x_deg = np.rad2deg(fov_x_rad)
      fov_y_deg = np.rad2deg(fov_y_rad)

      intrinsics_inv = np.linalg.inv(intrinsics)
      pcl = tonumpy(pixel2cam(torch.FloatTensor(depth[None, :, :]), torch.FloatTensor(intrinsics[None, :, :]))[0])

      valid_depth_mask = depth != 0

      world_annotated_gt_extrinsics = np.matmul(rotation_annotated, pcl.reshape((3, -1))).reshape(pcl.shape)
      floor_mask = labels == self.scannet_floor_id
      if floor_mask.sum() > 100:
        floor_points = world_annotated_gt_extrinsics[:, np.array(floor_mask, dtype='bool')]

        # to further remove outliers from regions badly annotated
        min_floor_height = np.quantile(floor_points[1], 0.3)
        max_floor_height = np.quantile(floor_points[1], 0.95)
        floor_points_good_height = np.logical_and(floor_points[1] > min_floor_height, floor_points[1] < max_floor_height)
        floor_mask[floor_mask] = floor_points_good_height
        floor_points = world_annotated_gt_extrinsics[:, np.array(floor_mask, dtype='bool')]
        # from all this, select some randomly to be faster when fitting the plane
        final_floor_indices = np.random.randint(0, floor_points.shape[1], 100)
        floor_points = floor_points[:, final_floor_indices]
        P = fit_plane_np(floor_points, robust=False)
        if P[1] > 0:
          P = -1 * P
        P = P / np.sqrt(P[0] ** 2 + P[1] ** 2 + P[2] ** 2)

        R = rotation_matrix_two_vectors(P[:3], np.array((0, -1, 0)))

        rotation = np.matmul(R, rotation_annotated)
        world_pcl = np.matmul(rotation, pcl.reshape((3, -1))).reshape(pcl.shape)
        # just the distance to the plane
        # http://mathworld.wolfram.com/Point-PlaneDistance.html
        camera_height = P[3]
      else:
        rotation = rotation_annotated
        camera_height = camera_height_annotated
        world_pcl = world_annotated_gt_extrinsics

      world_pcl = world_pcl - np.array((0, camera_height, 0))[:, None, None]
      if self.knn_normals:
        world_normals_from_knn = np.matmul(rotation, local_normals.reshape((3, -1))).reshape(local_normals.shape)

        world_normals = world_normals_from_knn[None, :, :-1, :-1]
        normals_mask = valid_depth_mask[None, :-1, :-1]

      else:
        world_normals, normals_mask = compute_normals_from_closest_image_coords(world_pcl[None, :, :, :], mask=valid_depth_mask[None, None, :, :])

      floor_normals_mask = np.array((floor_mask[:-1, :-1] * normals_mask)[0], dtype='bool')
      if floor_normals_mask.sum() > 50:
        floor_world_normals = world_normals[0, :, floor_normals_mask]
        mean_cos_similarity_floor = floor_world_normals[:, 1].mean()
        if mean_cos_similarity_floor < 0.95:
          raise Exception("Too many points on floor with cosine similarity with y axis < 0.9")

      pitch_rad, yaw_rad, roll_rad = mat2euler(rotation)
      pitch_deg = np.rad2deg(pitch_rad)
      roll_deg = np.rad2deg(roll_rad)
      params = np.array((fov_x_deg, camera_height, pitch_deg, roll_deg))

      to_return = dict(image=np.array(image / 255.0, dtype='float32'),
                       depth=np.array(depth, dtype='float32'),
                       mask=np.array(valid_depth_mask, dtype='float32'),
                       intrinsics=np.array(intrinsics, dtype='float32'),
                       intrinsics_inv=np.array(intrinsics_inv, dtype='float32'),
                       world_normals=np.array(world_normals[0], dtype='float32'),
                       normals_mask=np.array(normals_mask[0] * 1.0, dtype='float32'),
                       pcl=np.array(pcl, dtype='float32'),
                       world_pcl=np.array(world_pcl, dtype='float32'),
                       params=np.array(params, dtype='float32'),
                       path=color_path,
                       proper_scale=np.array((1), dtype=bool))

      self.last_return = to_return
      return self.last_return
    except Exception as e:
      if self.catch_exception:
        if self.last_return is None:
          raise e
        return self.last_return
      else:
        raise e


def compute_single_normals_knn(color_path, radius=DEFAULT_NN_RADIUS, max_nn=DEFAULT_MAX_NN):
  try:
    depth = png_16_bits_imread(color_to_depth_path(color_path)) / 1000

    ext_intrinsics_paths = color_to_extrinsics_and_intrinsics_path('/'.join(color_path.split('/')[:-1]))
    intrinsics_depth = np.loadtxt(ext_intrinsics_paths[-1])[:3, :3]

    intrinsics_inv = np.linalg.inv(intrinsics_depth)
    pcl = pixel2cam(torch.FloatTensor(depth[None, :, :]), torch.FloatTensor(intrinsics_inv[None, :, :]))[0]

    viewpoint = np.array((0, 0, 0))
    knn_computed_normals = compute_normals_from_pcl(tonumpy(pcl).reshape(3, -1), viewpoint, radius=radius,
                                                    max_nn=max_nn).reshape((3, *pcl.shape[-2:]))

    normals_file = color_to_knn_normals_path(color_path)
    normals_dir = '/'.join(normals_file.split('/')[:-1])

    os.makedirs(normals_dir, exist_ok=True)
    rounded_normals = quantize_normals(knn_computed_normals)
    np.savez_compressed(normals_file, computed_normals=rounded_normals, radius=radius, max_nn=max_nn)

  except Exception as e:
    print(e)
    print("Failed file: {}".format(color_path))


def compute_normals_knn(parallel=True):
  original_samples = set(ScannetWorld(240, 320, 'all').samples)
  dataset_knn_samples = set(ScannetWorld(240, 320, 'all', knn_normals=True).samples)

  missing_samples = list(original_samples.difference(dataset_knn_samples))

  if len(missing_samples) == 0:
    return

  from tqdm import tqdm

  from p_tqdm import p_map
  random.shuffle(missing_samples)
  if parallel:
    p_map(compute_single_normals_knn, missing_samples, num_cpus=5)
  else:
    for color_path in tqdm(missing_samples):
      compute_single_normals_knn(color_path)


if __name__ == '__main__':
  compute_normals_knn()
