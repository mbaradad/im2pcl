import copy
import time

from datasets.scannet_world_dataset import ScannetWorld
from datasets.sunrgbd_world_dataset import SUNRGBDWorld
from utils.geom_utils import *

from paths import *

BASE_CACHE_PATH = CACHE_PATH + '/cached_data'


# just produce samples of FOV
class NaiveFocalSampler():
  def __init__(self, min_focal=-1, max_focal=-1, focal_set=None):
    assert (min_focal == -1 and max_focal == -1 and type(focal_set) is list) or focal_set is None

    self.min_focal = min_focal
    self.max_focal = max_focal
    self.focal_set = focal_set

  def get_fov_deg(self):
    if self.min_focal != -1:
      return np.random.uninform(self.min_focal, self.max_focal)
    else:
      return np.random.choice(self.focal_set)


class MixedDataset(torch.utils.data.dataset.Dataset):
  def __init__(self, height, width, split,
               use_scannet=False, use_sunrgbd=False, use_nyu=False,
               use_knn_normals=True, focal_augmentation=False,
               min_fov_deg_augmentation=30, mixing_ratios=None, cache_results=False, fit_planercnn_normals=False):
    assert split in ['train', 'val', 'all']
    assert use_scannet + use_sunrgbd + use_nyu > 0

    if use_sunrgbd + use_nyu == 2:
      raise Exception("Both nyu and sunrgbd are activated, but nyu is a subset of sunrgbd!")

    common_dataset_args = {'height': height,
                           'width': width,
                           'split': split,
                           'knn_normals': use_knn_normals,
                           'catch_exception': False,
                           'try_random_on_exception': True}

    self.use_scannet = use_scannet
    self.use_sunrgbd = use_sunrgbd
    self.use_nyu = use_nyu

    self.split = split

    self.focal_augmentation = focal_augmentation
    self.min_fov_deg_augmentation = min_fov_deg_augmentation

    self.cache_results = cache_results
    self.cache_path = BASE_CACHE_PATH + '_' + str(height) + '_' + str(width)
    if self.cache_results and not os.path.exists(self.cache_path):
      os.makedirs(self.cache_path, exist_ok=True)

    self.datasets = dict()

    if use_sunrgbd or use_nyu:
      nyu_only = use_nyu and not use_sunrgbd
      self.datasets['sunrgbd'] = SUNRGBDWorld(nyu_only=nyu_only, **common_dataset_args)
    if use_scannet:
      self.datasets['scannet'] = ScannetWorld(**common_dataset_args)

    self.max_elems = -1
    self.total_elems = 0
    for k, v in self.datasets.items():
      self.max_elems = max(self.max_elems, len(v))
      self.total_elems += len(v)

    self.last_return = None
    self.ranges = None
    self.ranges = self.get_ranges()
    self.mixing_ratios = mixing_ratios
    if not mixing_ratios is None:
      assert len(mixing_ratios) == len(self.datasets)
      assert sum(mixing_ratios) == 1.0

    self.all_datasets = list(self.datasets.keys())
    self.all_datasets.sort()

    while self.last_return is None:
      items = list(self.get_sampler(10000))
      for item in items:
        self.__getitem__(item)
        if not self.last_return is None:
          break

  def use_cache(self, item):
    dataset_name = self.all_datasets[item[0]]
    return dataset_name in ['sunrgbd', 'scannet']

  def get_sampler(self, max_elems=-1):
    class CustomSampler(torch.utils.data.Sampler):
      def __init__(self, datasets, all_datasets, max_elems, mixing_ratios):
        super().__init__(None)
        self.max_elems = max_elems
        self.datasets = datasets
        self.all_datasets = all_datasets
        self.mixing_ratios = mixing_ratios
        if not mixing_ratios is None:
          raise Exception("NOT implemented!")

      def __len__(self):
        return self.max_elems

      def __iter__(self):
        dataset_index = np.random.randint(0, len(self.all_datasets), size=self.__len__())
        dataset_samples = np.zeros(self.__len__(), dtype='int32')
        for i in range(len(self.all_datasets)):
          actual_dataset_samples = np.random.randint(0, len(self.datasets[self.all_datasets[i]].samples), size=self.__len__())
          dataset_samples += (dataset_index == i) * actual_dataset_samples
        both = np.concatenate((dataset_index[None, :], dataset_samples[None, :])).transpose()
        return iter(both)

    sampler = CustomSampler(self.datasets, self.all_datasets, max_elems if max_elems > -1 else self.max_elems, mixing_ratios=self.mixing_ratios)
    return sampler

  def get_ranges(self):
    if self.ranges is None:
      items = [k for k in list(self.datasets.items())]
      k, v = items[0]
      ranges = v.get_ranges()
      for k, v in items[1:]:
        actual_ranges = v.get_ranges()
        for k in actual_ranges.keys():
          ranges[k] = (min(ranges[k][0], actual_ranges[k][0]), max(ranges[k][1], actual_ranges[k][1]))
      if self.focal_augmentation:
        ranges['FOV_deg'] = (self.min_fov_deg_augmentation, ranges['FOV_deg'][1])
      self.ranges = ranges
    return self.ranges

  def set_focal_range(self, min_fov_deg, max_fov_deg):
    if self.ranges is None:
      self.get_ranges()
    self.ranges['FOV_deg'] = (min_fov_deg, max_fov_deg)
    self.last_return = None

  def __len__(self):
    return self.total_elems

  def augment_focal(self, original_items):
    if original_items['dataset'] == 'gibson':
      return original_items
    else:
      items = copy.deepcopy(original_items)
      original_fov_deg = items['params'][0]
      augmented_fov_deg = np.random.uniform(self.ranges['FOV_deg'][0], self.ranges['FOV_deg'][1])
      zoom = original_fov_deg / augmented_fov_deg

      in_h, in_w = items['image'].shape[1:]

      affine_zoom = np.array(((zoom, 0, (1 - zoom) / 2 * in_w),
                              (0, zoom, (1 - zoom) / 2 * in_h)))

      items['image'] = from_cv2(cv2.warpAffine(to_cv2(items['image']), affine_zoom, dsize=(in_w, in_h), flags=cv2.INTER_LINEAR, borderValue=0))

      items['world_normals'] = from_cv2(cv2.warpAffine(to_cv2(items['world_normals']), affine_zoom, dsize=(in_w - 1, in_h - 1), flags=cv2.INTER_NEAREST, borderValue=0))
      items['pcl'] = from_cv2(cv2.warpAffine(to_cv2(items['pcl']), affine_zoom, dsize=(in_w, in_h), flags=cv2.INTER_NEAREST, borderValue=0))
      items['world_pcl'] = from_cv2(cv2.warpAffine(to_cv2(items['world_pcl']), affine_zoom, dsize=(in_w, in_h), flags=cv2.INTER_NEAREST, borderValue=0))

      items['depth'] = cv2.warpAffine(items['depth'], affine_zoom, dsize=(in_w, in_h), flags=cv2.INTER_NEAREST, borderValue=0)
      items['mask'] = cv2.warpAffine(items['mask'], affine_zoom, dsize=(in_w, in_h), flags=cv2.INTER_NEAREST, borderValue=0)
      items['normals_mask'] = cv2.warpAffine(items['normals_mask'], affine_zoom, dsize=(in_w - 1, in_h - 1), flags=cv2.INTER_NEAREST, borderValue=0)

      items['intrinsics'][0, 0] = items['intrinsics'][0, 0] * zoom
      items['intrinsics'][1, 1] = items['intrinsics'][1, 1] * zoom

      items['intrinsics_inv'] = np.linalg.inv(items['intrinsics'])

      items['params'][0] = augmented_fov_deg

      if items['image'].shape[1] != in_h or items['image'].shape[2] != in_w:
        raise Exception('Focal augmentation failed! Zoom: {}'.format(zoom))
      return items

  def _get_full_cache_name_(self, dataset_name, index):
    cache_filename = self.datasets[dataset_name].get_cache_name(index)
    full_path = self.cache_path + '/' + dataset_name + '/' + cache_filename + '.pckl'
    dir = '/'.join(full_path.split('/')[:-1])
    os.makedirs(dir, exist_ok=True)
    return full_path

  def __getitem__(self, item):
    # probably this is not thread safe, but just to get a rough number
    try:
      if len(self.datasets) == 1 and type(item) is int:
        item = (0, item)
      elif type(item) is int:
        raise Exception("Mixed dataset cannot be indexed by int if there are multiple datasets active!")
      dataset_name = self.all_datasets[item[0]]
      dataset = self.datasets[dataset_name]
      if self.use_cache(item):
        cache_name = self._get_full_cache_name_(dataset_name, item[1])
        error_on_load = False
        if os.path.exists(cache_name):
          try:
            to_return = load_from_pickle(cache_name)
            if 'semantics' in to_return.keys():
              error_on_load = True
          except:
            error_on_load = True
        if not os.path.exists(cache_name) or error_on_load:
          to_return = dataset.__getitem__(item[1])
          try:
            dump_to_pickle(cache_name, to_return)
          except:
            pass
      else:
        to_return = dataset.__getitem__(item[1])
      if len(to_return['normals_mask'].shape) == 3:
        to_return['normals_mask'] = to_return['normals_mask'][0]
      to_return['dataset'] = dataset_name
      sample_in_fov_range = to_return['params'][0] >= self.ranges['FOV_deg'][0] and \
                            to_return['params'][0] <= self.ranges['FOV_deg'][1]
      if self.focal_augmentation or not sample_in_fov_range:
        to_return = self.augment_focal(to_return)
      # image data augmentation
      # to_return[]
      to_return['log_depth'] = self.create_log_depth(to_return)
      self.last_return = to_return
      return self.last_return
    except Exception as e:
      if self.last_return is None:
        print("Exception in data loader for dataset {}".format(dataset_name))
        print("Exception: " + str(e))
      else:
        return self.last_return

  def create_log_depth(self, return_items):
    depth = return_items['depth']
    depth_mask = np.array(return_items['mask'], dtype='bool')

    log_depth = np.zeros(depth.shape, dtype='float32')

    assert not (depth[depth_mask] <= 0).any()
    log_depth[depth_mask] = np.array(np.log(depth[depth_mask]))

    return log_depth


def create_params_dict(params, postfix=''):
  params_dict = dict()
  params_dict['focal_deg' + postfix] = float(params[0])
  params_dict['height' + postfix] = float(params[1])
  params_dict['pitch_deg' + postfix] = float(params[2])
  params_dict['roll_deg' + postfix] = float(params[3])

  return params_dict


if __name__ == '__main__':
  width = 256
  height = 192
  torch.manual_seed(time.time())
  dataset = MixedDataset(height=height, width=width, split='train',
                         use_scannet=True, use_sunrgbd=False, use_nyu=False,
                         min_fov_deg_augmentation=30, focal_augmentation=False,
                         cache_results=True, fit_planercnn_normals=True)  # [0.02, 0.98])

  then = time.time()
  last = time.time()
  ranges = dataset.get_ranges()

  loader = torch.utils.data.DataLoader(
    dataset, batch_size=100,
    num_workers=100, shuffle=False)
  from tqdm import tqdm

  for k, batch in enumerate(tqdm(loader)):
    continue
  width = 384
  height = 288
  dataset = MixedDataset(height=height, width=width, split='train',
                         use_scannet=False, use_gibson=False, use_sunrgbd=False, use_nyu=True,
                         min_fov_deg_augmentation=30, focal_augmentation=False)  # [0.02, 0.98])

  then = time.time()
  last = time.time()
  ranges = dataset.get_ranges()

  loader = torch.utils.data.DataLoader(
    dataset, batch_size=1,
    num_workers=1, shuffle=True)
  from tqdm import tqdm

  for k, batch in enumerate(tqdm(loader)):
    continue

    print('Average time per batch: {}'.format((time.time() - then) / (k + 1)))
    print('Time per batch: {}'.format((time.time() - last)))
    print(k * 5)
    last = time.time()
    for batch_i in range(batch['image'].shape[0]):
      imshow(batch['image'][batch_i], title='image')
      valid_mask = batch['mask'][batch_i]
      imshow(valid_mask, title='depth_mask')
      imshow(batch['world_normals'][batch_i], title='normals')
      imshow(batch['normals_mask'][batch_i], title='normals_mask')
      show_pointcloud(batch['world_pcl'][batch_i], batch['image'][batch_i], title='world_coords_c', valid_mask=batch['mask'][batch_i].flatten())
      show_pointcloud(batch['pcl'][batch_i], batch['image'][batch_i], title='coords', valid_mask=batch['mask'][batch_i].flatten())

      valid_depth = batch['depth'][batch_i][valid_mask.bool()]
      valid_log_depth = batch['log_depth'][batch_i][valid_mask.bool()]
      visdom_histogram(valid_depth, title='depth_hist')
      visdom_histogram(valid_log_depth, title='valid_log_depth')

      print(valid_depth.min())

      if batch['dataset'] != 'gibson':
        actual_point_cloud = pixel2cam(totorch(batch['depth'][batch_i][None, :, :]), totorch(batch['intrinsics_inv'][batch_i][None, :, :]))[0, :, :, :]
        diff = actual_point_cloud - batch['pcl'][batch_i]
        print('Max diff computed vs cropped {}'.format(diff.max()))
        show_pointcloud(tonumpy(actual_point_cloud), batch['image'][batch_i], title='computed_pointcloud_from_depth')
        print('FOV deg: {}'.format(batch['params'][batch_i][0]))
      a = 1
