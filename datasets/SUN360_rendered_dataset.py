from datasets.scannet_world_dataset import POSE_STATS_PATH
from tqdm import tqdm

from paths import *
from utils.geom_utils import *

SUN360_IMAGES_PATH = SUN360_PATH + '/images/pano9104x4552'
SUN360_INDOOR_URLS_FOLDER = SUN360_PATH + '/www/Images/SUN360_urls_9104x4552/indoor'

RENDERED_RESSOLUTION = (288, 384)
COMPUTED_RESULTS_PATH = CACHE_PATH + '/sun360_projective_samples_{}_{}'.format(*RENDERED_RESSOLUTION)

RENDERS_PER_FILE = 6
os.makedirs(COMPUTED_RESULTS_PATH, exist_ok=True)


class SUN360RenderedDataset():
  def __init__(self, height=RENDERED_RESSOLUTION[0], width=RENDERED_RESSOLUTION[1]):
    self.height = height
    self.width = width
    assert RENDERED_RESSOLUTION[0] > height, 'Rendering ressolution should be higher than final one, so that the downscale reduces NN artifacts'
    assert RENDERED_RESSOLUTION[0] / height == RENDERED_RESSOLUTION[1] / width, 'Rendered aspect ratio is different than desired.'
    self.sun360_dataset = SUN360Dataset()
    all_pano_images = self.sun360_dataset.get_all_pano_images()
    self.all_renders = self.get_all_renders()
    assert len(self.all_renders) == RENDERS_PER_FILE * len(all_pano_images), 'Not all renders completed or not found. Run python datasets/SUN360_rendered_dataset.py to generate perspective renderings from SUN360!'
    return

  def get_all_renders(self):
    all_rendered_examples_file = COMPUTED_RESULTS_PATH + '/all_rendered_examples.txt'
    if os.path.exists(all_rendered_examples_file):
      all_rendered_examples = read_text_file_lines(all_rendered_examples_file)
    else:
      all_rendered_examples = find_all_files_recursively(COMPUTED_RESULTS_PATH, prepend_path=True, extension='.jpg')
      len(self.all_renders) == RENDERS_PER_FILE * len(all_rendered_examples)
      write_text_file_lines(all_rendered_examples, all_rendered_examples_file)
    return all_rendered_examples

  def __len__(self):
    return len(self.all_renders)

  def __getitem__(self, item):
    actual_render_file = self.all_renders[item]
    actual_render_params = actual_render_file.replace('.jpg', '_params.txt')
    img = cv2_imread(actual_render_file)
    img = best_centercrop_image(img, self.height, self.width, interpolation=cv2.INTER_LINEAR)
    params = read_text_file_lines(actual_render_params)[1].split(',')
    params = [float(k) for k in params]
    fov_x_deg = params[0]
    camera_height = -1
    pitch_deg = params[1]
    roll_deg = params[2]
    params = np.array((fov_x_deg, camera_height, pitch_deg, roll_deg))
    path = self.all_renders[item]
    to_return = {'image': np.array(img / 255.0, dtype='float32'),
                 'params': np.array(params, dtype='float32'),
                 'path': path}

    return to_return


class SUN360Dataset():
  def __init__(self):
    self.height = RENDERED_RESSOLUTION[0]
    self.width = RENDERED_RESSOLUTION[1]
    return

  def get_all_renders(self):
    all_renders_file = COMPUTED_RESULTS_PATH + '/all_renders.txt'
    if not os.path.exists(all_renders_file):
      all_renders = find_all_files_recursively(COMPUTED_RESULTS_PATH, True, extension='.jpg')
      write_text_file_lines(all_renders, all_renders_file)
    else:
      all_renders = read_text_file_lines(all_renders_file)
    return all_renders

  def get_all_pano_images(self):
    all_pano_examples_file = CACHE_PATH + '/all_pano_examples.txt'
    if os.path.exists(all_pano_examples_file):
      all_valid_examples = read_text_file_lines(all_pano_examples_file)
      return [example.split(',') for example in all_valid_examples]
    else:
      all_scenes_files = listdir(SUN360_INDOOR_URLS_FOLDER, True, extension='.txt')
      all_vaexamples = []
      for scene_file in all_scenes_files:
        type = scene_file.split('/')[-1].replace('.txt', '')
        all_scene_files = read_text_file_lines(scene_file)
        all_scene_files = [SUN360_IMAGES_PATH + '/' + k.split('/')[-1] for k in all_scene_files]
        all_vaexamples.extend(zip(all_scene_files, [type] * len(all_scene_files)))
      all_valid_examples = []
      print("Parsing all available pano images. This will only be run the first time!")
      for example in tqdm(all_vaexamples):
        if os.path.exists(example[0]):
          all_valid_examples.append(example)
      write_text_file_lines([','.join(example) for example in all_valid_examples], all_pano_examples_file)
    return all_valid_examples

  def _render_single(self, stats, pano_image):
    (fov_x_deg, pitch_deg, roll_deg) = stats
    yaw_deg = np.random.uniform(0, 360)

    pitch_rad = np.deg2rad(pitch_deg)
    roll_rad = np.deg2rad(roll_deg)
    yaw_rad = np.deg2rad(yaw_deg)

    fov_y_deg = int(fov_x_deg) * self.height / self.width
    fov_y_rad = np.deg2rad(fov_y_deg)

    pano_image_shape = (pano_image.shape[1], pano_image.shape[2])

    DI, sample_coords, sample_height, sample_width = get_pano_to_image_coords(yaw_rad=yaw_rad, pitch_rad=pitch_rad, fov_v=fov_y_rad, height=self.height,
                                                                              width=self.width, pano_shape=pano_image_shape, roll_rad=roll_rad)
    rendered_img = np.zeros((pano_image.shape[0], sample_height, sample_width), pano_image.dtype)
    rendered_img[:, DI[:, 1], DI[:, 0]] = pano_image[:, sample_coords[0], sample_coords[1]]

    return rendered_img

  def render_multiple(self, image_file, stats):
    assert type(stats) is list
    image = cv2_imread(image_file)
    samples = []
    for actual_stats in stats:
      rendered_example = self._render_single(actual_stats, image)
      samples.append((rendered_example, actual_stats))
    return samples


def generate_images():
  all_image_stats = list(load_from_pickle(POSE_STATS_PATH).values())
  # (fov_x_deg, camera_height, pitch_deg, roll_deg)
  sun360 = SUN360Dataset()

  all_images = sun360.get_all_pano_images()
  from tqdm import tqdm
  for image_and_type in tqdm(all_images):
    # sample RENDERS_PER_FILE camera parameters in dataset
    image_id = image_and_type[0].split('/')[-1].replace('.jpg', '')
    folder = COMPUTED_RESULTS_PATH + '/' + image_and_type[1] + '/' + image_id
    os.makedirs(folder, exist_ok=True)
    try:
      assert os.path.exists(image_and_type[0]), 'Image not found!'
      stats = [random.choice(all_image_stats) for _ in range(RENDERS_PER_FILE)]
      stats = [(stats[0], stats[2], stats[3]) for stats in stats]
      samples = sun360.render_multiple(image_and_type[0], stats)
      for i in range(RENDERS_PER_FILE):
        rendered_sample = samples[i]
        image_file = folder + '/' + str(i).zfill(2) + '.jpg'
        camera_params_file = folder + '/' + str(i).zfill(2) + '_params.txt'
        cv2_imwrite(rendered_sample[0], image_file)
        write_text_file_lines(['fov_x_deg, pitch_deg, roll_deg',
                               '{},{},{}'.format(rendered_sample[1][0], rendered_sample[1][1], rendered_sample[1][2])], camera_params_file)
    except Exception as e:
      write_text_file_lines([str(e)], folder + '_error')


if __name__ == '__main__':
  generate_images()
