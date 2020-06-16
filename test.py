from convolutional_head.eval.metrics import compute_depth_metrics
from convolutional_head.structure_regression_train import *
from datasets.SUN360_rendered_dataset import SUN360RenderedDataset


def create_parser():
  parser = argparse.ArgumentParser(description='Depth from single',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('-j', '--workers', default=0, type=int, metavar='N', help='number of dataloading workers')

  parser.add_argument('--imgs', type=str, nargs='+', default='')
  parser.add_argument('--img-folder', type=str, default='')
  parser.add_argument('--extension', type=str, default='jpg')

  parser.add_argument('--visdom-port', type=str, default='12890')

  parser.add_argument('--dataset-index', default='', type=str2intlist)
  parser.add_argument('--dataset', default="", type=str, choices=["scannet", "sunrgbd", "nyu", "sun360"])
  parser.add_argument('--split', default="val", type=str, choices=["train", "val"])

  parser.add_argument('--augment-focal', default="False", type=str2bool)
  parser.add_argument('--shuffle', default="False", type=str2bool)

  parser.add_argument('--compute-stats', default="False", type=str2bool)
  parser.add_argument('--plot', default="True", type=str2bool)
  parser.add_argument('-b', '--batch-size-per-gpu', default=10, type=int, metavar='N', help='mini-batch size')

  parser.add_argument('--test-path', default='checkpoints/full_fov',
                      help='path to continue training a previous model. reloads optimizer and nets.')
  parser.add_argument('--results-file', default="")

  parser.add_argument('--out-dir', default="scannet_test_results_dense")

  parser.add_argument('-d', '--debug', type=str2bool, default='False', help='debug mode')
  parser.add_argument('--gpus', type=str, default="1")

  return parser


class FakeMissingDepthGt():
  def __init__(self, warpped_dataset):
    self.loader = warpped_dataset

  def __len__(self):
    return len(self.loader)

  def __getitem__(self, item):
    to_return = self.loader.__getitem__(item)
    all_keys_to_return = ['image', 'depth', 'log_depth', 'mask', 'intrinsics', 'intrinsics_inv', 'world_normals', 'normals_mask',
                          'pcl', 'world_pcl', 'params', 'path', 'proper_scale']
    for k in set(all_keys_to_return).difference(set(to_return.keys())):
      to_return[k] = np.ones((1), dtype='float32')
    return to_return


def main():
  global args, best_error, n_iter_train, n_iter_val, gpus, global_vis
  parser = create_parser()
  args = parser.parse_args()

  global_vis = instantiate_visdom(args.visdom_port)

  with open(args.test_path + '/params.yml', 'r') as f:
    from yaml import Loader
    training_args = yaml.load(f, Loader=Loader)

  save_path = Path('test_' + training_args.save_path.split('/')[-1])

  gpus = select_gpus(args.gpus)
  args.batch_size = len(gpus) * args.batch_size_per_gpu
  args.__setattr__('hostname', socket.gethostname())
  env = save_path
  if 'PYCHARM_RUN' in os.environ.keys():
    # override parameters that makes debugging easier
    args.env = str('PYCHARM_RUN')  # + str(1)
    args.debug = True
    args.save_path = 'tmp' / env
  else:
    args.env = str(env)
    assert args.workers > 3
    args.save_path = 'tmp' / env

  # create model
  print("=> creating model")
  # input: image, output: depth, y, n_y and f, height, pitch
  assert (len(args.imgs) != 0) + (len(args.img_folder) != 0) <= 1, 'Either img_list or img_folder should be active!'
  if len(args.imgs) != 0:
    dataset_test = FakeMissingDepthGt(ImageFolderCenterCroppLoader([args.img], height=training_args.input_height, width=training_args.input_width))
  elif len(args.img_folder) != 0:
    dataset_test = FakeMissingDepthGt(ImageFolderCenterCroppLoader(args.img_folder, height=training_args.input_height, width=training_args.input_width, extension=args.extensions))
  elif args.dataset == 'sun360':
    dataset_test = FakeMissingDepthGt(SUN360RenderedDataset(height=training_args.input_height, width=training_args.input_width))
  else:
    common_train_val_args = {'split': args.split,
                             'height': training_args.input_height,
                             'width': training_args.input_width,
                             'use_sunrgbd': args.dataset == 'sunrgbd',
                             'use_scannet': args.dataset == 'scannet',
                             'use_nyu': args.dataset == 'nyu',
                             'focal_augmentation': training_args.augment_focal,
                             'use_knn_normals': training_args.use_knn_normals}
    dataset_test = MixedDataset(**common_train_val_args)
    dataset_test.set_focal_range(training_args.FOV_deg[0], training_args.FOV_deg[1])

  if len(args.dataset_index) > 0:
    sampler = args.dataset_index

    class SingleSampler(torch.utils.data.Sampler):
      def __init__(self):
        super().__init__(None)

      def __len__(self):
        return 1

      def __iter__(self):
        return iter(sampler)

    loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size,
                                         num_workers=args.workers, pin_memory=False, sampler=SingleSampler())
  else:
    loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=args.shuffle,
                                         num_workers=args.workers, pin_memory=False)

  depth_and_params = CoordConvHourglassModelWithRegressors(3, 1, 4)

  depth_and_params_path = args.test_path + '/depth_and_params_best.pth.tar'
  weights = torch.load(depth_and_params_path)
  depth_and_params.load_state_dict(weights['state_dict'], strict=True)

  logger = TermLogger(n_epochs=1, train_size=0, valid_size=0)
  logger.epoch_bar.start()

  depth_and_params = torch.nn.parallel.DataParallel(depth_and_params, device_ids=gpus).cuda()
  depth_and_params = depth_and_params.cuda()
  depth_and_params.eval()
  depth_metric_names = ['abs_rel', 'sq_rel', 'rmse', 'logmae', 'rmse_log', 'rmse_log10', 'log10', 'a1', 'a2', 'a3', 'rmse_mean_gt_depth']
  depth_metrics = AverageMeter(i=len(depth_metric_names), precision=5, names=depth_metric_names)

  camera_metric_names = ['fov_error', 'height_error', 'pitch_error', 'roll_error', 'angular_error_0', 'angular_error_1']
  camera_metrics = AverageMeter(i=len(camera_metric_names), precision=5, names=camera_metric_names)

  logger = TermLogger(n_epochs=1, train_size=0, valid_size=len(dataset_test))

  # we test what is the best scale that aligns predictions and gt.
  # It should be one if the sensors provide the same metric depth
  # To find it, we simply solve the the weighted least squares for a = sum_i{||(a*d - d^*)||}

  best_scale_numerator_sum = 0
  best_scale_denominator_sum = 0

  samples_camera_metrics_list = [['filename', *camera_metric_names]]
  samples_depth_metrics_list = [['filename', *depth_metric_names]]

  if len(args.results_file) > 0:
    results_file = open(args.results_file, "w")
  else:
    results_file = None
  with torch.no_grad():
    for i, batch in enumerate(tqdm(loader)):
      images, depth_gt, log_depth_gt, mask_gt, intrinsics_gt, intrinsics_inv_gt, normals_gt, normals_mask_gt, \
      coords_gt, world_coords_gt, params_gt, batch_items_proper_scale = batch['image'], batch['depth'], batch['log_depth'], batch['mask'], batch['intrinsics'], \
                                                                        batch['intrinsics_inv'], batch['world_normals'], batch['normals_mask'], \
                                                                        batch['pcl'], batch['world_pcl'], batch['params'], batch['proper_scale']

      paths = batch['path']

      # put necessary stuff to cuda
      images = images.cuda()

      predicted_world_coords, predicted_normals, predicted_depth, predicted_focal, predicted_height, predicted_pitch, predicted_roll, predicted_params = \
        predict_on_images(images, depth_and_params, training_args)

      batch_size = predicted_depth.shape[0]
      logger.valid_bar.update(i)

      gt_depth_available = not type(dataset_test) is FakeMissingDepthGt
      gt_camera_params_available = not type(dataset_test) is FakeMissingDepthGt or args.dataset == 'sun360'

      for b_i in range(batch_size):
        if args.compute_stats and gt_depth_available:
          computed_metrics = compute_depth_metrics(tonumpy(predicted_depth[b_i, 0]), tonumpy(depth_gt[b_i]), tonumpy(mask_gt[b_i].bool()))

          depth_metrics_to_store = []
          for metric_name in depth_metric_names:
            depth_metrics_to_store.append(computed_metrics[metric_name])
          depth_metrics.update(depth_metrics_to_store)
          depth_metrics_string = ''
          for metric_name in depth_metric_names:
            depth_metrics_string += '{}: {} ({}) '.format(*depth_metrics.get_val_and_avg_strings(append_names=True, names=[metric_name]))
          if not results_file is None:
            results_file.writelines([depth_metrics_string + '\n'])
          print(depth_metrics_string)

        if args.compute_stats and gt_camera_params_available:
          predicted_roll_rotations = zrotation_deg_torch(-1 * predicted_roll)
          predicted_pitch_rotations = xrotation_deg_torch(predicted_pitch)

          pitch_gt, roll_gt = params_gt[:, 2], params_gt[:, 3]
          gt_pitch_rotations = xrotation_deg_torch(pitch_gt).cuda()
          gt_roll_rotations = zrotation_deg_torch(-1 * roll_gt).cuda()

          non_rotated_upright = torch.FloatTensor([(0, -1, 0)] * predicted_pitch_rotations.shape[0]).cuda()[:, :, None]

          predicted_upright_0 = torch.bmm(predicted_pitch_rotations, torch.bmm(predicted_roll_rotations, non_rotated_upright))
          gt_upright_0 = torch.bmm(gt_pitch_rotations, torch.bmm(gt_roll_rotations, non_rotated_upright))

          predicted_upright_1 = torch.bmm(predicted_roll_rotations, torch.bmm(predicted_pitch_rotations, non_rotated_upright))
          gt_upright_1 = torch.bmm(gt_roll_rotations, torch.bmm(gt_pitch_rotations, non_rotated_upright))

          cosines_0 = (predicted_upright_0 * gt_upright_0).sum(-1).sum(-1)
          upright_error_angles_deg_0 = np.rad2deg(tonumpy(torch.acos(cosines_0)))

          cosines_1 = (predicted_upright_1 * gt_upright_1).sum(-1).sum(-1)
          upright_error_angles_deg_1 = np.rad2deg(tonumpy(torch.acos(cosines_1)))

          fov_errors = np.abs(tonumpy(predicted_focal.cpu() - params_gt[:, 0]))[b_i]
          height_errors = np.abs(tonumpy(predicted_height.cpu() - params_gt[:, 1]))[b_i]
          pitch_errors = np.abs(tonumpy(predicted_pitch.cpu() - params_gt[:, 2]))[b_i]
          roll_errors = np.abs(tonumpy(predicted_roll.cpu() - params_gt[:, 3]))[b_i]

          angular_error_0 = upright_error_angles_deg_0[b_i]
          angular_error_1 = upright_error_angles_deg_1[b_i]

          camera_metrics_string = ''
          camera_metrics_to_store = [fov_errors, height_errors, pitch_errors, roll_errors, angular_error_0, angular_error_1]
          camera_metrics.update(camera_metrics_to_store)

          for metric_name in camera_metric_names:
            median = np.median(camera_metrics.get_history(name=metric_name))
            camera_metrics_string += '{}: {} (avg: {}, med: {}) '.format(*camera_metrics.get_val_and_avg_strings(append_names=True, names=[metric_name]), median)
          if not results_file is None:
            results_file.writelines([camera_metrics_string + '\n'])
          print(camera_metrics_string)
        if args.plot:
          plot_func(predicted_depth[b_i], predicted_world_coords[b_i], predicted_normals[b_i], predicted_focal[b_i], predicted_height[b_i],
                    predicted_pitch[b_i], predicted_roll[b_i], depth_gt[b_i], world_coords_gt[b_i], normals_gt[b_i], mask_gt[b_i], normals_mask_gt[b_i],
                    params_gt[b_i], images[b_i], args, env_prefix='test_', title_prefix='', gt_available=gt_depth_available)
          print('params_gt: {}'.format(str(params_gt[b_i])))
          print('predicted_params: {}'.format(str(predicted_params[b_i])))

        if len(args.out_dir) > 0:
          output_dir = args.out_dir + '/' + args.test_path.split('/')[-1]
          os.makedirs(output_dir, exist_ok=True)
          things_to_save = dict(images=tonumpy(images[b_i]),
                                predicted_params=tonumpy(predicted_params[b_i]),
                                coords_predicted=tonumpy(predicted_world_coords[b_i]),
                                predicted_normals=tonumpy(predicted_normals[b_i]),
                                predicted_depth=tonumpy(predicted_depth[b_i]),
                                path=paths[b_i])

          if gt_depth_available:
            things_to_save['params_gt'] = tonumpy(params_gt[b_i])
            things_to_save['coords_gt'] = tonumpy(world_coords_gt[b_i])
            things_to_save['depth_gt'] = tonumpy(depth_gt[b_i])
            things_to_save['normals_gt'] = tonumpy(normals_gt[b_i])
            things_to_save['mask_gt'] = tonumpy(mask_gt[b_i])
            things_to_save['normals_mask_gt'] = tonumpy(normals_mask_gt[b_i])

          np.savez_compressed(output_dir + '/' + paths[b_i].replace('/', '_').replace('.jpg', ''), **things_to_save)

    # write_text_file_lines(samples_camera_metrics_list, args.save_path + '_{}_{}_results.csv'.format(args.dataset, args.split))
  if not results_file is None:
    results_file.close()


if __name__ == '__main__':
  parser = create_parser()
  args = parser.parse_args()
  if args.debug:
    with torch.autograd.detect_anomaly():
      main()
  else:
    main()
