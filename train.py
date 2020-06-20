import sys

sys.path.append('.')

from utils.visdom_utils import *

from utils.geom_utils import *
from utils.logging_utils import *
import time

import resource

# to avoid ancdata error for too many open files
# probably caused the loading of all the scenes from files by the PlanesDataset
resource.setrlimit(resource.RLIMIT_NOFILE, (131072, 131072))

import torch.optim
import torch.utils.data
from path import Path
from tensorboardX import SummaryWriter

import socket
import yaml

global args, global_vis

from losses import *
from datasets.mixed_dataset import MixedDataset, create_params_dict
from models.hourglass import HourglassModelWithRegressors as CoordConvHourglassModelWithRegressors


def create_parser():
  parser = argparse.ArgumentParser(description='Depth from single image', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('-j', '--workers', default=10, type=int, metavar='N', help='number of dataloading workers')
  parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
  parser.add_argument('--train-size', default=1000, type=int, metavar='N', help='manual training epoch size')
  parser.add_argument('--val-size', default=20, type=int, metavar='N', help='manual validation epoch size ')

  parser.add_argument('--input-height', default=192, type=int, metavar='N', help='input height to which the images will be resized')
  parser.add_argument('--input-width', default=256, type=int, metavar='N', help='input width to which the images will be resized')

  parser.add_argument('--visdom-port', default=10000, type=int, help='visdom port to use')
  parser.add_argument('--threaded-plotter', default="True", type=str2bool, help='whether to use a plotter in an independent thread (to not lock training when producing plotting)')
  parser.add_argument('--use-knn-normals', default="True", type=str2bool, help='whether to use knn-normals if set to True or normals computed with the closest two pixels in image space')

  parser.add_argument('--augment-focal', default="True", type=str2bool, help='whether to use focal augmentation (if True) or the dataset focal ranges')
  parser.add_argument('--min-fov', default=30, type=float, help='min focal to use if augment_focal. Maximum is set to to maximum of the dataset')

  parser.add_argument('--coords-weight', default=1, type=float)
  parser.add_argument('--normals-weight', default=1, type=float)

  parser.add_argument('-b', '--batch-size-per-gpu', default=10, type=int, metavar='N', help='mini-batch size')
  parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='initial learning rate')

  parser.add_argument('--plot-examples-freq', default=50, type=int, metavar='N', help='print frequency')

  parser.add_argument('--continue-training-path', dest='continue_training_path', default='', help='path to continue training a previous model. reloads optimizer and nets. if unset, trains from init.')
  parser.add_argument('--restart-optimization', default="False", type=str2bool, help='whether to restart the optimization or reload the optimizer from the checkpoint')
  parser.add_argument('--use-cache', default="True", type=str2bool)

  parser.add_argument('--seed', default=1337, type=int, help='seed for random functions, and network initialization')
  parser.add_argument('-d', '--debug', type=str2bool, default='False', help='debug mode')

  parser.add_argument('--gpus', type=str, default="0,1,2,3", help='gpus to use as indexed by the machine. accepts multigpu.')

  if 'PYCHARM_RUN' in os.environ.keys():
    parser.add_argument('--save_frequency', default=1, type=int)
  else:
    parser.add_argument('--save_frequency', default=1, type=int)
  return parser


def params_to_ranges(predicted_params, args):
  assert len(predicted_params.shape) == 2 and predicted_params.shape[-1] == 4

  predicted_params_sigmoid = torch.sigmoid(predicted_params[:, :3])
  focal = args.FOV_deg[0] + (args.FOV_deg[1] - args.FOV_deg[0]) * predicted_params_sigmoid[:, 0]
  pitch = args.pitch_deg[0] + (args.pitch_deg[1] - args.pitch_deg[0]) * predicted_params_sigmoid[:, 1]
  roll = args.roll_deg[0] + (args.roll_deg[1] - args.roll_deg[0]) * predicted_params_sigmoid[:, 2]
  height = predicted_params[:, 3]

  return torch.cat((focal[:, None], height[:, None], pitch[:, None], roll[:, None]), dim=1)


def plot_func(predicted_depth, predicted_world_coords, predicted_normals, predicted_focal, predicted_height, predicted_pitch, predicted_roll,
              depth_gt, world_coords_gt, normals_gt, mask_gt, normals_mask_gt, params_gt,
              images, args, env_prefix='', title_prefix='', gt_available=True, **kwargs):
  params = create_params_dict(
    [predicted_focal, predicted_height, predicted_pitch,
     predicted_roll], '_predicted')
  if gt_available:
    actual_params_gt = create_params_dict(params_gt, '_gt')
    params = {**params, **actual_params_gt}

  visdom_dict(params, title=title_prefix + 'cam_params', env=env_prefix + args.env)
  if gt_available:
    # same scale:
    depth_gt = tonumpy(depth_gt)
    max_depth, min_depth = depth_gt.max(), depth_gt.min()
    normalize_x = lambda x: (np.clip(tonumpy(x), min_depth, max_depth) - min_depth) / (max_depth - min_depth)
    imshow(normalize_x(depth_gt), title=title_prefix + 'depth_gt', env=env_prefix + args.env, normalize_image=False)
    imshow(normalize_x(predicted_depth), title=title_prefix + 'depth_predicted', env=env_prefix + args.env, normalize_image=False)
  else:
    imshow(predicted_depth, title=title_prefix + 'depth_predicted', env=env_prefix + args.env)
  image = images
  imshow(image, env=env_prefix + args.env, title=title_prefix + 'input')
  if gt_available:
    imshow(mask_gt, env=env_prefix + args.env, title=title_prefix + 'coords_mask')
    imshow(normals_mask_gt, env=env_prefix + args.env, title=title_prefix + 'normals_mask')

  predicted_y_normals = tonumpy(predicted_normals)
  imshow(np.clip((predicted_y_normals + 1) / 2 * 255.0, 0, 255), env=env_prefix + args.env, title=title_prefix + 'y_normals_predicted',
         normalize_image=False)

  if gt_available:
    gt_y_normals = tonumpy(normals_gt)
    imshow(np.clip((gt_y_normals + 1) / 2 * 255.0, 0, 255), env=env_prefix + args.env, title=title_prefix + 'y_normals_gt', normalize_image=False)

  pcl = tonumpy(predicted_world_coords).reshape(3, -1)
  pcl_colors = tonumpy(image).reshape(3, -1)
  pcl_colors = (pcl_colors - pcl_colors.min()) / (pcl_colors.max() - pcl_colors.min())
  pcl_colors = np.array(pcl_colors * 255, dtype='uint8')

  horizon_line_image_predicted_and_gt = draw_horizon_line(np.array(tonumpy(image) * 255, dtype='uint8'), rotation_mat=None,
                                                          pitch_angle=predicted_pitch,
                                                          roll_angle=predicted_roll, intrinsic=None, force_no_roll=False, color=(255, 0, 0))
  if (type(params_gt) is np.ndarray and params_gt.size > 1) or int(params_gt.nelement()) > 1:
    horizon_line_image_predicted_and_gt = draw_horizon_line(horizon_line_image_predicted_and_gt, rotation_mat=None,
                                                            pitch_angle=params_gt[2],
                                                            roll_angle=params_gt[3], intrinsic=None, force_no_roll=False, color=(0, 255, 0))
  imshow(horizon_line_image_predicted_and_gt, env=env_prefix + args.env, title=title_prefix + 'horizon_line')

  if not mask_gt is None and (type(mask_gt.gt) is np.ndarray and mask_gt.size > 1) or int(mask_gt.nelement()) > 1:
    valid_mask = tonumpy(mask_gt).reshape(3, -1)
  else:
    valid_mask = None
  show_pointcloud_several_pos(pcl, pcl_colors, title=title_prefix + 'predicted_world_pcl', env=env_prefix + args.env, valid_mask=valid_mask)
  if gt_available:
    gt_pcl = tonumpy(world_coords_gt).reshape(3, -1)
    show_pointcloud_several_pos(gt_pcl, pcl_colors, title=title_prefix + 'gt_world_pcl', env=env_prefix + args.env, valid_mask=valid_mask)


def show_pointcloud_several_pos(pcl, image, *args, top_crop_percentage=0, right_crop_percentage=0, **kwargs):
  # frontal
  original_title = kwargs['title']
  pcl = np.array(pcl)
  pcl[0] = -1 * pcl[0]
  pcl[1] = -1 * pcl[1]
  if 'axis_ranges' in kwargs.keys():
    final_axis_ranges = dict()
    final_axis_ranges['min_x'] = -1 * kwargs['axis_ranges']['max_x']
    final_axis_ranges['max_x'] = -1 * kwargs['axis_ranges']['min_x']

    final_axis_ranges['min_y'] = -1 * kwargs['axis_ranges']['max_y']
    final_axis_ranges['max_y'] = -1 * kwargs['axis_ranges']['min_y']

    final_axis_ranges['min_z'] = kwargs['axis_ranges']['min_z']
    final_axis_ranges['max_z'] = kwargs['axis_ranges']['max_z']

    kwargs['axis_ranges'] = final_axis_ranges

  original_pcl = np.array(pcl)
  original_image = np.array(image)
  # change x and y orientation for visualization:

  center = (0, 0, 0)
  up = (0, 1, 0)

  kwargs['title'] = original_title + '_frontal'

  eye = (0.1, -0.1, -10)
  show_pointcloud(pcl, image, *args, **kwargs, eye=eye, center=center, up=up, display_grid=(True, True, False))

  # top down
  eye = (0, 3, 0)
  up = (0, 0, 1)
  kwargs['title'] = original_title + '_top'
  if top_crop_percentage > 0 and len(original_pcl.shape) == 3:
    pcl = original_pcl[:, int(original_pcl.shape[1] * top_crop_percentage):, :]
    image = original_image[:, int(original_pcl.shape[1] * top_crop_percentage):, :]
  else:
    pcl = original_pcl
    image = original_image
  show_pointcloud(pcl, image, *args, **kwargs, eye=eye, center=center, up=up, display_grid=(True, False, True))

  kwargs['title'] = original_title + '_right'
  # right
  eye = (pcl[0].min(), 0, 0)
  center = (0, 0, 0)
  up = (0, 1, 0)
  if right_crop_percentage > 0 and len(original_pcl.shape) == 3:
    pcl = original_pcl[:, :, :int(original_pcl.shape[2] * (1 - right_crop_percentage))]
    image = original_image[:, :, :int(original_pcl.shape[2] * (1 - right_crop_percentage))]
  else:
    pcl = original_pcl
    image = original_image
  show_pointcloud(pcl, image, *args, **kwargs, eye=eye, center=center, up=up, display_grid=(False, True, True))


def predict_on_images(images, depth_and_params, args):
  predicted_log_depth, predicted_params_ranges = depth_and_params(images)
  # clamp to reasonable values to avoid nans
  predicted_log_depth = torch.clamp(predicted_log_depth, -6, 8)
  predicted_depth = torch.exp(predicted_log_depth)
  predicted_params = params_to_ranges(predicted_params_ranges, args)

  predicted_focal, predicted_height, predicted_pitch, predicted_roll = predicted_params[:, 0], predicted_params[:, 1], \
                                                                       predicted_params[:, 2], predicted_params[:, 3]

  batch_size = images.shape[0]
  predicted_intrinsics = fov_x_to_intrinsic_deg(predicted_focal, torch.FloatTensor([args.input_width] * batch_size).cuda(),
                                                torch.FloatTensor([args.input_height] * batch_size).cuda(),
                                                return_inverse=False)

  predicted_coords = pixel2cam(predicted_depth[:, 0, :, :], predicted_intrinsics)

  # at this point, predicted coords should be the same as coords_g if correct
  # rotate by roll and pitch and normalize using height
  roll_rotations = zrotation_deg_torch(-1 * predicted_roll)
  pitch_rotations = xrotation_deg_torch(predicted_pitch)

  predicted_world_rotated_coords = torch.bmm(roll_rotations, predicted_coords.reshape(batch_size, 3, -1))
  predicted_world_rotated_coords = torch.bmm(pitch_rotations, predicted_world_rotated_coords).reshape(batch_size, 3,
                                                                                                      args.input_height,
                                                                                                      args.input_width)

  zeros = torch.zeros_like(predicted_height)
  height_translation = torch.cat((zeros[:, None], predicted_height[:, None], zeros[:, None]), dim=-1)
  predicted_world_coords = predicted_world_rotated_coords - height_translation[:, :, None, None]

  predicted_normals = compute_normals_from_closest_image_coords(predicted_world_coords)

  return predicted_world_coords, predicted_normals, predicted_depth, predicted_focal, \
         predicted_height, predicted_pitch, predicted_roll, predicted_params


def main():
  global args, best_error, n_iter_train, n_iter_val, gpus, global_vis

  global_vis['vis'] = instantiate_visdom(args.visdom_port)

  torch.manual_seed(args.seed)
  random.seed(args.seed)

  best_error = float('inf')
  n_iter_train = 0
  n_iter_val = 0

  timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M:%S")
  gpus = select_gpus(args.gpus)
  args.batch_size = len(gpus) * args.batch_size_per_gpu

  save_path = Path('regression_train_{}_w_{}_h_{}_bs_{}_lr_{}_{}_cw_{}_nw_{}'.format(timestamp, args.input_width, args.input_height, args.batch_size,
                                                                                     args.lr, get_hostname(),
                                                                                     args.coords_weight, args.normals_weight))
  if args.augment_focal:
    save_path = save_path + '_af_min_fov_{}'.format(args.min_fov)

  if len(args.continue_training_path) > 0:
    save_path = save_path + '_continued'

  args.__setattr__('hostname', socket.gethostname())
  env = save_path
  if 'PYCHARM_RUN' in os.environ.keys():
    # override parameters that makes debugging easier
    args.pycharm_run = True
    args.env = PYCHARM_VISDOM  # + str(1)
    args.save_path = 'tmp' / env
  else:
    args.env = str(env)
    args.pycharm_run = False
    args.save_path = 'checkpoints/eccv' / env
    args.train_size > 100
    args.val_size > 50
    assert not args.debug
    assert args.threaded_plotter
    assert args.workers > 5

  print('=> will save everything to {}'.format(args.save_path))
  args.save_path.makedirs_p()

  if len(args.continue_training_path) > 0:
    with open(args.continue_training_path + '/params.yml', 'r') as f:
      from yaml import Loader
      previous_args = yaml.load(f, Loader=Loader)
    assert previous_args.input_width == args.input_width
    assert previous_args.input_height == args.input_height

  summary_writer = SummaryWriter(args.save_path)

  # create model
  print("=> creating model")
  depth_and_params = CoordConvHourglassModelWithRegressors(3, 1, 4)
  print("=> training model with {} parameters".format(count_trainable_parameters(depth_and_params, return_as_string=True)))

  common_train_val_args = {'height': args.input_height,
                           'width': args.input_width,
                           'use_scannet': True,
                           'use_knn_normals': args.use_knn_normals,
                           'focal_augmentation': args.augment_focal,
                           'cache_results': args.use_cache}

  dataset_train = MixedDataset(split='train', **common_train_val_args)
  dataset_val = MixedDataset(split='val', **common_train_val_args)

  param_ranges = dataset_train.get_ranges()
  for k, v in param_ranges.items():
    args.__setattr__(k, v)

  with open(args.save_path + '/params.yml', 'w') as outfile:
    yaml.dump(args, outfile, default_flow_style=False)

  # we use a big sampler to avoid recreating the iterator
  train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=args.batch_size,
    num_workers=args.workers, pin_memory=True, sampler=dataset_train.get_sampler(args.batch_size * args.train_size * args.epochs * 2))
  val_loader = torch.utils.data.DataLoader(
    dataset_val, batch_size=args.batch_size,
    num_workers=args.workers, pin_memory=True, sampler=dataset_val.get_sampler(args.batch_size * args.train_size * args.epochs * 2))

  if len(args.continue_training_path) > 0:
    print("=> using pre-trained weights ")
    depth_and_params_path = args.continue_training_path + '/depth_and_params_latest.pth.tar'
    weights = torch.load(depth_and_params_path)
    depth_and_params.load_state_dict(weights['state_dict'], strict=True)

  depth_and_params = torch.nn.parallel.DataParallel(depth_and_params, device_ids=gpus).cuda()
  optimizer_depth_and_params = torch.optim.Adam(depth_and_params.parameters(), args.lr)

  # decrease at 100k iterations:
  epoch_milestone = [int(1e5 / args.train_size), int(1.25e5 / args.train_size), int(1.5e5 / args.train_size)]
  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_depth_and_params, milestones=epoch_milestone, gamma=0.25)

  if len(args.continue_training_path) > 0 and not args.restart_optimization:
    # reload optimizer params and scheduler
    optimizer_state_dict = torch.load(args.continue_training_path + '/optimizer_depth_and_params_latest.pth.tar')
    optimizer_depth_and_params.load_state_dict(optimizer_state_dict)
    scheduler_state_dict = torch.load(args.continue_training_path + '/scheduler_latest.pth.tar')
    scheduler.load_state_dict(scheduler_state_dict)
    # one more than the registered, as we register just at the end
    init_epoch = scheduler.last_epoch + 1
    n_iter_train = init_epoch * args.train_size
    n_iter_val = init_epoch * args.val_size
  else:
    init_epoch = 0
  print('=> loading term')
  train_iters = args.train_size
  val_iters = args.val_size
  print("{} train iters; {} val iters".format(train_iters, val_iters))

  logger = TermLogger(n_epochs=args.epochs, train_size=train_iters, valid_size=val_iters)
  logger.epoch_bar.start()

  plotter = ThreadedVisdomPlotter(plot_func, use_threading=args.threaded_plotter)

  print("Creating iterators!")
  train_loader_iterator = train_loader.__iter__()
  val_loader_iterator = val_loader.__iter__()
  print("Ended creating iterators!")
  for epoch in range(init_epoch, args.epochs):
    print("Starting epoch: {}".format(epoch))
    print("lr: {}".format(optimizer_depth_and_params.state_dict()['param_groups'][0]['lr']))

    args.epoch = epoch
    logger.epoch_bar.update(epoch)

    # train for one epoch
    logger.reset_train_bar()

    train_losses, train_metrics = train_and_val(train_loader_iterator, depth_and_params, optimizer_depth_and_params,
                                                train_iters, logger, summary_writer, plotter, optimize=True, env_prefix='train_' if not args.pycharm_run else '')
    train_losses_metrics = [*train_losses.get_val_and_avg_strings(append_names=True), *train_metrics.get_val_and_avg_strings(append_names=True)]

    for i in range(0, len(train_losses_metrics), 3):
      name, val, avg = train_losses_metrics[i:i + 3]
      summary_writer.add_scalar('epoch_train_' + name, float(val), global_step=epoch)

    with torch.no_grad():
      val_losses, val_metrics = train_and_val(val_loader_iterator, depth_and_params, optimizer_depth_and_params,
                                              val_iters, logger, summary_writer, plotter, optimize=False, env_prefix='val_' if not args.pycharm_run else '')

    val_losses_metrics = [*val_losses.get_val_and_avg_strings(append_names=True), *val_metrics.get_val_and_avg_strings(append_names=True)]
    for i in range(0, len(val_losses_metrics), 3):
      name, val, avg = val_losses_metrics[i:i + 3]
      summary_writer.add_scalar('epoch_val_' + name, float(val), global_step=epoch)

    valid_metric = val_metrics.avg[0]

    summary_writer.add_scalar('lr', optimizer_depth_and_params.state_dict()['param_groups'][0]['lr'], global_step=epoch)
    scheduler.step(epoch=epoch)
    decisive_error = valid_metric

    is_best = (decisive_error <= best_error) and (not np.isnan(decisive_error))
    best_error = min(best_error, decisive_error)

    print('Best error: ' + str(best_error))

    nets_to_save = dict()
    nets_to_save['depth_and_params'] = {'state_dict': depth_and_params.module.state_dict(), 'epoch': epoch}
    nets_to_save['optimizer_depth_and_params'] = optimizer_depth_and_params.state_dict()
    nets_to_save['scheduler'] = scheduler.state_dict()

    save_checkpoint(args.save_path, nets_to_save, is_best=is_best)
  logger.epoch_bar.finish()


def train_and_val(data_loader_iterator, depth_and_params, optimizer_depth_and_params, total_iters, logger, summary_writer, plotter, optimize, env_prefix=''):
  global args, n_iter_train, n_iter_val
  times_hist = AverageMeter(precision=2, names=['data_time', 'batch_time'])
  losses_hist = AverageMeter(precision=2, names=['loss', 'coords_loss', 'normals_loss'])
  metrics_hist = AverageMeter(precision=2, names=['depth_rmse'])

  # switch to train mode
  if optimize:
    depth_and_params.train()
  else:
    depth_and_params.eval()
  t1 = time.time()
  i = 0
  while True:
    batch = data_loader_iterator.next()
    images, depth_gt, log_depth_gt, mask_gt, intrinsics_gt, intrinsics_inv_gt, normals_gt, normals_mask_gt, \
    coords_gt, world_coords_gt, params_gt, batch_items_proper_scale = batch['image'], batch['depth'], batch['log_depth'], batch['mask'], batch['intrinsics'], \
                                                                      batch['intrinsics_inv'], batch['world_normals'], batch['normals_mask'], \
                                                                      batch['pcl'], batch['world_pcl'], batch['params'], batch['proper_scale']

    # measure data loading time
    t0 = time.time()

    # put necessary stuff to cuda
    images = images.cuda()
    mask_gt = mask_gt.cuda()
    world_coords_gt = world_coords_gt.cuda()
    normals_gt = normals_gt.cuda()
    normals_mask_gt = normals_mask_gt.cuda()
    depth_gt = depth_gt.cuda()

    predicted_world_coords, predicted_normals, predicted_depth, predicted_focal, \
    predicted_height, predicted_pitch, predicted_roll, predicted_params = predict_on_images(images, depth_and_params, args)

    # mask if prediciton is too high, which can be caused by some normals
    mask_gt = mask_gt * (torch.abs(predicted_world_coords).sum(1) < 100).float()

    coords_regression_loss = MyRMSELoss(predicted_world_coords, world_coords_gt, mask_gt)
    normals_regression_loss = NormalsLoss(predicted_normals, normals_gt, normals_mask_gt)

    with torch.no_grad():
      depth_rmse = MyRMSELoss(predicted_depth[batch_items_proper_scale, 0],
                              depth_gt[batch_items_proper_scale].cuda(), mask_gt[batch_items_proper_scale])

    loss = args.coords_weight * coords_regression_loss + args.normals_weight * normals_regression_loss

    if loss != loss:
      print('Loss is nan!')
      optimizer_depth_and_params.zero_grad()
      if optimize:
        loss.backward()
      optimizer_depth_and_params.zero_grad()
      del loss
      continue

    if optimize:
      optimizer_depth_and_params.zero_grad()
      loss.backward()
      optimizer_depth_and_params.step()

    metrics_hist.update([float(depth_rmse)], args.batch_size)
    losses_hist.update([float(loss), float(coords_regression_loss), float(normals_regression_loss)])

    times_hist.update([t0 - t1, time.time() - t0])
    t1 = time.time()

    if i % args.plot_examples_freq == 0:
      print('Plotting')
      visdom_dict(args.__dict__, title='params', env=args.env)
      i_batch_to_plot = random.choice(np.where(tonumpy(batch_items_proper_scale))[0])

      plot_dict = dict(predicted_depth=predicted_depth[i_batch_to_plot], \
                       predicted_world_coords=predicted_world_coords[i_batch_to_plot], \
                       predicted_normals=predicted_normals[i_batch_to_plot], \
                       predicted_focal=predicted_focal[i_batch_to_plot], \
                       predicted_height=predicted_height[i_batch_to_plot], \
                       predicted_pitch=predicted_pitch[i_batch_to_plot], \
                       predicted_roll=predicted_roll[i_batch_to_plot], \
                       depth_gt=depth_gt[i_batch_to_plot], world_coords_gt=world_coords_gt[i_batch_to_plot],
                       normals_gt=normals_gt[i_batch_to_plot], mask_gt=mask_gt[i_batch_to_plot], normals_mask_gt=normals_mask_gt[i_batch_to_plot],
                       params_gt=params_gt[i_batch_to_plot], images=images[i_batch_to_plot], args=args, env_prefix=env_prefix, env=args.env)
      plotter.put_plot_dict(plot_dict)

    times_losses_metrics = [*times_hist.get_val_and_avg_strings(append_names=True),
                            *losses_hist.get_val_and_avg_strings(append_names=True),
                            *metrics_hist.get_val_and_avg_strings(append_names=True)]

    logger.train_writer.write(('{}:' + ' {}: {} ({})' * (len(times_losses_metrics) // 3))
                              .format('T' if optimize else 'V', *times_losses_metrics))
    for k in range(0, len(times_losses_metrics), 3):
      name, val, avg = times_losses_metrics[k:k + 3]
      summary_writer.add_scalar(env_prefix + name, float(val), n_iter_train if optimize else n_iter_val)

    if optimize:
      n_iter_train += 1
      logger.train_bar.update(i)
    else:
      n_iter_val += 1
      logger.valid_bar.update(i)

    if i >= total_iters - 1:
      break
    i = i + 1

  print('Epoch finished!')
  return losses_hist, metrics_hist


if __name__ == '__main__':
  global args, best_error, n_iter_train, n_iter_val, gpus
  parser = create_parser()
  args = parser.parse_args()

  if args.debug:
    with torch.autograd.detect_anomaly():
      main()
  else:
    main()
