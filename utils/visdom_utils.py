import random
import warnings

import matplotlib.pyplot as pyplt
from PIL import Image, ImageDraw
from utils.utils import *
from multiprocessing import Queue, Process
import cv2
import imageio
from skvideo.io import FFmpegWriter, FFmpegReader
import time
import tempfile

PYCHARM_VISDOM = 'PYCHARM_RUN'


def instantiante_visdom(port, server='http://localhost'):
  return visdom.Visdom(port=port, server=server, use_incoming_socket=True)


if not 'NO_VISDOM' in os.environ.keys():
  with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import visdom

    global_vis = instantiante_visdom(12890, server='http://visiongpu09')


def visdom_dict(dict_to_plot, title=None, window=None, env=PYCHARM_VISDOM, vis=None, simplify_floats=True):
  if vis is None:
    vis = global_vis
  opts = dict()
  if not title is None:
    opts['title'] = title
  vis.win_exists(title)
  if window is None:
    window = title
  dict_to_plot_sorted_keys = [k for k in dict_to_plot.keys()]
  dict_to_plot_sorted_keys.sort()
  html = '''<table style="width:100%">'''
  for k in dict_to_plot_sorted_keys:
    v = dict_to_plot[k]
    html += '<tr> <th>{}</th> <th>{}</th> </tr>'.format(k, v)
  html += '</table>'
  vis.text(html, win=window, opts=opts, env=env)


def visdom_default_window_title_and_vis(win, title, vis):
  if win is None and title is None:
    win = title = 'None'
  elif win is None:
    win = str(title)
  elif title is None:
    title = str(win)
  if vis is None:
    vis = global_vis
  return win, title, vis


def imshow_vis(im, title=None, win=None, env=None, vis=None):
  if vis is None:
    vis = global_vis
  opts = dict()
  win, title, vis = visdom_default_window_title_and_vis(win, title, vis)

  opts['title'] = title
  if im.dtype is np.uint8:
    im = im / 255.0
  vis.image(im, win=win, opts=opts, env=env)


def add_axis_to_image(im):
  fig = pyplt.figure()
  ax = fig.add_subplot(111)
  if len(im.shape) == 3 and im.shape[0] == 1:
    ax.imshow(im[0])
  else:
    ax.imshow(im)
  fig.canvas.draw()
  # Now we can save it to a numpy array.
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  data = data.transpose((2, 0, 1))
  return data

def myimresize(img, target_shape, interpolation_mode=cv2.INTER_NEAREST):
  assert interpolation_mode in [cv2.INTER_NEAREST, cv2.INTER_AREA]
  max = img.max();
  min = img.min()
  uint_mode = img.dtype == 'uint8'
  if max > min and not uint_mode:
    img = (img - min) / (max - min)
  if len(img.shape) == 3 and img.shape[0] in [1, 3]:
    if img.shape[0] == 3:
      img = np.transpose(cv2.resize(np.transpose(img, (1, 2, 0)), target_shape[::-1]), (2, 0, 1))
    else:
      img = cv2.resize(img[0], target_shape[::-1], interpolation=interpolation_mode)[None, :, :]
  else:
    img = cv2.resize(img, target_shape[::-1], interpolation=interpolation_mode)
  if max > min and not uint_mode:
    return (img * (max - min) + min)
  else:
    return img


def scale_image_biggest_dim(im, biggest_dim):
  # if it is a video, resize inside the video
  if im.shape[1] > im.shape[2]:
    scale = im.shape[1] / (biggest_dim + 0.0)
  else:
    scale = im.shape[2] / (biggest_dim + 0.0)
  target_imshape = (int(im.shape[1] / scale), int(im.shape[2] / scale))
  if im.shape[0] == 1:
    im = myimresize(im[0], target_shape=(target_imshape))[None, :, :]
  else:
    im = myimresize(im, target_shape=target_imshape)
  return im

def imshow(im, title='none', path=None, biggest_dim=None, normalize_image=True,
           max_batch_display=10, window=None, env=None, fps=10, vis=None,
           add_ranges=False, return_image=False, add_axis=False):
  if env is None:
    env = PYCHARM_VISDOM
  if type(im) is list:
    for k in range(len(im)):
      im[k] = tonumpy(im[k])
    im = np.array(im)
  if window is None:
    window = title
  if type(im) == 'string':
    # it is a path
    pic = Image.open(im)
    im = np.array(pic, dtype='float32')
  im = tonumpy(im)
  postfix = ''
  if im.dtype == np.bool:
    im = im * 1.0
  if add_ranges:
    postfix = '_max_{:.2f}_min_{:.2f}'.format(im.max(), im.min())
  if im.dtype == 'uint8':
    im = im / 255.0
  if len(im.shape) > 4:
    raise Exception('Im has more than 4 dims')
  if len(im.shape) == 4 and im.shape[0] == 1:
    im = im[0, :, :, :]
  if len(im.shape) == 3 and im.shape[-1] in [1, 3]:
    # put automatically channel first if its last
    im = im.transpose((2, 0, 1))

  if len(im.shape) == 2:
    # expand first if 1 channel image
    im = im[None, :, :]
  if not biggest_dim is None and len(im.shape) == 3:
    im = scale_image_biggest_dim(im, biggest_dim)
  if normalize_image and im.max() != im.min():
    im = (im - im.min()) / (im.max() - im.min())

  if add_axis:
    if len(im.shape) == 3:
      im = add_axis_to_image(im)
    else:
      for k in range(len(im)):
        im[k] = add_axis_to_image(im[k])

  if path is None:
    if window is None:
      window = title
    if len(im.shape) == 4:
      return vidshow_vis(im, title=title, window=window, env=env, vis=vis, biggest_dim=biggest_dim, fps=fps)
    else:
      imshow_vis(im, title=title + postfix, win=window, env=env, vis=vis)
  else:
    if len(im.shape) == 4:
      make_gif(im, path=path, fps=fps, biggest_dim=biggest_dim)
    else:
      imshow_matplotlib(im, path)
  if return_image:
    return im

def vidshow_vis(frames, title=None, window=None, env=None, vis=None, biggest_dim=None, fps=10):
  # if it does not work, change the ffmpeg. It was failing using anaconda ffmpeg default video settings,
  # and was switched to the machine ffmpeg.
  if vis is None:
    vis = global_vis
  if frames.shape[1] == 1 or frames.shape[1] == 3:
    frames = frames.transpose(0, 2, 3, 1)
  if frames.shape[-1] == 1:
    # if one channel, replicate it
    frames = np.tile(frames, (1, 1, 1, 3))
  if not frames.dtype is np.uint8:
    frames = np.array(frames * 255, dtype='uint8')
  videofile = '/tmp/%s.mp4' % next(tempfile._get_candidate_names())
  writer = MyVideoWriter(videofile, inputdict={'-r': str(fps)})
  for i in range(frames.shape[0]):
    if biggest_dim is None:
      actual_frame = frames[i]
    else:
      actual_frame = np.array(np.transpose(scale_image_biggest_dim(np.transpose(frames[i]), biggest_dim)), dtype='uint8')
    writer.writeFrame(actual_frame)
  writer.close()

  os.chmod(videofile, 0o777)
  vidshow_file_vis(videofile, title=title, window=window, env=env, vis=vis, fps=fps)
  return videofile

def vidshow_file_vis(videofile, title=None, window=None, env=None, vis=None, fps=10):
  # if it fails, check the ffmpeg version.
  # Depending on the ffmpeg version, sometimes it does not work properly.
  opts = dict()
  if not title is None:
    opts['title'] = title
    opts['caption'] = title
    opts['fps'] = fps
  if vis is None:
    vis = global_vis
  vis.win_exists(title)
  if window is None:
    window = title
  vis.video(videofile=videofile, win=window, opts=opts, env=env)

def cv2_imwrite(im, file, normalize=False, jpg_quality=None):
  if len(im.shape) == 3 and im.shape[0] == 3 or im.shape[0] == 4:
    im = im.transpose(1, 2, 0)
  if normalize:
    im = (im - im.min())/(im.max() - im.min())
    im = np.array(255.0*im, dtype='uint8')
  if jpg_quality is None:
    # The default jpg quality seems to be 95
    if im.shape[-1] == 3:
      cv2.imwrite(file, im[:,:,::-1])
    else:
      raise Exception('Alpha not working correctly')
      im_reversed = np.concatenate((im[:,:,3:0:-1], im[:,:,-2:-1]), axis=2)
      cv2.imwrite(file, im_reversed)
  else:
    cv2.imwrite(file, im[:,:,::-1], [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])

def imshow_matplotlib(im, path):
  cv2_imwrite(im, path)

def make_gif(ims, path, fps=None, biggest_dim=None):
  if ims.dtype != 'uint8':
    ims = np.array(ims*255, dtype='uint8')
  if ims.shape[1] in [1,3]:
    ims = ims.transpose((0,2,3,1))
  if ims.shape[-1] == 1:
    ims = np.tile(ims, (1,1,1,3))
  with imageio.get_writer(path) as gif_writer:
    for k in range(ims.shape[0]):
      #imsave(ims[k].mean()
      if biggest_dim is None:
        actual_im = ims[k]
      else:
        actual_im = np.transpose(scale_image_biggest_dim(np.transpose(ims[k]), biggest_dim))
      gif_writer.append_data(actual_im)
  if not fps is None:
    gif = imageio.mimread(path)
    imageio.mimsave(path, gif, fps=fps)


def draw_horizon_line(image, rotation_mat=None, pitch_angle=None, roll_angle=None, intrinsic=None, force_no_roll=False, color=(255, 0, 0)):
  # the horizon is the perpendicular of the line of the vanishing x and y axes
  if rotation_mat is None:
    roll_rotations = zrotation_deg(-1 * float(roll_angle))
    pitch_rotations = xrotation_deg(float(pitch_angle))
    rotation_mat = pitch_rotations @ roll_rotations
  assert image.dtype == 'uint8'
  height, width = image.shape[1:]
  image = np.ascontiguousarray(tonumpy(image))
  if intrinsic is None:
    intrinsic = np.array(((width, 0, width / 2.0),
                          (0, height, height / 2.0),
                          (0, 0, 1)))

  # the ground plane is the xy
  projection_mat = np.matmul(intrinsic, rotation_mat)
  vanishing_x = np.matmul(projection_mat, np.array((1, 0, 0)))
  vanishing_z = np.matmul(projection_mat, np.array((0, 0, 1)))

  if vanishing_x[2] == 0 or vanishing_z[2] == 0:
    vanishing_direction = np.array((1, 0))
  else:
    vanishing_x_image_plane = vanishing_x[:2] / vanishing_x[2]
    vanishing_z_image_plane = vanishing_z[:2] / vanishing_z[2]

    vanishing_direction = vanishing_z_image_plane - vanishing_x_image_plane

  if force_no_roll:
    vanishing_direction = np.array((1, 0))

  start_y = vanishing_z_image_plane[1] - vanishing_z_image_plane[1] * vanishing_direction[1] / vanishing_direction[0]
  finish_y = vanishing_z_image_plane[1] + (width - vanishing_z_image_plane[1]) * vanishing_direction[1] / vanishing_direction[0]
  im = Image.fromarray(image.transpose((1, 2, 0)))
  draw = ImageDraw.Draw(im)
  draw.line((0, start_y, im.size[0], finish_y), fill=color, width=int(math.ceil(height / 100)))
  image_with_line = np.array(im).transpose((2, 0, 1))
  return image_with_line


def prepare_pointclouds_and_colors(coords, colors, default_color=(0, 0, 0)):
  if type(coords) is list:
    for k in range(len(coords)):
      assert len(coords) == len(colors)
      coords[k], colors[k] = prepare_single_pointcloud_and_colors(coords[k], colors[k], default_color)
    return coords, colors
  else:
    return prepare_single_pointcloud_and_colors(coords, colors, default_color)


def prepare_single_pointcloud_and_colors(coords, colors, default_color=(0, 0, 0)):
  coords = tonumpy(coords)
  if colors is None:
    colors = np.array(default_color)[:, None].repeat(coords.size / 3, 1).reshape(coords.shape)
  colors = tonumpy(colors)
  if colors.dtype == 'float32':
    if colors.max() > 1.0:
      colors = np.array(colors, dtype='uint8')
    else:
      colors = np.array(colors * 255.0, dtype='uint8')
  if type(coords) is list:
    for k in range(len(colors)):
      if colors[k] is None:
        colors[k] = np.ones(coords[k].shape)
    colors = np.concatenate(colors, axis=0)
    coords = np.concatenate(coords, axis=0)

  assert coords.shape == colors.shape
  if len(coords.shape) == 3:
    coords = coords.reshape((3, -1))
  if len(colors.shape) == 3:
    colors = colors.reshape((3, -1))
  assert len(coords.shape) == 2
  if coords.shape[0] == 3:
    coords = coords.transpose()
    colors = colors.transpose()
  return coords, colors


def show_pointcloud(original_coords, original_colors=None, title='none', win=None, env=None,
                    markersize=3, max_points=10000, valid_mask=None, labels=None, default_color=(0, 0, 0),
                    projection="orthographic", center=(0, 0, 0), up=(0, -1, 0), eye=(0, 0, -2),
                    display_grid=(True, True, True), axis_ranges=None):
  if env is None:
    env = PYCHARM_VISDOM
  assert projection in ["perspective", "orthographic"]
  coords, colors = prepare_pointclouds_and_colors(original_coords, original_colors, default_color)
  if not type(coords) is list:
    coords = [coords]
    colors = [colors]
  if not valid_mask is None:
    if not type(valid_mask) is list:
      valid_mask = [valid_mask]
    assert len(valid_mask) == len(coords)
    for i in range(len(coords)):
      if valid_mask[i] is None:
        continue
      else:
        actual_valid_mask = np.array(valid_mask[i], dtype='bool').flatten()
        coords[i] = coords[i][actual_valid_mask]
        colors[i] = colors[i][actual_valid_mask]
  if not labels is None:
    if not type(labels) is list:
      labels = [labels]
    assert len(labels) == len(coords)
  for i in range(len(coords)):
    if max_points != -1 and coords[i].shape[0] > max_points:
      selected_positions = random.sample(range(coords[i].shape[0]), max_points)
      coords[i] = coords[i][selected_positions]
      colors[i] = colors[i][selected_positions]
      if not labels is None:
        labels[i] = [labels[i][k] for k in selected_positions]
      if not type(markersize) is int or type(markersize) is float:
        markersize[i] = [markersize[i][k] for k in selected_positions]
  # after this, we can compact everything into a single set of pointclouds. and do some more stuff for nicer visualization
  coords = np.concatenate(coords)
  colors = np.concatenate(colors)
  if not type(markersize) is int or type(markersize) is float:
    markersize = list_of_lists_into_single_list(markersize)
    assert len(coords) == len(markersize)
  if not labels is None:
    labels = list_of_lists_into_single_list(labels)
  if win is None:
    win = title
  plot_coords = coords
  from visdom import _markerColorCheck
  # we need to construct our own colors to override marker plotly options
  # and allow custom hover (to show real coords, and not the once used for visualization)
  visdom_colors = _markerColorCheck(colors, plot_coords, np.ones(len(plot_coords), dtype='uint8'), 1)
  # add the coordinates as hovertext
  hovertext = ['x:{:.2f}\ny:{:.2f}\nz:{:.2f}\n'.format(float(k[0]), float(k[1]), float(k[2])) for k in coords]
  if not labels is None:
    assert len(labels) == len(hovertext)
    hovertext = [hovertext[k] + ' {}'.format(labels[k]) for k in range(len(hovertext))]

  # to see all the options interactively, click on edit plot on visdom->json->tree
  # traceopts are used in line 1610 of visdom.__intit__.py
  # data.update(trace_opts[trace_name])
  # for layout options look at _opts2layout

  camera = {'up': {
    'x': str(up[0]),
    'y': str(up[1]),
    'z': str(up[2]),
  },
    'eye': {
      'x': str(eye[0]),
      'y': str(eye[1]),
      'z': str(eye[2]),
    },
    'center': {
      'x': str(center[0]),
      'y': str(center[1]),
      'z': str(center[2]),
    },
    'projection': {
      'type': projection
    }
  }

  global_vis.scatter(plot_coords, env=env, win=win,
                     opts={'webgl': True,
                           'title': title,
                           'name': 'scatter',
                           'layoutopts': {
                             'plotly': {
                               'scene': {
                                 'aspectmode': 'data',
                                 'camera': camera,
                                 'xaxis': {
                                   'tickfont': {
                                     'size': 14
                                   },
                                   'autorange': axis_ranges is None,
                                   'range': [str(axis_ranges['min_x']), str(axis_ranges['max_x'])] if not axis_ranges is None else [-1, -1],
                                   'showgrid': display_grid[0],
                                   'showticklabels': display_grid[0],
                                   'zeroline': display_grid[0],
                                   'title': {
                                     'text': 'x' if display_grid[0] else '',
                                     'font': {
                                       'size': 20
                                     }
                                   }
                                 },
                                 'yaxis': {
                                   'tickfont': {
                                     'size': 14
                                   },
                                   'autorange': axis_ranges is None,
                                   'range': [str(axis_ranges['min_y']), str(axis_ranges['max_y'])] if not axis_ranges is None else [-1, -1],
                                   'showgrid': display_grid[1],
                                   'showticklabels': display_grid[1],
                                   'zeroline': display_grid[1],
                                   'title': {
                                     'text': 'y' if display_grid[1] else '',
                                     'font': {
                                       'size': 20
                                     }
                                   }
                                 },
                                 'zaxis': {
                                   'tickfont': {
                                     'size': 14
                                   },
                                   'autorange': axis_ranges is None,
                                   'range': [str(axis_ranges['min_z']), str(axis_ranges['max_z'])] if not axis_ranges is None else [-1, -1],
                                   'showgrid': display_grid[2],
                                   'showticklabels': display_grid[2],
                                   'zeroline': display_grid[2],
                                   'title': {
                                     'text': 'z' if display_grid[2] else '',
                                     'font': {
                                       'size': 20
                                     }
                                   }
                                 }
                               }
                             }
                           },
                           'traceopts': {
                             'plotly': {
                               '1': {
                                 # custom ops
                                 # https://plot.ly/python/reference/#scattergl-transforms
                                 'hoverlabel': {
                                   'bgcolor': '#000000'
                                 },
                                 'hoverinfo': 'text',
                                 'hovertext': hovertext,
                                 'marker': {
                                   'sizeref': 1,
                                   'size': markersize,
                                   'symbol': 'dot',
                                   'color': visdom_colors[1],
                                   'line': {
                                     'color': '#000000',
                                     'width': 0,
                                   }
                                 }
                               },
                             }
                           }
                           })

  return


class ThreadedVisdomPlotter():
  # plot func receives a dict and gets what it needs to plot
  def __init__(self, plot_func, use_threading=True, queue_size=10, force_except=False):
    self.queue = Queue(queue_size)
    self.plot_func = plot_func
    self.use_threading = use_threading
    self.force_except = force_except

    def plot_results_process(queue, plot_func):
      # to avoid wasting time making videos
      while True:
        try:
          if queue.empty():
            time.sleep(1)
            if queue.full():
              print("Plotting queue is full!")
          else:
            actual_plot_dict = queue.get()
            env = actual_plot_dict['env']
            time_put_on_queue = actual_plot_dict.pop('time_put_on_queue')
            visdom_dict({"queue_put_time": time_put_on_queue}, title=time_put_on_queue, window='params', env=env)
            print("Plotting...")
            plot_func(**actual_plot_dict)
            continue
        except Exception as e:
          if self.force_except:
            raise e
          print('Plotting failed wiht exception: ')
          print(e)

    if self.use_threading:
      Process(target=plot_results_process, args=[self.queue, self.plot_func]).start()

  def _detach_tensor(self, tensor):
    if tensor.is_cuda:
      tensor = tensor.detach().cpu()
    tensor = np.array(tensor.detach())
    return tensor

  def _detach_dict_or_list_torch(self, list_or_dict):
    # We put things to cpu here to avoid er
    if type(list_or_dict) is dict:
      to_iter = list(list_or_dict.keys())
    elif type(list_or_dict) is list:
      to_iter = list(range(len(list_or_dict)))
    else:
      return list_or_dict
    for k in to_iter:
      if type(list_or_dict[k]) is torch.Tensor:
        list_or_dict[k] = self._detach_tensor(list_or_dict[k])
      else:
        list_or_dict[k] = self._detach_dict_or_list_torch(list_or_dict[k])
    return list_or_dict

  def clear_queue(self):
    while not self.queue.empty():
      self.queue.get()

  def is_queue_full(self):
    if not self.use_threading:
      return False
    else:
      return self.queue.full()

  def n_queue_elements(self):
    if not self.use_threading:
      return 0
    else:
      return self.queue.qsize()

  def put_plot_dict(self, plot_dict):
    try:
      assert type(plot_dict) is dict
      assert 'env' in plot_dict, 'Env to plot not found in plot_dict!'
      plot_dict = self._detach_dict_or_list_torch(plot_dict)
      if self.use_threading:
        timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M:%S")
        plot_dict['time_put_on_queue'] = timestamp
        self.queue.put(plot_dict)
      else:
        self.plot_func(**plot_dict)
    except Exception as e:
      if self.force_except:
        raise e
      print('Putting onto plot queue failed with exception:')
      print(e)


class MyVideoWriter():
  def __init__(self, file, fps=None, *args, **kwargs):
    if not fps is None:
      kwargs['inputdict'] = {'-r': str(fps)}
    self.video_writer = FFmpegWriter(file, *args, **kwargs)

  def writeFrame(self, im):
    if len(im.shape) == 3 and im.shape[0] == 3:
      transformed_image = im.transpose((1, 2, 0))
    elif len(im.shape) == 2:
      transformed_image = np.concatenate((im[:, :, None], im[:, :, None], im[:, :, None]), axis=-1)
    else:
      transformed_image = im
    self.video_writer.writeFrame(transformed_image)

  def close(self):
    self.video_writer.close()