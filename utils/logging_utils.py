import sys
import sys
sys.path.append('.')

import numpy as np
import progressbar
from blessings import Terminal


class TermLogger(object):
  def __init__(self, n_epochs, train_size, valid_size):
    self.n_epochs = n_epochs
    self.train_size = train_size
    self.valid_size = valid_size
    self.t = Terminal(force_styling=None)
    s = 10
    e = 1  # epoch bar position
    tr = 3  # train bar position
    ts = 6  # valid bar position
    h = self.t.height

    for i in range(10):
      print('')
    self.epoch_bar = progressbar.ProgressBar(maxval=n_epochs, fd=Writer(self.t, (0, h - s + e)))

    self.train_writer = Writer(self.t, (0, h - s + tr))
    self.train_bar_writer = Writer(self.t, (0, h - s + tr + 1))

    self.valid_writer = Writer(self.t, (0, h - s + ts))
    self.valid_bar_writer = Writer(self.t, (0, h - s + ts + 1))

    self.reset_train_bar()
    self.reset_valid_bar()

  def reset_train_bar(self):
    self.train_bar = progressbar.ProgressBar(maxval=self.train_size, fd=self.train_bar_writer)
    self.train_bar.start()

  def reset_valid_bar(self):
    self.valid_bar = progressbar.ProgressBar(maxval=self.valid_size, fd=self.valid_bar_writer)
    self.valid_bar.start()


class Writer(object):
  """Create an object with a write method that writes to a
  specific place on the screen, defined at instantiation.

  This is the glue between blessings and progressbar.
  """

  def __init__(self, t, location):
    """
    Input: location - tuple of ints (x, y), the position
                    of the bar in the terminal
    """
    self.location = location
    self.t = t

  def write(self, string):
    with self.t.location(*self.location):
      sys.stdout.write("\033[K")
      print(string)

  def flush(self):
    return


class AverageMeter(object):
  """Computes and stores the average and current value"""

  def __init__(self, i=1, precision=3, names=None):
    if i == 1 and not names is None:
      i = len(names)
    self.meters = i
    self.precision = precision
    self.reset(self.meters)
    self.names = names
    self.history = []
    self.last = None
    if not names is None:
      assert len(names) == i

  def reset(self, i):
    self.val = [0] * i
    self.avg = [0] * i
    self.sum = [0] * i
    self.count = [0] * i

  def update(self, val, n=1, losses_to_update=None):
    if not isinstance(val, list):
      val = [val]
    if not isinstance(n, list):
      n = [n] * len(val)
    val = [float(k) for k in val]
    assert (len(val) == self.meters)
    self.last = val
    self.history.append(val)
    for i, v in enumerate(val):
      self.val[i] = v
      self.sum[i] += v * n[i]
      self.count[i] += n[i]
      self.avg[i] = self.sum[i] / max(self.count[i], 1)

  def get_history(self, pos=-1, name=None):
    # xor
    assert type(pos) is int
    assert (not name is None) != (pos >= 0)
    if pos == -1:
      pos = self.names.index(name)
    return np.array(self.history)[:, pos]

  def get_val_strings(self):
    vals = ['{:.{}f}'.format(v, self.precision) for v in self.val]
    return vals

  def get_avg_strings(self):
    avgs = ['{:.{}f}'.format(a, self.precision) for a in self.avg]
    return avgs

  def get_val_and_avg_strings(self, names=None, append_names=False):
    val_strings = self.get_val_strings()
    avg_strings = self.get_avg_strings()
    if not names is None:
      indices = [self.names.index(name) for name in names]
      val_strings = [val_strings[k] for k in indices]
      avg_strings = [avg_strings[k] for k in indices]

    elems_per_val = (3 if append_names else 2)
    merged_val_avgs = [None] * (len(val_strings) * elems_per_val)
    if append_names:
      if not names is None:
        merged_val_avgs[0::3] = names
      else:
        merged_val_avgs[0::3] = self.names
    merged_val_avgs[elems_per_val - 2::elems_per_val] = val_strings
    merged_val_avgs[elems_per_val - 1::elems_per_val] = avg_strings
    return merged_val_avgs

  def __len__(self):
    return self.meters

  def __repr__(self):
    val = ' '.join(self.get_val_strings())
    avg = ' '.join(self.get_avg_strings())
    return '{} ({})'.format(val, avg)
