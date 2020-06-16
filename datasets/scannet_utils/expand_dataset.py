import argparse
import errno
import os
import subprocess
import sys

from p_tqdm import p_map

from paths import *

TRAIN_PATH = '{}/scans'.format(SCANNET_PATH)
TEST_PATH = '{}/scans_test'.format(SCANNET_PATH)
SCANNET_CODE_PATH = '{}/code/SensReader/python'.format(SCANNET_PATH)

parser = argparse.ArgumentParser(description='Depth from single gibson',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-n', '--n_split', required=True, type=int, metavar='N', help='split to process')

args = parser.parse_args()
N_MACHINES = 10
assert args.n_split >= 0 and args.n_split < N_MACHINES

print('Executing split {}'.format(args.n_split))
completed_folder = '{}/{}'.format(SCANNET_PATH, 'completed')
completed_filename = '{}/{}_{}'.format(completed_folder, str(args.n_split).zfill(2), str(N_MACHINES).zfill(2))
processing_filename = '{}/{}/{}'.format(SCANNET_PATH, 'processing', str(args.n_split).zfill(2))
if os.path.exists(completed_filename):
  print('Split has already been created!')
  print('Exiting!!!')
  exit(1)

if os.path.exists(processing_filename):
  print('Split is being processed!')
  print('Exiting!!!')
  exit(1)

sys.path.append(SCANNET_CODE_PATH)

FNULL = open(os.devnull, 'w')
retcode = subprocess.call('python2 --version', shell=True,
                          stdout=FNULL,
                          stderr=subprocess.STDOUT)
assert retcode == 0

# train things
all_scenes_path = [TRAIN_PATH + '/' + k for k in os.listdir(TRAIN_PATH) if os.path.isdir(TRAIN_PATH + '/' + k)]


def make_dir_without_checking(folder):
  try:
    os.makedirs(folder)
  except OSError as e:
    if e.errno != errno.EEXIST:
      raise  # Reraise if failed for reasons other than existing already


def process_train_scene(scene_path):
  make_dir_without_checking(processing_filename)
  try:
    os.chdir(scene_path)

    zip_files = [k for k in os.listdir('.') if '.zip' in k]
    for zip in zip_files:
      subprocess.call('unzip -o ' + zip, shell=True, stdout=FNULL, stderr=subprocess.STDOUT)
    sense_file = [k for k in os.listdir('.') if '.sens' in k][0]
    subprocess.call('python2 ' + SCANNET_CODE_PATH + '/reader.py --filename ' + sense_file + ' ' + \
                    '--output_path . --export_depth_images --export_color_images ' + \
                    '--export_poses --export_intrinsics',
                    shell=True)

  except Exception as e:
    print('Error on file {}'.format(scene_path))
    print(e)


all_scenes_path.sort()


def chunk_list(seq, num):
  avg = len(seq) / float(num)
  out = []
  last = 0.0

  while last < len(seq):
    out.append(seq[int(last):int(last + avg)])
    last += avg
  return out


parallel = True

chunks = chunk_list(all_scenes_path, N_MACHINES)
chunk_to_process = chunks[args.n_split]
if parallel:
  p_map(process_train_scene, chunk_to_process, num_cpus=10)
else:
  for k in chunk_to_process:
    process_train_scene(k)

if not os.path.exists(completed_folder):
  os.makedirs(completed_folder)
with open(completed_filename, 'a'):
  os.utime(completed_filename, None)
