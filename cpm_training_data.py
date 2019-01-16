import cv2
from skimage.io import imshow
import matplotlib.pyplot as plt
from utils import cpm_utils
from utils import tf_utils
import numpy as np
import math
import tensorflow as tf
import time
import random
import os, shutil, sys, ipdb
from tqdm import tqdm
from utils.cpm_utils import get_ground_truth_params


tfr_file = '/mnt/monkey_data/Experiment6/train.tfrecord'

DEBUG = False
IMG_SIZE = 368
HEATMAP_SIZE = 46
NUM_OF_JOINTS = 13
GAUSSIAN_RADIUS = 1


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int32List(value=[value]))


def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))



tfr_writer = tf.python_io.TFRecordWriter(tfr_file)

img_count = 0

dataset_dir = sys.argv[1]
gt_content = open('{}/labels.txt'.format(dataset_dir), 'r').readlines()

corrupted_imgs = 0
corrupted_paths = []
for line in tqdm(gt_content):
    output_image, output_heatmaps, cur_joints_x, cur_joints_y, vis = get_ground_truth_params(dataset_dir, line, IMG_SIZE, HEATMAP_SIZE, NUM_OF_JOINTS, GAUSSIAN_RADIUS)
    if output_image is None:
        corrupted_imgs += 1
        continue

    coords_set = np.concatenate((np.reshape(cur_joints_x, (-1, 1)),np.reshape(cur_joints_y, (-1, 1))), axis=1)

    output_image_raw = output_image.astype(np.uint8).tostring()
    output_heatmaps_raw = output_heatmaps.flatten().tolist()
    output_coords_raw = coords_set.flatten().tolist()

    raw_sample = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image': _bytes_feature(output_image_raw),
                    'heatmaps': _float32_feature(output_heatmaps_raw)
                    }
                )
            )

    tfr_writer.write(raw_sample.SerializeToString())

    img_count += 1


tfr_writer.close()
print(corrupted_paths)
print(corrupted_imgs)
