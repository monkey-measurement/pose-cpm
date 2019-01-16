import tensorflow as tf
import numpy as np
from utils.cpm_utils import get_ground_truth_params
from cpm_net import *
import cv2
from PIL import Image, ImageDraw
import time
import math
import sys
from tqdm import tqdm
import ipdb
import random
from matplotlib import pyplot as plt 

input_size = 368 
hmap_size = 46
cmap_radius = 21
num_joints = 13
stages = 6
threshold = 1

joints = ['nose','head','neck','RShoulder','RHand','Lshoulder','Lhand','hip','RKnee','RFoot','LKnee','Lfoot','tail']
joint_indexes = { joint: i for i, joint in enumerate(joints) }

dataset_dir = sys.argv[1]
lines = open('{}/viz_labels.txt'.format(dataset_dir), 'r').readlines()
# lines = open('{}/labels.txt'.format(dataset_dir), 'r').readlines()

input_data = tf.placeholder(dtype=tf.float32, shape=[None, input_size, input_size, 3], name='input_image')
model = CPM_Model(stages, num_joints + 1)
model.build_model(input_data, 1)
saver = tf.train.Saver()

sess = tf.Session()
# ckpt_path = tf.train.latest_checkpoint('checkpoints')
ckpt_path = 'checkpoints/model.ckpt-0'
print("Restoring from {}".format(ckpt_path))
saver.restore(sess, ckpt_path)

PCKh = np.array([])
head_sizes = np.array([])
for i in tqdm(range(len(lines))):
    line = lines[i]
    image, gt_heatmaps, cur_joints_x, cur_joints_y, vis = get_ground_truth_params(dataset_dir, line, input_size, hmap_size, num_joints, gaussian_radius=1)
    if image is None:
        # corrupted_imgs += 1
        continue
    else:
        image = image.astype(np.float32)
    head_loc = np.array([cur_joints_x[joint_indexes['head']], cur_joints_y[joint_indexes['head']]])
    neck_loc = np.array([cur_joints_x[joint_indexes['neck']], cur_joints_y[joint_indexes['neck']]])
    head_size = np.array([np.linalg.norm(head_loc - neck_loc)])
    if not head_size:
        continue
    head_sizes = np.concatenate((head_sizes, head_size))

    input_img = np.expand_dims(image, axis=0) / 255.
    heatmap = sess.run(model.current_heatmap,
                       feed_dict={'input_image:0': input_img})
    heatmap = np.squeeze(heatmap)

    truth = np.array(list(zip(cur_joints_x, cur_joints_y)))
    preds = np.array([np.unravel_index(np.argmax(heatmap[:, :, joint]),(heatmap.shape[0], heatmap.shape[1])) for joint in range(num_joints)])[:,[1,0]]          # reverse x,y
    dist = np.linalg.norm(truth - preds, axis=1) # / head_size[0]
    dist = dist[vis]    # only visible joints matter
    PCKh = np.concatenate((PCKh, dist))

    # viz code
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    scale = input_size / hmap_size
    colors = [ 
            (255, 255, 0),   # aqua
            (255, 0, 255),   # fuchsia
            (0, 0, 255),     # red
            (0, 255, 0),     # lime
            (226, 43, 138),  # blueviolet
            (0, 255, 255)    # yellow
            ]
    joint_pairs = [ 
            ('nose','head'), 
            ('head','neck'),
            ('neck','RShoulder'),
            ('RShoulder','RHand'),
            ('neck','Lshoulder'),
            ('Lshoulder','Lhand'),
            ('neck','hip'),
            ('hip','RKnee'),
            ('RKnee','RFoot'),
            ('hip','LKnee'),
            ('LKnee','Lfoot'),
            ('hip','tail')
            ]
    for j,(jt1,jt2) in enumerate(joint_pairs):
        cam = int(line.split(' ')[0].split('/')[-1][5:12])
        pt1 = tuple(int(round(pt * scale)) for pt in preds[joint_indexes[jt1],:])
        pt2 = tuple(int(round(pt * scale)) for pt in preds[joint_indexes[jt2],:])
        # plt.figure()
        # plt.imshow(image / 255.)
        # plt.figure()
        # plt.imshow(gt_heatmaps[:,:,joint_indexes[jt1]], cmap='hot', interpolation='nearest')
        # plt.imshow(gt_heatmaps[:,:,joint_indexes[jt2]], cmap='hot', interpolation='nearest')
        # plt.figure()
        # plt.imshow(heatmap[:,:,joint_indexes[jt1]], cmap='hot', interpolation='nearest')
        # plt.imshow(heatmap[:,:,joint_indexes[jt2]], cmap='hot', interpolation='nearest')
        # plt.show()
        cv2.line(image, pt1, pt2, colors[j % 6], 3)
        # cv2.circle(image, pt1, 5, colors[j % 6])
        # plt.imshow(image / 255.)
        # plt.show()
    # TODO First line un-comment
    cv2.imwrite("preds/{}/image{:03d}.jpg".format(cam,i), image)
    # cv2.imwrite("preds/image{:03d}.jpg".format(i), image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # sys.exit()

    if i % 100 == 0:
        print(PCKh.mean()) #, head_sizes.mean())

ipdb.set_trace()
