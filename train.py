import tensorflow as tf
import numpy as np
import cv2
from utils import cpm_utils, tf_utils
from cpm_net import *
from matplotlib import pyplot as plt


tfr_data_files = '/mnt/monkey_data/Experiment6/train.tfrecord'
input_size = 368
heatmap_size = 46
stages =6 
# center_radius =21 
num_of_joints = 13
batch_size = 10
max_iterations = 200000
lr =0.0001 
lr_decay_rate = 0.9 
lr_decay_step = 5000
color_channel ='RGB'
log_every = 10
save_every = 1000
train_dir = 'checkpoints'


batch_x, batch_y = tf_utils.read_batch_cpm(tfr_data_files, input_size, heatmap_size, num_of_joints, batch_size)
input_placeholder = tf.placeholder(dtype=tf.float32, shape=(batch_size, input_size, input_size, 3), 
                                   name='input_placeholer')
cmap_placeholder = tf.placeholder(dtype=tf.float32, shape=(batch_size, input_size, input_size, 1),
                                  name='cmap_placeholder')
hmap_placeholder = tf.placeholder(dtype=tf.float32,shape=(batch_size, heatmap_size,heatmap_size,num_of_joints + 1),
                                  name='hmap_placeholder')

model = CPM_Model(stages, num_of_joints + 1)
model.build_model(input_placeholder, batch_size)
model.build_loss(hmap_placeholder, lr, lr_decay_rate, lr_decay_step)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    saver = tf.train.Saver(max_to_keep=None)
    ckpt_path = tf.train.latest_checkpoint(train_dir)
    if ckpt_path:
        print('Restoring from {}'.format(ckpt_path))
        saver.restore(sess, ckpt_path)
    else:
        init = tf.global_variables_initializer()
        sess.run(init)

    '''
    model.load_weights_from_file(pretrained_model, sess, finetune=True)

    for variable in tf.trainable_variables():
        with tf.variable_scope('', reuse=True):
                var = tf.get_variable(variable.name.split(':0')[0])
                print(variable.name, np.mean(sess.run(var)))           
    '''

    # Summary Writer
    summary_writer = tf.summary.FileWriter(train_dir)

    global_step = sess.run(model.global_step)
    while global_step < max_iterations:
        # Read in batch data
        batch_x_np, batch_y_np = sess.run([batch_x,batch_y])
 
        # Recreate heatmaps
        #gt_heatmap_np = cpm_utils.make_gaussian_batch(batch_y_np, heatmap_size, 2)

        # Update once
        stage_losses_np, total_loss_np, _, summary, current_lr, \
        stage_heatmap_np, global_step = sess.run([model.stage_loss,
                                                      model.total_loss,
                                                      model.train_op,
                                                      model.merged_summary,
                                                      model.lr,
                                                      model.stage_heatmap,
                                                      model.global_step
                                                      ],
                                                     feed_dict={input_placeholder: batch_x_np,
                                                                hmap_placeholder: batch_y_np})
        # plt.figure()
        # plt.imshow(batch_x_np[0,:,:,:])
        # plt.figure()
        # plt.imshow(batch_y_np[0,:,:,0], cmap='hot', interpolation='nearest')
        # plt.figure()
        # plt.imshow(stage_heatmap_np[5][0,:,:,0], cmap='hot', interpolation='nearest')
        # plt.show()
        # import ipdb; ipdb.set_trace()

        summary_writer.add_summary(summary, global_step=global_step)

        if global_step % log_every == 0:
            print("Step: {}, Loss = {}".format(global_step, total_loss_np))
        if global_step % save_every == 0:
            saver.save(sess=sess, save_path="{}/model.ckpt".format(train_dir), global_step=global_step)
            print('--- Model checkpoint saved ---')

    summary_writer.close()
    coord.request_stop()
    coord.join(threads)

print('Training done.')
