"""Training script for the WaveNet network."""

from __future__ import print_function

from datetime import datetime
import json
import os
import sys
import time

import tensorflow as tf
from tensorflow.python.training.queue_runner import add_queue_runner
from wavenet import WaveNetModel, TextReader

BATCH_SIZE = 1
LOGDIR_ROOT = './logdir'
CHECKPOINT_EVERY = 500
NUM_STEPS = 4000
LEARNING_RATE = 0.001
WAVENET_PARAMS = './wavenet_params.json'
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SAMPLE_SIZE = 4
L2_REGULARIZATION_STRENGTH = 0
DATA_DIR = './data/data_chid'
restore_from = None
logdir = None
logdir_root = None


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


def get_default_logdir(logdir_root):
    logdir = os.path.join(logdir_root, 'train', STARTED_DATESTRING)
    return logdir


def validate_directories(logdir,logdir_root,restore_from):
    logdir_root = logdir_root
    if logdir_root is None:
        logdir_root = LOGDIR_ROOT

    if logdir is None:
        logdir = get_default_logdir(logdir_root)
        print('Using default logdir: {}'.format(logdir))

    if restore_from is None:
        restore_from = logdir
    return {
        'logdir': logdir,
        'logdir_root': logdir_root,
        'restore_from': restore_from
    }


def main():
    
    l2_REGULARIZATION_STRENGTH = L2_REGULARIZATION_STRENGTH
    restore_from = None
    logdir = None
    logdir_root = None

    try:
        directories = validate_directories(logdir,logdir_root,restore_from)
    except ValueError as e:
        print("Some arguments are wrong:")
        print(str(e))
        return
    
    logdir = directories['logdir']
    restore_from = directories['restore_from']

    with open(WAVENET_PARAMS, 'r') as f:
        wavenet_params = json.load(f)

    coord = tf.train.Coordinator()

    with tf.name_scope('create_inputs'):
        reader = TextReader(
            DATA_DIR,
            coord,
            sample_size=SAMPLE_SIZE)
        text_batch = reader.dequeue(BATCH_SIZE)

    net = WaveNetModel(
        batch_size=BATCH_SIZE,
        dilations=wavenet_params["dilations"],
        filter_width=wavenet_params["filter_width"],
        residual_channels=wavenet_params["residual_channels"],
        dilation_channels=wavenet_params["dilation_channels"],
        skip_channels=wavenet_params["skip_channels"],
        quantization_channels=wavenet_params["quantization_channels"],
        use_biases=wavenet_params["use_biases"])
    if l2_REGULARIZATION_STRENGTH == 0:
        l2_REGULARIZATION_STRENGTH = None
    loss = net.loss(text_batch, l2_REGULARIZATION_STRENGTH)
    optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    trainable = tf.trainable_variables ()
    optim = optimizer.minimize(loss, var_list=trainable)

    # Set up session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    init = tf.initialize_all_variables()
    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver()

    saved_global_step = load(saver, sess, restore_from)
    if saved_global_step is None:
        saved_global_step = -1
        raise
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess)

    step = None

    try:
        last_saved_step = saved_global_step
        for step in range(saved_global_step + 1, NUM_STEPS):
            start_time = time.time()
            loss_value, _ = sess.run([loss, optim])
            print("fin step", step)
            duration = time.time() - start_time
            print('step {:d} - loss = {:.3f}, ({:.3f} sec/step)'
                  .format(step, loss_value, duration))

            if step % CHECKPOINT_EVERY == 0:
                save(saver, sess, logdir, step)
                last_saved_step = step

    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()
    finally:
        save(saver, sess, logdir, step)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
