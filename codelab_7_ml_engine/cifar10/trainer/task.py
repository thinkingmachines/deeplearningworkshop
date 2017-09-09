# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import, division, print_function

import json
import os
import time
from datetime import datetime

import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_def_utils, tag_constants

from . import model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('job-dir', '/tmp/cifar10',
                           """A Google Cloud Storage path in which to store"""
                           """training outputs and other data needed for"""
                           """training.""")
tf.app.flags.DEFINE_integer('log-frequency', 10,
                            """How often to log results to the console.""")
tf.app.flags.DEFINE_integer('max-steps', 100000,
                            """Number of batches to run.""")


def start_server(cluster, task):
  return tf.train.Server(cluster, job_name=task.type, task_index=task.index)


def train(cluster=None, task=None):
  train_dir = os.path.join(FLAGS.job_dir, 'train')
  model_dir = os.path.join(FLAGS.job_dir, 'model')

  if cluster:
    server = start_server(cluster, task)
    master = server.target
    device_fn = tf.train.replica_device_setter(cluster=cluster)
    config = tf.ConfigProto(log_device_placement=False)
  else:
    master = ''
    device_fn = ''
    config = None

  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      images, labels = model.distorted_inputs()

    with tf.device(device_fn):
      # Build a Graph that computes the logits predictions from the
      # inference model.
      logits = model.inference(images)

      # Calculate loss.
      loss = model.loss(logits, labels)

      # Build a Graph that trains the model with one batch of examples and
      # updates the model parameters.
      train_op = model.train(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    hooks = [tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
             tf.train.NanTensorHook(loss), _LoggerHook()]

    with tf.train.MonitoredTrainingSession(
      master=master,
      is_chief=task.type == 'master',
      checkpoint_dir=train_dir,
      hooks=hooks,
      config=config,
    ) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)
        # FIXME: Add eval

  export(train_dir, model_dir)


def export(train_dir, model_dir):
  with tf.Session(graph=tf.Graph()) as sess:
    inputs, outputs = model.build_prediction_graph()
    signature_def_map = {
      'serving_default': signature_def_utils.predict_signature_def(
        inputs, outputs),
    }

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        model.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    ckpt = tf.train.get_checkpoint_state(train_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
    else:
      print('No checkpoint file found')
      return

    builder = saved_model_builder.SavedModelBuilder(model_dir)
    builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING],
                                         signature_def_map=signature_def_map)
    builder.save(False)


def main(argv=None):
  model.maybe_download_and_extract()

  tf_config = os.environ.get('TF_CONFIG')

  # If TF_CONFIG is not available run local
  if not tf_config:
    return train()

  tf_config_json = json.loads(tf_config)

  cluster_data = tf_config_json.get('cluster')
  cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None

  task_data = tf_config_json.get('task', {})
  task = type('TaskSpec', (object,), task_data)

  if task.type == 'ps':
    server = start_server(cluster, task)
    return server.join()
  elif task.type in ['master', 'worker']:
    return train(cluster, task)
  else:
    raise ValueError('Invalid task_type %s' % (task.type,))


if __name__ == '__main__':
  tf.app.run()
