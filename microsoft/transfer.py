# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Binary for training Tensorflow models on the YouTube-8M dataset."""

import json
import os
import time
import sys
import multiprocessing

sys.path.insert(0, 'youtube_8m')
import eval_util, export_model, losses, frame_level_models, video_level_models, readers, utils
#from youtube_8m import eval_util, export_model, losses, frame_level_models, video_level_models, readers, utils
# eval_util, export_model, losses, frame_level_models, video_level_models, readers, utils

import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.lib.io import file_io
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
from tensorflow.python.client import device_lib

from train import validate_class_name, get_input_data_tensors, find_class_by_name, task_as_string, start_server, get_reader, ParameterServer, Trainer

FLAGS = flags.FLAGS

if __name__ == "__main__":
    # Dataset flags.
    flags.DEFINE_string("train_dir", "/tmp/yt8m_model/",
                        "The directory to save the model files in.")
    flags.DEFINE_string(
        "train_data_pattern", "",
        "File glob for the training dataset. If the files refer to Frame Level "
        "features (i.e. tensorflow.SequenceExample), then set --reader_type "
        "format. The (Sequence)Examples are expected to have 'rgb' byte array "
        "sequence feature as well as a 'labels' int64 context feature.")
    flags.DEFINE_string("feature_names", "mean_rgb", "Name of the feature "
                        "to use for training.")
    flags.DEFINE_string("feature_sizes", "1024",
                        "Length of the feature vectors.")

    # Model flags.
    flags.DEFINE_bool(
        "frame_features", False,
        "If set, then --train_data_pattern must be frame-level features. "
        "Otherwise, --train_data_pattern must be aggregated video-level "
        "features. The model must also be set appropriately (i.e. to read 3D "
        "batches VS 4D batches.")
    flags.DEFINE_string(
        "model", "LogisticModel",
        "Which architecture to use for the model. Models are defined "
        "in models.py.")
    flags.DEFINE_bool(
        "start_new_model", False,
        "If set, this will not resume from a checkpoint and will instead create a"
        " new model instance.")

    # Training flags.
    flags.DEFINE_integer("num_gpu", 1,
                         "The maximum number of GPU devices to use for training. "
                         "Flag only applies if GPUs are installed")
    flags.DEFINE_integer("batch_size", 1024,
                         "How many examples to process per batch for training.")
    flags.DEFINE_string("label_loss", "CrossEntropyLoss",
                        "Which loss function to use for training the model.")
    flags.DEFINE_float(
        "regularization_penalty", 1.0,
        "How much weight to give to the regularization loss (the label loss has "
        "a weight of 1).")
    flags.DEFINE_float("base_learning_rate", 0.01,
                       "Which learning rate to start with.")
    flags.DEFINE_float("learning_rate_decay", 0.95,
                       "Learning rate decay factor to be applied every "
                       "learning_rate_decay_examples.")
    flags.DEFINE_float("learning_rate_decay_examples", 4000000,
                       "Multiply current learning rate by learning_rate_decay "
                       "every learning_rate_decay_examples.")
    flags.DEFINE_integer("num_epochs", None,
                         "How many passes to make over the dataset before "
                         "halting training.")
    flags.DEFINE_integer("max_steps", None,
                         "The maximum number of iterations of the training loop.")
    flags.DEFINE_integer("export_model_steps", 1000,
                         "The period, in number of steps, with which the model "
                         "is exported for batch prediction.")

    # Other flags.
    flags.DEFINE_string("base_checkpoint", "model.ckpt-0",
                        "pretrained checkpoint")
    flags.DEFINE_integer("num_readers", multiprocessing.cpu_count(),
                         "How many threads to use for reading input files.")
    flags.DEFINE_string("optimizer", "AdamOptimizer",
                        "What optimizer class to use.")
    flags.DEFINE_float("clip_gradient_norm", 1.0, "Norm to clip gradients to.")
    flags.DEFINE_bool(
        "log_device_placement", False,
        "Whether to write the device on which every op will run into the "
        "logs on startup.")

def build_graph(reader,
                model,
                train_data_pattern,
                label_loss_fn=losses.CrossEntropyLoss(),
                batch_size=1000,
                base_learning_rate=0.01,
                learning_rate_decay_examples=1000000,
                learning_rate_decay=0.95,
                optimizer_class=tf.train.AdamOptimizer,
                clip_gradient_norm=1.0,
                regularization_penalty=1,
                num_readers=1,
                num_epochs=None):
    """Creates the Tensorflow graph.

    This will only be called once in the life of
    a training model, because after the graph is created the model will be
    restored from a meta graph file rather than being recreated.

    Args:
      reader: The data file reader. It should inherit from BaseReader.
      model: The core model (e.g. logistic or neural net). It should inherit
             from BaseModel.
      train_data_pattern: glob path to the training data files.
      label_loss_fn: What kind of loss to apply to the model. It should inherit
                  from BaseLoss.
      batch_size: How many examples to process at a time.
      base_learning_rate: What learning rate to initialize the optimizer with.
      optimizer_class: Which optimization algorithm to use.
      clip_gradient_norm: Magnitude of the gradient to clip to.
      regularization_penalty: How much weight to give the regularization loss
                              compared to the label loss.
      num_readers: How many threads to use for I/O operations.
      num_epochs: How many passes to make over the data. 'None' means an
                  unlimited number of passes.
    """

    global_step = tf.Variable(0, trainable=False, name="global_step")

    local_device_protos = device_lib.list_local_devices()
    gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
    gpus = gpus[:FLAGS.num_gpu]
    num_gpus = len(gpus)

    if num_gpus > 0:
        logging.info("Using the following GPUs to train: " + str(gpus))
        num_towers = num_gpus
        device_string = '/gpu:%d'
    else:
        logging.info("No GPUs found. Training on CPU.")
        num_towers = 1
        device_string = '/cpu:%d'

    learning_rate = tf.train.exponential_decay(
        base_learning_rate,
        global_step * batch_size * num_towers,
        learning_rate_decay_examples,
        learning_rate_decay,
        staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = optimizer_class(learning_rate)
    unused_video_id, model_input_raw, labels_batch, num_frames = (
        get_input_data_tensors(
            reader,
            train_data_pattern,
            batch_size=batch_size * num_towers,
            num_readers=num_readers,
            num_epochs=num_epochs))
    tf.summary.histogram("model/input_raw", model_input_raw)

    feature_dim = len(model_input_raw.get_shape()) - 1

    model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)

    tower_inputs = tf.split(model_input, num_towers)
    tower_labels = tf.split(labels_batch, num_towers)
    tower_num_frames = tf.split(num_frames, num_towers)
    tower_gradients = []
    tower_predictions = []
    tower_label_losses = []
    tower_reg_losses = []
    for i in range(num_towers):
        # For some reason these 'with' statements can't be combined onto the same
        # line. They have to be nested.
        with tf.device(device_string % i):
            with (tf.variable_scope(("tower"), reuse=True if i > 0 else None)):
                with (slim.arg_scope([slim.model_variable, slim.variable], device="/cpu:0" if num_gpus != 1 else "/gpu:0")):
                #with (slim.arg_scope([slim.model_variable, slim.variable], device="/cpu:0")):
                    result = model.create_model(
                        tower_inputs[i],
                        num_frames=tower_num_frames[i],
                        vocab_size=reader.num_classes,
                        labels=tower_labels[i])
                    for variable in slim.get_model_variables():
                        tf.summary.histogram(variable.op.name, variable)

                    predictions = result["predictions"]
                    tower_predictions.append(predictions)

                    if "loss" in result.keys():
                        label_loss = result["loss"]
                    else:
                        label_loss = label_loss_fn.calculate_loss(
                            predictions, tower_labels[i])

                    if "regularization_loss" in result.keys():
                        reg_loss = result["regularization_loss"]
                    else:
                        reg_loss = tf.constant(0.0)

                    reg_losses = tf.losses.get_regularization_losses()
                    if reg_losses:
                        reg_loss += tf.add_n(reg_losses)

                    tower_reg_losses.append(reg_loss)

                    # Adds update_ops (e.g., moving average updates in batch normalization) as
                    # a dependency to the train_op.
                    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                    if "update_ops" in result.keys():
                        update_ops += result["update_ops"]
                    if update_ops:
                        with tf.control_dependencies(update_ops):
                            barrier = tf.no_op(name="gradient_barrier")
                            with tf.control_dependencies([barrier]):
                                label_loss = tf.identity(label_loss)

                    tower_label_losses.append(label_loss)

                    # Incorporate the L2 weight penalties etc.
                    final_loss = regularization_penalty * reg_loss + label_loss
                    new_train_variables = [var for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if '/rnn/' not in var.name]
                    gradients = optimizer.compute_gradients(final_loss,
                                                            var_list=new_train_variables,
                                                            colocate_gradients_with_ops=False)
                    tower_gradients.append(gradients)
    label_loss = tf.reduce_mean(tf.stack(tower_label_losses))
    tf.summary.scalar("label_loss", label_loss)
    if regularization_penalty != 0:
        reg_loss = tf.reduce_mean(tf.stack(tower_reg_losses))
        tf.summary.scalar("reg_loss", reg_loss)
    merged_gradients = utils.combine_gradients(tower_gradients)

    if clip_gradient_norm > 0:
        with tf.name_scope('clip_grads'):
            merged_gradients = utils.clip_gradient_norms(
                merged_gradients, clip_gradient_norm)

    train_op = optimizer.apply_gradients(
        merged_gradients, global_step=global_step)

    tf.add_to_collection("global_step", global_step)
    tf.add_to_collection("loss", label_loss)
    tf.add_to_collection("predictions", tf.concat(tower_predictions, 0))
    tf.add_to_collection("input_batch_raw", model_input_raw)
    tf.add_to_collection("input_batch", model_input)
    tf.add_to_collection("num_frames", num_frames)
    tf.add_to_collection("labels", tf.cast(labels_batch, tf.float32))
    tf.add_to_collection("train_op", train_op)


class TransferTrainer(Trainer):
    """A Trainer to train a Tensorflow graph."""

    def recover_model(self, meta_filename):
        return tf.train.import_meta_graph(meta_filename, clear_devices=True)  

    def run(self, start_new_model=False):
        """Performs training on the currently defined Tensorflow graph.

        Returns:
          A tuple of the training Hit@1 and the training PERR.
        """
        if self.is_master and start_new_model:
            self.remove_training_directory(self.train_dir)

        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)

        model_flags_dict = {
            "model": FLAGS.model,
            "feature_sizes": FLAGS.feature_sizes,
            "feature_names": FLAGS.feature_names,
            "frame_features": FLAGS.frame_features,
            "label_loss": FLAGS.label_loss,
        }
        flags_json_path = os.path.join(FLAGS.train_dir, "model_flags.json")
        if file_io.file_exists(flags_json_path):
            existing_flags = json.load(
                file_io.FileIO(flags_json_path, mode="r"))
            if existing_flags != model_flags_dict:
                logging.error("Model flags do not match existing file %s. Please "
                              "delete the file, change --train_dir, or pass flag "
                              "--start_new_model",
                              flags_json_path)
                logging.error("Ran model with flags: %s",
                              str(model_flags_dict))
                logging.error("Previously ran with flags: %s",
                              str(existing_flags))
                exit(1)
        else:
            # Write the file.
            with file_io.FileIO(flags_json_path, mode="w") as fout:
                fout.write(json.dumps(model_flags_dict))

        target, device_fn = self.start_server_if_distributed()

        meta_filename = self.get_meta_filename(start_new_model, self.train_dir)

        # get the baseline model
        with tf.Graph().as_default() as baseline_graph, tf.Session() as baseline_sess:
            baseline_checkpoint = FLAGS.base_checkpoint
            #print("BASELINE:" + baseline_checkpoint)
            bl_ck= baseline_checkpoint+'.meta'
            print("BASELINE: " + bl_ck)
            baseline_saver = self.recover_model(bl_ck)
            baseline_saver.restore(baseline_sess, baseline_checkpoint)
            baseline_transferable_variables = [var for var in tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES) if '/rnn/' in var.name]
            baseline_graph.add_to_collections('TRANSFERABLE_VARIABLES',
                                              baseline_transferable_variables)
            baseline_transferable_weights = baseline_sess.run(
                tf.get_collection('TRANSFERABLE_VARIABLES'))
            #print("TRANSFERRABLE VARIABLES: " + str(baseline_transferable_variables))
            #print("TRANSFERRABLE VARIABLES WEIGHTS: " + str(baseline_transferable_weights))

        with tf.Graph().as_default() as graph:
            if meta_filename:
                # means that we don't need to copy paste the transferable weights
                saver = self.recover_model(meta_filename)

            with tf.device(device_fn), tf.Session() as sess:
                if not meta_filename:
                    print('building new model')
                    # means that we NEED to copy paste the weights from the base model
                    saver = self.build_model(self.model, self.reader)
                    print('new model has been built')
                    transferable_variables = [var for var in graph.get_collection(
                        tf.GraphKeys.TRAINABLE_VARIABLES) if '/rnn/' in var.name]
                    graph.add_to_collections(
                        'TRANSFERABLE_VARIABLES', transferable_variables)
                    for variable, weight in zip(graph.get_collection('TRANSFERABLE_VARIABLES')[0], baseline_transferable_weights[0]):
                        # variable.load(weight, sess)
                        variable.assign(weight)
                    print('variables were transferred')

                global_step = tf.get_collection("global_step")[0]
                loss = tf.get_collection("loss")[0]
                predictions = tf.get_collection("predictions")[0]
                labels = tf.get_collection("labels")[0]
                train_op = tf.get_collection("train_op")[0]
                init_op = tf.global_variables_initializer()

        sv = tf.train.Supervisor(
            graph,
            logdir=self.train_dir,
            init_op=init_op,
            is_chief=self.is_master,
            global_step=global_step,
            save_model_secs=15 * 60,
            save_summaries_secs=120,
            saver=saver)

        logging.info("%s: Starting managed session.",
                     task_as_string(self.task))
        with sv.managed_session(target, config=self.config) as sess:
            try:
                logging.info("%s: Entering training loop.",
                             task_as_string(self.task))
                while (not sv.should_stop()) and (not self.max_steps_reached):
                    batch_start_time = time.time()
                    _, global_step_val, loss_val, predictions_val, labels_val = sess.run(
                        [train_op, global_step, loss, predictions, labels])
                    seconds_per_batch = time.time() - batch_start_time
                    examples_per_second = labels_val.shape[0] / \
                        seconds_per_batch

                    if self.max_steps and self.max_steps <= global_step_val:
                        self.max_steps_reached = True

                    if self.is_master and global_step_val % 10 == 0 and self.train_dir:
                        eval_start_time = time.time()
                        hit_at_one = eval_util.calculate_hit_at_one(
                            predictions_val, labels_val)
                        perr = eval_util.calculate_precision_at_equal_recall_rate(predictions_val,
                                                                                  labels_val)
                        gap = eval_util.calculate_gap(
                            predictions_val, labels_val)
                        eval_end_time = time.time()
                        eval_time = eval_end_time - eval_start_time

                        logging.info("training step " + str(global_step_val) + " | Loss: " + ("%.2f" % loss_val) +
                                     " Examples/sec: " + ("%.2f" % examples_per_second) + " | Hit@1: " +
                                     ("%.2f" % hit_at_one) + " PERR: " + ("%.2f" % perr) +
                                     " GAP: " + ("%.2f" % gap))
                        #print("TRANSFERRABLE VARIABLES: " + str(transferable_variables))
                        #print("TRANSFERRABLE VARIABLES WEIGHTS: " + str(baseline_transferable_weights))

                        sv.summary_writer.add_summary(
                            utils.MakeSummary(
                                "model/Training_Hit@1", hit_at_one),
                            global_step_val) 
                        sv.summary_writer.add_summary(
                            utils.MakeSummary("model/Training_Perr", perr), global_step_val)
                        sv.summary_writer.add_summary(
                            utils.MakeSummary("model/Training_GAP", gap), global_step_val)
                        sv.summary_writer.add_summary(
                            utils.MakeSummary("global_step/Examples/Second",
                                              examples_per_second), global_step_val)
                        sv.summary_writer.flush()

                        # Exporting the model every x steps
                        time_to_export = ((self.last_model_export_step == 0) or
                                          (global_step_val - self.last_model_export_step
                                           >= self.export_model_steps))

                        if self.is_master and time_to_export:
                            self.export_model(
                                global_step_val, sv.saver, sv.save_path, sess)
                            self.last_model_export_step = global_step_val
                    else:
                        logging.info("training step " + str(global_step_val) + " | Loss: " +
                                     ("%.10f" % loss_val) + " Examples/sec: " + ("%.2f" % examples_per_second))
                        #print("TRANSFERRABLE VARIABLES: " + str(transferable_variables))
                        #print("TRANSFERRABLE VARIABLES WEIGHTS: " + str(baseline_transferable_weights))
            except tf.errors.OutOfRangeError:
                logging.info("%s: Done training -- epoch limit reached.",
                             task_as_string(self.task))

        logging.info("%s: Exited training loop.", task_as_string(self.task))
        sv.stop()

def main(unused_argv):
    # Load the environment.
    env = json.loads(os.environ.get("TF_CONFIG", "{}"))

    # Load the cluster data from the environment.
    cluster_data = env.get("cluster", None)
    cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None

    # Load the task data from the environment.
    task_data = env.get("task", None) or {"type": "master", "index": 0}
    task = type("TaskSpec", (object,), task_data)

    # Logging the version.
    logging.set_verbosity(tf.logging.INFO)
    logging.info("%s: Tensorflow version: %s.",
                 task_as_string(task), tf.__version__)

    # Dispatch to a master, a worker, or a parameter server.
    if not cluster or task.type == "master" or task.type == "worker":
        model = find_class_by_name(FLAGS.model,
                                   [frame_level_models, video_level_models])()

        reader = get_reader()

        model_exporter = export_model.ModelExporter(
            frame_features=FLAGS.frame_features,
            model=model,
            reader=reader)

        TransferTrainer(cluster, task, FLAGS.train_dir, model, reader, model_exporter,
                        FLAGS.log_device_placement, FLAGS.max_steps,
                        FLAGS.export_model_steps).run(start_new_model=FLAGS.start_new_model)

    elif task.type == "ps":
        ParameterServer(cluster, task).run()
    else:
        raise ValueError("%s: Invalid task_type: %s." %
                         (task_as_string(task), task.type))


if __name__ == "__main__":
    app.run()
