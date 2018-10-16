# Copyright 2017 The TensorFlow Authors All Rights Reserved.
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

r"""A simple demonstration of running VGGish in inference mode.

This is intended as a toy example that demonstrates how the various building
blocks (feature extraction, model definition and loading, postprocessing) work
together in an inference context.

A WAV file (assumed to contain signed 16-bit PCM samples) is read in, converted
into log mel spectrogram examples, fed into VGGish, the raw embedding output is
whitened and quantized, and the postprocessed embeddings are optionally written
in a SequenceExample to a TFRecord file (using the same format as the embedding
features released in AudioSet).

Usage:
  # Run a WAV file through the model and print the embeddings. The model
  # checkpoint is loaded from vggish_model.ckpt and the PCA parameters are
  # loaded from vggish_pca_params.npz in the current directory.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file

  # Run a WAV file through the model and also write the embeddings to
  # a TFRecord file. The model checkpoint and PCA parameters are explicitly
  # passed in as well.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file \
                                    --tfrecord_file /path/to/tfrecord/file \
                                    --checkpoint /path/to/model/checkpoint \
                                    --pca_params /path/to/pca/params

  # Run a built-in input (a sine wav) through the model and print the
  # embeddings. Associated model files are read from the current directory.
  $ python vggish_inference_demo.py
"""

from __future__ import print_function

import numpy as np
from scipy.io import wavfile
import six
import tensorflow as tf

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
import os
import csv
import sys 
from collections import deque

flags = tf.app.flags

flags.DEFINE_string(
    'wav_file', None,
    'Path to a wav file. Should contain signed 16-bit PCM samples. '
    'If none is provided, a synthetic sound is used.')

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'pca_params', 'vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

flags.DEFINE_string(
    'tfrecord_file', None,
    'Path to a TFRecord file where embeddings will be written.')

flags.DEFINE_string(
    'target_directory', None,
    'Path to a multiple directories containing WAV files.')

flags.DEFINE_string(
    'subdirectory', None,
    'Path to WAV files.')

flags.DEFINE_string(
    'tf_directory', None,
    'Path to tfrecords')

flags.DEFINE_string(
    'labels_file', None,
    'Path to csv file that contains label ids'
)

FLAGS = flags.FLAGS

def get_last_row(csv_filename):
    with open(csv_filename, 'r') as f:
        try:
            lastrow = deque(csv.reader(f), 1)[0]
        except IndexError:  # empty file
            lastrow = None
        return lastrow

def embedding(wav, tf_record_filename, labels_filename):

    # name of subdirectory
    sub_name = wav.split('/')[-2]
    print("SUB_NAME: " + sub_name)

    # WAV Filename
    if type(wav) == str:
        wav_filename = wav.rsplit('/',1)[-1]
        label = 0
    else:
        wav_filename = wav

    # Acquiring Label ID
    if labels_filename:
        csv_file = csv.reader(open(labels_filename, "rt", encoding="utf8"), delimiter=",")
        for row in csv_file:
            if sub_name.title() in row[2]:
                print(row)
                label = int(row[0])
                break

    # Need to append to csv file if label is STILL 0
    if label == 0:
        print("Label is still 0. Will append new entry in labels CSV file.")
        last_row = get_last_row(labels_filename)
        row = [int(last_row[0])+1, '/m/t3st/', sub_name.title()]     
        with open(labels_filename,'a') as fd:
            writer=csv.writer(fd)
            writer.writerow(row)

    ############################################################################################
    batch = vggish_input.wavfile_to_examples(wav)
    #print(batch)

    ############################################################################################
    # Prepare a postprocessor to munge the model embeddings.
    pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)

    ############################################################################################

    # If needed, prepare a record writer to store the postprocessed embeddings.
    if FLAGS.tfrecord_file:
        writer = tf.python_io.TFRecordWriter(tf_record_filename)
    #if FLAGS.tf_directory:
        #writer = tf.python_io.TFRecordWriter(tf_record_filename)
    else:
        writer = tf.python_io.TFRecordWriter(tf_record_filename)

    with tf.Graph().as_default(), tf.Session() as sess:
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)

        # Run inference and postprocessing.
        [embedding_batch] = sess.run([embedding_tensor],
                                    feed_dict={features_tensor: batch})
        #print(embedding_batch)
        postprocessed_batch = pproc.postprocess(embedding_batch)
        #print(postprocessed_batch)

        # Write the postprocessed embeddings as a SequenceExample, in a similar
        # format as the features released in AudioSet. Each row of the batch of
        # embeddings corresponds to roughly a second of audio (96 10ms frames), and
        # the rows are written as a sequence of bytes-valued features, where each
        # feature value contains the 128 bytes of the whitened quantized embedding.
        if type(wav) == str:
            seq_example = tf.train.SequenceExample(
                context=tf.train.Features(feature={
                    'video_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[wav_filename.encode()])),
                    'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
                }),
                feature_lists=tf.train.FeatureLists(feature_list={
                        vggish_params.AUDIO_EMBEDDING_FEATURE_NAME: tf.train.FeatureList(feature=[tf.train.Feature(bytes_list=tf.train.BytesList(value=[embedding.tobytes()]))
                                    for embedding in postprocessed_batch
                                ]
                            )
                    }
                )
            )
            print(seq_example)
            if writer:
                writer.write(seq_example.SerializeToString())
        else:
            seq_example = tf.train.SequenceExample(
            feature_lists=tf.train.FeatureLists(feature_list={
                    vggish_params.AUDIO_EMBEDDING_FEATURE_NAME: tf.train.FeatureList(feature=[tf.train.Feature(bytes_list=tf.train.BytesList(value=[embedding.tobytes()]))
                                for embedding in postprocessed_batch
                            ]
                        )
                }
            )
        )
            print(seq_example)
            if writer:
                writer.write(seq_example.SerializeToString())

    if writer:
        writer.close()

def main(_):
  # In this simple example, we run the examples from a single audio file through
  # the model. If none is provided, we generate a synthetic input.
  if FLAGS.wav_file:
      #check to see if target_directory or subdirectory or tf_directory were used. If so, raise Exception.
      if FLAGS.target_directory or FLAGS.subdirectory or FLAGS.tf_directory:
        raise ValueError('If specifying the --wav_file argument, be sure to not use any of the following: --target_directory, --subdirectory, --tf_directory')
      else:
        wav_file = FLAGS.wav_file
        tf_recordfile = FLAGS.tf_recordfile
        labels_filename = FLAGS.labels_file
        embedding(wav_file, tf_recordfile, labels_filename)

  elif FLAGS.target_directory and not FLAGS.subdirectory:
      #check to see if tf_directory is provided. If not, raise Exception.
      if FLAGS.tfrecord_file:
          raise ValueError('If specifying the --subdirectory argument, be sure to also include the --tf_directory argument')
      if FLAGS.tf_directory:
          target_directory = FLAGS.target_directory
          #print("TARGET DIRECTORY: " + target_directory)
          tf_directory = FLAGS.tf_directory
          #print("TF RECORD DIRECTORY: " + tf_directory)
          labels_filename = FLAGS.labels_file
          #Iterate through each subdirectory
          subdirectories = []
          for dirs in os.walk(target_directory):
              subdirectories.append(dirs[0])
              for sub in subdirectories[1:]:
                #print("SUBDIRECTORY: " + sub)
                #Iterate through each file in subdirectory
                for filename in os.listdir(sub):
                 if filename.endswith(".wav"):
                     #print("FILENAME: " + filename)
                     subdirectory_name = sub.rsplit('/',1)[-1]
                     wav_filename = filename.rsplit('.',1)[0]
                     tf_recordfile = tf_directory + "/" + subdirectory_name + "_" + wav_filename + ".tfrecord"
                     #print("TF_RECORD_FILENAME: " + tf_recordfile)
                     embedding(target_directory + "/" + subdirectory_name + "/" + filename, tf_recordfile, labels_filename)
                     continue
                 else:
                     continue

  elif FLAGS.subdirectory and not FLAGS.target_directory:
      #check to see if target_directory or subdirectory or tf_directory were used. If so, raise Exception.
      if FLAGS.tfrecord_file:
          raise ValueError('If specifying the --subdirectory argument, be sure to also include the --tf_directory argument')
      if FLAGS.tf_directory:
        subdirectory = FLAGS.subdirectory
        #print("SUBDIRECTORY: " + subdirectory)
        labels_filename = FLAGS.labels_file
        for filename in os.listdir(subdirectory):
            if filename.endswith(".wav"):
                #print("FILENAME: " + filename)
                tf_directory = FLAGS.tf_directory
                #print("TF RECORD DIRECTORY: " + tf_directory)
                subdirectory_name = subdirectory.rsplit('/',1)[-1]
                wav_filename = filename.rsplit('.',1)[0]
                tf_recordfile = tf_directory + "/" + subdirectory_name + "_" + wav_filename + ".tfrecord"
                #print("TF_RECORD_FILENAME: " + tf_recordfile)
                embedding(subdirectory + "/" + filename, tf_recordfile, labels_filename)
                continue
            else:
                continue 

  elif FLAGS.target_directory and FLAGS.subdirectory:
    target_directory = FLAGS.target_directory
    subdirectory = FLAGS.subdirectory 
    labels_filename = FLAGS.labels_file
    for filename in os.listdir(subdirectory):
        if filename.endswith(".wav"):
            #print("FILENAME: " + filename)
            tf_directory = FLAGS.tf_directory
            #print("TF RECORD DIRECTORY: " + tf_directory)
            subdirectory_name = subdirectory.rsplit('/',1)[-1]
            wav_filename = filename.rsplit('.',1)[0]
            tf_recordfile = tf_directory + "/" + subdirectory_name + "_" + wav_filename + ".tfrecord"
            #print("TF_RECORD_FILENAME: " + tf_recordfile)
            embedding(subdirectory + "/" + filename, tf_recordfile, labels_filename)
            continue
        else:
            continue 
    
  else:
    # Write a WAV of a sine wav into an in-memory file object.
    num_secs = 5
    freq = 1000
    sr = 44100
    t = np.linspace(0, num_secs, int(num_secs * sr))
    x = np.sin(2 * np.pi * freq * t)
    # Convert to signed 16-bit samples.
    samples = np.clip(x * 32768, -32768, 32767).astype(np.int16)
    wav_file = six.BytesIO()
    wavfile.write(wav_file, sr, samples)
    wav_file.seek(0)
    tf_recordfile = "sin_wav_tfrecord"
    embedding(wav_file, tf_recordfile, "no_labels_file")

if __name__ == '__main__':
    tf.app.run()