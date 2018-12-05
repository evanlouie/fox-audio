import json
import os
import utils
from tensorflow.python.lib.io import file_io
from inference_json_old import inference_app
from vggish_inference import embedding, embedding_from_wav_data
import configparser
import readers

config = configparser.ConfigParser()
config.read('environment.properties')

train_dir = config['DEFAULT']['TRAINING_DIRECTORY']
data_pattern = config['DEFAULT']['DATA_PATTERN']
out_file_location = config['DEFAULT']['OUTPUT_FILE_PATH']
batch_size = int(config['DEFAULT']['BATCH_SIZE'])
top_k = int(config['DEFAULT']['TOP_K'])
wav = config['TF']['WAV']
tfrecord_filename = config['TF']['TFRECORD']

flags = { 'json_out' : True, 'movie_title' : 'Deadpool', 'class_csv_path': 'class_labels_indices_custom.csv' }
flags_dict_file = os.path.join(train_dir, "model_flags.json")
flags_dict = json.loads(file_io.FileIO(flags_dict_file, "r").read())
feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
        flags_dict["feature_names"], flags_dict["feature_sizes"]
    )
reader = readers.YT8MFrameFeatureReader(
            feature_names=feature_names, feature_sizes=feature_sizes
        )

def get_tfrecord():
  print("** get_tfrecord **")
  return embedding_from_wav_data(wav, tfrecord_filename)

def get_inf_json():
  print("** get_inf_json **")
  return inference_app(reader, train_dir, data_pattern, out_file_location, batch_size, top_k, flags)

def get_tfrecord_from_file(wav_filename, wav_data):
  return embedding_from_wav_data(wav_filename, wav_data, tfrecord_filename)
