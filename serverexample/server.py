import json
import os
import utils
from tensorflow.python.lib.io import file_io
from inference_json import inference_app
from vggish_inference import embedding
from inference_json import inference
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


flags_dict_file = os.path.join(train_dir, "model_flags.json")
#if not file_io.file_exists(flags_dict_file):
#        raise IOError("Cannot find %s. Did you run eval.py?" % flags_dict_file)
flags_dict = json.loads(file_io.FileIO(flags_dict_file, "r").read())
feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
        flags_dict["feature_names"], flags_dict["feature_sizes"]
    )


#feature_sizes=[float(1024), float(128)]
#feature_names=["mean_rgb", "mean_audio"]
reader = readers.YT8MFrameFeatureReader(
            feature_names=feature_names, feature_sizes=feature_sizes
        )
flags = { 'json_out' : True }

def get_tfrecord():
  print("** get_tfrecord **")

  return embedding(wav, tfrecord_filename)

def get_inf_json():
  print("** get_inf_json **")
  #inference(reader, train_dir, data_pattern, out_file_location, batch_size, top_k)
  return inference_app(reader, train_dir, data_pattern, out_file_location, batch_size, top_k, flags)

def get_tfrecord_from_file(wav_file):
  flags = dict()
  flags['ff'] = 'gunshot'
  return embedding_from_wav_data(wav_file, tfrecord_filename, flags)

'''class Server:

  def server_running(self):
    return('Server is running...')
  
  def get_tfrecord(self):
    tfrecord = embedding(wav, tfrecord_filename)
    return(tfrecord)

  def get_inference(self):
    inference_json = inference("reader", train_dir, tfrecord_filename, out_file_location, batch_size, top_k)
    print(inference_json)
    return(inference_json)'''
