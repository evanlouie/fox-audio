import json
from inference_json import inference
from vggish_inference import embedding, embedding_from_wav_data
import configparser

config = configparser.ConfigParser()
config.read('environment.properties')

train_dir = config['DEFAULT']['TRAINING_DIRECTORY']
data_pattern = config['DEFAULT']['DATA_PATTERN']
out_file_location = config['DEFAULT']['OUTPUT_FILE_PATH']
batch_size = config['DEFAULT']['BATCH_SIZE']
top_k = config['DEFAULT']['TOP_K']
wav = config['TF']['WAV']
tfrecord_filename = config['TF']['TFRECORD']

def get_tfrecord():
  print("** get_tfrecord **")

  return embedding(wav, tfrecord_filename)


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
