import json
from inference_json import inference
import configparser

config = configparser.ConfigParser()
config.read('environment.properties')

train_dir = config['DEFAULT']['TRAIN_DIRECTORY']
data_pattern = config['DEFAULT']['DATA_PATTERN']
out_file_location = config['DEFAULT']['OUTPUT_FILE_PATH']
batch_size = config['DEFAULT']['BATCH_SIZE']
top_k = config['DEFAULT']['TOP_K']

class Server:

  def server_running(self):
    return('Server is running...')

  def get_inference(self):
    inference_json = inference('reader', train_dir, data_pattern, out_file_location, batch_size, top_k)
    print(inference_json)
    return(inference_json)
