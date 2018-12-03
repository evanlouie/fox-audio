import os 
import numpy as np
import pandas as pd
import json
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import librosa
from Config import Config
from keras import losses, models, optimizers
from keras.activations import relu, softmax
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras.layers import (Convolution2D, GlobalAveragePooling2D, BatchNormalization, Flatten,
                          GlobalMaxPool2D, MaxPool2D, concatenate, Activation, Input, Dense)
from keras.utils import Sequence, to_categorical
from keras import backend as K
import argparse

def main(_):
    outputData = {}
    outputData['modelType'] ='keras_gunshot_v'
    outputData['modelVersion'] = args.model_folder
    outputData['inferenceData'] = []

    for filename in os.listdir(args.wav_dir):
        if filename.endswith(".wav"):
            pred_list = []
            print("Scoring: " + filename)
            for i in range(config.n_folds):
                model = get_2d_conv_model(config)
                model.load_weights(args.model_folder + 'best_%d.h5' % i)
                X = np.empty(shape=(1, config.dim[0], config.dim[1], 1))
                input_length = config.audio_length
                data, _ = librosa.core.load(
                    args.wav_dir + filename, sr=config.sampling_rate, res_type="kaiser_fast")

                # Random offset / Padding
                if len(data) > input_length:
                    max_offset = len(data) - input_length
                    offset = np.random.randint(max_offset)
                    data = data[offset:(input_length+offset)]
                else:
                    if input_length > len(data):
                        max_offset = input_length - len(data)
                        offset = np.random.randint(max_offset)
                    else:
                        offset = 0
                        data = np.pad(data, (offset, input_length - len(data) - offset), "constant")

                data = librosa.feature.mfcc(data, sr=config.sampling_rate, n_mfcc=config.n_mfcc)
                data = np.expand_dims(data, axis=-1)
                X[0,] = data
                pred = model.predict(X)
                pred_list.append(pred)
            prediction = np.ones_like(pred_list[0])
            for pred in pred_list:
                prediction = prediction*pred
            prediction = prediction**(1./len(pred_list))
            pred = pd.DataFrame(prediction)
            df = pred
            df.columns = ['Other', 'Gunshot_or_gunfire']
            jsondata = {}
            jsondata['labelData'] = []
            jsondata['file'] = filename
            jsondata['labelData']+=({'label':'Other', 'confidence':float(df['Other'][0])}, {'label':'gunshot', 'confidence': float(df['Gunshot_or_gunfire'][0])}, {'label':'pred_label','confidence':float(prediction[:, 1][0])})
            outputData['inferenceData'].append(jsondata)

    with open(args.json_filepath, 'w') as outfile:
        json.dump(outputData, outfile)

def get_2d_conv_model(config):

    nclass = config.n_classes

    inp = Input(shape=(config.dim[0], config.dim[1], 1))
    x = Convolution2D(32, (4, 10), padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Convolution2D(32, (4, 10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Convolution2D(32, (4, 10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Convolution2D(32, (4, 10), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = MaxPool2D()(x)

    x = Flatten()(x)
    x = Dense(64)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    out = Dense(nclass, activation=softmax)(x)

    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(config.learning_rate)

    model.compile(
        optimizer=opt, loss=losses.categorical_crossentropy, metrics=['acc'])
    return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        'Infer. ex: python inference_2dcnn --model_folder="2dcnn/" --wav_file="test/deadpool1_00-01-30.000.wav" --json_filepath="infer.txt"')
    parser.add_argument(
        "--model_folder", help="directory to store models", type=str, default=R'2dcnn/')
    parser.add_argument("--wav_file", help="path to a wav file", type=str,
                        default=R'test/deadpool1_00-01-30.000.wav')
    parser.add_argument(
        "--json_filepath", help="directory to store predictions", type=str, default=R'infer.json')
    parser.add_argument(
        "--wav_dir", help="Target directory of wav files. Will default to 'test/' if none is passed.", default=R'test/',)
    args = parser.parse_args()


    config = Config(sampling_rate=44100, audio_duration=2, n_folds=10,
                    learning_rate=0.001, use_mfcc=True, n_mfcc=40)

#    args = parser
    main(args)
