# Copyright 2018 Microsoft All Rights Reserved.
#
# Author NathanielRose
# ==============================================================================

r"""A simple helper script that formats Audioset Inference CSV files to a norma-
lized consumable json to be used in model inference result comparisons. The scr-
pt expects a directory of csv files processed my the Youtube-8m Frame Level mod-
els and outputs a directory of respected json files converted by this script.
Usage:
  # Run a directory of csv files as the input.
  # Specify a targeted output directory for the json files.

  $ python yt8m_inference_json_formatter.py --csv_files /path/to/csv/files \
                                    --json_files /path/to/targed/json/file/directory \

  # Future code will include: Model Metrics, Threshold for Audioset Label Confidence
"""
from __future__ import print_function

import os
import csv
import glob
import json
import pandas as pd
import tensorflow as tf
from collections import deque

flags = tf.app.flags

flags.DEFINE_string(
    "csv_files", None, "Path to directory of YT8m csv inference file scores."
)

flags.DEFINE_string(
    "json_files", None, "Path to target directory for output json files.")

FLAGS = flags.FLAGS


def main(_):
    # check to see if flags have been set, otherwise throw ValueError
    if FLAGS.csv_files:
            if FLAGS.json_files:
                #for csvfile in glob.glob('InferencedModels/*json'):
                csv_files = FLAGS.csv_files
                json_files= FLAGS.json_files
                for filename in os.listdir(csv_files):
                    if filename.endswith(".csv"):
                        print("INPUT FILENAME: " + filename)
                        file_location = str(csv_files + "/" + filename)
                        df = pd.read_csv(filename)
                        audiosetData = []
                        data = {}
                        for index, row in df.iterrows():
                            #Specific to Audiset CSV Inferences
                            rowStr = row['LabelConfidencePairs']
                            rowLabels = rowStr.split()
                            data['VideoId'] = row['VideoId']
                            data['Label_Data'] = {}
                            for newVal in range(0, int(len(rowLabels))): 
                                    #Check to see if the value is a confidence or a Label index
                                    if newVal % 2 == 0:
                                        if(newVal <=1):
                                            x=0
                                        else:
                                            x = newVal/2
                                    insertLabel = str("label_"+str(int(x)))
                                    insertLabelConfidence = "labelConfidenceRate_"+str(int(x))
                                    #Add value to Label_Data array
                                    if newVal % 2 == 0:
                                        data['Label_Data'].update({insertLabel:rowLabels[newVal]})
                                    else:
                                        data['Label_Data'].update({insertLabelConfidence:rowLabels[newVal]})

                            audiosetData.append(data)
                        parsed_filename = filename.rsplit(".", 1)[0]
                        output_location = str(json_files + "/" + parsed_filename + ".json")
                        print("OUTPUT FILENAME: "+output_location)
                        #Write to json file directory
                        with open(output_location, 'w') as jsonOut:
                            json.dump(audiosetData, jsonOut)
                    else:
                        continue
            else:
                raise ValueError(
                    "No directory for json files specified. Please set the --json_files to a path for converted JSON files to be outputted to."
                )
    else:
        raise ValueError(
                    "No directory for csv files specified. Please set the --csv_files to a path for YT8M inference scores."
                )

if __name__ == "__main__":
    tf.app.run()