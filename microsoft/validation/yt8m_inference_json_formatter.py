# Copyright 2018 Microsoft All Rights Reserved.
#
# Active Learning Team
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
import json
import argparse
import pandas as pd
from collections import deque


def main(_):
    # Check to see if flags have been set, otherwise throw ValueError
    if args.csv_dir:
        # make directory for json files if directory not set
        if not os.path.exists(json_dir):
            os.makedirs(json_dir)
        # Iterate through each csv file in the directory
        csv_files = args.csv_dir
        json_files = json_dir
        for filename in os.listdir(csv_files):
            if filename.endswith(".csv"):
                print("INPUT FILENAME: " + filename)
                df = pd.read_csv(filename)
                audiosetData = []
                # Iterate through each file label data
                for index, row in df.iterrows():
                    # Validate Audiset CSV Inferences
                    data = {}
                    rowStr = row["LabelConfidencePairs"]
                    rowLabels = rowStr.split()
                    data["VideoId"] = row["VideoId"]
                    data["Label_Data"] = {}
                    # Iterate through each LabelConfidentPair value seperated by space and convert to json record
                    for newVal in range(0, int(len(rowLabels))):
                        is_even = newVal % 2 == 0
                        # Check to see if the string in the current row is a confidence or a Label index
                        if is_even:
                            x = 0 if newVal <= 1 else newVal / 2
                        insertLabel = str("label_" + str(int(x)))
                        insertLabelConfidence = "labelConf_" + str(int(x))
                        # Add value to Label_Data list
                        if is_even:
                            data["Label_Data"].update({insertLabel: rowLabels[newVal]})
                        else:
                            data["Label_Data"].update(
                                {insertLabelConfidence: rowLabels[newVal]}
                            )
                    audiosetData.append(data)
                # print(audiosetData)
                parsed_filename = filename.rsplit(".", 1)[0]
                output_location = str(json_files + "/" + parsed_filename + ".json")
                print("OUTPUT FILENAME: " + output_location)
                # Write to json file in output directory
                with open(output_location, "w") as jsonOut:
                    json.dump(audiosetData, jsonOut)
            else:
                continue

    else:
        raise ValueError(
            "No directory for csv files specified. Please set the --csv_files to a path for YT8M inference scores."
        )


if __name__ == "__main__":

    def parse_args():
        """Parse out the required arguments to run as script"""
        parser = argparse.ArgumentParser(
            description="Convert Audioset csv inference scores to a normalized json"
        )
        parser.add_argument(
            "csv_dir",
            help="Target directory of Audioset Inference CSV files. Will default to inputCSV if none is passed.",
            default="inputCsv",
        )
        parser.add_argument(
            "--json_dir",
            help="Output path for formatted json files. Will be created if does not exist (default: outputJson)",
            default="outputJson",
        )
        return parser.parse_args()

    # Parse user input
    args = parse_args()
    csv_dir = args.csv_dir
    json_dir = args.json_dir
    main(args)
