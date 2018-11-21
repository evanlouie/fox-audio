# Model Inference Comparison and Validationtarter Code

The scripts in this directory provide tools to automate the comparison of models using a normalized json and scripts that help to format model inferences for comparison. Once compared use the metrics to identify which model requires additional train data.


## Inference Scripts with Normalization

This repository contains 2 inference scripts that format the model inferences to a normalized json.
-  [`inference_json.py`](inference_json.py) - is dependent on the Youtube8m model directory. Pass the `--json_out` argument to your script and the inference will be formatted.
- [`k_inference_json.py`](k_inference_json.py) - is dependent on the Keras custom Gunshot model directory. Currently accepts individual input files for inferencing.

## Model Comparison

The Model Inference Comparison and Validation script can be executed using the following arguments below:

> `python mc&v.py --inference_score_dir --canonical_json --model_index_json --compare_video --compare_label`

### `inference_score_dir`

This is the directory of model inferences where each json file has been normalized in the format below:

``` json
{
  "VideoId": "some-identifier.wav",
  "Label_Data": [
    {
      "label_0" "427",
      "labelConf_0": "0.4131"
    },
    {
      "label_1": "213",
      "labelConf_1": ".3121"
    },
    {
      "label_2": "0",
      "labelConf_2": ".2421"
    }
  ]
}
```

### `canonical_json`

This is a canonical validation json file that stores an array of string labels associated with respective audio frames. In this example a frame is a 10 second interval and the videoId is reflective of the frame.

``` json
[
  {
    "movie": "Deadpool1",
    "data": [
      {
        "Label_Array": "gunshot, speech",
        "VideoId": "deadpool1_00-07-50.000"
      },
      {
        "Label_Array": "gunshot",
        "VideoId": "deadpool1_00-13-00.000"
      },
      {
        "Label_Array": "gunshot",
        "VideoId": "deadpool1_00-11-00.000"
      }
    ]
  },
  {
    "movie": "Deadpool2",
    "data": [
      {
        "Label_Array": "gunshot",
        "VideoId": "deadpool2_00-02-00.000"
      },
      {
        "Label_Array": "gunshot",
        "VideoId": "deadpool2_00-03-20.000"
      },
      {
        "Label_Array": "gunshot",
        "VideoId": "deadpool2_00-04-20.000"
      },
      {
        "Label_Array": "gunshot",
        "VideoId": "deadpool2_00-33-10.000"
      }
    ]
  }
]
```

### `model_index_json`

The model index json stores each respective model index labels based on a knowledge graph for scoring the accuracy of the model.

``` json
[
  {
    "model": "AudioSet",
    "model_index": [
      {
        "model_label_str": "gunshot",
        "model_label_vals": "427, 428, 429, 430",
        "kg_mid": "/m/032s66, /m/04zjc, /m/02z32qm, /m/0_1c"
      },
      {
        "model_label_str": "explosion",
        "model_label_vals": "426",
        "kg_mid": "/m/014zdl"
      }
    ]
  },
  {
    "model": "KerasGunshot_CustomModel1",
    "model_index": [
      {
        "model_label_str": "gunshot",
        "model_label_vals": "gunshot",
        "kg_mid": "/m/032s66, /m/04zjc, /m/02z32qm, /m/0_1c"
      },
      {
        "model_label_str": "other",
        "model_label_vals": "other",
        "kg_mid": "n/a"
      },
      {
        "model_label_str": "pred_label",
        "model_label_vals": "pred_label",
        "kg_mid": "n/a"
      }
    ]
  }
]
```

### `compare_video`

The string that you wish to compare model accuracy on a specific movie. The movie must be already entered in the `canonical_json`

### `compare_label`

A string with the label that you wish to compare model accuracy on a specific sound.

## Future features:

- Move model index to directory base
- Persisted Dataframes
- Validation Cells for Gunshot (Inputs for Validation)
- Migrate to a script