# Gunshot Detection using AudioSet and YouTube8m Starter Code

This repository contains the code and helper tools necessary to create an LSTM model based on the
AudioSet data set and the YouTube8m starter code.

## Packages

This repository contains 3 top level packages: `audioset`, `youtube_8m`, and `microsoft`.

### AudioSet

The `audioset` package contains code required to utilize the VGGish audio features AudioSet was based
on.

### YouTube8m

The `youtube_8m` package contains the code pertaining to the actual model which will be generated
on the AudioSet data set.

### Microsoft

The `microsoft` package contains any custom code and helper scripts which can prove useful through
the modelling process.

## Quickstart

This repository is tested against Python `3.6.6`, TensorFlow does not support >= `3.7.x`

Assuming `python` currently points to a `3.6.6` installation:

```bash
# Setup a project level python installation in .env
pip install --upgrade pip
pip install --upgrade virtualenv
virtualenv .env

# Activate/inject virtualenv. To disable run `deactivate`
source .env/bin/activate
```

If you are on a machine with a GPU available, add the `tensorflow-gpu` package to utilize it.
We do not include it by default as it causes runtime errors if on a host that does not have a GPU.

```bash
# Install `tensorflow-gpu` if available on machine
pip install --upgrade tensorflow-gpu
```

Install project dependencies and download required `vggish` files

```bash
# Downgrade to setuptools==39.1.0 (Tensorflow requirement)
pip install setuptools==39.1.0
pip install -r requirements.txt
curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt
curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz
```

Download and extract the files associated with AudioSet

```bash
curl -O http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv
curl -O http://storage.googleapis.com/us_audioset/youtube_corpus/v1/features/features.tar.gz
tar xvzf features.tar.gz
```

### Training an LSTM model

```sh
# Ensure the output directory for your model exists
mkdir -p output/lstm

# Wait till the model training reaches an acceptable loss level. Typically you want to train till 0.01.
# Once an acceptable loss level is reached, use <ctrl-c> to exit the script.
python youtube_8m/retrain.py \
  --frame_features \
  --model=LstmModel \
  --feature_names=audio_embedding \
  --feature_sizes=128 \
  --train_data_pattern=audioset_v1_embeddings/bal_train/*.tfrecord \
  --train_dir=output/lstm \
  --base_learning_rate=0.001 \
  --num_epochs=1000
```

### Evaluating

You will need to evaluate the model against the `eval` training set provided from AudioSet.

```sh
# Backup the original model directory*
now=`date +%Y-%m-%d.%H:%M:%S`
tar cvzf lstm-${now}.tar.gz output/lstm

# Run eval
python youtube_8m/eval.py \
  --eval_data_pattern=audioset_v1_embeddings/eval/*.tfrecord \
  --train_dir=output/lstm \
  --run_once
```

\*note: you will not be able to run `train.py` against the model directory again after eval. Make sure to backup the
directory prior to running eval.

### Inference / Prediction

To do inference, the model will require the movie to be converted to 10 second WAV files; for this guide, we assume that
the `*.wav` files are in a directory called `movie_wav_files`

```sh
# Ensure output directory for movie TFRecord files exists
mkdir -p output/data_prep/movie_as_vggish

# Convert our movie *.wav files to vggish sequence examples
python microsoft/vggish_inference.py \
  --tf_directory=output/data_prep/movie_as_vggish \
  --checkpoint=vggish_model.ckpt \
  --pca_params=vggish_pca_params.npz \
  --labels_file=class_labels_indices.csv \
  --subdirectory=movie_wav_files

# Run inference against our now vggish converted movie wav files
python youtube_8m/inference.py \
  --output_file=predictions.csv \
  --input_data_pattern=output/data_prep/movie_as_vggish/*.tfrecord \
  --train_dir=output/lstm

# Print the results (only showing explosion predictions; which are associated to labels 426-431)
cat predictions.csv | grep -P '(\,|\s)?(426|427|428|429|430|431)\s'
```

### Transfer Learning

To perform transfer learning, you will need to acquire checkpoint files from a previously trained LSTM model. The following files are required from a pre-trained model:

`checkpoint`
`model.ckpt-0.data-00000-of-00001`
`model.ckpt-0.index`
`model.ckpt-0.meta`

and optionally, later checkpoint files:

`model.ckpt-310.data-00000-of-00001`
`model.ckpt-310.index`
`model.ckpt-310.meta`

To execute the transfer.py, run the following command:

```
python microsoft\transfer.py \
  --frane_features \
  --model=LstmModel \
  --feature_names=audtio_embedding \
  --feature_sizes=128 \
  --train_data_pattern=output/data_prep/movie_as_vggish/*.tfrecord \
  --train_dir=output/new_lstm \
  --base_learning_rate=0.001 \
  --export_model_steps=100 \
  --base_checkpoint=lstm/dir/model.ckpt-310 \
  --start_new_model

```

You can then proceed with evaluating and predicting the results as described above.
