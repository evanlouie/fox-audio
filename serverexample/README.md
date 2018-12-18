# Flask server example

This repo contains starter code for a Web API for:

- converting a wav file to a tfrecord
- inference on a wav file
- bulk conversion and inference on multiple wav files

### Requirements

- Trained Tensorflow model
- `inference_model` files (This implies that evaluation and inference must be performed on the trained model)
- Eval features directory (e.g. `audioset_v1_embeddings/eval`). 
**Note:** Copy this directory inside `serverexample`. The path for the eval features is the path to tfrecords that were used to perform evaluation. This path *_must_* be a relative path (e.g. `audioset_v1_embeddings/eval`), and *_not_* the full absolute path (e.g. `C:/User/youtube8m/audioset_v1_embeddings/eval`). If the full path was used during evaluation, eval may need to be performed again (along with inference) using relative paths for the argument `--eval_data_pattern`.
- The csv file with the labels associated with the classifier model (e.g. `class_labels_indices.csv`)

### Additional setup:
1. `cd serverexample`
2. Change the directories in the `environment.properties` file to the appropriate directories.

`TRAINING_DIRECTORY` is the path to the trained model.

`DATA_PATTERN` is the path that stores the tfrecords upon generated.

`OUTPUT_FILE_PATH` is the path to the json output file that contains inference on uploaded file(s).

`CSV_FILE_PATH` is the path to the csv labels file

3. Install the requirements listed on `requirements.txt`. You can use pip for this:
    `pip install -r /path/to/requirements.txt`
4. `curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt`
5. `curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz`


### Run on localhost
1. `cd serverexample`
2. Run the following command in the terminal to start the server: `python -m flask run`

The server should now be running on localhost:5000.

You can use tools like [Fiddler](https://www.telerik.com/fiddler) or [Postman](https://www.getpostman.com/) to call the API.

#### Sample Web page
You can test the server by launching the sample web page to call the various APIs. 
1. Launch a browser
2. Go to localhost:5000
3. Upload wav file(s)
