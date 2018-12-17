# Flask server example

This repo contains starter code for a Web API for:

- converting a wav file to a tfrecord
- inference on a wav file
- bulk inference on several wav files

### Requirements

- Visual Studio Code: https://code.visualstudio.com/
- Trained audioset model directory
- Run eval.py on the trained model
- Audioset eval features directory (`audioset_v1_embeddings/eval`). **Note:** Add this directory inside the `serverexample` directory.
- The csv file with the labels that correspond to the classifier model.

### Additional setup:
1. Change the directories in the `environment.properties` file to the directories from your machine
2. `cd serverexample`
3. Install the requirements listed on `requirements.txt`. You can use pip for this.
4. `curl -O https://storage.googleapis.com/audioset/vggish_model.ckpt`
5. `curl -O https://storage.googleapis.com/audioset/vggish_pca_params.npz`


### Run on localhost
1. `cd serverexample`
2. Run the following command in the terminal to start the server: `python -m flask run`

The server should now be running on localhost:5000

You can use tools like Fiddler or Postman to call the API.

#### Sample Web page
You can test the server by launching the sample web page to call the various APIs. 
1. Launch a browser
2. Go to localhost:5000
