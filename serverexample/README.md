# Flask server example

This repo contains starter code for a Web API for:

- converting a wav file to a tfrecord
- inference on a wav file
- bulk inference on several wav files

### Requirements

- Visual Studio Code: https://code.visualstudio.com/
- Trained audioset model directory
- Run eval.py on the trained model
- Audioset eval features directory


### To run on localhost

1. Change the directories in the `environment.properties` file to the directories from your machine
2. cd to the serverexample directory
3. Install the requirements listed on `requirements.txt`. You can use pip for this.
4. Run the following command in the terminal to start the server: `python -m flask run`




### Setup
