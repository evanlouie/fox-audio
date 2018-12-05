from flask import Flask, request, flash, jsonify, abort, Response
import os
import server
import io
from scipy.io import wavfile

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, Flask!"

@app.route("/inference", methods=['POST'])
def inference():
    #To-do: error handling for min requirement files

    uploaded_file_name = next(iter(request.files))
    uploaded_file = request.files[uploaded_file_name]
    wav_filename = uploaded_file.filename

    # Get the content from the uploaded wav
    uploaded_file.seek(0)
    wav_content = uploaded_file.read()

    # rate, signal = wavfile.read(io.BytesIO(wav_content))
    server.get_tfrecord_from_file(wav_filename, io.BytesIO(wav_content))
    json = server.get_inf_json()
    print(json)
    if uploaded_file.filename == '':
            flash('No selected file')
    
    return "Finished."
