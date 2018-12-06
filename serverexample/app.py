from flask import Flask, request, flash, jsonify, abort, Response, render_template
import os
import server
import io
from scipy.io import wavfile
import uuid

app = Flask(__name__)
app.debug = True

guid = str(uuid.uuid4())

@app.route("/", methods=['POST', 'GET'])
def home():
    if request.method == 'GET':
        return render_template("index.html")
    
    uploaded_file_name = next(iter(request.files))
    uploaded_file = request.files[uploaded_file_name]
    wav_filename = uploaded_file.filename

    uploaded_file.seek(0)
    wav_content = uploaded_file.read()

    server.get_tfrecord_from_file(wav_filename, io.BytesIO(wav_content))
    json = server.get_inf_json()

    #return jsonify(json)
    return render_template("index.html", json=str(json))

@app.route("/inference", methods=['POST'])
def inference():
    #To-do: error handling for min requirement files

    uploaded_files_name_property = next(iter(request.files))
    files = request.files.getlist(uploaded_files_name_property)
    #uploaded_file = request.files[uploaded_file_name]
    #wav_filename = uploaded_file.filename

    for wav_file in files:
        wav_filename = wav_file.filename
        # Get the content from the uploaded wav
        wav_file.seek(0)
        wav_content = wav_file.read()

        server.get_tfrecord_from_file(wav_filename, io.BytesIO(wav_content), guid + "-" + wav_filename + ".tfrecord")
        
        #return jsonify(json)
    json = server.get_inf_json(guid + "*.tfrecord")
    print(str(json))
    return("Finished")
