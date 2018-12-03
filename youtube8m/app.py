from flask import Flask, request, flash, jsonify
import os

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, Flask!"

@app.route("/inference", methods=['POST'])
def inference():
    f = request.files['fieldNameHere']

    f.save(os.path.join('upload', 'testing.wav'))
    if f.filename == '':
            flash('No selected file')
    
    return "Finished."