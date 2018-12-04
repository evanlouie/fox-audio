from flask import Flask, request, flash, jsonify
import os
import server

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, Flask!"

@app.route("/inf", methods=['POST'])
def inf():
    #result = server2.test()
    #server.get_tfrecord()
    ##s = Server()
    #s.get_tfrecord()
    server.get_tfrecord()
    return "Inference API"

@app.route("/inference", methods=['POST'])
def inference():
    #server.get_tfrecord()
    f = request.files['fieldNameHere']

    f.save(os.path.join('upload', 'testing.wav'))
 
    if f.filename == '':
            flash('No selected file')
    
    return "Finished."