# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import io
import json
import os
import signal
import sys
import traceback

import flask
import pandas as pd
import whisper
import numpy as np
import torch
import base64

import logging

# Configure logging to log to CloudWatch
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger.info(f"DEVICE that is used during inference is {DEVICE}")

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
class AsrService(object):
    model = None  # Where we keep the model when it's loaded
    is_loaded = False


    @classmethod
    def get_model(cls):
        if not cls.is_loaded:
            model = whisper.load_model(os.path.join(model_path, 'model/whisper-gpu.pt'))
            model = model.to(DEVICE)
            cls.options = whisper.DecodingOptions(language="en", without_timestamps=True, fp16 = False)
            cls.model = model
            cls.is_loaded = True
        logger.info(f'whisper model has been loaded to this device: {cls.model.device.type}')
        return {'model': cls.model, 'options': cls.options}

    @classmethod
    def predict(cls, input, options):
        clf = cls.get_model()
        audio = whisper.pad_or_trim(input.flatten()).to(DEVICE)
        mel = whisper.log_mel_spectrogram(audio)
        options = whisper.DecodingOptions(language="en", without_timestamps=True, fp16 = False)
        output = clf["model"].decode(mel, options)
        return str(output.text)
        

def input_fn(request):
    data = json.loads(request.data)
    audio_base64 = data.get("audio_base64")
    sample_rate = data.get("sample_rate")
    audio_bytes = base64.b64decode(audio_base64)
    # Convert the audio bytes to a numpy array
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).copy().astype(np.float32) 
    audio_tensor = torch.from_numpy(audio_np)
    return audio_tensor, sample_rate


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route("/ping", methods=["GET"])
def ping():
    model_info = AsrService.get_model()  # Retrieve model information
    if model_info is not None:
        result = {
            "model_loaded": True,
            "device": model_info["model"].device.type
        }
        status = 200
    else:
        result = {"model_loaded": False}
        status = 404

    return flask.jsonify(result), status


@app.route("/invocations", methods=["POST"])
def transformation():
    if flask.request.content_type == "application/json":
        audio_tensor, sample_rate = input_fn(flask.request)
        transcribe = AsrService.predict(audio_tensor, sample_rate)
        result = json.dumps({"results": transcribe})
        return flask.Response(response=result, status=200, mimetype="application/json")
    else:
        return flask.Response(
            response="This predictor only supports JSON data", 
            status=415, 
            mimetype="text/plain"
        )

