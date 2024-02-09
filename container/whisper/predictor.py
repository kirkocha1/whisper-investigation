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

import subprocess

import base64
import atexit
import time
import multiprocessing
import psutil
import logging
from datetime import datetime, timedelta


DEADLINE_TIME = datetime.now()
MODEL_NAME = "whisper.pt"

def current_device():
    if torch.cuda.is_available():
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
    else:
        print("CUDA (GPU) is not available.")


def reset_deadline():
    global DEADLINE_TIME
    DEADLINE_TIME = datetime.now() + timedelta(minutes=5)

# Configure logging to log to CloudWatch
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info(f"PYTHON VERSION: {sys.version_info}")

prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")

PID = os.getpid()


def get_worker_processes(parent_pid):
    child_processes = []
    for process in psutil.process_iter(attrs=['pid', 'ppid', 'name']):
        try:
            process_info = process.info
            pid = process_info['pid']
            ppid = process_info['ppid']
            name = process_info['name']
            if ppid == parent_pid:
                child_processes.append({"pid": pid, "name": name})
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    return child_processes


def get_gpu_info():
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            gpu_info = result.stdout
        else:
            gpu_info = f"Error running nvidia-smi: {result.stderr}"
    except Exception as e:
        gpu_info = f"Error: {str(e)}"
    return gpu_info


def obtain_device():
    child_processe_ids = list(map(lambda child : child["pid"], get_worker_processes(os.getppid())))
    worker_core = child_processe_ids.index(PID)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_core)
    if torch.cuda.is_available():    
        device = torch.device("cuda")
        current_device()
        logger.info(f"DEVICE that is used during inference is {device}, worker core: {worker_core}")
    else:
        device = torch.device("cpu")
        logger.info(f"DEVICE that is used during inference is {device}")
    logger.info(f"Pytorch version that is used by service is {torch.__version__}")
    return device


def shutdown_callback():
    logger.info("inference service is shut down")
    torch.cuda.empty_cache()


atexit.register(shutdown_callback)

def cuda_stats():
    memory_stats = torch.cuda.memory_stats()
    total_memory = torch.cuda.get_device_properties(0).total_memory
    gpu_info = get_gpu_info()
    allocated = memory_stats['allocated_bytes.all.current'] / 1024**2
    active = memory_stats['active.all.current'] / 1024**2
    reserved = memory_stats['reserved_bytes.all.current'] / 1024**2
    inactive = memory_stats['inactive_split_bytes.all.current'] / 1024**2

    logger.info(f"Total GPU Memory: {total_memory / 1024**2} MB")
    logger.info(f"Allocated GPU Memory: {allocated} MB")
    logger.info(f"Active GPU Memory: {active} MB")
    logger.info(f"Reserved GPU Memory: {reserved} MB")
    logger.info(f"Inactive GPU Memory: {inactive} MB")
    logger.info(f"GPU info: {gpu_info}")

    return {
            "total_memory": total_memory, 
            "allocated": allocated, 
            "active": active, 
            "reserved": reserved, 
            "inactive": inactive,
            "gpu_info":  gpu_info
        }

def load_model():
    logger.info(f"loading whisper model, PID {PID}, device type {DEVICE.type}")
    model = whisper.load_model(os.path.join(model_path, MODEL_NAME))
    model = model.to(DEVICE)
    logger.info(f"model was loaded now it is synced with device {DEVICE.type}")
    options = whisper.DecodingOptions(language="en", without_timestamps=True, fp16 = False)
    return {'model': model, 'options': options}

DEVICE = obtain_device()
whisper_model = load_model()

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
class AsrService(object):
    model = None  # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        return whisper_model

    @classmethod
    def predict(cls, input, options):
        clf = cls.get_model()
        audio = whisper.pad_or_trim(input.flatten()).to(DEVICE)
        mel = whisper.log_mel_spectrogram(audio).to(DEVICE)
        options = whisper.DecodingOptions(language="en", without_timestamps=True, fp16 = False)
        output = clf["model"].decode(mel, options)
        return str(output.text)
    
    @classmethod
    def memory_stat(cls):
        if torch.cuda.is_available():
            return cuda_stats()
        else:
            print("CUDA (GPU) is not available.")
        

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
            "device": model_info["model"].device.type,
            "pid": PID
        }
        current_time = datetime.now()
        if current_time > DEADLINE_TIME and torch.cuda.is_available():
            metrics = AsrService.memory_stat()
            result.update(metrics)
            reset_deadline()
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

