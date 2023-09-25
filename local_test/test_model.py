import os
import multiprocessing
import numpy as np
import torch
import pandas as pd
import whisper
import torchaudio
import sagemaker
import time
from tqdm.notebook import tqdm
import base64
import requests
import json

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class LibriSpeech(torch.utils.data.Dataset):
    """
    A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """
    def __init__(self, split="test-clean", device=DEVICE):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=os.path.expanduser("~/.cache"),
            url=split,
            download=True,
        )
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        assert sample_rate == 16000
        audio = whisper.pad_or_trim(audio.flatten()).to(self.device)
        mel = whisper.log_mel_spectrogram(audio)
        
        return (mel, text)
    

def test_model():
    dataset = LibriSpeech("test-clean")
    audio, sample_rate, text, _, _, _ = dataset.dataset[0]

    # Convert the audio data to a Base64-encoded string
    audio_base64 = base64.b64encode(audio.numpy().tobytes()).decode("utf-8")
    
    data = {
        "audio_base64": audio_base64,
        "sample_rate": sample_rate,
    }

    url = "http://localhost:8080/invocations"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=json.dumps(data), headers=headers)

    if response.status_code == 200:
        result = json.loads(response.text)
        print(f"Example Transcription: \n{result}")
    else:
        print(f"Error: {response.status_code}")


def load_ignore():
            # content_type = 'application/json'
        # endpoint_name = 'whisper-xlarge-1-gpu-endpoint'

        # response = sagemaker_runtime.invoke_endpoint(
        #     EndpointName=endpoint_name,
        #     ContentType=content_type,
        #     Body=request_payload
        # )
        # response_body = response['Body'].read()
        # response_data = json.loads(response_body)

        # Process the response_data as per your application's needs
        # print(response_data)                                       
        # status_code = response['ResponseMetadata']['HTTPStatusCode']

    pass

if __name__ == "__main__":
    test_model()