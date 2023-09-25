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


# model = whisper.load_model("large")

# print(
#     f"Model is {'multilingual' if model.is_multilingual else 'English-only'} "
#     f"and has {sum(np.prod(p.shape) for p in model.parameters()):,} parameters."
# )

def test_model():
    dataset = LibriSpeech("test-clean")
    audio, sample_rate, text, _, _, _ = dataset.dataset[0]

    # Convert the audio data to a Base64-encoded string
    audio_base64 = base64.b64encode(audio.numpy().tobytes()).decode("utf-8")
    # Determine the values for batch_size, num_channels, and num_features
    batch_size = 1
    num_channels = audio.shape[0]  # Number of channels in the audio data
    num_features = audio.shape[1]  # Number of features in the audio data
    
    print(f"num_channels: {num_channels} num_features: {num_features}")
    
    # Prepare the data in JSON format
    data = {
        "audio_base64": audio_base64,
        "sample_rate": sample_rate,
    }

    # Send a POST request to the HTTP service
    url = "http://localhost:8080/invocations"
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=json.dumps(data), headers=headers)

    if response.status_code == 200:
        result = json.loads(response.text)
        print(f"Example Transcription: \n{result}")
    else:
        print(f"Error: {response.status_code}")

if __name__ == "__main__":
    test_model()