from locust import HttpUser, task, between
from locust import events
from test_model import LibriSpeech
import base64
import json
import boto3
import datetime
import time
import random
import requests

sagemaker_runtime = boto3.client('sagemaker-runtime')


class Test(HttpUser):
    # wait_time = between(1, 2)  # Wait time between requests (in seconds)
    # Specify your SageMaker endpoint URL
    host = "https://runtime.sagemaker.us-east-1.amazonaws.com"

    @staticmethod
    def total_time(start_time) -> float:
        return int((time.time() - start_time) * 1000)
    
    @task
    def send_inference_request(self):
        dataset = LibriSpeech("test-clean")
        random_number = random.randint(0, 50)
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_maxsize=100)  # Increase the pool size as needed
        session.mount('https://', adapter)
        
        audio, sample_rate, text, _, _, _ = dataset.dataset[random_number]

        # Convert the audio data to a Base64-encoded string
        audio_base64 = base64.b64encode(audio.numpy().tobytes()).decode("utf-8")

        data = {
            "audio_base64": audio_base64,
            "sample_rate": sample_rate,
        }

        request_payload = json.dumps(data)
        
        content_type = 'application/json'
        endpoint_name = 'whisper-gpu-3-endpoint'

        try:
            start_time = time.time()
            asr_response = sagemaker_runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType=content_type,
                Body=request_payload
            )
            
            events.request.fire(
                request_type="POST",
                name="/endpoints/whisper-xlarge-1-gpu-endpoint/invocations",
                response_time=Test.total_time(start_time),
                response_length=len(asr_response),
                response=asr_response
            )

        except Exception as e:
            events.request.fire(
                request_type="POST",
                name="/endpoints/whisper-xlarge-1-gpu-endpoint/invocations",
                response_time=Test.total_time(start_time),
                exception=e,
                response_length=len(asr_response),
                response=asr_response
            )
