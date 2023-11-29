Steps to set up endpoint:

Please use pytorch 2.0.0 and torchaudio 2.0.1

- choose proper notebook kernel
- run prepare_model.ipynb
-- check what device is used
- go to container folder
-- run build_and_push.sh "whisper-{device}"
- run endpoint_setup.ipynb


After model is deployed and sagemaker endpoint is up and running

- run test_endpoint.ipynb