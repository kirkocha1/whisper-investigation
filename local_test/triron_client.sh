#!/bin/sh

docker pull nvcr.io/nvidia/tritonserver:23.10-py3-sdk
docker run -it --rm --net=host nvcr.io/nvidia/tritonserver:23.10-py3-sdk