ARG REGION=us-east-1

FROM pytorch/pytorch:latest

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    nginx \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set up Python 3 as the default
RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install \
    gunicorn \
    flask \
    openai-whisper \
    numpy \
    pandas \
    nvgpu \
    psutil \
    nvidia-smi

ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY whisper /opt/program
WORKDIR /opt/program

