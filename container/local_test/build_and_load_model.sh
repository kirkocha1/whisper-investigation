#!/usr/bin/env bash

# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
image=$1

if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name>"
    exit 1
fi

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi


# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
region=${region:-us-west-2}
echo $region

login_to_public_ecr() {
    aws ecr get-login-password \
    --region us-east-1 \
    | docker login \
    --username AWS \
    --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
    
}

login_to_public_ecr

docker build -f Dockerfile-model -t ${image} . 
docker tag ${image} model
