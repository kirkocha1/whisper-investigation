#!/usr/bin/env bash

# This script shows how to build the Docker image from gpu optimized sagemaker image and push it to ECR to be ready for use
# by SageMaker.

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
image=$1

auth_script=".aws_login.sh"

# Check if the file exists
if [ -f "$auth_script" ]; then
    # If the file exists, execute it
    ./$auth_script
else
    echo "The file $auth_script does not exist."
fi

if [ "$image" == "" ]
then
    echo "Usage: $0 <image-name>"
    exit 1
fi

chmod +x whisper/serve

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

login_to_own_ecr () {
    fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"

    # If the repository doesn't exist in ECR, create it.

    aws ecr describe-repositories --repository-names "${image}" > /dev/null 2>&1

    if [ $? -ne 0 ]
    then
        aws ecr create-repository --repository-name "${image}" > /dev/null
    fi

    # Get the login command from ECR and execute it directly
    aws ecr get-login-password --region "${region}" | docker login --username AWS --password-stdin "${account}".dkr.ecr."${region}".amazonaws.com | echo $fullname
 
}

login_to_public_ecr() {
    aws ecr get-login-password \
    --region us-east-1 \
    | docker login \
    --username AWS \
    --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
    
}

login_to_public_ecr

docker build -t ${image} .

image_name="$(login_to_own_ecr)"
echo "image name $image_name"

docker tag ${image} ${image_name}
docker push ${image_name}
