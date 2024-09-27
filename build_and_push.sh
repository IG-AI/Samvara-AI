#!/bin/bash

# Build the Docker image
echo "Building the Docker image..."
docker build -t samvara-docker .

# Tag the Docker image for GCR
echo "Tagging the Docker image for GCR..."
docker tag samvara-docker gcr.io/samvara-ai-minimal-setup/samvara-docker:latest

# Authenticate Docker with Google Cloud
echo "Authenticating Docker with Google Cloud..."
gcloud auth configure-docker

# Push the Docker image to GCR
echo "Pushing the Docker image to GCR..."
docker push gcr.io/samvara-ai-minimal-setup/samvara-docker:latest

# Verify the image was pushed successfully
echo "Docker image pushed successfully to GCR: gcr.io/samvara-ai-minimal-setup/samvara-docker:latest"
