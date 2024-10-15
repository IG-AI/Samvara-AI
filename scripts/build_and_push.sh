#!/bin/bash
# Build and push Docker image to GCR
IMAGE_NAME="gcr.io/samvara-ai/samvara-docker"

echo "Building Docker image..."
docker build -t $IMAGE_NAME .

echo "Pushing Docker image to GCR..."
docker push $IMAGE_NAME

echo "Docker image pushed successfully!"
