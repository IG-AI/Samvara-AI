#!/bin/bash

# Function to log error messages and exit
log_error() {
    echo "[ERROR] $1"
    exit 1
}

# Enable the Artifact Registry API if not already enabled
echo "Enabling Artifact Registry API..."
gcloud services enable artifactregistry.googleapis.com || log_error "Failed to enable Artifact Registry API. Please ensure you have the right permissions."

# Update and install dependencies
echo "Updating and installing dependencies..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io || log_error "Failed to install Docker."

# Start and enable Docker
echo "Starting and enabling Docker..."
sudo systemctl start docker
sudo systemctl enable docker

# Authenticate Docker with Google Cloud
echo "Authenticating Docker with Google Cloud..."
gcloud auth configure-docker || log_error "Failed to configure Docker authentication with Google Cloud."

# Set project ID and image details
PROJECT_ID="samvara-ai-minimal-setup"
IMAGE_NAME="samvara-docker"
IMAGE_TAG="latest"

# Pull the Docker image from Google Container Registry (GCR)
echo "Pulling Docker image: gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}"
sudo docker pull gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG} || log_error "Failed to pull Docker image. Ensure the image exists and the Artifact Registry API is enabled."

# Run the Docker container with GPU support
echo "Running the Docker container with GPU support..."
sudo docker run --gpus all -d --name samvara-container \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e TF_FORCE_GPU_ALLOW_GROWTH=true \
  -p 8888:8888 \
  gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG} bash || log_error "Failed to run the Docker container."

# Check if the container is running
echo "Checking if the container is running..."
sudo docker ps || log_error "No running containers found."

# Test TensorFlow GPU access inside the container
echo "Testing TensorFlow GPU access..."
sudo docker exec -it samvara-container python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))" || log_error "Failed to verify TensorFlow GPU access."

# Test PyTorch GPU access inside the container
echo "Testing PyTorch GPU access..."
sudo docker exec -it samvara-container python3 -c "import torch; print(torch.cuda.is_available())" || log_error "Failed to verify PyTorch GPU access."

# Output container logs (optional, can remove if not needed)
echo "Showing container logs..."
sudo docker logs samvara-container
