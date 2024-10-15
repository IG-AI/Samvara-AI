#!/bin/bash

# Starting Samvara-AI Docker container with the required settings
echo "Starting Samvara-AI..."
docker run -v ~/Samvara-AI/.storage/memory:/container_path/memory \
           -v ~/Samvara-AI/.storage/cache:/container_path/cache \
           -v ~/Samvara-AI/.storage/data:/container_path/data \
           --user $(id -u):$(id -g) \
           -it samvara-ai-gpu
