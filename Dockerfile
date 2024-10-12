# Base image with TensorFlow 2.12.0 and GPU support
FROM tensorflow/tensorflow:2.12.0-gpu

# Install necessary libraries and tools
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA Container Toolkit (for GPU usage in containers)
RUN curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add - && \
    curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu18.04/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list && \
    apt-get update && apt-get install -y nvidia-container-toolkit && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up Python environment
RUN pip install --upgrade pip

# Install project-specific dependencies
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install -r requirements.txt

# Copy the Samvara-AI project files
COPY . /app

# Expose necessary ports
EXPOSE 8888 6006

# Command to run the Samvara-AI training script
CMD ["python", "main.py"]
