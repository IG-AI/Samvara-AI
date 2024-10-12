# Base image with CUDA and cuDNN pre-installed (TensorFlow GPU version)
FROM tensorflow/tensorflow:2.10.0-gpu

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

# Set up Python environment and virtualenv
RUN pip install --upgrade pip virtualenv

# Create a virtual environment
RUN virtualenv /app/venv

# Activate the virtual environment and install project-specific dependencies
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN /bin/bash -c "source /app/venv/bin/activate && pip install --no-cache-dir --root-user-action=ignore -r requirements.txt"

# Copy the Samvara-AI project files
COPY . /app

# Expose necessary ports
EXPOSE 8888 6006

# Command to run the Samvara-AI training script within the virtual environment
CMD ["/bin/bash", "-c", "source /app/venv/bin/activate && python main.py"]
