# Base image with CUDA and cuDNN pre-installed (TensorFlow GPU version)
FROM tensorflow/tensorflow:2.10.0-gpu

# Create a non-root user with sudo privileges
ARG USERNAME=samvarauser
ARG USER_UID=1001
ARG USER_GID=$USER_UID

# Install necessary libraries and tools
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    python3-pip \
    sudo \
    && rm -rf /var/lib/apt/lists/*

# Create a new user and set appropriate permissions
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Switch to the new user
USER $USERNAME
WORKDIR /home/$USERNAME/app

# Set up Python environment
RUN pip install --upgrade pip

# Install project-specific dependencies
COPY requirements.txt /home/$USERNAME/app/requirements.txt
RUN pip install -r requirements.txt

# Set up directories with proper permissions
RUN mkdir -p /home/$USERNAME/app/checkpoints \
    && chmod -R 775 /home/$USERNAME/app/checkpoints \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME/app/checkpoints

# Copy the Samvara-AI project files
COPY . /home/$USERNAME/app
RUN chmod -R 775 /home/$USERNAME/app

# Expose necessary ports
EXPOSE 8888 6006

# Command to run the Samvara-AI training script
CMD ["python", "main.py"]

