# Base image with CUDA and cuDNN pre-installed (TensorFlow GPU version)
FROM tensorflow/tensorflow:2.10.0-gpu

# Create a non-root user with sudo privileges
ARG USERNAME=samvarauser
ARG USER_UID=1001
ARG USER_GID=$USER_UID

# Install necessary libraries and tools, including locales
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    python3-pip \
    sudo \
    locales \
    && rm -rf /var/lib/apt/lists/*

# Set up the locale to en_US.UTF-8
RUN locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8

# Create a new user and set appropriate permissions
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Switch to the non-root user
USER $USERNAME
WORKDIR /home/$USERNAME/Samvara-AI

# Set up Python environment
RUN pip install --upgrade pip

# Install project-specific dependencies
COPY requirements.txt /home/$USERNAME/Samvara-AI/requirements.txt
RUN pip install -r requirements.txt

# Copy the Samvara-AI project files
COPY . /home/$USERNAME/Samvara-AI

# Expose necessary ports
EXPOSE 8888 6006

# Set the entrypoint script to run_samvara.sh for further setup
ENTRYPOINT ["/home/samvarauser/Samvara-AI/scripts/run_samvara.sh"]

COPY scripts/run_samvara.sh /home/samvarauser/Samvara-AI/scripts/run_samvara.sh
