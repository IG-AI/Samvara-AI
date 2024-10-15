# Base image with CUDA and cuDNN pre-installed (TensorFlow GPU version)
FROM tensorflow/tensorflow:2.10.0-gpu as base

# Create a non-root user with sudo privileges
ARG USERNAME=samvarauser
ARG USER_UID=1001
ARG USER_GID=$USER_UID

# Install necessary libraries and tools, including locales and apt-utils
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
    apt-utils \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN /usr/bin/python3 -m pip install --upgrade pip

# Set up the locale to en_US.UTF-8
RUN locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8

# Create a new user and set appropriate permissions
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Define Samvara-AI directory as environment variable
ENV SAMVARA_DIR="/home/$USERNAME/Samvara-AI"
ENV PATH="/home/$USERNAME/.local/bin:$PATH"
ENV PYTHONPATH="$SAMVARA_DIR:$PYTHONPATH"  # Set PYTHONPATH here

# Set working directory to Samvara-AI directory
WORKDIR $SAMVARA_DIR

# Copy the requirements file before switching users
COPY requirements.txt $SAMVARA_DIR/requirements.txt

# Switch to the newly created user
USER $USERNAME

# Set up Python environment and create a virtual environment
RUN python3 -m venv /home/$USERNAME/venv \
    && /home/$USERNAME/venv/bin/pip install --upgrade pip \
    && /home/$USERNAME/venv/bin/pip install -r $SAMVARA_DIR/requirements.txt

# Switch back to root to copy scripts
USER root

# Copy the Samvara-AI project files
COPY . $SAMVARA_DIR

# Copy all .sh scripts from the scripts directory and make them executable
RUN chmod +x $SAMVARA_DIR/scripts/*.sh

# Ensure user can write to SAMVARA_DIR
RUN chown -R $USERNAME:$USERNAME $SAMVARA_DIR

# Switch to non-root user to run the alias setup script
USER $USERNAME
RUN bash $SAMVARA_DIR/scripts/setup_alias.sh

# Expose necessary ports
EXPOSE 8888 6006

# Set up a dynamic entrypoint
ENTRYPOINT ["/bin/bash", "-c", "$SAMVARA_DIR/scripts/setup_dynamic_dir.sh && $SAMVARA_DIR/scripts/run_samvara.sh"]
