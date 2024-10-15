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

# Set up Python environment
RUN pip install --upgrade pip

# Define Samvara-AI directory as environment variable
ENV SAMVARA_DIR="/home/$USERNAME/Samvara-AI"

# Set working directory to Samvara-AI directory
WORKDIR $SAMVARA_DIR

# Install project-specific dependencies
COPY requirements.txt $SAMVARA_DIR/requirements.txt
RUN pip install -r requirements.txt

# Copy the Samvara-AI project files
COPY . $SAMVARA_DIR
ENV PYTHONPATH="$SAMVARA_DIR:$PYTHONPATH"

# Switch to root for setting permissions
USER root
RUN mkdir -p $SAMVARA_DIR/.storage/memory \
    && mkdir -p $SAMVARA_DIR/.storage/cache \
    && mkdir -p $SAMVARA_DIR/.storage/data \
    && mkdir -p $SAMVARA_DIR/checkpoints \
    && chmod -R 755 $SAMVARA_DIR/.storage \
    && chmod -R 755 $SAMVARA_DIR/checkpoints

# Make all scripts executable under root
RUN chmod +x $SAMVARA_DIR/scripts/*.sh

# Copy the alias setup script and make it executable
COPY scripts/setup_alias.sh $SAMVARA_DIR/scripts/setup_alias.sh
RUN chmod +x $SAMVARA_DIR/scripts/setup_alias.sh

# Switch back to non-root user to run the alias setup script
USER $USERNAME
RUN bash $SAMVARA_DIR/scripts/setup_alias.sh

# Expose necessary ports
EXPOSE 8888 6006

# Set up a dynamic entrypoint
ENTRYPOINT ["/bin/bash", "-c", "$SAMVARA_DIR/scripts/setup_dynamic_dir.sh && $SAMVARA_DIR/scripts/run_samvara.sh"]
