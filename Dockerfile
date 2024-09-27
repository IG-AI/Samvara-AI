# Use a base image with TensorFlow and PyTorch
FROM tensorflow/tensorflow:2.12.0-gpu

# Install PyTorch and other dependencies
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

# Install PennyLane (for quantum-inspired models)
RUN pip install pennylane cirq

# Set environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Default command (bash shell)
CMD ["bash"]
