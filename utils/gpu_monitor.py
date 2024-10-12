# utils/gpu_monitor.py

import subprocess
import time
from threading import Thread
import logging

# Function to log GPU usage
def log_gpu_usage():
    """Log the current GPU usage using nvidia-smi."""
    try:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
        logging.info(result.stdout.decode())
    except FileNotFoundError:
        logging.warning("nvidia-smi not found. Ensure NVIDIA drivers are installed and nvidia-smi is available.")

# Function to monitor GPU usage during training at regular intervals
def monitor_gpu_during_training(interval=60):
    """Monitor GPU usage at the specified interval."""
    while True:
        log_gpu_usage()
        time.sleep(interval)

# Function to start GPU monitoring in a separate thread
def start_gpu_monitoring(interval=60):
    """Start monitoring GPU usage in a separate thread."""
    gpu_monitor_thread = Thread(target=monitor_gpu_during_training, args=(interval,))
    gpu_monitor_thread.daemon = True  # Daemonize thread so it will not block the main program from exiting
    gpu_monitor_thread.start()
    return gpu_monitor_thread
