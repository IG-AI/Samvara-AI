import subprocess
import time
from threading import Thread

# Function to log GPU usage
def log_gpu_usage():
    result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE)
    print(result.stdout.decode())

# Function to monitor GPU usage during training
def monitor_gpu_during_training(interval=60):
    while True:
        log_gpu_usage()
        time.sleep(interval)

# Function to start GPU monitoring in a separate thread
def start_gpu_monitoring(interval=60):
    gpu_monitor_thread = Thread(target=monitor_gpu_during_training, args=(interval,))
    gpu_monitor_thread.start()
    return gpu_monitor_thread
