# utils/debugger.py

import os
import subprocess
import logging
from datetime import datetime

# Set up logging to capture output to a file for later analysis
log_file = f"debug_log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
logging.basicConfig(filename=log_file, level=logging.DEBUG, format='%(asctime)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(console_handler)

def run_command(command):
    """Utility to run a system command and log the output."""
    try:
        logging.info(f"Running command: {command}")
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info(f"Command output:\n{result.stdout.decode()}")
        logging.info(f"Command error (if any):\n{result.stderr.decode()}")
        return result.stdout.decode(), result.stderr.decode()
    except subprocess.CalledProcessError as e:
        logging.error(f"Command '{command}' failed with exit code {e.returncode}")
        logging.error(f"Command output:\n{e.output.decode()}")
        logging.error(f"Command error:\n{e.stderr.decode()}")
        return None, e.stderr.decode()

def check_docker_container(container_name):
    """Check if the Docker container exists and get its status."""
    logging.info(f"Checking Docker container: {container_name}")
    command = f"docker ps -a --filter 'name={container_name}' --format '{{{{.Status}}}}'"
    stdout, stderr = run_command(command)
    if stdout:
        logging.info(f"Docker container status: {stdout.strip()}")
    else:
        logging.error(f"Failed to get Docker container status for {container_name}")
    return stdout.strip() if stdout else None

def check_docker_logs(container_name):
    """Check the logs of the Docker container for detailed error messages."""
    logging.info(f"Checking Docker logs for container: {container_name}")
    command = f"docker logs {container_name}"
    stdout, stderr = run_command(command)
    if stdout:
        logging.info(f"Docker logs:\n{stdout.strip()}")
    if stderr:
        logging.error(f"Docker error logs:\n{stderr.strip()}")

def check_file_exists(file_path):
    """Check if a file exists and log the result."""
    if os.path.exists(file_path):
        logging.info(f"File exists: {file_path}")
    else:
        logging.error(f"File does not exist: {file_path}")

def check_docker_image(image_name):
    """Check if the specified Docker image exists."""
    logging.info(f"Checking if Docker image exists: {image_name}")
    command = f"docker images --filter=reference={image_name} --format '{{{{.Repository}}}}:{{{{.Tag}}}}'"
    stdout, stderr = run_command(command)
    if stdout.strip():
        logging.info(f"Docker image found: {stdout.strip()}")
    else:
        logging.error(f"Docker image not found: {image_name}")
    return stdout.strip() if stdout else None

def debug():
    logging.info("Starting Debugger for Samvara-AI")
    
    # Check if the necessary Docker image exists
    image_name = "samvara-ai-gpu"
    check_docker_image(image_name)

    # Check if the necessary Docker container exists
    container_name = "samvara-ai-gpu"
    container_status = check_docker_container(container_name)

    # Check the status of the Samvara run script
    samvara_script_path = "/home/samvarauser/Samvara-AI/scripts/run_samvara.sh"
    check_file_exists(samvara_script_path)

    # Get Docker logs if the container exists
    if container_status:
        check_docker_logs(container_name)

    # Check GPU status if applicable
    check_gpu_status()

    logging.info("Debugging complete. Check the log file for details.")
    print(f"Debugging completed. Check the log file: {log_file}")

def check_gpu_status():
    """Check if the GPU is available and log the details."""
    logging.info("Checking GPU status...")
    command = "nvidia-smi"
    stdout, stderr = run_command(command)
    if stdout:
        logging.info(f"GPU status:\n{stdout.strip()}")
    if stderr:
        logging.error(f"Error while checking GPU status:\n{stderr.strip()}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "-x":
        debug()
    else:
        print("Usage: python debugger.py -x")
