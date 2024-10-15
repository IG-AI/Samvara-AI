# utils/debugger.py

import os
import subprocess
import random
import tensorflow as tf

def check_directory_permissions(path):
    if not os.path.exists(path):
        return f"Directory {path} does not exist."
    elif not os.access(path, os.W_OK):
        return f"Permission denied: {path} is not writable."
    return f"Directory {path} is accessible and writable."

def check_docker_container_exists(container_name):
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if result.stdout.decode().strip() == container_name:
            return f"Container with the name {container_name} already exists."
        return f"Container {container_name} does not exist."
    except Exception as e:
        return f"Error checking Docker container: {str(e)}"

def check_docker_running():
    try:
        result = subprocess.run(["systemctl", "is-active", "docker"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.stdout.decode().strip() == "active":
            return "Docker service is running."
        return "Docker service is not running."
    except Exception as e:
        return f"Error checking Docker service: {str(e)}"

def check_python_packages(container_name, package_name):
    try:
        result = subprocess.run(
            ["docker", "exec", container_name, "pip", "show", package_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            return f"Package {package_name} is missing inside the container {container_name}."
        return f"Package {package_name} is installed in container {container_name}."
    except Exception as e:
        return f"Error checking package {package_name}: {str(e)}"

def check_gpu_availability():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        return f"Available GPUs: {len(gpus)}"
    return "No GPU detected."

def check_docker_volume(container_name, volume_path):
    try:
        result = subprocess.run(
            ["docker", "exec", container_name, "ls", volume_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            return f"Volume {volume_path} not mounted correctly in {container_name}: {result.stderr.decode().strip()}"
        return f"Volume {volume_path} is mounted correctly in {container_name}."
    except Exception as e:
        return f"Error checking Docker volume {volume_path}: {str(e)}"

def check_docker_image_integrity(image_name):
    try:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.RootFS.Layers}}", image_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            return f"Error inspecting Docker image {image_name}: {result.stderr.decode().strip()}"
        layers = result.stdout.decode().strip()
        if layers:
            return f"Docker image {image_name} integrity is intact."
        return f"Docker image {image_name} has issues with layers."
    except Exception as e:
        return f"Error checking Docker image integrity: {str(e)}"

def randomize_directory_permissions(path):
    try:
        permissions = [0o777, 0o755, 0o700, 0o400, 0o000]  # Full, read/write, read-only, etc.
        random_permission = random.choice(permissions)
        os.chmod(path, random_permission)
        return f"Set random permissions {oct(random_permission)} for {path}."
    except Exception as e:
        return f"Error setting random permissions for {path}: {str(e)}"

def random_command_execution(container_name):
    random_commands = [
        "ls /", "whoami", "python --version", "cat /proc/cpuinfo", "nvidia-smi"
    ]
    random_command = random.choice(random_commands)
    try:
        result = subprocess.run(
            ["docker", "exec", container_name, random_command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            return f"Command {random_command} failed in {container_name}: {result.stderr.decode().strip()}"
        return f"Command {random_command} succeeded in {container_name}:\n{result.stdout.decode().strip()}"
    except Exception as e:
        return f"Error executing random command {random_command}: {str(e)}"

def random_file_operations(container_name, base_path):
    try:
        file_path = f"{base_path}/test_file_{random.randint(1, 10000)}.txt"
        # Randomly create or delete
        if random.choice([True, False]):
            result = subprocess.run(
                ["docker", "exec", container_name, "touch", file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            if result.returncode != 0:
                return f"Failed to create file {file_path} in {container_name}: {result.stderr.decode().strip()}"
            return f"File {file_path} created successfully in {container_name}."
        else:
            result = subprocess.run(
                ["docker", "exec", container_name, "rm", "-f", file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            if result.returncode != 0:
                return f"Failed to delete file {file_path} in {container_name}: {result.stderr.decode().strip()}"
            return f"File {file_path} deleted successfully in {container_name}."
    except Exception as e:
        return f"Error during file operation in {container_name}: {str(e)}"

def check_tensorflow_env():
    try:
        result = subprocess.run(
            ["python", "-c", "'import tensorflow as tf; print(tf.config.list_physical_devices())'"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            return f"TensorFlow environment check failed: {result.stderr.decode().strip()}"
        return f"TensorFlow environment is correctly configured:\n{result.stdout.decode().strip()}"
    except Exception as e:
        return f"Error checking TensorFlow environment: {str(e)}"

# Add to debugger log
print(check_tensorflow_env())

def test_random_http_request():
    import requests
    urls = ["https://www.google.com", "https://www.github.com", "https://www.tensorflow.org"]
    random_url = random.choice(urls)
    try:
        response = requests.get(random_url)
        if response.status_code == 200:
            return f"Successfully connected to {random_url}."
        return f"Failed to connect to {random_url}."
    except Exception as e:
        return f"Error during HTTP request to {random_url}: {str(e)}"

# Add to debugger log
print(test_random_http_request())

def check_file_access(container_name, file_path):
    try:
        # Test write access
        result = subprocess.run(
            ["docker", "exec", container_name, "echo", "'test'", ">", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            return f"Write access failed for {file_path} in {container_name}: {result.stderr.decode().strip()}"
        
        # Test read access
        result = subprocess.run(
            ["docker", "exec", container_name, "cat", file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            return f"Read access failed for {file_path} in {container_name}: {result.stderr.decode().strip()}"
        return f"File access test succeeded for {file_path} in {container_name}."
    except Exception as e:
        return f"Error during file access test for {file_path}: {str(e)}"

# Add to debugger log
print(check_file_access("samvara-container", "/container_path/memory/test_file.txt"))

def random_tensorflow_operations():
    try:
        random_operations = [
            "tf.matmul(tf.ones((3, 3)), tf.ones((3, 3)))",
            "tf.reduce_sum(tf.random.normal([1000]))",
            "tf.linalg.inv(tf.random.normal([3, 3]))"
        ]
        random_operation = random.choice(random_operations)
        result = subprocess.run(
            ["python", "-c", f"'import tensorflow as tf; result = {random_operation}; print(result)'"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            return f"Random TensorFlow operation {random_operation} failed: {result.stderr.decode().strip()}"
        return f"Random TensorFlow operation {random_operation} succeeded:\n{result.stdout.decode().strip()}"
    except Exception as e:
        return f"Error performing random TensorFlow operation: {str(e)}"

# Add to debugger log
print(random_tensorflow_operations())

def check_docker_env_vars(container_name):
    try:
        result = subprocess.run(
            ["docker", "exec", container_name, "printenv"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            return f"Error checking environment variables in {container_name}: {result.stderr.decode().strip()}"
        return f"Environment variables in {container_name}:\n{result.stdout.decode().strip()}"
    except Exception as e:
        return f"Error checking Docker environment variables: {str(e)}"

# Add to debugger log
print(check_docker_env_vars("samvara-container"))

def check_gpu_memory_usage():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        if result.returncode != 0:
            return f"Error checking GPU memory usage: {result.stderr.decode().strip()}"
        return f"GPU memory usage:\n{result.stdout.decode().strip()}"
    except Exception as e:
        return f"Error checking GPU memory usage: {str(e)}"

# Add to debugger log
print(check_gpu_memory_usage())


# Running all checks in one place
def run_debugger():
    print("Starting Debugger for Samvara-AI")

    # Check directories
    print(check_directory_permissions("/container_path/memory"))
    print(check_directory_permissions("/container_path/cache"))
    print(check_directory_permissions("/container_path/data"))

    # Check Docker status
    print(check_docker_running())
    print(check_docker_container_exists("samvara-container"))

    # Check Python package inside container
    print(check_python_packages("samvara-container", "scikit-learn"))

    # Check GPU availability
    print(check_gpu_availability())

    # Docker image and volume checks
    print(check_docker_image_integrity("samvara-ai-gpu"))
    print(check_docker_volume("samvara-container", "/container_path/memory"))
    print(check_docker_volume("samvara-container", "/container_path/cache"))
    print(check_docker_volume("samvara-container", "/container_path/data"))

    # Random tests
    print(randomize_directory_permissions("/tmp/test_memory"))
    print(random_command_execution("samvara-container"))
    print(random_file_operations("samvara-container", "/container_path/memory"))


    # Test TensorFlow environment setup
    print(check_tensorflow_env())

    # Test random network communication
    print(test_random_http_request())

    # Test read/write access on Docker mounted volumes
    print(check_file_access("samvara-container", "/container_path/memory/test_file.txt"))

    # Perform random TensorFlow operations
    print(random_tensorflow_operations())

    # Check Docker environment variables
    print(check_docker_env_vars("samvara-container"))

    # Check GPU memory usage
    print(check_gpu_memory_usage())


if __name__ == "__main__":
    run_debugger()
