#!/bin/bash

# Default variables
SCREEN_SESSION=""
SAMVARA_DIR="${SAMVARA_DIR:-/home/$USER/Samvara-AI}"

# Define environment variables for storage paths
MEMORY_DIR="$SAMVARA_DIR/.storage/memory"
CACHE_DIR="$SAMVARA_DIR/.storage/cache"
DATA_DIR="$SAMVARA_DIR/.storage/data"

# Function to clean up orphaned processes and memory
cleanup() {
    echo "Cleaning up resources..."
    DOCKER_PS=$(docker ps -a --filter "name=samvara-ai-gpu" --format "{{.ID}}")
    if [ ! -z "$DOCKER_PS" ]; then
        echo "Stopping running containers..."
        docker stop $DOCKER_PS
        docker rm $DOCKER_PS
    fi
    [ -d "$CACHE_DIR" ] && rm -rf "$CACHE_DIR"/*
    [ -d "$MEMORY_DIR" ] && rm -rf "$MEMORY_DIR"/*
}

# Ensure cleanup happens on script exit or errors
trap cleanup EXIT

# Parse arguments
while [[ "$1" != "" ]]; do
    case "$1" in
        -s | --screen ) shift
                        SCREEN_SESSION="$1"
                        ;;
        -h | --help )   echo "Usage: ./run_in_screen.sh [-s session_name | --screen session_name]"
                        exit 0
                        ;;
        * )             break
    esac
    shift
done

# Display paths
echo "Running Samvara-AI in the following directories:"
echo "Memory: $MEMORY_DIR"
echo "Cache: $CACHE_DIR"
echo "Data: $DATA_DIR"

# Create directories if they don't exist
mkdir -p "$MEMORY_DIR" "$CACHE_DIR" "$DATA_DIR"

# Ensure permissions for directories
sudo chown -R $(id -u):$(id -g) "$MEMORY_DIR" "$CACHE_DIR" "$DATA_DIR"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed or not found in the path."
    exit 1
fi

# Run Samvara-AI in Docker
run_docker() {
    echo "Starting Samvara-AI Docker container..."
    docker run -v "$MEMORY_DIR:/container_path/memory" \
               -v "$CACHE_DIR:/container_path/cache" \
               -v "$DATA_DIR:/container_path/data" \
               --user $(id -u):$(id -g) \
               -it samvara-ai-gpu "$@" || { echo "Error running Samvara-AI"; cleanup; exit 1; }
}

# If -s or --screen is passed, run in screen session
if [ -n "$SCREEN_SESSION" ]; then
    echo "Running Samvara-AI in screen session: $SCREEN_SESSION"
    screen -S "$SCREEN_SESSION" -dm bash -c "$(declare -f run_docker); run_docker"
    echo "Screen session '$SCREEN_SESSION' started."
else
    run_docker
fi

cleanup
