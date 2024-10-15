#!/bin/bash

# Define environment variables for storage paths
SAMVARA_DIR="${SAMVARA_DIR:-$HOME/Samvara-AI}"
MEMORY_DIR="$SAMVARA_DIR/.storage/memory"
CACHE_DIR="$SAMVARA_DIR/.storage/cache"
DATA_DIR="$SAMVARA_DIR/.storage/data"

# Parse flags
SCREEN_SESSION=""

while [[ "$1" != "" ]]; do
    case "$1" in
        -s | --screen ) shift
                        SCREEN_SESSION="$1"
                        ;;
        -h | --help )   echo "Usage: ./run_script.sh [-s session_name | --screen session_name]"
                        exit 0
                        ;;
        * )             break
    esac
    shift
done

# Ensure directories exist
mkdir -p "$MEMORY_DIR" "$CACHE_DIR" "$DATA_DIR"

# Run Docker container
if [ -n "$SCREEN_SESSION" ]; then
    echo "Running in screen session: $SCREEN_SESSION"
    screen -S "$SCREEN_SESSION" -dm docker run -v "$MEMORY_DIR:/container_path/memory" \
                                              -v "$CACHE_DIR:/container_path/cache" \
                                              -v "$DATA_DIR:/container_path/data" \
                                              -it samvara-ai-gpu
else
    docker run -v "$MEMORY_DIR:/container_path/memory" \
               -v "$CACHE_DIR:/container_path/cache" \
               -v "$DATA_DIR:/container_path/data" \
               -it samvara-ai-gpu
fi
