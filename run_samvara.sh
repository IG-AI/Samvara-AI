#!/bin/bash

# Function to display help message
show_help() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -s, --screen       Run the Docker container inside a screen session."
    echo "  -h, --help         Display this help message and exit."
    exit 0
}

# Parse command line options
while [[ "$1" != "" ]]; do
    case $1 in
        -s | --screen )
            screen_flag=true
            ;;
        -h | --help )
            show_help
            ;;
        * )
            break
            ;;
    esac
    shift
done

# Check if the --screen or -s flag is set
if [[ "$screen_flag" == true ]]; then
    # Start the Docker container inside a screen session
    screen -S samvara-session -dm bash -c "docker run -v /path/to/volume:/container/path:rw --user $(id -u):$(id -g) -it samvara-ai-gpu $@"
    echo "Started Docker container in a screen session named 'samvara-session'."
else
    # Run the Docker container normally
    docker run -v /path/to/volume:/container/path:rw --user $(id -u):$(id -g) -it samvara-ai-gpu "$@"
fi
