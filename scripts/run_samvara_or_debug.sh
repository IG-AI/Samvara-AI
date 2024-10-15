#!/bin/bash

# Log the arguments to a file for debugging
echo "Arguments: $@" >> ~/Samvara-AI/debug.log

# Check if the first argument is -x to run the debugger
if [[ "$1" == "-x" ]]; then
    echo "Running debugger for Samvara-AI..." >> ~/Samvara-AI/debug.log
    python3 ~/Samvara-AI/utils/debugger.py -x
else
    # Otherwise, run the Samvara-AI Docker container
    echo "Running Samvara-AI in Docker..." >> ~/Samvara-AI/debug.log
    ~/Samvara-AI/scripts/run_samvara.sh
fi
