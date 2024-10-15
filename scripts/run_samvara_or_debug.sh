#!/bin/bash

# Log the arguments to a file for debugging
echo "Arguments: $@" >> ~/Samvara-AI/debug.log

# Ensure SAMVARA_DIR is set
if [ -z "$SAMVARA_DIR" ]; then
    export SAMVARA_DIR="$(cd "$(dirname "$0")/.." && pwd)"
fi

echo "SAMVARA_DIR is set to: $SAMVARA_DIR" >> ~/Samvara-AI/debug.log

# Add SAMVARA_DIR to PYTHONPATH
export PYTHONPATH="$SAMVARA_DIR/utils:$SAMVARA_DIR:$PYTHONPATH"

echo "PYTHONPATH: $PYTHONPATH" >> ~/Samvara-AI/debug.log  # Log PYTHONPATH

# Check if the first argument is -x to run the debugger
if [[ "$1" == "-x" || "$1" == "--debug" ]]; then
    echo "Running debugger for Samvara-AI..." >> ~/Samvara-AI/debug.log
    python3 "$SAMVARA_DIR/utils/debugger.py" "$@"
else
    # Otherwise, run the Samvara-AI Docker container
    echo "Running Samvara-AI in Docker..." >> ~/Samvara-AI/debug.log
    bash "$SAMVARA_DIR/scripts/run_samvara.sh"
fi
