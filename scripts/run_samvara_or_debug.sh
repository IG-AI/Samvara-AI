#!/bin/bash

# Log the arguments to a file for debugging
echo "Arguments: $@" >> ~/Samvara-AI/debug.log

# Call the debugger or run Samvara-AI based on arguments
if [[ "$1" == "-x" || "$1" == "--debug" ]]; then
    echo "Running debugger for Samvara-AI..." >> ~/Samvara-AI/debug.log
    python3 $SAMVARA_DIR/utils/debugger.py "$@"
elif [[ "$1" == "-s" || "$1" == "--screen" ]]; then
    echo "Running Samvara-AI on screen: $2" >> ~/Samvara-AI/debug.log
    python3 $SAMVARA_DIR/utils/debugger.py "$@"
else
    # Otherwise, run Samvara-AI normally
    echo "Running Samvara-AI..." >> ~/Samvara-AI/debug.log
    $SAMVARA_DIR/scripts/run_samvara.sh
fi
