#!/bin/bash

# Ensure SAMVARA_DIR is set
if [ -z "$SAMVARA_DIR" ]; then
    export SAMVARA_DIR="$(cd "$(dirname "$0")/.." && pwd)"
fi

# Add SAMVARA_DIR to PYTHONPATH
export PYTHONPATH="$SAMVARA_DIR:$PYTHONPATH"

# Print PYTHONPATH for debugging
echo "SAMVARA_DIR set to: $SAMVARA_DIR"
echo "PYTHONPATH: $PYTHONPATH"

# Run the Samvara-AI or Debug script
./scripts/run_samvara_or_debug.sh "$@"
