#!/bin/bash
# Check if a screen session named 'samvara_training' is already running
if screen -list | grep -q "samvara_training"; then
  echo "A screen session named 'samvara_training' is already running."
else
  # Start a new screen session named 'samvara_training' and run the Python script
  echo "Starting a new screen session named 'samvara_training'..."
  screen -dmS samvara_training bash -c "python3 main.py; exec bash"
  echo "Samvara model training is running in a screen session."
fi
