#!/bin/bash
# Start the Google Cloud VM
echo "Starting VM instance..."
gcloud compute instances start samvara-vm --zone=europe-west4-a

echo "VM started."
