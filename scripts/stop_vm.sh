#!/bin/bash
# Stop the Google Cloud VM
echo "Stopping VM instance..."
gcloud compute instances stop samvara-vm --zone=europe-west4-a

echo "VM stopped."
