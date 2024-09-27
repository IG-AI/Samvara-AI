#!/bin/bash

# Function to log error messages and exit
log_error() {
    echo "[ERROR] $1"
    exit 1
}

# Check if Artifact Registry API is already enabled
echo "Checking if Artifact Registry API is enabled..."
API_STATUS=$(gcloud services list --enabled --filter="artifactregistry.googleapis.com" --format="value(config.name)")

if [ "$API_STATUS" == "artifactregistry.googleapis.com" ]; then
    echo "Artifact Registry API is already enabled."
else
    echo "Artifact Registry API is not enabled. Attempting to enable it..."
    
    # Attempt to enable Artifact Registry API with the current account
    gcloud services enable artifactregistry.googleapis.com

    # Check if enabling the API succeeded
    if [ $? -eq 0 ]; then
        echo "Artifact Registry API enabled successfully."
    else
        echo "[ERROR] Failed to enable Artifact Registry API with the current service account."

        # Attempt to log in with a user account to enable the API
        echo "Attempting to authenticate with a Google Cloud user account to enable the API..."

        gcloud auth login || log_error "User authentication failed. Please log in with a Google Cloud user account."

        # Set the authenticated user account as the active account
        gcloud config set account $(gcloud auth list --filter=status:ACTIVE --format="value(account)") || log_error "Failed to set the active account."

        # Attempt to enable the Artifact Registry API again
        gcloud services enable artifactregistry.googleapis.com || log_error "Failed to enable Artifact Registry API even with user account."

        echo "Artifact Registry API enabled successfully with the user account."

        # Optionally, switch back to the Compute Engine service account
        echo "Switching back to the Compute Engine service account..."
        gcloud config set account 1037346874620-compute@developer.gserviceaccount.com || log_error "Failed to switch back to the Compute Engine service account."

        echo "Switched back to the Compute Engine service account."
    fi
fi

echo "Proceeding with the rest of your setup..."
# Rest of your script can continue here, for example:
# - Pull Docker image
# - Run Docker container
# etc.
