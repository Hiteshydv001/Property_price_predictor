#!/usr/bin/env bash
# exit on error
set -o errexit

# Install Python dependencies
pip install -r requirements.txt

# --- THIS IS THE FIX ---
# Use sudo to grant permissions for installing system packages
sudo apt-get update
sudo apt-get install -y git-lfs
# ---------------------

# Pull the large files
git lfs pull