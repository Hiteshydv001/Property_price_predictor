#!/usr/bin/env bash
# exit on error
set -o errexit

# Install Python dependencies
pip install -r requirements.txt

# Install Git LFS
apt-get update && apt-get install -y git-lfs

# Pull the large files
git lfs pull