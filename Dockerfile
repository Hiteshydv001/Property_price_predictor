# --- Stage 1: The "Builder" ---
# This stage will handle cloning the repo and pulling LFS files.
FROM python:3.11-slim as builder

# Install git and git-lfs
RUN apt-get update && apt-get install -y git git-lfs

# Set the working directory
WORKDIR /app

# Initialize an empty git repository and enable LFS
RUN git init && git lfs install

# This is the key: Add your GitHub remote. 
# Replace the URL with your actual GitHub repository URL.
RUN git remote add origin https://github.com/Hiteshydv001/Property_price_predictor.git

# Fetch only the LFS metadata without downloading the actual files yet
RUN git lfs fetch origin main

# Checkout the LFS pointers into the working directory
RUN git lfs checkout

# --- Stage 2: The "Final App" ---
# This is the clean, final container for our application.
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install Python dependencies first (this layer is cached)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and the LFS files from the "builder" stage
COPY . .
COPY --from=builder /app .

# Expose the port Gunicorn will run on
EXPOSE 80

# Define the command to run your app
CMD ["gunicorn", "--workers", "3", "--timeout", "300", "app.main:app", "--bind", "0.0.0.0:80"]