# --- Stage 1: The "Builder" ---
# This stage will handle cloning the repo and pulling LFS files.
FROM python:3.11-slim as builder

# Install git and git-lfs
RUN apt-get update && apt-get install -y git git-lfs

# Set the working directory
WORKDIR /app

# Clone the repository. This also pulls the LFS files automatically.
RUN git clone https://github.com/Hiteshydv001/Property_price_predictor.git .

# --- Stage 2: The "Final App" ---
# This is the clean, final container for our application.
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# --- THIS IS THE FIX ---
# Install the required OpenMP library for LightGBM before installing Python packages
RUN apt-get update && apt-get install -y libgomp1
# --------------------------

# Install Python dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the fully cloned application with LFS files from the "builder" stage
COPY --from=builder /app .

# Expose the port Gunicorn will run on
EXPOSE 80

# Define the command to run your app
CMD ["gunicorn", "--workers", "3", "--timeout", "300", "app.main:app", "--bind", "0.0.0.0:80"]