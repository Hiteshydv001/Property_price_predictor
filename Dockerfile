# --- Base Image for the Final App ---
# We define the final image first and install dependencies.
# This makes the build process more efficient.
FROM python:3.11-slim as base

# Install the required OpenMP library for LightGBM runtime
RUN apt-get update && apt-get install -y libgomp1

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# --- Builder Stage ---
# This stage will handle cloning the repo and pulling LFS files.
FROM python:3.11-slim as builder

# Install git and git-lfs
RUN apt-get update && apt-get install -y git git-lfs

# Set the working directory
WORKDIR /app

# Clone the repository. This also pulls the LFS files automatically.
RUN git clone https://github.com/Hiteshydv001/Property_price_predictor.git .


# --- Final Stage ---
# This is the final, clean container for our application.
FROM base as final

WORKDIR /app

# Copy the fully cloned application with LFS files from the "builder" stage
COPY --from=builder /app .

# Expose the port Gunicorn will run on
EXPOSE 80

# Define the command to run your app
CMD ["gunicorn", "--workers", "3", "--timeout", "300", "app.main:app", "--bind", "0.0.0.0:80"]