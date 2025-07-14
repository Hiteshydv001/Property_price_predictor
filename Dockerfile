# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install git and git-lfs (the correct way for Docker)
RUN apt-get update && apt-get install -y git git-lfs

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# This is the crucial step: pull the large files into the container image
RUN git lfs pull

# Make port 80 available to the world outside this container
EXPOSE 80

# Define the command to run your app using Gunicorn
CMD ["gunicorn", "--workers", "3", "--timeout", "300", "app.main:app", "--bind", "0.0.0.0:80"]