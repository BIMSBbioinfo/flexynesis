# Use an official Python base image
FROM python:3.11-slim

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    bash \
    wget \
    && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Set the working directory inside the container
WORKDIR /app

# Install Jupyter Notebook and your package
RUN pip install --no-cache-dir \
    jupyterlab \
    flexynesis 

# Copy Jupyter notebooks into the container
COPY examples/tutorials /app/notebooks

# Expose Jupyter Notebook port
EXPOSE 8888

# Start with a bash shell by default
CMD ["bash"]

