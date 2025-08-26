# Use an official Python base image
FROM python:3.11-slim

# Install necessary system dependencies, including bash-completion
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    bash \
    wget \
    vim \
    less \
    git \
    curl \
    bash-completion \
    && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

# Enable bash completion
RUN echo "source /usr/share/bash-completion/bash_completion" >> /etc/bash.bashrc

# Set the working directory inside the container
WORKDIR /app

# Install Jupyter Notebook and your package
RUN pip install --no-cache-dir \
    jupyterlab \
    flexynesis \
    snakemake

# Copy Jupyter notebooks into the container
COPY examples/tutorials /app/notebooks

# Expose Jupyter Notebook port
EXPOSE 8888

# Start with a bash shell by default
CMD ["/bin/bash", "-l"]