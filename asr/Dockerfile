# Dockerfile for building the ASR image with GPU support.

# Use an NVIDIA CUDA base image with CUDA 12.2.2.
#FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore

# Install Python 3.10 and pip.
# Ubuntu 22.04 (the base of this CUDA image) includes Python 3.10.
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Explicitly create and set the working directory
RUN mkdir -p /workspace
WORKDIR /workspace

# Installs your dependencies.
# NOTE: CRITICAL - Ensure your requirements.txt specifies GPU-compatible versions
# of your machine learning libraries compatible with CUDA 12.2.
# For example:
# - For PyTorch: Find the correct install command for CUDA 12.2 on https://pytorch.org/get-started/locally/
#   (e.g., pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 or similar for 12.2)
# - For TensorFlow: Use a GPU-enabled package (e.g., tensorflow[and-cuda]) and ensure compatibility with CUDA 12.2.
# Copy only requirements.txt
RUN python3 -m pip install -U pip
COPY requirements.txt /workspace/requirements.txt
RUN python3 -m pip install --no-cache-dir -r /workspace/requirements.txt

# Copies specific source files from your local 'src' directory
# (relative to Dockerfile) directly into /workspace.
# These paths must exactly match your local file paths relative to the Dockerfile.
COPY src/asr_server.py /workspace/asr_server.py
COPY src/asr_manager.py /workspace/asr_manager.py
COPY src/__init__.py /workspace/__init__.py

# Set PYTHONPATH to include the /workspace directory, where your modules now reside.
ENV PYTHONPATH="/workspace"

# COPY TRAINED MODEL
# This path is relative to the Docker build context (where Dockerfile is - i.e., til-25-data-chefs/asr/).
COPY models_OLD/asr_finetuned_model_small_aug/model_and_processor_files /app/model/

# Set the MODEL_DIR environment variable
ENV MODEL_DIR="/app/model"

# Starts your model server.
# 'asr_server.py' (from your local src/asr_server.py) is now directly in /workspace
CMD ["uvicorn", "asr_server:app", "--host", "0.0.0.0", "--port", "5001"]