# Dockerfile for building the ASR image.

FROM python:3.10-slim

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore

# Explicitly create and set the working directory
RUN mkdir -p /workspace
WORKDIR /workspace

# Installs your dependencies.
# Copy only requirements.txt
RUN pip install -U pip
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt

# Copies specific source files from your local 'src' directory
# (relative to Dockerfile) directly into /workspace.
# These paths must exactly match your local file paths relative to the Dockerfile.
COPY src/asr_server.py /workspace/asr_server.py
COPY src/asr_manager.py /workspace/asr_manager.py
COPY src/__init__.py /workspace/__init__.py

# Set PYTHONPATH to include the /workspace directory, where your modules now reside.
ENV PYTHONPATH="/workspace"

# COPY TRAINED MODEL
# This path is relative to the Docker build context (where Dockerfile is).
# This operation will also be slow if the models directory is part of the huge context,
# but it's a necessary copy.
COPY ../models/asr_finetuned_model_small_aug_VER1/model_and_processor_files /app/model/

# Set the MODEL_DIR environment variable
ENV MODEL_DIR="/app/model"

# Starts your model server.
# 'asr_server.py' (from your local src/asr_server.py) is now directly in /workspace
CMD ["uvicorn", "asr_server:app", "--host", "0.0.0.0", "--port", "5001"]