# Dockerfile for building the ASR image.

# The base image, an example deep learning VM.
FROM python:3.10-slim

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace

# Installs your dependencies.
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copies your source files from the local 'src' directory directly into /workspace
COPY src/. /workspace/ # Note the added '.' to copy contents

# Ensure PYTHONPATH includes the workspace where your modules now reside directly
ENV PYTHONPATH "${PYTHONPATH}:/workspace"

# COPY TRAINED MODEL
# This path should be relative to the Docker build context (where Dockerfile is).
# It copies your ASR model to /app/model inside the container.
# Assuming 'models' directory is a sibling of 'src' and Dockerfile is in 'asr/'
COPY ../models/asr_finetuned_model_small_aug_VER1/model_and_processor_files /app/model/

# Set the MODEL_DIR environment variable
ENV MODEL_DIR /app/model

# Starts your model server.
# 'asr_server.py' is now directly in /workspace
CMD ["uvicorn", "asr_server:app", "--host", "0.0.0.0", "--port", "5001"]