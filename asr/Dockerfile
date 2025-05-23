# Dockerfile for building the ASR image.


# The base image, an example deep learning VM.
# For a full list, see https://us-docker.pkg.dev/deeplearning-platform-release/gcr.io/
# For info, see https://cloud.google.com/deep-learning-vm/docs/images#supported-frameworks
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

# Copies your source files.
# This will copy src/asr_manager.py (with both classes) and src/asr_server.py
COPY src .

#COPY TRAINED MODEL
# This copies your ASR model to /app/model inside the container
COPY ../models/asr_finetuned_model_small_aug/model_and_processor_files /app/model/

# Set the MODEL_DIR environment variable
# ASRManager will use this if no model_path_override is given during instantiation
ENV MODEL_DIR /app/model

# Starts your model server.
# This assumes 'app' is an instance of FastAPI (or similar) in asr_server.py
CMD uvicorn asr_server:app --port 5001 --host 0.0.0.0