# Dockerfile for building the OCR image using standard pre-trained models.

# Base image: NVIDIA CUDA image
# For paddlepaddle-gpu==3.0.0 (which uses CUDA 12.x libraries)
# CUDA 12.2.2 and cuDNN 8 provides a compatible environment.
# The first FROM line is ignored, the second one is effective.
FROM us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.2-2.py310
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04


# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV DEBIAN_FRONTEND=noninteractive

# --- Environment Variables for Model Paths ---
# Base directory for all models within the image
ENV MODELS_BASE_DIR=/opt/paddleocr_models

# Specific model directory names for STANDARD models
ENV DET_MODEL_SUBDIR=det/en/en_PP-OCRv3_det_infer
ENV CLS_MODEL_SUBDIR=cls/en/ch_ppocr_mobile_v2.0_cls_infer
ENV LAYOUT_MODEL_SUBDIR=layout/en/picodet_lcnet_x1_0_fgd_layout_infer
# ENV FINETUNED_REC_MODEL_DIR_NAME=my_custom_finetuned_ocrv4_rec # Not used in this config
# ENV REC_MODEL_SUBDIR=rec/en/${FINETUNED_REC_MODEL_DIR_NAME} # Not used in this config

# Default to GPU 0 within the container
ENV CUDA_VISIBLE_DEVICES=0

WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    python3-dev \
    wget \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Update alternatives for python and pip to use python3.10
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create final model directories for STANDARD models
# The 'rec/en' directory might still be used by PaddleOCR for caching default downloaded models.
RUN mkdir -p ${MODELS_BASE_DIR}/${DET_MODEL_SUBDIR} && \
    mkdir -p ${MODELS_BASE_DIR}/rec/en && \
    mkdir -p ${MODELS_BASE_DIR}/${CLS_MODEL_SUBDIR} && \
    mkdir -p ${MODELS_BASE_DIR}/${LAYOUT_MODEL_SUBDIR} && \
    mkdir -p ${MODELS_BASE_DIR}/dicts
    # NO backslash or && on the last command of the RUN block

# Parent directories for downloaded model extraction (DET, CLS, LAYOUT)
# These are temporary staging parent dirs before specific model dirs are populated.
# This can also be a single RUN command.
RUN mkdir -p ${MODELS_BASE_DIR}/det/en && \
    mkdir -p ${MODELS_BASE_DIR}/cls/en && \
    mkdir -p ${MODELS_BASE_DIR}/layout/en
    # NO backslash or && on the last command of the RUN block

# Download and extract BASE PaddleOCR models (Detection, Classification, Layout)
# Detection Model (en_PP-OCRv3_det_infer)
RUN wget https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar -O /tmp/det_model.tar && \
    tar -xvf /tmp/det_model.tar -C ${MODELS_BASE_DIR}/det/en/ && \
    rm /tmp/det_model.tar

# Classification Model (ch_ppocr_mobile_v2.0_cls_infer)
RUN wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar -O /tmp/cls_model.tar && \
    tar -xvf /tmp/cls_model.tar -C ${MODELS_BASE_DIR}/cls/en/ && \
    rm /tmp/cls_model.tar

# Layout Model (picodet_lcnet_x1_0_fgd_layout_infer)
RUN wget https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_infer.tar -O /tmp/layout_model.tar && \
    tar -xvf /tmp/layout_model.tar -C ${MODELS_BASE_DIR}/layout/en/ && \
    rm /tmp/layout_model.tar

# --- Copy your fine-tuned recognition model (SECTION REMOVED/COMMENTED) ---
# This section is intentionally omitted as we are using standard pre-trained models.
# If you had a `my_finetuned_rec_model_output/` directory in your build context,
# the COPY command for it would be here, but it's not needed for this configuration.
# COPY my_finetuned_rec_model_output/ ${MODELS_BASE_DIR}/${REC_MODEL_SUBDIR}/ # Example of what's removed

# Copy your custom character dictionary (SECTION REMOVED/COMMENTED)
# This is not needed when using standard pre-trained English recognition model.
# COPY src/custom_char_dict.txt ${MODELS_BASE_DIR}/dicts/custom_char_dict.txt # Example of what's removed

# Install Python dependencies
RUN pip install --no-cache-dir -U pip setuptools wheel

# Install a NumPy version compatible with older OpenCV/PaddleOCR builds (NumPy 1.x series)
# This should be done BEFORE paddleocr and requirements.txt to ensure compatibility.
RUN pip install --no-cache-dir "numpy<2.0"

# Install paddlepaddle-gpu using the specific index URL
RUN pip install --no-cache-dir paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# Install an older version of paddleocr (e.g., 2.7.x)
RUN pip install --no-cache-dir "paddleocr~=2.7.0"

# Copy and install the rest of your application requirements
COPY requirements.txt .
# If requirements.txt also specifies numpy, the earlier install of "numpy<2.0" will likely satisfy it
# or pip will handle it. If requirements.txt forces numpy>=2.0, that would be a conflict.
RUN pip install --no-cache-dir -r requirements.txt

# Copies your application source files from src/ to /workspace/src/
# This includes ocr_server.py, ocr_manager.py.
COPY src/ /workspace/src/

# Copy and run the verification script (assuming it's in the Docker build context root)
# Ensure docker_verify.py is adapted for standard models if it checks specific model paths.
# COPY docker_verify.py /workspace/docker_verify.py
# RUN python /workspace/docker_verify.py

# Expose the port the app runs on
EXPOSE 5003

# Starts your model server.
# The `src.` prefix for `ocr_server:app` is because WORKDIR is /workspace and files are in /workspace/src/
CMD ["uvicorn", "src.ocr_server:app", "--host", "0.0.0.0", "--port", "5003"]