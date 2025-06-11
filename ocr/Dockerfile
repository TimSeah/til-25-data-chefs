# Dockerfile for building the OCR image with standard pre-downloaded PaddleOCR models

#FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
FROM nvcr.io/nvidia/pytorch:25.03-py3

LABEL maintainer="lolkabash"
LABEL description="OCR Engine with PaddleOCR, FastAPI, GPU, and pre-downloaded standard models. Current Time: 2025-05-24 21:54:37 UTC"

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV DEBIAN_FRONTEND=noninteractive

# --- Environment Variables for Model Paths ---
ENV MODELS_BASE_DIR=/opt/paddleocr_models

# Standard Pre-trained Model Paths
ENV DET_MODEL_SUBDIR=det/en/en_PP-OCRv3_det_infer
# Using PP-OCRv4 Rec model
ENV REC_MODEL_SUBDIR=rec/en/en_PP-OCRv4_rec_infer
ENV CLS_MODEL_SUBDIR=cls/en/ch_ppocr_mobile_v2.0_cls_infer
ENV LAYOUT_MODEL_SUBDIR=layout/en/picodet_lcnet_x1_0_fgd_layout_infer

ENV CUDA_VISIBLE_DEVICES=0
WORKDIR /workspace

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

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Create ALL final model directories (target for extraction)
RUN mkdir -p ${MODELS_BASE_DIR}/${DET_MODEL_SUBDIR} && \
    mkdir -p ${MODELS_BASE_DIR}/${REC_MODEL_SUBDIR} && \
    mkdir -p ${MODELS_BASE_DIR}/${CLS_MODEL_SUBDIR} && \
    mkdir -p ${MODELS_BASE_DIR}/${LAYOUT_MODEL_SUBDIR}

# Parent directories for downloaded model extraction (where tar -C extracts to)
RUN mkdir -p ${MODELS_BASE_DIR}/det/en && \
    mkdir -p ${MODELS_BASE_DIR}/rec/en && \
    mkdir -p ${MODELS_BASE_DIR}/cls/en && \
    mkdir -p ${MODELS_BASE_DIR}/layout/en

# Download and extract Detection Model (en_PP-OCRv3_det_infer)
RUN wget https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar -O /tmp/det_model.tar && \
    tar -xvf /tmp/det_model.tar -C ${MODELS_BASE_DIR}/det/en/ && \
    echo "--- Listing contents after DET model extraction ---" && \
    echo "Listing ${MODELS_BASE_DIR}/det/en/:" && \
    ls -lR ${MODELS_BASE_DIR}/det/en/ && \
    echo "Listing ${MODELS_BASE_DIR}/${DET_MODEL_SUBDIR}/:" && \
    ls -l ${MODELS_BASE_DIR}/${DET_MODEL_SUBDIR}/ && \
    rm /tmp/det_model.tar

# Download and extract Recognition Model (en_PP-OCRv4_rec_infer)
# AND download its dictionary from GitHub
RUN wget https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar -O /tmp/rec_model.tar && \
    tar -xvf /tmp/rec_model.tar -C ${MODELS_BASE_DIR}/rec/en/ && \
    rm /tmp/rec_model.tar && \
    echo "Downloading en_dict.txt for recognition model..." && \
    wget https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/en_dict.txt -O ${MODELS_BASE_DIR}/${REC_MODEL_SUBDIR}/en_dict.txt && \
    echo "--- Listing contents after REC model extraction AND en_dict.txt download ---" && \
    echo "Listing ${MODELS_BASE_DIR}/rec/en/:" && \
    ls -lR ${MODELS_BASE_DIR}/rec/en/ && \
    echo "Listing ${MODELS_BASE_DIR}/${REC_MODEL_SUBDIR}/ (expected location of en_dict.txt):" && \
    ls -l ${MODELS_BASE_DIR}/${REC_MODEL_SUBDIR}/

# Download and extract Classification Model (ch_ppocr_mobile_v2.0_cls_infer)
RUN wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar -O /tmp/cls_model.tar && \
    tar -xvf /tmp/cls_model.tar -C ${MODELS_BASE_DIR}/cls/en/ && \
    echo "--- Listing contents after CLS model extraction ---" && \
    echo "Listing ${MODELS_BASE_DIR}/cls/en/:" && \
    ls -lR ${MODELS_BASE_DIR}/cls/en/ && \
    echo "Listing ${MODELS_BASE_DIR}/${CLS_MODEL_SUBDIR}/:" && \
    ls -l ${MODELS_BASE_DIR}/${CLS_MODEL_SUBDIR}/ && \
    rm /tmp/cls_model.tar

# Download and extract Layout Model (picodet_lcnet_x1_0_fgd_layout_infer)
RUN wget https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_infer.tar -O /tmp/layout_model.tar && \
    tar -xvf /tmp/layout_model.tar -C ${MODELS_BASE_DIR}/layout/en/ && \
    echo "--- Listing contents after LAYOUT model extraction ---" && \
    echo "Listing ${MODELS_BASE_DIR}/layout/en/:" && \
    ls -lR ${MODELS_BASE_DIR}/layout/en/ && \
    echo "Listing ${MODELS_BASE_DIR}/${LAYOUT_MODEL_SUBDIR}/ (expected location of inference.pdmodel):" && \
    ls -l ${MODELS_BASE_DIR}/${LAYOUT_MODEL_SUBDIR}/ && \
    rm /tmp/layout_model.tar


# Install Python dependencies
RUN pip install --no-cache-dir -U pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /workspace/src/

# If you have a docker_verify.py script, you can copy and run it here
# COPY docker_verify.py /workspace/docker_verify.py
# RUN python /workspace/docker_verify.py

EXPOSE 5003
CMD ["uvicorn", "src.ocr_server:app", "--host", "0.0.0.0", "--port", "5003"]
