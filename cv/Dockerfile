# Base Python image
FROM nvcr.io/nvidia/pytorch:25.03-py3

WORKDIR /app

# Install missing system libraries for OpenCV and its dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install a specific, compatible set of PyTorch 2.7 libraries for GPU (CUDA 12.4)
# Note: PyTorch provides wheels for major CUDA versions. The cu124 wheel is compatible with the CUDA 12.x toolkit.
#RUN pip install --no-cache-dir \
 #   torch==2.7.0 \
  #  torchvision==0.22.0 \
   # torchaudio==2.7.0 \
    #--index-url https://download.pytorch.org/whl/cu124

# Copy your application code and requirements file
COPY ./requirements.txt ./requirements.txt 
COPY ./src/cv_manager.py ./cv_manager.py
COPY ./src/cv_server.py ./cv_server.py
COPY ./src/best.pt ./best.pt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port and run command
EXPOSE 5002
CMD ["uvicorn", "cv_server:app", "--host", "0.0.0.0", "--port", "5002"]