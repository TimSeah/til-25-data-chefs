# Base Python image
FROM python:3.10-slim

WORKDIR /app

# Install missing system libraries for OpenCV and its dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    # ^^^ CHANGED THIS LINE from libgthread-2.0-0
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch 
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

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