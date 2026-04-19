FROM mcr.microsoft.com/devcontainers/python:3.10

# Switch to root to install core packages
USER root

# Install system dependencies required by OpenCV (often needed by YOLO backend)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Switch back to vscode user
USER vscode

# Install PyTorch with CUDA 12.1 explicitly to leverage cluster GPUs effectively
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Requirements will be installed via postCreateCommand in devcontainer.json
