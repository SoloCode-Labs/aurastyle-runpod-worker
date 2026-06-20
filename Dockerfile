# Use official RunPod PyTorch base image with CUDA pre-installed
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /

# Set environment variables for huggingface and python
ENV HF_HOME=/runpod-volume/huggingface
ENV PYTHONUNBUFFERED=1

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Reinstall PyTorch with CUDA 12.8 wheel to support Blackwell (sm_120) GPUs
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --force-reinstall

# Copy download script (optional utility, no longer run at build-time to keep image small)
COPY download_model.py .
# RUN python3 download_model.py # Commented out to support Option A: models download dynamically into /cache (Network Volume)

# Copy source code
COPY src /src

# Clear entrypoint to bypass base image's start.sh
ENTRYPOINT []

# Command to run the handler
CMD ["python3", "-u", "/src/handler.py"]

