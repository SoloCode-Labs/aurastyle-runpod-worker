# Use official RunPod PyTorch base image with CUDA pre-installed
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /

# Set environment variables for huggingface and python
ENV HF_HOME=/cache/huggingface
ENV PYTHONUNBUFFERED=1

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy download script (optional utility, no longer run at build-time to keep image small)
COPY download_model.py .
# RUN python3 download_model.py # Commented out to support Option A: models download dynamically into /cache (Network Volume)

# Copy source code
COPY src /src

# Command to run the handler
CMD ["python3", "-u", "/src/handler.py"]
