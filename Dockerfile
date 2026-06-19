# Use a base image with PyTorch and CUDA support
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src /src

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the handler
CMD ["python", "-u", "/src/handler.py"]
