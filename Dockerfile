# Use a lightweight python image for development and testing the pipeline
FROM python:3.10-slim

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
