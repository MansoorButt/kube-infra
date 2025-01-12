FROM python:3.8-slim

WORKDIR /app

# Install required packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create logs directory
RUN mkdir -p /app/logs

# Copy client code
COPY client.py .

# Default command (will be overridden by k8s job)
CMD ["python", "client.py"]