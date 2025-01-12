FROM python:3.8-slim

WORKDIR /app

# Install required packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create logs directory
RUN mkdir -p /app/logs

# Copy server code
COPY server.py .

# Expose the port
EXPOSE 58000

# Default command (will be overridden by k8s deployment)
CMD ["python", "server.py"]