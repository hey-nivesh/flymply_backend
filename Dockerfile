FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
# gcc/g++ might be needed for some python packages like bitsandbytes
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=5000
# MODEL_PATH, etc. can be overridden at runtime

# Expose port
EXPOSE 5000

# Run with Gunicorn using the config file
CMD ["gunicorn", "-c", "gunicorn_config.py", "app:app"]
