FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (only if required)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (better cache)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Env
ENV PYTHONUNBUFFERED=1

# Render sets PORT dynamically, so DO NOT hardcode ENV PORT=5000 here
# EXPOSE is optional in Render but harmless
EXPOSE 5000

# Run Gunicorn (1 worker + longer timeout + bind to Render PORT)
CMD ["sh", "-c", "gunicorn -w 1 -b 0.0.0.0:${PORT} --timeout 120 app:app"]
