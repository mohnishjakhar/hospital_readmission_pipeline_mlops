# ---------------------------------------------------------------------------
# Dockerfile - Healthcare Readmission Prediction Inference Service (FastAPI)
# ---------------------------------------------------------------------------
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

WORKDIR /app

# Install system runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specification and install
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code, models, and docs
COPY src/ /app/src/
COPY models/ /app/models/
COPY dashboard/ /app/dashboard/
COPY docs/ /app/docs/

# Expose REST inference port
EXPOSE 8000

# Health check to ensure microservice readiness
HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start production uvicorn ASGI server
CMD ["uvicorn", "src.ml_serve:app", "--host", "0.0.0.0", "--port", "8000"]
