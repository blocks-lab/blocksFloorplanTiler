# Multi-stage Dockerfile for Blocks Floorplan Tiler Service
# Optimized for Azure Container Apps deployment

# Stage 1: Builder - Install dependencies
FROM python:3.11-slim as builder

# Install system dependencies required for Pillow and PyMuPDF
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libwebp-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime - Lightweight final image
FROM python:3.11-slim

# Install runtime libraries only (smaller footprint)
RUN apt-get update && apt-get install -y \
    libjpeg62-turbo \
    libpng16-16 \
    libwebp7 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create app directory
WORKDIR /app

# Copy application code
COPY app.py .
COPY worker.py .
COPY pdf_annotation.py .

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=8000

# Expose port (Azure Container Apps will inject PORT env var)
EXPOSE 8000

# Run the application
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
