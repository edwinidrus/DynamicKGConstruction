# =============================================================================
# DynamicKGConstruction - Dockerfile
# =============================================================================
# Multi-stage build for optimal image size
#
# Build: docker build -t dynamickg .
# Run:   docker run -p 8000:8000 dynamickg
# =============================================================================

FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# =============================================================================
# Builder stage - install dependencies
# =============================================================================
FROM base as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# =============================================================================
# Production stage
# =============================================================================
FROM base as production

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

# Create output directories
RUN mkdir -p /app/build_docling /app/chunks_output /app/kg_output /app/uploads && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Default environment variables
ENV HOST=0.0.0.0 \
    PORT=8000 \
    OUTPUT_DIR=/app/build_docling \
    CHUNKS_OUTPUT_DIR=/app/chunks_output \
    KG_OUTPUT_DIR=/app/kg_output \
    UPLOAD_DIR=/app/uploads \
    LLM_PROVIDER=ollama \
    LLM_MODEL=qwen2.5:32b \
    EMBEDDINGS_MODEL=nomic-embed-text \
    OLLAMA_BASE_URL=http://ollama:11434

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
