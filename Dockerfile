# ============================================================
# Stage 1: Model Export (heavy — includes PyTorch)
# ============================================================
FROM python:3.11-slim AS model-builder

WORKDIR /build

# Install export dependencies
COPY requirements-export.txt ./
RUN pip install --no-cache-dir -r requirements-export.txt

# Export the embedding model to ONNX
COPY scripts/export_model.py ./scripts/
RUN python scripts/export_model.py --output ./models/onnx


# ============================================================
# Stage 2: Runtime (lean — no PyTorch)
# ============================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# System deps for faiss
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Install Python runtime dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the exported ONNX model from the builder stage
COPY --from=model-builder /build/models/onnx ./models/onnx

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/

# Copy sample data (will be overridden by volume mount in production)
COPY data/ ./data/

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); exit(0 if r.status_code == 200 else 1)"

# Run the API server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
