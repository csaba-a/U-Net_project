# Multi-stage Dockerfile for UNet Training Framework
# This demonstrates production-ready containerization practices

# Stage 1: Base image with CUDA support
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -s /usr/bin/python3.9 /usr/bin/python
RUN ln -s /usr/bin/python3.9 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip

# Stage 2: Development environment
FROM base as development

# Install development dependencies
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipywidgets \
    pytest \
    pytest-cov \
    black \
    flake8 \
    mypy \
    pre-commit \
    tensorboard \
    wandb \
    mlflow

# Stage 3: Production environment
FROM base as production

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/checkpoints /app/logs /app/visualizations

# Set permissions
RUN chmod +x /app/scripts/*.sh

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import torch; print('CUDA available:', torch.cuda.is_available())" || exit 1

# Expose ports for TensorBoard and Jupyter
EXPOSE 6006 8888

# Default command
CMD ["python", "train.py"]

# Stage 4: Inference optimized image
FROM production as inference

# Install additional inference dependencies
RUN pip install --no-cache-dir \
    onnx \
    onnxruntime-gpu \
    tensorrt \
    opencv-python-headless

# Copy only inference-related code
COPY inference.py .
COPY models/ ./models/
COPY utils/ ./utils/

# Create inference script
RUN echo '#!/bin/bash\npython inference.py "$@"' > /app/run_inference.sh && \
    chmod +x /app/run_inference.sh

# Default command for inference
CMD ["/app/run_inference.sh"]

# Stage 5: Testing environment
FROM development as testing

# Copy test files
COPY tests/ ./tests/

# Install test dependencies
RUN pip install --no-cache-dir \
    pytest-xdist \
    pytest-benchmark \
    pytest-mock \
    coverage

# Create test script
RUN echo '#!/bin/bash\npytest tests/ -v --cov=unet_training --cov-report=html' > /app/run_tests.sh && \
    chmod +x /app/run_tests.sh

# Default command for testing
CMD ["/app/run_tests.sh"]

# Stage 6: Documentation environment
FROM base as documentation

# Install documentation dependencies
RUN pip install --no-cache-dir \
    sphinx \
    sphinx-rtd-theme \
    myst-parser \
    sphinx-autodoc-typehints \
    sphinx-gallery

# Copy documentation files
COPY docs/ ./docs/
COPY README.md .

# Build documentation
RUN cd docs && make html

# Serve documentation
EXPOSE 8000
CMD ["python", "-m", "http.server", "8000", "--directory", "docs/_build/html"] 