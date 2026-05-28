# Reproducible scientific execution container for AdaptiveQEC
FROM python:3.11-slim

LABEL maintainer="AdaptiveQEC Research Team"
LABEL description="Hermetic execution environment for reproducible quantum error correction benchmarks"

# Install system build dependencies for C++ matching libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy dependency configuration
COPY pyproject.toml .

# Install pinned core numerical & quantum libraries to eliminate version drift
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    stim==1.16.0 \
    pymatching==2.4.0 \
    numpy==2.4.6 \
    scipy==1.17.1 \
    qiskit==2.5.2 \
    qiskit-ibm-runtime==0.49.0

# Copy full repository
COPY . .

# Install package in editable mode with dev dependencies
RUN pip install --no-cache-dir -e ".[dev]"

# Runtime budget: ~3 minutes for full benchmark validation
ENV RUNTIME_BUDGET_MINUTES=3

# Default entrypoint runs reproduction pipeline
CMD ["make", "reproduce"]
