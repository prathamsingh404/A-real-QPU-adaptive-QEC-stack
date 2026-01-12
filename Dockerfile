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
