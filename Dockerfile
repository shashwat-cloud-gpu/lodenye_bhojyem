# Dockerfile for SynapseFS: Linux POSIX FUSE & Benchmarking Environment
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies & libfuse
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    fuse \
    libfuse-dev \
    pkg-config \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Configure FUSE permissions for non-root / container access
RUN echo 'user_allow_other' >> /etc/fuse.conf

WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt /workspace/
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source repository
COPY . /workspace/
RUN pip3 install -e .

CMD ["python3", "-m", "unittest", "discover", "-s", "tests"]
