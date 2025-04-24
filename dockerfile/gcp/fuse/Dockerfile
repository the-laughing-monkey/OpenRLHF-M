FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

# Upgrade pip
RUN pip install --upgrade pip

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    vim \
    build-essential \
    net-tools \
    iputils-ping \
    netcat \
    lsof \
    fuse \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install gcsfuse for GCS bucket mounting
RUN curl -L https://github.com/GoogleCloudPlatform/gcsfuse/releases/download/v1.1.0/gcsfuse_1.1.0_amd64.deb > gcsfuse.deb && \
    dpkg -i gcsfuse.deb && \
    rm gcsfuse.deb

# Install Python dependencies
RUN pip install --no-cache-dir \
    wheel \
    packaging \
    setuptools \
    huggingface_hub \
    ray[default] \
    vllm==0.7.3 \
    flash-attn --no-build-isolation \
    wandb

# Clone OpenRLHF-M repository
RUN git clone https://github.com/the-laughing-monkey/OpenRLHF-M.git /app/OpenRLHF-M

# Install OpenRLHF-M
WORKDIR /app/OpenRLHF-M
RUN pip install -e .

# The training script is already in the repository at:
# /app/OpenRLHF-M/examples/scripts/tests/train_grpo_ray_qwen2_5_vl_mathv60k_multinode_gcp.sh

# Create directories for GCS mounting
RUN mkdir -p /mnt/gcs-cache /mnt/gcs-datasets

# Set working directory
WORKDIR /app/OpenRLHF-M

# Default command
CMD ["/bin/bash"]
