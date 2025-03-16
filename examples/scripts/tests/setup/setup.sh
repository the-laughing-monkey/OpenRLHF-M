#!/bin/bash

# Setup script for installing python packages for OpenRLHF-M
# Assumes you are in the activated virtual environment.
# This script follows instructions from sections 4, 5, and 6 of the deployment guide.

#############################
# 1. Torch Installation (Section 5)
#############################

# Detect CUDA version using nvcc if available.
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
else
    CUDA_VERSION="none"
fi

echo "Detected CUDA version: $CUDA_VERSION"

# Select torch installation command based on CUDA version.
if [[ "$CUDA_VERSION" == 12* ]]; then
    echo "CUDA 12 detected. Using simple installation for torch."
    TORCH_INSTALL="pip install torch torchvision torchaudio"
else
    if [ -d "/usr/local/cuda-12.7" ]; then
        echo "CUDA 12 not detected via nvcc, but /usr/local/cuda-12.7 exists. Using complicated installation for torch with CUDA 12.1 wheels."
        TORCH_INSTALL="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    else
        echo "CUDA version does not match expected setups. Falling back to default torch installation."
        TORCH_INSTALL="pip install torch torchvision torchaudio"
    fi
fi

echo "Installing torch packages with command: $TORCH_INSTALL"
# Execute the torch installation command.
eval $TORCH_INSTALL

#############################
# 2. Install Additional Python Packages (Section 4)
#############################

# Define the remaining required python packages
PACKAGES="kdown loralib loguru lightning-utilities jsonlines itsdangerous isort grpcio docker-pycreds dill blinker av absl-py tensorboard qwen_vl_utils pandas multiprocess levenshtein latex2sympy2_extended gitdb flask math-verify gitpython wandb transformers torchmetrics deepspeed datasets bitsandbytes accelerate transformers_stream_generator peft optimum openrlhf"

echo "Installing additional packages: $PACKAGES"
pip install $PACKAGES

echo "Core python package installation complete."

#############################
# 3. Install vLLM and OpenRLHF (Section 6)
#############################

# Install vLLM explicitly
echo "Installing vLLM==0.7.3"
pip install vllm==0.7.3

# Determine the repository root directory
if command -v git &> /dev/null; then
    REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null)
fi

if [ -z "$REPO_ROOT" ]; then
    # Fallback: assume this script is in examples/scripts/tests/setup and navigate up to the repo root
    REPO_ROOT=$(cd "$(dirname "$0")/../../../.." && pwd)
fi

echo "Changing directory to repository root: $REPO_ROOT"
cd "$REPO_ROOT"

# Install OpenRLHF (without the vLLM extra)
echo "Installing OpenRLHF package"
pip install .

# Install ray with default extras to ensure dashboard dependencies
echo "Installing ray[default]"
pip install 'ray[default]'

# Install flash-attention
echo "Installing flash-attn with no build isolation"
pip install flash-attn --no-build-isolation

echo "Setup complete." 