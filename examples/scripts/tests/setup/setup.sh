#!/bin/bash

# Setup script for installing python packages for OpenRLHF-M
# Assumes you are in the activated virtual environment.
# This script follows instructions from sections 4, 5, and 6 of the deployment guide.

#############################
# Set Working Root
#############################

WORKING_DIR="/data"

#############################
# 1. Install Core Python Packages (Section 4)
#############################

# Define the core required python packages
PACKAGES="pip wheel packaging setuptools"

echo "Installing core python packages: $PACKAGES"
pip install $PACKAGES

echo "Core python package installation complete."


#############################
# 2. Torch Installation (Section 5)
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
# 3. Install vLLM and OpenRLHF (Section 6)
#############################

# Install vLLM explicitly
echo "Installing vLLM==0.7.3"
pip install vllm==0.7.3

# Set repository directory in WORKING_DIR
REPO_DIR="$WORKING_DIR/OpenRLHF-M"

if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning OpenRLHF-M repository into $REPO_DIR"
    git clone https://github.com/the-laughing-monkey/OpenRLHF-M.git "$REPO_DIR"
fi

echo "Changing directory to repository: $REPO_DIR"
cd "$REPO_DIR"

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