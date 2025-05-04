#!/bin/bash

# Setup script for installing python packages for OpenRLHF-M
# Assumes you are in the activated virtual environment.
# This script follows instructions from sections 4, 5, and 6 of the deployment guide.

#############################
# Set Working Root
#############################

WORKING_DIR="$(pwd)" # Use current working directory

#############################
# 1. Install Core Python Packages (Section 4)
#############################

# Define the core required python packages
PACKAGES="pip wheel packaging setuptools huggingface_hub ring_flash_attn"

echo "Upgrading pip"
pip install --upgrade pip

echo "Installing core python packages: $PACKAGES"
pip install $PACKAGES

echo "Core python package installation complete."


#############################
# 2. Torch Installation (Section 5)
#############################

# Function to get the maximum GPU compute capability detected using nvidia-smi
get_max_compute_capability() {
    if command -v nvidia-smi &> /dev/null; then
        # Get compute capability for all GPUs, sort numerically, get the highest. Handle potential errors.
        MAX_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | sort -n | tail -n 1)
        if [ -n "$MAX_CC" ]; then
            echo "$MAX_CC"
        else
            echo "none" # nvidia-smi query failed or returned empty
        fi
    else
        echo "none" # nvidia-smi not found
    fi
}

echo "Detecting GPU Compute Capability..."
MAX_COMPUTE_CAPABILITY=$(get_max_compute_capability)
echo "Detected Max Compute Capability: $MAX_COMPUTE_CAPABILITY"

# Select torch installation command based on detected hardware
if [[ "$MAX_COMPUTE_CAPABILITY" == 10.0* ]]; then
    echo "NVIDIA B200 (Compute Capability 10.0) detected."
    echo "Installing latest PyTorch compatible with CUDA 12.1+ for B200 support."
    # PyTorch 2.7+ is needed for Blackwell (sm_100). Using the cu121 index url.
    # Ensure the host machine has compatible NVIDIA drivers (e.g., 555+ for B200).
    TORCH_INSTALL_CMD="pip install --no-cache-dir --force-reinstall torch>=2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"

# If B200 not detected, fall back to nvcc check
elif command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \\([0-9]*\\.[0-9]*\\).*/\\1/p')
    echo "Detected CUDA version via nvcc: $CUDA_VERSION"

    # Select torch installation command based on CUDA version via nvcc.
    if [[ "$CUDA_VERSION" == 12* ]]; then
        echo "CUDA 12 (via nvcc) detected. Using PyTorch wheel for CUDA 12.1."
        TORCH_INSTALL_CMD="pip install --no-cache-dir --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    else
        echo "CUDA version $CUDA_VERSION (via nvcc) detected, not CUDA 12. Using default PyTorch installation."
        TORCH_INSTALL_CMD="pip install --no-cache-dir --force-reinstall torch torchvision torchaudio"
    fi

# Fallback if no B200 detected and nvcc isn't present
else
    echo "B200 not detected and nvcc not found. Using default PyTorch installation."
    TORCH_INSTALL_CMD="pip install --no-cache-dir --force-reinstall torch torchvision torchaudio"
fi

# Uninstall existing torch versions first
echo "Uninstalling existing torch, torchvision, torchaudio..."
pip uninstall torch torchvision torchaudio -y

# Execute the torch installation command.
echo "Installing torch packages with command: $TORCH_INSTALL_CMD"
eval $TORCH_INSTALL_CMD


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