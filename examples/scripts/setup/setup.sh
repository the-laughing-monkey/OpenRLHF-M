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
    echo "Attempting to install PyTorch 2.7.0 with CUDA 12.8 for B200 support."
    # PyTorch 2.7.0 is needed for Blackwell (sm_100). Using the cu128 index url.
    # Ensure the host machine has compatible NVIDIA drivers (e.g., 555+ for B200).
    TORCH_INSTALL_CMD="pip install --no-cache-dir --force-reinstall torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128"

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
# 3. Build and Install vLLM from Source (compatible with current PyTorch)
#############################

##echo "Building vLLM from source against installed PyTorch..."
## Remove any pre-installed wheels that pin older torch versions
##pip uninstall -y vllm xformers intel_extension_for_pytorch >/dev/null 2>&1 || true
## Install vLLM directly from GitHub, forcing a source build so that the
## C++/CUDA extensions are compiled against the just-installed torch headers.
##VLLM_COMMIT=${VLLM_COMMIT:-main}
##pip install --no-cache-dir --no-binary :all: --no-build-isolation --no-deps \
##    "git+https://github.com/vllm-project/vllm.git@$VLLM_COMMIT"

#############################
# 4. Install OpenRLHF and its Dependencies
#############################

# Set repository directory in WORKING_DIR
REPO_DIR="$WORKING_DIR/OpenRLHF-M"

if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning OpenRLHF-M repository into $REPO_DIR"
    git clone https://github.com/the-laughing-monkey/OpenRLHF-M.git "$REPO_DIR"
fi

echo "Changing directory to repository: $REPO_DIR"
cd "$REPO_DIR"

# Conditional vLLM and OpenRLHF installation
if [[ "$MAX_COMPUTE_CAPABILITY" == 10.0* ]]; then
    echo "NVIDIA B200 (Blackwell) detected. PyTorch 2.7.0 should be installed."
    
    echo "Uninstalling any existing vLLM to ensure a clean build from source..."
    pip uninstall -y vllm || true # Continue if vllm is not installed

    # Create a temporary directory for cloning and building vLLM
    VLLM_TMP_DIR=$(mktemp -d -t vllm_build-XXXXXXXXXX)
    echo "Cloning latest vLLM repository into $VLLM_TMP_DIR..."
    git clone https://github.com/vllm-project/vllm.git "$VLLM_TMP_DIR"
    
    ORIGINAL_PWD=$(pwd) # Should be $REPO_DIR
    cd "$VLLM_TMP_DIR"
    echo "Building and installing vLLM from source (this may take 5-10 minutes)..."
    # Optional: export MAX_JOBS=6 # To limit compilation jobs
    # Optional: export VLLM_INSTALL_PUNICA_KERNELS=1 # For multi-LoRA capability
    pip install -e .
    cd "$ORIGINAL_PWD" # Change back to the original directory (OpenRLHF-M)
    
    echo "Cleaning up vLLM source directory: $VLLM_TMP_DIR"
    rm -rf "$VLLM_TMP_DIR"

    echo "Installing OpenRLHF package (will use the vLLM built from source)"
    pip install . 
else
    echo "Blackwell GPU not detected."
    echo "Installing OpenRLHF with the latest vLLM extra specified by OpenRLHF."
    pip install .[vllm_latest] 
fi

# Install ray with default extras to ensure dashboard dependencies
echo "Installing ray[default]"
pip install 'ray[default]'

# Install flash-attention
echo "Installing flash-attn with no build isolation"
pip install flash-attn --no-build-isolation


echo "Setup complete."

# Verify installed versions
echo "Installed package versions:"
python -c "import torch; print(f'torch: {torch.__version__}')"
python -c "import vllm; print(f'vllm: {vllm.__version__}')"
python -c "import flash_attn; print(f'flash-attn: {flash_attn.__version__}')"
python -c "import ray; print(f'ray: {ray.__version__}')"