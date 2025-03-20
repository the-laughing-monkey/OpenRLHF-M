#!/bin/bash

# Simple script to verify CUDA, NCCL environment and compile standalone NCCL test

# Test if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Please ensure CUDA toolkit is installed and nvcc is in your PATH."
    exit 1
fi

# Check CUDA version
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
echo "Found CUDA version: $CUDA_VERSION"

# Check for NCCL
NCCL_FOUND=0
if [ -d "/usr/local/nccl" ]; then
    echo "Found NCCL in /usr/local/nccl"
    export NCCL_HOME="/usr/local/nccl"
    NCCL_FOUND=1
elif [ -f "/usr/include/nccl.h" ]; then
    echo "Found NCCL in /usr/include"
    NCCL_FOUND=1
elif [ -f "/usr/local/cuda/include/nccl.h" ]; then
    echo "Found NCCL in /usr/local/cuda/include"
    NCCL_FOUND=1
fi

if [ $NCCL_FOUND -eq 0 ]; then
    echo "Warning: Could not find NCCL in standard locations."
    echo "If you have NCCL installed in a non-standard location, please set NCCL_HOME environment variable."
    echo "Continuing anyway, compilation may fail if NCCL is not available."
fi

# Create a simple CUDA test program to verify CUDA works
echo "Compiling a simple CUDA test program..."
cat << 'EOF' > test_cuda.cu
#include <stdio.h>

__global__ void hello_kernel() {
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Found %d CUDA devices\n", deviceCount);
    
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s\n", i, prop.name);
    }
    
    printf("CUDA test successful!\n");
    return 0;
}
EOF

# Compile the test program
nvcc test_cuda.cu -o test_cuda
if [ $? -ne 0 ]; then
    echo "Failed to compile the CUDA test program. Please check your CUDA installation."
    rm -f test_cuda.cu test_cuda
    exit 1
fi

# Run the test program
./test_cuda
if [ $? -ne 0 ]; then
    echo "The CUDA test program failed to run. There might be issues with your CUDA environment."
    rm -f test_cuda.cu test_cuda
    exit 1
else
    echo "CUDA test program compiled and ran successfully."
fi

rm -f test_cuda.cu test_cuda

# Compile the standalone NCCL test
echo "Compiling standalone NCCL test..."

# Create the source directory if it doesn't exist
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$SCRIPT_DIR/standalone_nccl_test.cu" ]; then
    echo "Error: $SCRIPT_DIR/standalone_nccl_test.cu not found."
    echo "Please ensure the standalone NCCL test source file exists."
    exit 1
fi

# Compile the standalone test
NCCL_INCLUDE=""
NCCL_LIB=""

if [ -n "$NCCL_HOME" ]; then
    echo "Using NCCL_HOME: $NCCL_HOME"
    NCCL_INCLUDE="-I$NCCL_HOME/include"
    NCCL_LIB="-L$NCCL_HOME/lib"
fi

nvcc -o standalone_nccl_test "$SCRIPT_DIR/standalone_nccl_test.cu" $NCCL_INCLUDE $NCCL_LIB -lnccl -lcudart

if [ $? -ne 0 ]; then
    echo "Failed to compile standalone NCCL test. Check the build logs for errors."
    exit 1
fi

# Verify the build output
if [ ! -f "standalone_nccl_test" ]; then
    echo "Compilation completed but standalone_nccl_test binary not found."
    exit 1
fi

echo "Standalone NCCL test compiled successfully. Your environment is ready for NCCL testing."
echo "Run the test with: ./standalone_nccl_test" 