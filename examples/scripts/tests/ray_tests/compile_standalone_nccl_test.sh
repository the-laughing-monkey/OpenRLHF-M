#!/bin/bash

# Script to compile the standalone NCCL test

# Check if nvcc is available
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc (NVIDIA CUDA Compiler) not found. Please ensure CUDA toolkit is installed."
    exit 1
fi

# Check if NCCL is available
if [ ! -d "/usr/local/nccl" ] && [ ! -f "/usr/include/nccl.h" ] && [ ! -d "/usr/local/cuda/include/nccl.h" ]; then
    echo "Warning: NCCL may not be installed in standard locations. If compilation fails, set NCCL_HOME env variable."
fi

# Set NCCL_HOME if provided as an environment variable
NCCL_INCLUDE=""
NCCL_LIB=""

if [ -n "$NCCL_HOME" ]; then
    echo "Using NCCL_HOME: $NCCL_HOME"
    NCCL_INCLUDE="-I$NCCL_HOME/include"
    NCCL_LIB="-L$NCCL_HOME/lib"
elif [ -d "/usr/local/nccl" ]; then
    echo "Using NCCL from /usr/local/nccl"
    NCCL_INCLUDE="-I/usr/local/nccl/include"
    NCCL_LIB="-L/usr/local/nccl/lib"
fi

echo "Compiling standalone NCCL test..."
nvcc -o standalone_nccl_test examples/scripts/tests/ray_tests/standalone_nccl_test.cu $NCCL_INCLUDE $NCCL_LIB -lnccl -lcudart

if [ $? -ne 0 ]; then
    echo "Failed to compile standalone NCCL test. Please check errors above."
    exit 1
fi

echo "Standalone NCCL test compiled successfully: standalone_nccl_test"
echo "Run with: ./standalone_nccl_test"
exit 0 