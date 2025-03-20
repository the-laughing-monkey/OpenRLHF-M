#!/bin/bash
# Script to run the standalone NCCL test on Ray

# Check for Ray address
if [ -z "$RAY_ADDRESS" ]; then
    if [ -z "$HEAD_POD_ID" ]; then
        echo "Error: Either RAY_ADDRESS or HEAD_POD_ID must be set."
        exit 1
    fi
    RAY_ADDRESS="http://${HEAD_POD_ID}.runpod.internal:8265"
fi

# Ensure the standalone NCCL test is compiled
if [ ! -f "./standalone_nccl_test" ]; then
    echo "Standalone NCCL test binary not found. Compiling..."
    bash examples/scripts/tests/ray_tests/compile_standalone_nccl_test.sh
    if [ $? -ne 0 ]; then
        echo "Failed to compile standalone NCCL test."
        exit 1
    fi
fi

# Set NCCL_SOCKET_IFNAME from environment or use default
if [ -z "$NCCL_SOCKET_IFNAME" ]; then
    export NCCL_SOCKET_IFNAME="lo,eth0,podnet1"
fi
echo "Using NCCL network interfaces: $NCCL_SOCKET_IFNAME"

# Submit the job to Ray using the Python script
echo "Submitting standalone NCCL test job to Ray..."
ray job submit --address="$RAY_ADDRESS" --no-wait -- python examples/scripts/tests/ray_tests/nccl_ray_test.py

echo "Job submitted to Ray. Check Ray dashboard for results." 