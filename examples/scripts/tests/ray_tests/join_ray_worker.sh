#!/bin/bash
#=============================================================================
# Join Ray Worker Node to Cluster for Testing
#=============================================================================
# This script joins a worker node to an existing Ray cluster.
#=============================================================================

# Get hostname
NODE_HOSTNAME=$(hostname)
RAY_TEMP_DIR="/data/cache-ray-${NODE_HOSTNAME}"

# Check for HEAD_POD_ID
if [ -z "${HEAD_POD_ID}" ]; then
  echo "ERROR: HEAD_POD_ID environment variable is not set."
  echo "Please set it to the head node's pod ID from RunPod dashboard."
  echo "Example: export HEAD_POD_ID=abc123"
  exit 1
fi

# Set up internal DNS name for head node
export HEAD_NODE_IP="${HEAD_POD_ID}.runpod.internal"
export RAY_PORT="6379"
echo "Using head node internal DNS: ${HEAD_NODE_IP}"

# Stop any existing Ray processes
echo "Stopping any existing Ray processes..."
ray stop --force || true
sleep 2

# Clean up Ray temp directory
echo "Creating Ray temp directory: ${RAY_TEMP_DIR}"
rm -rf "${RAY_TEMP_DIR}" || true
mkdir -p "${RAY_TEMP_DIR}"

# First, run the networking test to verify connectivity
echo "Testing connectivity to head node..."
bash examples/scripts/tests/check_runpod_networking.sh

# Ask for confirmation before joining
echo
echo "Do you want to join the Ray cluster at ${HEAD_NODE_IP}:${RAY_PORT}? (y/n)"
read -r response
if [[ ! "$response" =~ ^[Yy]$ ]]; then
  echo "Aborting. Fix connectivity issues first."
  exit 0
fi

# Join Ray cluster
echo "Joining Ray cluster at ${HEAD_NODE_IP}:${RAY_PORT}..."
ray start --address="${HEAD_NODE_IP}:${RAY_PORT}" \
  --num-gpus=2 \
  --temp-dir="${RAY_TEMP_DIR}"

# Check Ray status
echo "Checking Ray status..."
ray status

echo
echo "Worker node has joined the Ray cluster."
echo "To stop Ray:"
echo "ray stop" 