#!/bin/bash
#=============================================================================
# Start Ray Head Node for Testing
#=============================================================================
# This script starts a Ray head node with proper networking configuration
# for testing connectivity between nodes.
#=============================================================================

# Get hostname
NODE_HOSTNAME=$(hostname)
RAY_TEMP_DIR="/data/cache-ray-${NODE_HOSTNAME}"

# Add HEAD_POD_ID check and set HEAD_NODE_IP
if [ -z "${HEAD_POD_ID}" ]; then
  echo "ERROR: HEAD_POD_ID environment variable is not set."
  exit 1
fi
export HEAD_NODE_IP="${HEAD_POD_ID}.runpod.internal"
export RAY_PORT="6379"
export DASHBOARD_PORT="8265"
echo "Using head node internal DNS: ${HEAD_NODE_IP}"

# Stop any existing Ray processes
echo "Stopping any existing Ray processes..."
ray stop --force || true
sleep 2

# Clean up Ray temp directory
echo "Creating Ray temp directory: ${RAY_TEMP_DIR}"
rm -rf "${RAY_TEMP_DIR}" || true
mkdir -p "${RAY_TEMP_DIR}"

# Start Ray head node
echo "Starting Ray head node with Global Networking on ${HEAD_NODE_IP}"
ray start --head --node-ip-address="${HEAD_NODE_IP}" --port="${RAY_PORT}" --dashboard-port="${DASHBOARD_PORT}" --dashboard-host="0.0.0.0" --num-gpus=2 --temp-dir="${RAY_TEMP_DIR}"

# Check Ray status
echo "Checking Ray status..."
ray status

echo
echo "Ray head node is running. Worker nodes can join using:"
echo "ray start --address=${HEAD_NODE_IP}:${RAY_PORT} --num-gpus=2"
echo
echo "To test connectivity from worker nodes, run:"
echo "bash examples/scripts/tests/check_runpod_networking.sh"
echo
echo "To stop Ray:"
echo "ray stop" 