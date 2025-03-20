#!/bin/bash
# Simple Ray Startup Script for Multi-Node Training and NCCL Testing
# 
# This script detects whether the node is a head or a worker node based on the RAY_WORKER env variable.
#   - If head (RAY_WORKER is unset or "0"):
#       • Stops any existing Ray instance.
#       • Starts Ray as the head node on port 6379 (with dashboard at port 8265).
#       • Runs 'ray status' to display cluster connectivity.
#       • Derives HEAD_NODE DNS from HEAD_POD_ID.
#       • Submits the standalone NCCL test job using Ray.
#   - If worker (RAY_WORKER is "1"):
#       • Stops any existing Ray instance.
#       • Waits for the head node (derived from HEAD_POD_ID) to be reachable.
#       • Joins the Ray cluster, explicitly specifying its own IP address.
#
# Required environment variables:
#   HEAD_POD_ID       -- The head node's pod id (used to derive its DNS as <HEAD_POD_ID>.runpod.internal)
#   RAY_WORKER        -- Set to 1 for worker nodes; head node if unset or "0".
#   NCCL_SOCKET_IFNAME -- (Optional) Comma-separated list of interfaces (e.g. "lo,eth0,podnet1").

# Determine if this is a head node.
if [ -z "$RAY_WORKER" ] || [ "$RAY_WORKER" = "0" ]; then
    echo "Running as HEAD NODE."

    # Stop any existing Ray instances.
    ray stop

    # Start Ray head node.
    echo "Starting Ray head node..."
    MY_IP=$(hostname -I | awk '{print $1}')
    echo "Local node IP: ${MY_IP}"
    ray start --head --node-ip-address ${MY_IP} --port=6379 --dashboard-port=8265
    sleep 5  # Allow the cluster to stabilize.
    
    echo "Ray cluster status:"
    ray status

    # Verify that HEAD_POD_ID is set.
    if [ -z "$HEAD_POD_ID" ]; then
        echo "Error: HEAD_POD_ID must be set for head node."
        exit 1
    fi

    # Derive the head node DNS from HEAD_POD_ID.
    HEAD_NODE="${HEAD_POD_ID}.runpod.internal"
    echo "HEAD_NODE DNS set to: ${HEAD_NODE}"

    # (Optional) Set a default for NCCL_SOCKET_IFNAME if not already set.
    if [ -z "$NCCL_SOCKET_IFNAME" ]; then
        export NCCL_SOCKET_IFNAME="lo,eth0,podnet1"
    fi
    echo "Using NCCL interfaces: ${NCCL_SOCKET_IFNAME}"
    
    # Compile the standalone NCCL test if it doesn't exist
    if [ ! -f "./standalone_nccl_test" ]; then
        echo "Compiling standalone NCCL test..."
        bash examples/scripts/tests/ray_tests/setup_test_nccl_env.sh
        if [ $? -ne 0 ]; then
            echo "Failed to compile standalone NCCL test."
            exit 1
        fi
    fi

    # Submit the standalone NCCL test job via Ray
    echo "Submitting standalone NCCL test job..."
    bash examples/scripts/tests/ray_tests/run_standalone_nccl_test.sh
else
    echo "Running as WORKER NODE."

    # Verify that HEAD_POD_ID is set.
    if [ -z "$HEAD_POD_ID" ]; then
        echo "Error: HEAD_POD_ID must be set for worker nodes."
        exit 1
    fi
    HEAD_NODE="${HEAD_POD_ID}.runpod.internal"
    echo "HEAD_NODE DNS derived as: ${HEAD_NODE}"

    # Stop any existing Ray instances.
    ray stop

    # Wait until the Ray head node is reachable.
    echo "Waiting for head node $HEAD_NODE:6379 to become reachable..."
    while ! nc -z $HEAD_NODE 6379; do
        echo "Head node not yet reachable, waiting 5 seconds..."
        sleep 5
    done

    echo "Head node is reachable. Starting Ray worker..."
    # Get local IP address of the worker node.
    MY_IP=$(hostname -I | awk '{print $1}')
    echo "Local node IP: $MY_IP"

    ray start --address=$HEAD_NODE:6379 --node-ip-address=$MY_IP

    echo "Ray worker successfully joined the cluster."

    echo "Ray cluster status:"
    ray status
    
    # Compile the standalone NCCL test on this worker node too
    if [ ! -f "./standalone_nccl_test" ]; then
        echo "Compiling standalone NCCL test on worker node..."
        bash examples/scripts/tests/ray_tests/setup_test_nccl_env.sh
        if [ $? -ne 0 ]; then
            echo "Failed to compile standalone NCCL test on worker node."
            exit 1
        fi
    fi
    
    echo "Worker node setup complete and ready for Ray tasks."
fi