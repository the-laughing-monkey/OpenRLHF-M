#!/bin/bash
# NCCL Ray Test Script
#
# This script tests NCCL communication across nodes using a baseline NCCL AllReduce test.
# It is intended to be run as a Ray job to verify connectivity between a head node and a worker node.
#
# Usage:
#   ./nccl_ray_test.sh -i interfaces HEAD_NODE WORKER_NODE
#
# Options:
#   -i interfaces : (Optional) Comma-separated list of network interfaces for NCCL (e.g., "lo,eth0,podnet1").
#                  If not specified, it defaults to "lo,eth0,podnet1".
#
# Notes:
#   - HEAD_NODE and WORKER_NODE should be the hostnames or IP addresses of the head and worker nodes, respectively.
#   - The script will automatically clone and build the nccl-tests suite if it is not found at "./nccl-tests/build/all_reduce_perf".
#     The repository is cloned from https://github.com/NVIDIA/nccl-tests and built with 'make MPI=1 NCCL_HOME=/usr/local/nccl'.
#   - It is recommended to set relevant NCCL environment variables before running this test (e.g., NCCL_SOCKET_FAMILY=IPv4, NCCL_DEBUG=TRACE).
#   - The mpirun command has been modified to disable SSH strict host key verification by passing the argument:
#         -mca plm_rsh_args "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
#     This avoids host key verification issues when launching remote processes.
#
# Example:
#   ./nccl_ray_test.sh -i lo,eth0,podnet1 headnode.example.com workernode.example.com
#

# Default value for NCCL interfaces
interfaces="eth0,podnet1"

# Parse options
while getopts ":i:" opt; do
    case $opt in
        i)
            interfaces="$OPTARG"
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            exit 1
            ;;
    esac
done
shift $((OPTIND - 1))

# Check for HEAD_NODE and WORKER_NODE arguments.
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 -i interfaces HEAD_NODE WORKER_NODE"
    exit 1
fi

HEAD_NODE=$1
WORKER_NODE=$2

# If no interfaces specified, use default.
if [ -z "$interfaces" ]; then
    interfaces="lo,eth0,podnet1"
fi

echo "==========================================================="
echo "NCCL Ray Test Script"
echo "Using NCCL interfaces: $interfaces"
echo "HEAD_NODE: $HEAD_NODE"
echo "WORKER_NODE: $WORKER_NODE"
echo "==========================================================="

# Check if nccl-tests is built; if not, clone and build it automatically.
if [ ! -f "./nccl-tests/build/all_reduce_perf" ]; then
    echo "nccl-tests not found at ./nccl-tests/build/all_reduce_perf."
    echo "Cloning and building nccl-tests automatically..."
    if [ ! -d "./nccl-tests" ]; then
        git clone https://github.com/NVIDIA/nccl-tests.git
        if [ $? -ne 0 ]; then
            echo "Failed to clone nccl-tests from GitHub."
            exit 1
        fi
    fi
    cd nccl-tests
    make MPI=1 NCCL_HOME=/usr/local/nccl
    if [ $? -ne 0 ]; then
        echo "Failed to build nccl-tests."
        exit 1
    fi
    cd ..
fi

# Allow mpirun to run as root by setting these environment variables:
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Run the NCCL AllReduce test using mpirun.
echo "Running NCCL AllReduce test..."
mpirun -np 2 \
    -mca plm_rsh_args "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
    -H ${HEAD_NODE}:1,${WORKER_NODE}:1 \
    -x NCCL_SOCKET_IFNAME="$interfaces" \
    ./nccl-tests/build/all_reduce_perf -b 8 -e 256M -f 2 -g 1
