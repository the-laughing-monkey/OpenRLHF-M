#!/bin/bash
# NCCL Ray Cluster Test Script with DNS resolution support
#
# Usage:
#   ./nccl_ray_test.sh [-i interface] [-d] HEAD_NODE WORKER_NODE
#
#   -i interface : Specify the network interface to be used by NCCL 
#                  (default is eth0)
#   -d           : Enable DNS resolution test mode (i.e. when passing hostnames).
#
# HEAD_NODE and WORKER_NODE can be IP addresses or hostnames.
#
# This script will:
#   1) Set NCCL environment variables (including NCCL_SOCKET_IFNAME)
#   2) Optionally perform DNS tests to verify resolution of hostnames
#   3) Verify GPU visibility and topology on the local node
#   4) Clone and build NCCL tests if not already present
#   5) Run the NCCL AllReduce performance test via mpirun across the two nodes
#   6) Optionally check the Ray cluster status
#
# References:
#   - [NVIDIA NCCL Troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting.html)
#   - [NCCL and Multiple NICs](https://github.com/NVIDIA/nccl/issues/412)

usage="Usage: $0 [-i interface] [-d] HEAD_NODE WORKER_NODE"
dns_enabled=0
interface="eth0,podnet1"

# Parse options
while getopts ":i:d" opt; do
    case $opt in
        i)
            interface="$OPTARG"
            ;;
        d)
            dns_enabled=1
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            echo "$usage"
            exit 1
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            echo "$usage"
            exit 1
            ;;
    esac
done
shift $((OPTIND - 1))

if [ "$#" -ne 2 ]; then
    echo "HEAD_NODE and WORKER_NODE are required."
    echo "$usage"
    exit 1
fi

HEAD_NODE=$1
WORKER_NODE=$2

# Export NCCL environment variables
export NCCL_DEBUG=TRACE
export NCCL_IB_HCA=mlx5
export NCCL_SOCKET_IFNAME=${interface}
# Optionally, force IPv4 for network communications:
export NCCL_SOCKET_FAMILY=IPv4

echo "========================================"
echo "Using network interface: ${interface}"
echo "HEAD_NODE: ${HEAD_NODE}"
echo "WORKER_NODE: ${WORKER_NODE}"
echo "========================================"

# If DNS mode enabled, perform a DNS resolution test.
if [ "$dns_enabled" -eq 1 ]; then
    echo "Performing DNS resolution tests..."
    HEAD_IP_RES=$(getent hosts $HEAD_NODE | awk '{ print $1 }')
    WORKER_IP_RES=$(getent hosts $WORKER_NODE | awk '{ print $1 }')
    echo "HEAD_NODE ($HEAD_NODE) resolves to: ${HEAD_IP_RES}"
    echo "WORKER_NODE ($WORKER_NODE) resolves to: ${WORKER_IP_RES}"
    echo "Detailed nslookup of HEAD_NODE:"
    nslookup $HEAD_NODE
    echo "Detailed nslookup of WORKER_NODE:"
    nslookup $WORKER_NODE
fi

echo ""
echo "=== GPU Verification ==="
nvidia-smi
echo ""
nvidia-smi topo -m

# Clone and build NCCL tests if necessary
if [ ! -d nccl-tests ]; then
    echo "Cloning nccl-tests repository..."
    git clone https://github.com/NVIDIA/nccl-tests.git
    if [ $? -ne 0 ]; then
        echo "Failed to clone NCCL tests repository."
        exit 1
    fi
    echo "Building nccl-tests..."
    cd nccl-tests
    make MPI=1 NCCL_HOME=/usr/local/nccl
    cd ..
fi

# Run NCCL AllReduce test across nodes using mpirun.
echo "=== Running NCCL AllReduce Test across nodes ==="
mpirun -np 2 \
    -H ${HEAD_NODE}:1,${WORKER_NODE}:1 \
    -x NCCL_DEBUG \
    -x NCCL_IB_HCA \
    -x NCCL_SOCKET_IFNAME \
    -x NCCL_SOCKET_FAMILY \
    ./nccl-tests/build/all_reduce_perf -b 8 -e 256M -f 2 -g 1

# Optionally, verify the Ray cluster communication.
echo "=== Ray Cluster Check ==="
# Assuming the Ray head node is available at port 6379 on HEAD_NODE.
ray start --address=${HEAD_NODE}:6379
sleep 5
ray status
ray list nodes
